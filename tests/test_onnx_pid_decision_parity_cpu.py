"""Deployment preserves rollout PID decisions and observed-zero measurements."""

from dataclasses import replace

import pytest
import torch
import numpy as np

from hypertagging.basf2_integration.runtime import (
    NODE_KIND_TRACK,
    OnnxModelBundle,
    _common_block,
)
from hypertagging.deployment.export_onnx import (
    ExportConfiguration,
    NORMALIZED_BLOCKS,
    _LevelOnnxWrapper,
    _example_inputs,
    _restore_deployment_model,
    export_checkpoint_bundle,
)
from hypertagging.deployment.onnx_contract import (
    FEATURE_NAMES,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
)
from hypertagging.models.ablation import build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import SCHEMA_VERSION_V4, feature_spec_v4
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.model_config import ModelArchitecture
from tests.test_onnx_full_decay_beam_cpu import _toy_leaves


def _checkpoint(training_mode):
    architecture = ModelArchitecture("parity_fixture", 16, 4, 2, 1, 32, 0.0, 1.0, 2)
    torch.manual_seed(319)
    model = build_ablation_model(
        "full_revised", n_features=len(FEATURE_NAMES["common"]),
        n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
        n_queries=2, max_cardinality=6, n_heads=2, n_context_layers=1,
        ffn_dim=32, pid_kinematics_mode=training_mode,
        type_conditioned_daughter_relation_bias=False,
    )
    specification = feature_spec_v4()
    return {
        "preprocessing_schema_version": SCHEMA_VERSION_V4,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_specification": specification,
        "runtime_model_contracts": specification["runtime_model_contracts"],
        "feature_contract": {
            "model_feature_contract_hash": specification["model_feature_contract_hash"],
            "pid_reconstruction_mode": training_mode,
            "reconstruction_constraint_policy": ReconstructionConstraintPolicy().to_dict(),
        },
        "config": {
            "pid_kinematics_mode": training_mode,
            "rollout_pid_kinematics_mode": "soft_decision_hard_construction",
            "rollout_pid_temperature": 0.5,
        },
        "architecture": architecture.to_dict(),
        "model_state_dict": model.state_dict(),
        "data_compatible_performance": True,
        "normalizer_state": {
            block: {
                "count": torch.ones(len(FEATURE_NAMES[block])),
                "mean": torch.zeros(len(FEATURE_NAMES[block])),
                "standard_deviation": torch.ones(len(FEATURE_NAMES[block])),
            }
            for block in NORMALIZED_BLOCKS
        },
    }


@pytest.mark.parametrize("training_mode", [
    "soft_expectation", "hard", "straight_through_hard", "temperature_softmax",
])
def test_export_uses_rollout_decisions_and_preserves_training_mode(training_mode):
    payload = _checkpoint(training_mode)
    model, _, policy, decision_mode, temperature = _restore_deployment_model(payload)
    assert model.pid_kinematics_mode == training_mode
    assert payload["feature_contract"]["pid_reconstruction_mode"] == training_mode
    assert decision_mode == "soft_expectation"
    example = _example_inputs(
        ExportConfiguration(levels=(1, 2), max_nodes=8, max_sources=8),
        policy, target_level=2,
    )
    inputs = tuple(example[name] for name in MODEL_INPUT_NAMES)
    exported = _LevelOnnxWrapper(
        model, target_level=2, pid_kinematics_mode=decision_mode,
        pid_temperature=temperature,
    )
    offline_rollout = _LevelOnnxWrapper(
        model, target_level=2, pid_kinematics_mode="soft_expectation",
        pid_temperature=temperature,
    )
    with torch.inference_mode():
        export_outputs = exported(*inputs)
        reference_outputs = offline_rollout(*inputs)
        for actual, expected in zip(export_outputs, reference_outputs):
            torch.testing.assert_close(actual, expected)
        # The fixture exercises a real kinematic difference, so equality above
        # cannot pass merely because raw tracks or PID refinement were absent.
        hard_outputs = _LevelOnnxWrapper(
            model, target_level=2, pid_kinematics_mode="hard",
            pid_temperature=temperature,
        )(*inputs)
    assert (export_outputs[-1] - hard_outputs[-1]).abs().max() > 1e-3


def test_common_confidence_preserves_observed_zero_and_missingness():
    leaf = _toy_leaves()[0]
    common = (0.0,) * len(FEATURE_NAMES["common"])
    measured = tuple(index == 11 for index in range(len(common)))
    explicit = replace(leaf, common_features=common, common_availability=measured)
    assert _common_block(explicit)[1][11]
    assert not _common_block(replace(explicit, common_availability=(False,) * len(common)))[1][11]
    assert not _common_block(leaf)[1][11]
    track_mask = (True,) + (False,) * (len(FEATURE_NAMES["track"]) - 1)
    track = replace(leaf, kind_id=NODE_KIND_TRACK, track_availability=track_mask)
    assert _common_block(track)[1][11]
    assert not _common_block(replace(track, track_availability=(False,) * len(track_mask)))[1][11]
    mother = replace(leaf, node_id=4, level=1, daughter_ids=(0, 1))
    assert _common_block(mother)[1][11]


def test_exported_graph_accepts_changing_track_counts_and_signs(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    payload = _checkpoint("hard")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    configuration = ExportConfiguration(levels=(1, 2), max_nodes=8, max_sources=8)
    manifest_path = export_checkpoint_bundle(
        checkpoint, tmp_path / "bundle", configuration=configuration,
    )
    bundle = OnnxModelBundle(manifest_path, intra_op_threads=2)
    assert bundle.manifest["pid_inference"]["decision_mode"] == "soft_expectation"
    model, _, policy, _, temperature = _restore_deployment_model(payload)
    # All variants run through the same graph for each level. The trace has
    # two tracks; deployed events can contain none, one, or more of either sign.
    for level in (1, 2):
        base = _example_inputs(configuration, policy, target_level=level)
        for charges in ((), (1,), (-1,), (1, -1), (1, 1), (-1, -1), (1, -1, 1, -1)):
            example = {name: value.clone() for name, value in base.items()}
            example["leaf_kinematics_mode_ids"][0, :4] = 2
            example["node_kind_ids"][0, :4] = 2
            example["pid_labels"][0, :4] = 2
            example["charge"][0, :4] = 0
            example["track_availability"][0, :4] = False
            for position, charge in enumerate(charges):
                example["leaf_kinematics_mode_ids"][0, position] = 0
                example["node_kind_ids"][0, position] = NODE_KIND_TRACK
                example["pid_labels"][0, position] = 0
                example["charge"][0, position] = charge
                example["track_availability"][0, position] = True
            example["common_features"][..., 5] = example["charge"]
            reference = _LevelOnnxWrapper(
                model, target_level=level, pid_kinematics_mode="soft_expectation",
                pid_temperature=temperature,
            )
            with torch.inference_mode():
                expected = reference(*(example[key] for key in MODEL_INPUT_NAMES))
            observed = bundle.score(level, {key: value.numpy() for key, value in example.items()})
            for output_name, value in zip(MODEL_OUTPUT_NAMES, expected):
                np.testing.assert_allclose(
                    getattr(observed, output_name), value.numpy(), rtol=3e-4, atol=3e-5,
                    err_msg=f"level={level}, charges={charges}, output={output_name}",
                )
