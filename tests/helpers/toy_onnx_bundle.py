"""Build a tiny deterministic ONNX bundle for full-depth beam-search tests.

The two level models deliberately encode a non-greedy decision:

* level 1 offers ``pi0`` at confidence 0.80 and ``K_S0`` at 0.70 from the
  same two leaves;
* level 2 gives the pi0 branch a root confidence of 0.10, but the K_S0
  branch a root confidence of 0.95.

A width-two beam must therefore retain both branches and select K_S0 even
though it was not the locally best first-level prediction.  The files are
small standards-compliant ONNX graphs, so the same bundle can be executed by
``onnx.reference.ReferenceEvaluator`` in unit tests and ONNX Runtime inside
the CVMFS basf2 release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from hypertagging.deployment.onnx_contract import (
    BUNDLE_FORMAT_VERSION,
    FEATURE_NAMES,
    MODEL_FAMILY,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
    file_sha256,
    with_manifest_hash,
)


# Include charge-conjugate pion hypotheses so the same deterministic bundle
# can exercise reconstructed Track feature extraction in the basf2 smoke.
PID_TOKENS = (0, 300553, 111, 310, 211, -211)
MAX_NODES = 8
MAX_SOURCES = 4
N_QUERIES = 2
MAX_CARDINALITY = 4


def _logit(probability: float) -> np.float32:
    return np.float32(math.log(probability / (1.0 - probability)))


def _constant(name: str, value: np.ndarray, output: str | None = None) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output or name],
        name=name,
        value=numpy_helper.from_array(np.asarray(value), name=f"{name}_value"),
    )


def _value_info(name: str, shape: Sequence[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _base_outputs() -> dict[str, np.ndarray]:
    return {
        "object_logits": np.full((1, N_QUERIES), -12.0, dtype=np.float32),
        "type_logits": np.full(
            (1, N_QUERIES, len(PID_TOKENS)), -12.0, dtype=np.float32
        ),
        "pointer_logits": np.full(
            (1, N_QUERIES, MAX_NODES), -12.0, dtype=np.float32
        ),
        "cardinality_logits": np.full(
            (1, N_QUERIES, MAX_CARDINALITY + 1), -12.0, dtype=np.float32
        ),
        "confidence_logits": np.full((1, N_QUERIES), -12.0, dtype=np.float32),
        "leaf_pid_logits": np.zeros(
            (1, MAX_NODES, len(PID_TOKENS)), dtype=np.float32
        ),
        "current_p4": np.zeros((1, MAX_NODES, 4), dtype=np.float32),
    }


def _make_model(
    *,
    name: str,
    inputs: Iterable[onnx.ValueInfoProto],
    nodes: Sequence[onnx.NodeProto],
) -> onnx.ModelProto:
    output_shapes = {
        "object_logits": (1, N_QUERIES),
        "type_logits": (1, N_QUERIES, len(PID_TOKENS)),
        "pointer_logits": (1, N_QUERIES, MAX_NODES),
        "cardinality_logits": (1, N_QUERIES, MAX_CARDINALITY + 1),
        "confidence_logits": (1, N_QUERIES),
        "leaf_pid_logits": (1, MAX_NODES, len(PID_TOKENS)),
        "current_p4": (1, MAX_NODES, 4),
    }
    graph = helper.make_graph(
        list(nodes),
        name,
        list(inputs),
        [_value_info(output, output_shapes[output]) for output in MODEL_OUTPUT_NAMES],
    )
    model = helper.make_model(
        graph,
        producer_name="hypertagging-toy-test-bundle",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Keep compatibility with the ONNX Runtime version bundled by the selected
    # basf2 light release; the graph uses no newer IR feature.
    model.ir_version = 8
    checker.check_model(model, full_check=True)
    return model


def _level_one_model() -> onnx.ModelProto:
    outputs = _base_outputs()
    outputs["object_logits"][0, :] = _logit(0.99)
    outputs["type_logits"][0, 0, 2] = 12.0  # pi0
    outputs["type_logits"][0, 1, 3] = 12.0  # K_S0
    outputs["pointer_logits"][0, :, :2] = 12.0
    outputs["cardinality_logits"][0, :, 2] = 12.0
    outputs["confidence_logits"][0, 0] = _logit(0.80)
    outputs["confidence_logits"][0, 1] = _logit(0.70)
    nodes = [_constant(f"level1_{name}", outputs[name], name) for name in MODEL_OUTPUT_NAMES]
    # Keeping one genuine runtime input exercises manifest/session agreement;
    # constant test predictions intentionally do not depend on its value.
    input_info = helper.make_tensor_value_info(
        "node_mask", TensorProto.BOOL, [1, MAX_NODES]
    )
    return _make_model(name="toy_level_1", inputs=[input_info], nodes=nodes)


def _level_two_model() -> onnx.ModelProto:
    outputs = _base_outputs()
    outputs["object_logits"][0, 0] = _logit(0.99)
    outputs["type_logits"][0, :, 1] = 12.0  # Upsilon(4S) root
    outputs["pointer_logits"][0, 0, 2:5] = 12.0
    outputs["cardinality_logits"][0, 0, 3] = 12.0

    nodes: list[onnx.NodeProto] = []
    for name in MODEL_OUTPUT_NAMES:
        if name == "confidence_logits":
            continue
        nodes.append(_constant(f"level2_{name}", outputs[name], name))

    nodes.extend(
        [
            _constant("level2_position", np.asarray(4, dtype=np.int64), "position"),
            helper.make_node(
                "Gather",
                inputs=["pid_labels", "position"],
                outputs=["branch_token"],
                name="read_level_one_mother_token",
                axis=1,
            ),
            _constant("level2_ks_token", np.asarray(3, dtype=np.int64), "ks_token"),
            helper.make_node(
                "Equal",
                inputs=["branch_token", "ks_token"],
                outputs=["is_ks_branch_flat"],
                name="is_ks_branch",
            ),
            _constant("level2_axes", np.asarray([1], dtype=np.int64), "axes"),
            helper.make_node(
                "Unsqueeze",
                inputs=["is_ks_branch_flat", "axes"],
                outputs=["is_ks_branch"],
                name="make_branch_condition_2d",
            ),
            _constant(
                "level2_ks_confidence",
                np.asarray([[_logit(0.95), _logit(0.001)]], dtype=np.float32),
                "ks_confidence",
            ),
            _constant(
                "level2_pi0_confidence",
                np.asarray([[_logit(0.10), _logit(0.001)]], dtype=np.float32),
                "pi0_confidence",
            ),
            helper.make_node(
                "Where",
                inputs=["is_ks_branch", "ks_confidence", "pi0_confidence"],
                outputs=["confidence_logits"],
                name="branch_dependent_root_confidence",
            ),
        ]
    )
    input_info = helper.make_tensor_value_info(
        "pid_labels", TensorProto.INT64, [1, MAX_NODES]
    )
    return _make_model(name="toy_level_2", inputs=[input_info], nodes=nodes)


def build_toy_onnx_bundle(directory: str | Path) -> Path:
    """Create the deterministic two-level bundle and return its manifest path."""

    bundle_dir = Path(directory)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    models = {1: _level_one_model(), 2: _level_two_model()}
    model_entries: dict[str, dict[str, object]] = {}
    for level, model in models.items():
        model_path = bundle_dir / f"toy_level_{level}.onnx"
        onnx.save_model(model, model_path)
        model_entries[str(level)] = {
            "file": model_path.name,
            "sha256": file_sha256(model_path),
            "runtime_inputs": [value.name for value in model.graph.input],
        }

    zeros = lambda width: [0.0] * width
    ones = lambda width: [1.0] * width
    payload = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "contract": {
            "max_nodes": MAX_NODES,
            "max_sources": MAX_SOURCES,
            "n_queries": N_QUERIES,
            "max_cardinality": MAX_CARDINALITY,
            "levels": [1, 2],
            "pid_tokens": list(PID_TOKENS),
            "feature_names": {
                block: list(names) for block, names in FEATURE_NAMES.items()
            },
            "model_input_names": list(MODEL_INPUT_NAMES),
            "model_output_names": list(MODEL_OUTPUT_NAMES),
        },
        "models": model_entries,
        "normalization": {
            block: {
                "mean": zeros(len(FEATURE_NAMES[block])),
                "standard_deviation": ones(len(FEATURE_NAMES[block])),
            }
            for block in ("common", "track", "cluster", "composite")
        },
        "reconstruction_policy": {
            "beam_score": "sum_confidence",
            "beam_width": 2,
            "max_proposals_per_hypothesis": 2,
            "minimum_daughters": 2,
            "object_threshold": 0.5,
            "pointer_threshold": 0.5,
            "confidence_threshold": 0.05,
            "type_probability_threshold": 0.5,
            "daughter_cardinality_policy": "predicted",
            "cardinality_insufficient_policy": "invalid",
            "use_learned_confidence": True,
            "reject_recursive_source_conflicts": True,
            "require_lower_level_context": True,
            "root_tokens": [1],
            "root_requires_all_sources": True,
            "static_allowed_mother_tokens": [1, 2, 3],
            "allowed_mother_types_by_level": [[1, [2, 3]], [2, [1]]],
            "empirical_type_prior_mode": "hard",
            "initial_state_policy": "upsilon4s",
            "token_charge": [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            "mother_charge_compatibility": "hard",
            "mother_charge_tolerance": 1e-6,
            "valid_leaf_node_kinds": [1, 2, 4, 5],
            "valid_composite_node_kinds": [3],
            "allow_fixed_hypothesis_unknown_kind": True,
            "loose_physical_constraints": [],
        },
        "runtime": {
            "basf2_releases": ["light-2607-kasei"],
            "providers": ["CPUExecutionProvider"],
        },
        "test_fixture": {
            "description": "non-greedy two-level full-decay beam fixture",
            "locally_best_first_token": 2,
            "globally_best_first_token": 3,
        },
    }
    manifest = with_manifest_hash(payload)
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


@dataclass
class ReferenceSession:
    """Small InferenceSession-compatible wrapper around ONNX's reference VM."""

    filename: str

    def __post_init__(self) -> None:
        self.model = onnx.load(self.filename)
        checker.check_model(self.model, full_check=True)
        self.evaluator = ReferenceEvaluator(self.model)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=value.name) for value in self.model.graph.input]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=value.name) for value in self.model.graph.output]

    def run(
        self, output_names: Sequence[str], feeds: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        return [np.asarray(value) for value in self.evaluator.run(list(output_names), feeds)]


def reference_session_factory(filename: str, _options: object) -> ReferenceSession:
    return ReferenceSession(filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(build_toy_onnx_bundle(args.output_directory))


if __name__ == "__main__":
    main()
