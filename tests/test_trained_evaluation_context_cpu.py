from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hypertagging.data.dataset_index import build_dataset_index, load_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.evaluation import load_trained_evaluation_context
from hypertagging.evaluation.trained_context import (
    _evaluation_split_assignment,
    _preflight_evaluation_data_binding,
    _require_source_role_manifest_binding,
    _resolve_checkpoint_event_selection,
    _select_evaluation_events,
)
from hypertagging.models.ablation import build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import (
    ParquetEventWriter, feature_spec_v4, iter_event_records_v4,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import save_training_checkpoint
from hypertagging.training.data_module import build_real_data_module
from hypertagging.training.model_config import MODEL_PRESETS


def test_evaluation_split_assignment_honors_source_role_manifest():
    config = SourceAwareSplitConfig()
    event = SimpleNamespace(
        event_uid="manifest-event",
        source_file="manifest-validation.root",
        source_category="mixed",
    )
    assert _evaluation_split_assignment(
        event,
        split_config=config,
        event_split_overrides={},
        source_split_overrides={event.source_file: "validation"},
    ) == "validation"


def test_evaluation_split_assignment_honors_event_override_first():
    config = SourceAwareSplitConfig()
    event = SimpleNamespace(
        event_uid="pilot-repair-event",
        source_file="manifest-validation.root",
        source_category="mixed",
    )
    assert _evaluation_split_assignment(
        event,
        split_config=config,
        event_split_overrides={event.event_uid: "train"},
        source_split_overrides={event.source_file: "validation"},
    ) == "train"


def test_source_role_index_rejects_raw_parquet_data(tmp_path):
    index = {"selection_contract": {"mode": "source_role_manifest"}}

    with pytest.raises(ValueError, match="training-selection manifest"):
        _require_source_role_manifest_binding([tmp_path / "events.parquet"], index)


def test_trained_context_delegates_to_shared_metadata_preflight(
    tmp_path, monkeypatch
):
    import hypertagging.evaluation.trained_context as trained_context_module

    expected = {"selection_contract": {"mode": "source_role_manifest"}}
    calls = []

    def fake_preflight(data, dataset_index):
        calls.append((data, dataset_index))
        return expected

    monkeypatch.setattr(
        trained_context_module,
        "preflight_dataset_index_data_binding",
        fake_preflight,
    )
    data = tmp_path / "selection.json"
    index = tmp_path / "index.json"

    assert _preflight_evaluation_data_binding(data, index) is expected
    assert calls == [(data, index)]


def test_checkpoint_rollout_cohort_restores_uid_order_before_category_limit():
    cohort = ("uid-a", "uid-b", "uid-c", "uid-d")
    events = [
        SimpleNamespace(event_uid="uid-d", source_category="mixed"),
        SimpleNamespace(event_uid="uid-b", source_category="charged"),
        SimpleNamespace(event_uid="uid-c", source_category="mixed"),
        SimpleNamespace(event_uid="uid-a", source_category="mixed"),
    ]

    class Module:
        selection_manifest_hash = "manifest-hash"

        @staticmethod
        def iter_events(split, shuffle=False):
            assert split == "validation"
            assert not shuffle
            yield from events

    payload = {
        "validation_selection": {
            "split": "validation",
            "rollout_event_uids": list(cohort),
            "event_uids": list(cohort),
            "deterministic": True,
            "selection_manifest_hash": "manifest-hash",
        }
    }
    selected_uids, mode = _resolve_checkpoint_event_selection(
        payload,
        data_module=Module(),
        split="validation",
        requested="auto",
    )
    selected = _select_evaluation_events(
        Module(),
        split="validation",
        max_events=2,
        requested_categories={"mixed"},
        cohort_uids=selected_uids,
    )

    assert mode == "checkpoint_rollout"
    assert [event.event_uid for event in selected] == ["uid-a", "uid-c"]


def test_checkpoint_rollout_cohort_rejects_missing_uid_when_stream_exhausts():
    events = [SimpleNamespace(event_uid="uid-a", source_category="mixed")]

    class Module:
        @staticmethod
        def iter_events(split, shuffle=False):
            assert split == "validation"
            assert not shuffle
            yield from events

    with pytest.raises(ValueError, match="uid-missing"):
        _select_evaluation_events(
            Module(),
            split="validation",
            max_events=2,
            requested_categories=set(),
            cohort_uids=("uid-a", "uid-missing"),
        )


def test_trained_evaluation_restores_normalization_and_heldout_contract(tmp_path):
    source = write_notebook_fixture_v4(tmp_path / "source.parquet", row_group_size=1)
    shards = []
    for index, record in enumerate(iter_event_records_v4(source)):
        record["source_file"] = f"fixture-{index}.root"
        shard = tmp_path / f"events-{index}.parquet"
        with ParquetEventWriter(shard, event_buffer_size=1) as writer:
            writer.write_event(record)
        shards.append(shard)
    split_config = SourceAwareSplitConfig(
        train_fraction=0.5,
        validation_fraction=0.5,
        test_fraction=0.0,
        group_by_source_file=False,
        seed=1,
    )
    # Choose a deterministic seed that places the two fixture UIDs on both sides.
    for seed in range(1, 1000):
        candidate = SourceAwareSplitConfig(
            train_fraction=0.5, validation_fraction=0.5, test_fraction=0.0,
            group_by_source_file=False, seed=seed,
        )
        from hypertagging.data.splitting import stable_split_name
        assigned = {
            stable_split_name(record, candidate)
                for shard in shards for record in iter_event_records_v4(shard)
        }
        if assigned == {"train", "validation"}:
            split_config = candidate
            break
    index_path = build_dataset_index(
        shards, tmp_path / "index.json", split_config=split_config
    )
    module = build_real_data_module(
        shards, dataset_index=index_path, split_config=split_config,
        seed=split_config.seed, required_splits=("train", "validation"),
    )
    architecture = MODEL_PRESETS["tiny_cpu"]
    model = build_ablation_model(
        "full_revised", n_features=12, n_types=len(PDG_TOKENS),
        hidden_dim=architecture.d_model, hyper_dim=architecture.hyper_dim,
        n_queries=architecture.n_queries, max_cardinality=architecture.max_cardinality,
        n_heads=architecture.n_heads, n_context_layers=architecture.n_context_layers,
        curvature=architecture.curvature, ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
    )
    policy = ReconstructionConstraintPolicy(initial_state_policy="upsilon4s")
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint.pt", model=model, encoder=model.encoder,
        config={
            "ablation": "full_revised", "seed": split_config.seed,
            "target_policy": "complete_only", "max_events": None,
            "pid_kinematics_mode": None,
            "rollout_pid_kinematics_mode": "hard",
        },
        normalizer_state=module.normalization_state(),
        split_manifest_hash=module.split_manifest_hash,
        feature_contract={
            "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
            "model_feature_contract_hash": feature_spec_v4()["model_feature_contract_hash"],
            "pid_reconstruction_mode": "soft_expectation",
            "reconstruction_constraint_policy": policy.to_dict(),
        },
        data_order_contract={
            "dataset_index_hash": load_dataset_index(index_path)["index_hash"]
        },
        architecture=architecture.to_dict(),
    )
    context = load_trained_evaluation_context(
        checkpoint=checkpoint, data=shards, dataset_index=index_path,
        split="validation", max_events=2,
    )
    assert context.events
    assert context.report_metadata["evaluation_split"] == "validation"
    assert context.report_metadata["evaluated_event_uids"] == [
        event.event_uid for event in context.events
    ]
    assert context.constraint_policy.initial_state_policy == "upsilon4s"
    assert context.rollout_pid_kinematics_mode == "hard"
    assert context.pid_kinematics_mode == "soft_expectation"
    assert context.model.pid_kinematics_mode == "soft_expectation"
    assert context.report_metadata["pid_kinematics_mode"] == "soft_expectation"
    assert torch.equal(
        context.model.runtime_feature_normalizer.common_mean,
        module.normalizers["common"].mean,
    )
    batch = context.collated_batch()
    assert batch["runtime_features_are_raw"]
    assert torch.isfinite(batch["track_features"]).all()

    filtered = load_trained_evaluation_context(
        checkpoint=checkpoint,
        data=shards,
        dataset_index=index_path,
        split="validation",
        max_events=2,
        source_categories=(context.events[0].source_category,),
    )
    assert filtered.events
    assert {
        event.source_category for event in filtered.events
    } == {context.events[0].source_category}

    conflicting = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conflicting["config"]["pid_kinematics_mode"] = "hard"
    conflicting_checkpoint = tmp_path / "conflicting-pid-mode.pt"
    torch.save(conflicting, conflicting_checkpoint)
    with pytest.raises(ValueError, match="conflicts with authoritative"):
        load_trained_evaluation_context(
            checkpoint=conflicting_checkpoint,
            data=shards,
            dataset_index=index_path,
            split="validation",
            max_events=2,
        )
