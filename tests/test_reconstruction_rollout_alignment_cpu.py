from __future__ import annotations

import json

import pytest
import torch

from hypertagging.data.heterogeneous import heterogeneous_from_level_event
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.training import reconstruction_trainer
from hypertagging.training.checkpoint_selection import (
    reconstruction_selection_contract,
)
from hypertagging.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from hypertagging.training.pretrain_trainer import ContextualPretrainingModel
from hypertagging.training.pretrained_transfer import load_pretrained_encoder
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    _collate_context_batches,
    _normalized_weighted_mean,
    _predicted_context_rollout_max_level,
    train_level_reconstruction,
    validate_reconstruction,
)


def test_level_weighted_mean_changes_emphasis_without_changing_scale() -> None:
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    assert _normalized_weighted_mean(losses, [1.0, 1.0]) == pytest.approx(2.0)
    assert _normalized_weighted_mean(losses, [1.0, 3.0]) == pytest.approx(2.5)


class _ValidationDataModule:
    def __init__(self) -> None:
        self.events = [heterogeneous_from_level_event(tiny_level_events()[0])]
        self.split_counts = {"validation": 1}
        self.allowed_types_by_level = {}

    def iter_events(self, *_args, **_kwargs):
        return iter(self.events)

    def normalize_batch(self, batch):
        return batch


def _validation_model() -> LevelAutoregressiveReconstructor:
    return LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=4,
        n_heads=4,
        n_context_layers=1,
    )


def test_explicit_rollout_policy_is_checkpoint_selection_identity() -> None:
    contract = reconstruction_selection_contract(
        best_metric="predicted_edge_f1",
        best_mode="max",
        max_validation_events=2000,
        rollout_validation_events=1000,
        rollout_validate_every=500,
        rollout_pid_kinematics_mode="soft_decision_hard_construction",
        rollout_pid_temperature=0.5,
        target_policy="complete_only",
        constraint_policy={},
        rollout_continue_through_empty_levels=True,
        rollout_max_level=6,
        rollout_root_types=(1,),
        rollout_exclusive_final=True,
        rollout_use_learned_confidence=True,
        rollout_object_threshold=0.35,
        rollout_pointer_threshold=0.4,
    )

    assert contract["version"] == "reconstruction-checkpoint-selection-v8"
    assert contract["rollout_validate_at_final_step"] is True
    assert contract["rollout_configuration"] == {
        "max_level": 6,
        "root_types": [1],
        "exclusive_final": True,
        "exclusive_resolution": "greedy",
        "bounded_weighted_set_packing": "diagnostic_only",
        "learned_confidence": True,
        "policy_identity": contract["rollout_configuration"]["policy_identity"],
    }
    assert contract["thresholds"]["object_probability"] == pytest.approx(0.35)
    assert contract["thresholds"]["daughter_pointer_probability"] == pytest.approx(
        0.4
    )


def test_validation_threads_one_explicit_policy_to_every_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    real_level_rollout = reconstruction_trainer.level_rollout

    def recording_level_rollout(*args, **kwargs):
        observed.append((kwargs["mode"], kwargs["config"]))
        return real_level_rollout(*args, **kwargs)

    monkeypatch.setattr(
        reconstruction_trainer, "level_rollout", recording_level_rollout
    )
    result = validate_reconstruction(
        _validation_model(),
        _ValidationDataModule(),
        device=torch.device("cpu"),
        max_validation_events=1,
        rollout_validation_events=1,
        validation_batch_size=1,
        rollout_continue_through_empty_levels=True,
        rollout_max_level=6,
        rollout_root_types=(1,),
        rollout_exclusive_final=True,
        rollout_use_learned_confidence=True,
        rollout_object_threshold=0.35,
        rollout_pointer_threshold=0.4,
    )

    assert result["rollout_validation_events"] == 1
    assert [mode for mode, _config in observed] == [
        "teacher_forced",
        "predicted",
        "predicted",
        "scheduled",
    ]
    assert all(config.max_level == 6 for _mode, config in observed)
    assert all(config.root_types == (1,) for _mode, config in observed)
    assert all(config.exclusive_final for _mode, config in observed)
    assert all(config.use_learned_confidence for _mode, config in observed)
    assert all(config.confidence_trained for _mode, config in observed)
    assert all(config.object_threshold == 0.35 for _mode, config in observed)
    assert all(config.pointer_threshold == 0.4 for _mode, config in observed)


def test_balanced_micro_rollout_stops_before_selected_level_and_obeys_cap() -> None:
    config = ReconstructionConfig(
        data="unused",
        output_dir="unused",
        level_sampling_mode="balanced_level_replay",
        rollout_max_level=6,
        rollout_root_types=(1,),
        rollout_exclusive_final=True,
        rollout_use_learned_confidence=True,
    )

    assert _predicted_context_rollout_max_level(
        config, selected_target_level=1, valid_levels=list(range(1, 7))
    ) == 0
    assert _predicted_context_rollout_max_level(
        config, selected_target_level=4, valid_levels=list(range(1, 7))
    ) == 3
    assert _predicted_context_rollout_max_level(
        config, selected_target_level=8, valid_levels=list(range(1, 9))
    ) == 6


def test_dynamic_context_collation_pads_runtime_structural_validity() -> None:
    """Reproduce the mixed 48/34-node phase-35 step-1690 collation."""

    contexts = []
    for node_count in (48, 34):
        active = torch.ones((1, node_count), dtype=torch.bool)
        structurally_valid = torch.zeros((1, node_count), dtype=torch.bool)
        structurally_valid[:, : node_count // 2] = True
        contexts.append(
            {
                "active": active,
                "node_mask": active.clone(),
                "common_features": torch.zeros((1, node_count, 12)),
                "runtime_structurally_valid": structurally_valid,
            }
        )

    batch = _collate_context_batches(contexts)

    assert batch["runtime_structurally_valid"].shape == (2, 48)
    assert batch["runtime_structurally_valid"][0, :24].all()
    assert not batch["runtime_structurally_valid"][0, 24:].any()
    assert batch["runtime_structurally_valid"][1, :17].all()
    assert not batch["runtime_structurally_valid"][1, 17:].any()
    assert not batch["node_mask"][1, 34:].any()


def test_explicit_policy_forces_rollout_validation_at_final_step(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = write_notebook_fixture_v3(tmp_path / "tiny-final-rollout.parquet")
    observed_rollout_limits: list[int] = []
    real_validate = reconstruction_trainer.validate_reconstruction

    def recording_validate(*args, **kwargs):
        observed_rollout_limits.append(int(kwargs["rollout_validation_events"]))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        reconstruction_trainer, "validate_reconstruction", recording_validate
    )
    result = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "training"),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            rollout_validate_every=2,
            rollout_validation_events=1,
            checkpoint_every=2,
            scheduled_sampling_probability=1.0,
            scheduled_sampling_duration_steps=0,
            rollout_continue_through_empty_levels=True,
            rollout_max_level=6,
            rollout_root_types=(1,),
            rollout_exclusive_final=True,
            rollout_use_learned_confidence=True,
        )
    )

    assert observed_rollout_limits == [1]
    assert result.metrics["rollout_validation_events"] == 1
    checkpoint = load_training_checkpoint(result.checkpoint)
    selection = checkpoint["training_state"]["checkpoint_selection_contract"]
    assert selection["version"] == "reconstruction-checkpoint-selection-v8"
    assert selection["rollout_validate_at_final_step"] is True
    assert checkpoint["validation_selection"]["rollout_was_run"] is True


def test_non_rollout_validation_checkpoint_metadata_is_json_finite(
    tmp_path,
) -> None:
    data = write_notebook_fixture_v3(tmp_path / "tiny-no-rollout.parquet")
    result = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "training-no-rollout"),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            rollout_validate_every=2,
            rollout_validation_events=1,
            checkpoint_every=2,
        )
    )

    checkpoint = load_training_checkpoint(result.checkpoint)
    eligibility = checkpoint["training_state"][
        "last_rollout_checkpoint_eligibility"
    ]
    assert not eligibility["eligible"]
    assert set(eligibility["evaluated_metrics"].values()) == {None}
    assert checkpoint["validation_selection"]["rollout_was_run"] is False
    json.dumps(eligibility, allow_nan=False)


def test_exact_leaf_pid_transfer_requires_every_destination_key(tmp_path) -> None:
    source = ContextualPretrainingModel(d_model=16, hyper_dim=4)
    checkpoint = save_training_checkpoint(
        tmp_path / "source.pt", model=source, encoder=source.encoder
    )
    target = _validation_model()
    report = load_pretrained_encoder(
        target.encoder,
        checkpoint,
        leaf_pid_head=target.leaf_pid_head,
        transfer_leaf_pid_head=True,
        require_exact_leaf_pid_transfer=True,
    )
    assert report.leaf_pid_loaded_keys == ("bias", "weight")
    assert report.leaf_pid_missing_keys == ()
    assert report.leaf_pid_unexpected_keys == ()
    assert report.leaf_pid_shape_mismatches == ()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state_dict"].pop("leaf_pid_head.bias")
    incomplete = tmp_path / "incomplete-leaf.pt"
    torch.save(payload, incomplete)
    with pytest.raises(ValueError, match="exact leaf PID transfer contract failed"):
        load_pretrained_encoder(
            target.encoder,
            incomplete,
            leaf_pid_head=target.leaf_pid_head,
            transfer_leaf_pid_head=True,
            require_exact_leaf_pid_transfer=True,
        )


def test_small_candidate_enforces_configured_encoder_coverage_before_optimizer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = write_notebook_fixture_v3(tmp_path / "tiny-transfer.parquet")
    incomplete = tmp_path / "incomplete-encoder.pt"
    torch.save({"encoder_state_dict": {}, "model_state_dict": {}}, incomplete)
    optimizer_called = False

    def forbidden_optimizer(*_args, **_kwargs):
        nonlocal optimizer_called
        optimizer_called = True
        raise AssertionError("optimizer must not be constructed")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    with pytest.raises(ValueError, match="pretrained encoder transfer contract failed"):
        train_level_reconstruction(
            ReconstructionConfig(
                data=str(data),
                output_dir=str(tmp_path / "failed-training"),
                pretrained_encoder=str(incomplete),
                model_preset="small_candidate",
                max_steps=1,
                batch_size=2,
                allow_legacy_conflated=True,
                minimum_encoder_transfer_coverage=0.9,
            )
        )
    assert optimizer_called is False
