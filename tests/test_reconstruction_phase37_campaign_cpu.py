from __future__ import annotations

import json
from pathlib import Path

from scripts.run_reconstruction_phase35 import _training_config
from scripts.run_reconstruction_phase37 import ARM_ROLES, CHECKPOINT_TRACKS


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "configs/reconstruction/ht_reconstruction_phase37_20260906.json"
)
COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase37_validation_cohort_20260906.json"
)


def test_phase37_cohort_is_untouched_and_sealed_test_free() -> None:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    assert cohort["role"] == "validation"
    assert cohort["sealed_test_role_access"] == "forbidden"
    assert cohort["checkpoint_selection_event_uid_count"] == 2_000
    assert cohort["evaluation_event_uid_count"] == 100
    assert len(cohort["checkpoint_selection_event_uids"]) == 2_000
    assert len(cohort["evaluation_event_uids"]) == 100
    assert cohort["event_uids"] == cohort["evaluation_event_uids"]
    assert cohort["all_required_overlaps_zero"] is True
    assert set(cohort["overlap_audit"].values()) == {0}
    assert cohort["source_bindings"]["phase36_cohort"][
        "checkpoint_selection_event_uid_count"
    ] == 2_000
    assert cohort["source_bindings"]["phase36_cohort"][
        "evaluation_event_uid_count"
    ] == 100


def test_phase37_preregisters_single_factor_arms_and_hard_gates() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    selection = prereg["checkpoint_selection_contract"]
    assert selection["primary"] == {
        "filename": "best.pt",
        "metric": "micro_complete_target_efficiency",
        "mode": "max",
    }
    assert {item["metric"] for item in selection["independent_tracks"]} == {
        "micro_complete_target_efficiency",
        "predicted_depth_fraction",
        "predicted_tree_validity_rate",
    }
    assert tuple(item["role"] for item in prereg["arms"]) == ARM_ROLES
    common = prereg["common_training_contract"]
    reference = prereg["arms"][0]["overrides"]
    candidate = prereg["arms"][1]["overrides"]
    assert common["rollout_pointer_threshold"] == 0.35
    assert common["unrepresentable_target_policy"] == "recovery_objective"
    assert common["level_loss_weights"][-1] == [6, 3.0]
    assert reference["auxiliary_teacher_weight"] == 0.25
    assert candidate["auxiliary_teacher_weight"] == 0.50
    assert set(reference) == set(candidate) == {
        "freeze_pretrained_encoder_steps",
        "auxiliary_teacher_weight",
    }
    gates = prereg["post_training_gates"]
    assert gates["minimum_full_root_completion_numerator"] == 1
    assert gates["minimum_full_lcag_numerator"] == 1
    assert gates["minimum_exact_mother_coverage_numerator"] == 1
    assert gates["minimum_full_source_recall"] == 0.15
    assert gates["minimum_full_source_precision"] == 0.70
    assert gates["minimum_half_source_precision"] == 0.45
    assert gates["minimum_half_perfect_lcag_numerator"] == 1
    assert gates["minimum_half_root_pid_accuracy"] == 0.02
    assert gates["minimum_predicted_depth_fraction"] == 1.0
    assert gates["minimum_complete_target_efficiency"] == 0.025
    assert prereg["authority"]["longer_run_authorized"] is False
    assert prereg["authority"]["promotion_authorized"] is False
    assert prereg["authority"]["sealed_test_access_authorized"] is False


def test_phase37_training_config_preserves_fractional_level_weights(
    tmp_path: Path,
) -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    candidate = {
        **prereg["common_training_contract"],
        **prereg["arms"][1]["overrides"],
    }
    config = _training_config(
        config=candidate,
        runtime={
            "selection_manifest": "selection.json",
            "dataset_index": "index.json",
            "checkpoint": "source.pt",
        },
        training_output=tmp_path / "training",
        validation_exclusions=("excluded",),
    )
    assert config.max_steps == 2_188
    assert config.best_metric == "micro_complete_target_efficiency"
    assert config.level_loss_weights == (
        (1, 1.0),
        (2, 1.0),
        (3, 1.25),
        (4, 1.5),
        (5, 2.0),
        (6, 3.0),
    )
    assert config.recovery_objective_weight == 2.0
    assert config.auxiliary_teacher_weight == 0.50
    assert config.rollout_pointer_threshold == 0.35
    assert config.rollout_min_depth_fraction == 0.0
    assert config.rollout_min_complete_target_efficiency == 0.0


def test_phase37_runner_requires_all_independent_tracks() -> None:
    assert CHECKPOINT_TRACKS == {
        "final": "checkpoint.pt",
        "best": "best.pt",
        "best_complete_target": "best_rollout_complete_target_efficiency.pt",
        "best_depth": "best_rollout_depth_fraction.pt",
        "best_tree_validity": "best_rollout_tree_validity.pt",
    }


def test_phase37_full_decay_binds_manifest_and_preregistered_thresholds() -> None:
    source = (ROOT / "scripts/run_reconstruction_phase37_full_decay.py").read_text(
        encoding="utf-8"
    )
    assert '"manifest_version": "hypertagging-reconstruction-evaluation-cohort-v1"' in source
    assert 'contract["config"]["rollout_object_threshold"]' in source
    assert 'contract["config"]["rollout_pointer_threshold"]' in source
    for metric in (
        "full_source_recall",
        "full_source_precision",
        "half_source_precision",
        "half_perfect_lcag",
        "half_root_pid_accuracy",
    ):
        assert metric in source
