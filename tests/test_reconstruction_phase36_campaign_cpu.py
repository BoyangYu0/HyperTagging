from __future__ import annotations

import json
from pathlib import Path

from scripts.run_reconstruction_phase35 import _training_config
from scripts.run_reconstruction_phase36 import ARM_ROLES, CHECKPOINT_TRACKS


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "configs/reconstruction/ht_reconstruction_phase36_20260905.json"
)
COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase36_validation_cohort_20260905.json"
)


def test_phase36_cohort_is_untouched_and_sealed_test_free() -> None:
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


def test_phase36_preregisters_metric_tracks_arms_and_hard_gates() -> None:
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
    candidate = prereg["arms"][1]["overrides"]
    assert candidate["rollout_pointer_threshold"] == 0.35
    assert candidate["unrepresentable_target_policy"] == "recovery_objective"
    assert candidate["level_loss_weights"][-1] == [6, 3.0]
    gates = prereg["post_training_gates"]
    assert gates["minimum_full_root_completion_numerator"] == 1
    assert gates["minimum_full_lcag_numerator"] == 1
    assert gates["minimum_exact_mother_coverage_numerator"] == 1
    assert gates["minimum_predicted_depth_fraction"] > 0
    assert gates["minimum_complete_target_efficiency"] > 0
    assert prereg["authority"]["longer_run_authorized"] is False
    assert prereg["authority"]["promotion_authorized"] is False
    assert prereg["authority"]["sealed_test_access_authorized"] is False


def test_phase36_training_config_preserves_fractional_level_weights(
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
    assert config.rollout_pointer_threshold == 0.35
    assert config.rollout_min_depth_fraction == 0.0
    assert config.rollout_min_complete_target_efficiency == 0.0


def test_phase36_runner_requires_all_independent_tracks() -> None:
    assert CHECKPOINT_TRACKS == {
        "final": "checkpoint-step-2188.pt",
        "best": "best.pt",
        "best_complete_target": "best_rollout_complete_target_efficiency.pt",
        "best_depth": "best_rollout_depth_fraction.pt",
        "best_tree_validity": "best_rollout_tree_validity.pt",
    }
