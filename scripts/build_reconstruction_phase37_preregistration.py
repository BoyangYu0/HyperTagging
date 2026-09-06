#!/usr/bin/env python3
"""Freeze the two-arm phase-37 scientific campaign preregistration."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reconstruction_phase35_evaluation_cohort import (  # noqa: E402
    atomic_json,
    sha256,
)


PARENT = ROOT / "configs/reconstruction/ht_reconstruction_phase36_20260905.json"
COHORT = ROOT / "configs/reconstruction/ht_reconstruction_phase37_validation_cohort_20260906.json"
STUDY_ID = "phase37-hierarchy-aware-reconstruction-20260906"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    if (
        cohort.get("study_id") != STUDY_ID
        or cohort.get("all_required_overlaps_zero") is not True
        or any(cohort.get("overlap_audit", {}).values())
    ):
        raise RuntimeError("phase37 untouched cohort is not ready")

    common = deepcopy(parent["common_training_contract"])
    common.update(
        {
            "seed": 20260906,
            "best_metric": "micro_complete_target_efficiency",
            "best_mode": "max",
            "unrepresentable_target_policy": "recovery_objective",
            "auxiliary_teacher_weight": 0.25,
            "level_loss_weights": [
                [1, 1.0],
                [2, 1.0],
                [3, 1.25],
                [4, 1.5],
                [5, 2.0],
                [6, 3.0],
            ],
            "recovery_objective_weight": 2.0,
            "rollout_object_threshold": 0.5,
            "rollout_pointer_threshold": 0.35,
            "rollout_min_tree_validity": 0.999,
            "rollout_min_p4_closure": 1.0,
            # Primary checkpoint selection remains evaluable even when the
            # campaign fails the later-run/promotion authorization gates.
            "rollout_min_depth_fraction": 0.0,
            "rollout_min_complete_target_efficiency": 0.0,
        }
    )
    common["balanced_level_replay_contract"]["seed"] = 20260906
    payload = {
        "preregistration_version": "hypertagging-reconstruction-phase37-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "scientific_question": (
            "Does increasing auxiliary teacher anchoring from 0.25 to 0.50, "
            "while holding the successful phase36 weighted/recovery contract fixed, "
            "improve exact hierarchy recovery without losing rollout recall?"
        ),
        "parent_phase36": {"path": str(PARENT.relative_to(ROOT)), "sha256": sha256(PARENT)},
        "data_binding": deepcopy(parent["data_binding"]),
        "untouched_validation_cohort": {
            "path": str(COHORT.relative_to(ROOT)),
            "sha256": sha256(COHORT),
            "checkpoint_selection_event_uid_count": cohort[
                "checkpoint_selection_event_uid_count"
            ],
            "checkpoint_selection_event_uids_sha256": cohort[
                "checkpoint_selection_event_uids_sha256"
            ],
            "evaluation_event_uid_count": cohort["evaluation_event_uid_count"],
            "evaluation_event_uids_sha256": cohort[
                "evaluation_event_uids_sha256"
            ],
            "all_required_overlaps_zero": True,
            "sealed_test_role_access": "forbidden",
        },
        "common_training_contract": common,
        "checkpoint_selection_contract": {
            "primary": {
                "filename": "best.pt",
                "metric": "micro_complete_target_efficiency",
                "mode": "max",
            },
            "independent_tracks": [
                {
                    "filename": "best_rollout_complete_target_efficiency.pt",
                    "metric": "micro_complete_target_efficiency",
                    "mode": "max",
                },
                {
                    "filename": "best_rollout_depth_fraction.pt",
                    "metric": "predicted_depth_fraction",
                    "mode": "max",
                },
                {
                    "filename": "best_rollout_tree_validity.pt",
                    "metric": "predicted_tree_validity_rate",
                    "mode": "max",
                },
            ],
            "allowed_selection_steps": [500, 1000, 1500, 2000, 2188],
        },
        "arms": [
            {
                "role": "phase36_weighted_recovery_reference",
                "hypothesis": (
                    "Repeating the successful phase36 weighted/recovery contract on a "
                    "fresh cohort provides the paired reference."
                ),
                "overrides": {
                    "freeze_pretrained_encoder_steps": 1094,
                    "auxiliary_teacher_weight": 0.25,
                },
            },
            {
                "role": "teacher_anchor_050",
                "hypothesis": (
                    "Stronger auxiliary teacher anchoring improves exact mother/link "
                    "assignment and half-tree precision while retaining the phase36 "
                    "depth and source-recall gains."
                ),
                "overrides": {
                    "freeze_pretrained_encoder_steps": 1094,
                    "auxiliary_teacher_weight": 0.50,
                },
            },
        ],
        "post_training_gates": {
            "minimum_full_root_completion_numerator": 1,
            "minimum_full_lcag_numerator": 1,
            "minimum_exact_mother_coverage_numerator": 1,
            "minimum_full_source_recall": 0.15,
            "minimum_full_source_precision": 0.70,
            "minimum_half_source_precision": 0.45,
            "minimum_half_perfect_lcag_numerator": 1,
            "minimum_half_root_pid_accuracy": 0.02,
            "minimum_predicted_depth_fraction": 1.0,
            "minimum_complete_target_efficiency": 0.025,
            "minimum_tree_validity": 0.999,
            "minimum_p4_closure": 1.0,
            "all_gates_required": True,
        },
        "authority": {
            "training_authorized": True,
            "validation_payload_access_authorized": True,
            "scheduler_authorized": True,
            "submission_authorized": True,
            "scientific_validation_authorized": True,
            "source_checkpoint_mutation_authorized": False,
            "automatic_promotion": False,
            "longer_run_authorized": False,
            "promotion_authorized": False,
            "sealed_test_access_authorized": False,
            "sealed_test_request_authorized": False,
        },
        "execution_policy": {
            "task_count": 2,
            "global_concurrency": 2,
            "gres": "gpu:h100nvl:1",
            "cpus_per_task": 8,
            "memory": "64G",
            "time": "24:00:00",
            "requeue": False,
            "maximum_restarts": 0,
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preregistration: {output}")
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
