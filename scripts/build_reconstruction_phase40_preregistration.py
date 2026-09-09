#!/usr/bin/env python3
"""Freeze the two-arm phase-40 scientific campaign preregistration."""

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


PARENT = ROOT / "configs/reconstruction/ht_reconstruction_phase39_20260907.json"
COHORT = ROOT / "configs/reconstruction/ht_reconstruction_phase40_validation_cohort_20260907.json"
SELECTION = ROOT / "configs/training_selection/production_1m_20260812/train_070k_phase40.json"
DATASET_INDEX = ROOT / "artifacts/experiment_readiness/reconstruction_phase40_20260907/train_070k.complete_only.index.json"
STUDY_ID = "phase40-hierarchy-aware-reconstruction-20260907"


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
        raise RuntimeError("phase40 untouched cohort is not ready")

    common = deepcopy(parent["common_training_contract"])
    common.update(
        {
            "seed": 20260909,
            "max_steps": 4376,
            "lr_schedule_total_steps": 4376,
            "checkpoint_every": 1000,
            "rollout_validate_every": 1000,
            "freeze_leaf_pid_head_steps": 4376,
            "scheduled_sampling_duration_steps": 2188,
            "scheduled_sampling_zero_based_step_range": [0, 4375],
            "optimizer_steps_per_train_size_equivalent": 1094,
            "four_train_size_equivalent_slot_target": 280000,
            "replay_slot_budget": 280064,
            "replay_slot_counts_by_level": {
                "1": 46678,
                "2": 46678,
                "3": 46677,
                "4": 46677,
                "5": 46677,
                "6": 46677,
            },
            "train_split_event_count": 70000,
            "training_budget_semantics": (
                "four_train_size_equivalent_level_conditioned_replay_slots_"
                "on_exact_doubled_training_data"
            ),
            "best_metric": "micro_complete_target_efficiency",
            "best_mode": "max",
            "unrepresentable_target_policy": "recovery_objective",
            "auxiliary_teacher_weight": 0.50,
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
    common["balanced_level_replay_contract"]["seed"] = 20260909
    common["balanced_level_replay_contract"]["materialized_train_event_count"] = 70000
    common["balanced_level_replay_contract"]["materialized_train_uid_sha256"] = (
        "f57ef10b1f2249f1abf9e38c641730ccf71dc919273021dbc315a86167102bf0"
    )
    common["balanced_level_replay_contract"]["eligible_pool_counts_by_level"] = {
        "1": 55323,
        "2": 32352,
        "3": 25942,
        "4": 20061,
        "5": 13220,
        "6": 1744,
    }
    common["balanced_level_replay_contract"]["eligible_pool_uid_sha256_by_level"] = {
        "1": "a4bd6b19439cfe4146713fb8c91e4a6bb9d10eb3adb07c2512f873a0c754c836",
        "2": "1c292746fad717298e0a796d603b1109aa8b9262ab487add44d0ea1fa5bd0443",
        "3": "29925713b07f5c99c6cff3355530db9de2b8fffefc0f231245f30da512849441",
        "4": "f39a49fe0b9b397717fa24afb7939af6a0fc05fff08026e633e48b4b4f5f5194",
        "5": "d09cce173d1b5f4aa6695c5c89c1bbb4ec001198d4b4ac53a7e5cc7e5e083a4e",
        "6": "97887987a77dad4c13fea2682a0de7888d212c0059adb25eaea5a189d6986011",
    }
    common["balanced_level_replay_contract"]["planned_schedule"] = {
        "start_slot": 0,
        "slot_count": 280064,
        "end_slot_exclusive": 280064,
        "level_counts": dict(common["replay_slot_counts_by_level"]),
        "max_minus_min_level_count": 1,
    }
    data_binding = {
        "selection_manifest": str(SELECTION.relative_to(ROOT)),
        "selection_manifest_sha256": sha256(SELECTION),
        "dataset_index": str(DATASET_INDEX.relative_to(ROOT)),
        "dataset_index_sha256": sha256(DATASET_INDEX),
        "included_splits": ["train", "validation"],
        "train_events": 70000,
        "validation_events": 50000,
        "sealed_test_opened": False,
    }
    payload = {
        "preregistration_version": "hypertagging-reconstruction-phase40-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "scientific_question": (
            "Does doubling reconstruction data and training exposure improve "
            "hierarchy recovery, and does added per-level query capacity provide "
            "a further controlled gain without exhausting precision?"
        ),
        "parent_phase39": {"path": str(PARENT.relative_to(ROOT)), "sha256": sha256(PARENT)},
        "data_binding": data_binding,
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
            "allowed_selection_steps": [1000, 2000, 3000, 4000, 4376],
        },
        "arms": [
            {
                "role": "doubled_data_control",
                "hypothesis": (
                    "The phase39 pointer-32 architecture receives exactly twice the "
                    "training data and twice its four-size-equivalent step budget."
                ),
                "overrides": {
                    "freeze_pretrained_encoder_steps": 2188,
                    "pointer_positive_weight": 32.0,
                },
            },
            {
                "role": "doubled_data_query_scale",
                "hypothesis": (
                    "More decoder query slots reduce competition among simultaneous "
                    "mothers while leaving encoder size, data, optimizer, and losses fixed."
                ),
                "overrides": {
                    "freeze_pretrained_encoder_steps": 2188,
                    "pointer_positive_weight": 32.0,
                    "n_queries_by_level": [
                        [1, 24],
                        [2, 12],
                        [3, 8],
                        [4, 6],
                        [5, 4],
                        [6, 2],
                    ],
                },
            },
        ],
        "post_training_gates": {
            "minimum_full_root_completion_numerator": 2,
            "minimum_full_lcag_numerator": 2,
            "minimum_exact_mother_coverage_numerator": 2,
            "minimum_full_source_recall": 0.17,
            "minimum_full_source_precision": 0.75,
            "minimum_half_source_recall": 0.19,
            "minimum_half_source_precision": 0.45,
            "minimum_half_lcag_numerator": 10,
            "minimum_half_perfect_lcag_numerator": 3,
            "minimum_half_root_pid_accuracy": 0.02,
            "minimum_predicted_depth_fraction": 1.2,
            "minimum_complete_target_efficiency": 0.03,
            "minimum_tree_validity": 0.999,
            "minimum_p4_closure": 1.0,
            "all_gates_required": True,
        },
        "evaluation_contract": {
            "cpu_threads": 1,
            "deterministic_algorithms": True,
            "primary_evaluation_repeats": 2,
            "require_exact_primary_repeat_equality": True,
            "root_completion_stability_policy": (
                "require_at_least_two_completed_roots_in_each_identical_primary_repeat"
            ),
            "beam_search": {
                "enabled": True,
                "scope": "full",
                "beam_width": 4,
                "max_events": 20,
                "max_level": 6,
                "model_only_rankings": [
                    "learned_confidence_sum",
                    "learned_confidence_mean",
                    "average_link_probability",
                    "normalized_joint_log_probability",
                ],
                "oracle_at_k_diagnostic_only": True,
            },
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
            "time": "36:00:00",
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
