#!/usr/bin/env python3
"""Freeze the controlled phase-41 level-1 pointer-balance campaign."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (SRC, ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.build_reconstruction_phase35_evaluation_cohort import (  # noqa: E402
    atomic_json,
    sha256,
)
from hypertagging.data.capacity import production_capacity_report  # noqa: E402
from hypertagging.training.model_config import MODEL_PRESETS  # noqa: E402


PARENT = ROOT / "configs/reconstruction/ht_reconstruction_phase40r1_20260908.json"
COHORT = ROOT / "configs/reconstruction/ht_reconstruction_phase41_validation_cohort_20260909.json"
SELECTION = ROOT / "configs/training_selection/production_1m_20260812/train_070k_phase40.json"
DATASET_INDEX = ROOT / "artifacts/experiment_readiness/reconstruction_phase40_20260907/train_070k.complete_only.index.json"
THRESHOLD_DIAGNOSTIC = ROOT / "artifacts/codex/reconstruction_phase40r1_threshold_diagnostic_20260909.json"
STUDY_ID = "phase41-level1-pointer-balance-20260909"
PHASE41_COHORT_STUDY_ID = STUDY_ID
PHASE40_EVIDENCE = (
    (
        "doubled_data_control",
        ROOT / "artifacts/runs/ht-reconstruction-phase40r1-20260908/doubled_data_control/16378817/result.json",
        ROOT / "artifacts/runs/ht-reconstruction-phase40r1-20260908/doubled_data_control/16378817/full-decay-gates.json",
        ROOT / "artifacts/slurm/reconstruction-phase40r1/jobs/16378817/attempt-00/receipt.json",
    ),
    (
        "doubled_data_query_scale",
        ROOT / "artifacts/runs/ht-reconstruction-phase40r1-20260908/doubled_data_query_scale/16378818/result.json",
        ROOT / "artifacts/runs/ht-reconstruction-phase40r1-20260908/doubled_data_query_scale/16378818/full-decay-gates.json",
        ROOT / "artifacts/slurm/reconstruction-phase40r1/jobs/16378818/attempt-00/receipt.json",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    threshold_diagnostic = json.loads(
        THRESHOLD_DIAGNOSTIC.read_text(encoding="utf-8")
    )
    if (
        cohort.get("study_id") != PHASE41_COHORT_STUDY_ID
        or cohort.get("all_required_overlaps_zero") is not True
        or any(cohort.get("overlap_audit", {}).values())
    ):
        raise RuntimeError("phase41 untouched cohort is invalid")
    if (
        threshold_diagnostic.get("conclusion")
        != "INFERENCE_THRESHOLD_INSUFFICIENT_HIERARCHY_TRAINING_REQUIRED"
        or threshold_diagnostic.get("sealed_test_accessed") is not False
        or threshold_diagnostic.get("promotion_authorized") is not False
        or threshold_diagnostic.get("phase41_candidate")
        != {
            "pointer_threshold": 0.35,
            "object_threshold": 0.6,
            "role": "preregistered_fresh_cohort_candidate",
            "post_hoc_promotion_forbidden": True,
        }
    ):
        raise RuntimeError("phase40r1 threshold diagnostic is invalid")

    phase40_evidence = []
    for role, result_path, gates_path, receipt_path in PHASE40_EVIDENCE:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "completed"
            or receipt.get("terminal_stage") != "full_decay_complete"
            or receipt.get("exit_status") != 0
            or receipt.get("sealed_test_accessed") is not False
            or result.get("status") != "training_completed"
            or gates.get("all_preregistered_gates_passed") is not False
            or gates.get("promotion_authorized") is not False
            or gates.get("sealed_test_request_authorized") is not False
        ):
            raise RuntimeError(f"phase40r1 evidence is incomplete: {role}")
        phase40_evidence.append(
            {
                "arm_role": role,
                "training_result": str(result_path.relative_to(ROOT)),
                "training_result_sha256": sha256(result_path),
                "full_decay_gates": str(gates_path.relative_to(ROOT)),
                "full_decay_gates_sha256": sha256(gates_path),
                "receipt": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": sha256(receipt_path),
                "optimizer_steps": result["optimizer_steps"],
                "all_gates_passed": False,
                "sealed_test_accessed": False,
            }
        )

    common = deepcopy(parent["common_training_contract"])
    common.update(
        {
            "seed": 20260910,
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
            "rollout_object_threshold": 0.6,
            "rollout_pointer_threshold": 0.35,
            "rollout_min_tree_validity": 0.999,
            "rollout_min_p4_closure": 1.0,
            # Primary checkpoint selection remains evaluable even when the
            # campaign fails the later-run/promotion authorization gates.
            "rollout_min_depth_fraction": 0.0,
            "rollout_min_complete_target_efficiency": 0.0,
            "pointer_positive_weights_by_level": [],
        }
    )
    common["balanced_level_replay_contract"]["seed"] = 20260910
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
    arms = [
        {
            "role": "pointer32_control",
            "hypothesis": (
                "The phase40r1 winning decoder is reproduced on the untouched "
                "phase41 cohort with its scalar pointer-positive weight of 32."
            ),
            "overrides": {
                "freeze_pretrained_encoder_steps": 2188,
                "pointer_positive_weight": 32.0,
            },
        },
        {
            "role": "level1_pointer24",
            "hypothesis": (
                "Reducing only the level-1 pointer-positive weight from 32 to 24 "
                "will reduce low-level over-linking while preserving the higher-level "
                "pointer recall needed for complete roots."
            ),
            "overrides": {
                "freeze_pretrained_encoder_steps": 2188,
                "pointer_positive_weight": 32.0,
                "pointer_positive_weights_by_level": [[1, 24.0]],
            },
        },
    ]
    index = json.loads(DATASET_INDEX.read_text(encoding="utf-8"))
    capacity_reports = {}
    for arm in arms:
        role = arm["role"]
        config = {**common, **arm["overrides"]}
        preset = MODEL_PRESETS[config["model_preset"]]
        report = production_capacity_report(
            index,
            global_n_queries=preset.n_queries,
            global_max_cardinality=int(config.get("max_cardinality", preset.max_cardinality)),
            n_queries_by_level={int(k): int(v) for k, v in config["n_queries_by_level"]},
            max_cardinality_by_level={
                int(k): int(v) for k, v in config["max_cardinality_by_level"]
            },
            target_policy="complete_only",
        )
        if (
            report["production_training_allowed"] is not True
            or report["query_overflow_count"] != 0
            or report["cardinality_overflow_count"] != 0
        ):
            raise RuntimeError(f"phase41 capacity admission failed: {role}")
        capacity_reports[role] = report

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
        "preregistration_version": "hypertagging-reconstruction-phase41-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "scientific_question": (
            "Does a level-specific pointer class-balance correction improve exact "
            "full-decay topology without sacrificing the phase40r1 control's depth, "
            "source recall, or tree validity?"
        ),
        "parent_phase40r1": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": sha256(PARENT),
        },
        "phase40r1_closeout_basis": {
            "classification": "completed_controlled_study_no_promotion",
            "selected_next_factor": "level_1_pointer_positive_weight_32_to_24",
            "selection_reason": (
                "The phase40r1 control beat query scaling, but level-1 pointer "
                "precision was 0.1664 at 0.7481 recall and exact hierarchy gates "
                "remained below threshold."
            ),
            "query_scale_continuation_authorized": False,
            "longer_run_authorized": False,
            "sealed_test_accessed": False,
            "evidence": phase40_evidence,
            "threshold_diagnostic": {
                "path": str(THRESHOLD_DIAGNOSTIC.relative_to(ROOT)),
                "sha256": sha256(THRESHOLD_DIAGNOSTIC),
                "phase41_policy": "PREREGISTERED_FRESH_COHORT_CANDIDATE_ONLY",
            },
        },
        "capacity_admission": {
            "dataset_index": str(DATASET_INDEX.relative_to(ROOT)),
            "dataset_index_sha256": sha256(DATASET_INDEX),
            "reports_by_arm": capacity_reports,
        },
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
        "arms": arms,
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
            "unified_full_evaluation": {
                "runner": "scripts/run_full_reconstruction_evaluation_suite.py",
                "strict_checkpoint_direct": True,
                "strict_repeat": True,
                "contracted_topology_diagnostic": True,
                "beam_search": True,
            },
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
