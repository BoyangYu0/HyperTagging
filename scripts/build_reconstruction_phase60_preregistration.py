#!/usr/bin/env python3
"""Preregister one matched replication after replenishing independent validation."""

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_reconstruction_phase46_preregistration import binding
from hypertagging.data.capacity import production_capacity_report
from hypertagging.training.model_config import MODEL_PRESETS


def main():
    previous = ROOT / "configs/reconstruction/ht_reconstruction_phase59_20260924.json"
    p = json.loads(previous.read_text())
    ep = ROOT / "artifacts/codex/reconstruction_phase59_closeout_20260925.json"
    rp = ROOT / "artifacts/codex/reconstruction_phase59_retained_metrics_20260925.json"
    cp = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase60_validation_cohort_20260925.json"
    )
    epd = json.loads(ep.read_text())
    ret = json.loads(rp.read_text())
    c = json.loads(cp.read_text())
    assert (
        epd["status"] == "FEASIBILITY_COMPLETE"
        and epd["metric_completeness"] == "COMPLETE"
    )
    assert epd["independent_validation"] and epd["strict_selection_overlap"] == 0
    assert ret["all_returned_beam_candidates_checked"]
    for key in (
        "parent_phase58",
        "phase58_closeout_basis",
        "phase58_retained_metric_basis",
        "unchanged_gate_thresholds_are_diagnostic_on_25_events",
    ):
        p.pop(key)
    p.update(
        study_id="phase60-fresh-validation-pretraining-replication-20260925",
        preregistration_version="hypertagging-reconstruction-phase60-preregistration-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question="Replicate late PID 0.2 versus 0.1 at fixed parent 2 on independently sourced fresh selection and strict cohorts; test stability and downstream topology jointly. Both arms share the required level-5 cardinality repair 13 to 14. No new dose sweep, duration increase or training-size contrast.",
        pilot_classification="EXPLORATORY_FRESH_VALIDATION_REPLICATION",
    )
    p["scientific_source_boundary"]["limitation"] = (
        "Both arms initialize the original encoder81096 parameters with fresh train-only normalization, optimizer, schedule, RNG and memory. "
        "Only the late pretraining leaf-PID weight differs between arms; parent weight2 and budgets are matched. "
        "Both share the level5 cardinality limit14 required by the expanded index, with global limit16 unchanged. "
        "Other direct-target policy incompatibilities remain. No physical momentum-resolution or cross-phase causal claim."
    )
    p["common_training_contract"]["max_cardinality_by_level"] = [
        [level, 14 if level == 5 else limit]
        for level, limit in p["common_training_contract"]["max_cardinality_by_level"]
    ]
    p["shared_capacity_repair"] = {
        "level": 5,
        "previous_limit": 13,
        "new_limit": 14,
        "global_limit_unchanged": 16,
        "applies_to_both_arms": True,
        "observed_overflow_targets_at_old_limit": 1,
        "reason": "Expanded validation exposes one 14-daughter target; include it rather than dropping the target or weakening the admission gate.",
        "cross_phase_causal_comparison_valid": False,
    }
    for cfg in (p["common_training_contract"], p["pretraining_refinement"]["config"]):
        cfg["seed"] = 20260927
    p["common_training_contract"]["balanced_level_replay_contract"]["seed"] = 20260927
    selection = (
        ROOT
        / "configs/training_selection/phase60_validation_expansion_20260925/train_070k.json"
    )
    index_path = (
        ROOT
        / "artifacts/experiment_readiness/reconstruction_phase60_20260925/train_070k.complete_only.index.json"
    )
    index = json.loads(index_path.read_text())
    for key, path in [("selection_manifest", selection), ("dataset_index", index_path)]:
        p["data_binding"][key] = str(path.relative_to(ROOT))
        p["data_binding"][key + "_sha256"] = binding(path)["sha256"]
    p["data_binding"]["validation_events"] = 100000
    p["pretraining_refinement"]["config"]["data"] = str(selection.relative_to(ROOT))
    p["pretraining_refinement"]["config"]["dataset_index"] = str(
        index_path.relative_to(ROOT)
    )
    p["pretraining_refinement"]["comparison_limitation"] = (
        "Fresh matched 1000 selection and 100 strict events; conditional single-seed comparison, no causal cross-phase contrast."
    )
    p["pretraining_refinement"]["validation_population"] = (
        "fresh_1000_shared_checkpoint_selection_cohort"
    )
    for arm in p["arms"]:
        arm["hypothesis"] = (
            "Replicate the registered late-PID contrast on fresh sources; require joint topology and forest evidence, not weighted-loss reduction."
        )
    p["parent_phase59"] = binding(previous)
    p["phase59_closeout_basis"] = {
        **binding(ep),
        "classification": "feasibility_complete_no_promotion",
        "selected_next_factor": "fresh_validation_replication",
        "sealed_test_accessed": False,
    }
    p["phase59_retained_metric_basis"] = {
        **binding(rp),
        "version": ret["version"],
        "evaluator_revision": ret["evaluator_revision"],
    }
    p["validation_cohort"] = {
        **binding(cp),
        **{
            k: c[k]
            for k in (
                "checkpoint_selection_event_uid_count",
                "checkpoint_selection_event_uids_sha256",
                "evaluation_event_uid_count",
                "evaluation_event_uids_sha256",
                "all_required_overlaps_zero",
                "permitted_selection_reuse_count",
            )
        },
        "sealed_test_role_access": "forbidden",
    }
    p["validation_budget"] = {
        "validation_role_events": 100000,
        "previously_used": 50000,
        "new_independent_validation_events": 50000,
        "newly_reserved": 1100,
        "reused_selection_events": 0,
        "selection_events": 1000,
        "rollout_selection_events": 1000,
        "strict_events": 100,
        "reconstruction_excluded_events": 99000,
        "remaining_untouched_after_phase60": 48900,
        "limitation": "New source files and fresh seed prohibit causal cross-phase comparisons. One bounded pair; no automatic chain.",
    }
    p["evaluation_contract"]["max_events"] = 100
    p["fresh_statistics_binding"] = binding(
        index_path.parent / "fresh-statistics-audit.json"
    )
    p["statistics_reuse"] = {
        "normalizer_scope": "train",
        "training_payloads_unchanged": True,
        "new_index_full_record_scan": True,
        "validation_never_fits_statistics": True,
    }
    p["decision_rules"].update(
        dataset_size="Hold the same 70000 training payloads. Expand only independent validation; no controlled learning curve establishes benefit from training growth.",
        pretraining="Both pilot arms completed but 25 strict events are insufficient for quality preference. Replicate 0.2 versus 0.1 once with fresh validation and seed, retaining dominance 20 fail guard.",
        stop_dose_tuning="No further PID dose search. One replication only; assess joint topology, coherent forests, uncertainty and failure evidence before allocating again.",
        authority="User authorized Phase59 review, publication and the next bounded pair on 2026-09-25. No promotion, sealed-test access or automatic chain.",
    )
    common = p["common_training_contract"]
    preset = MODEL_PRESETS[common["model_preset"]]
    capacity = production_capacity_report(
        index,
        global_n_queries=preset.n_queries,
        global_max_cardinality=common["max_cardinality"],
        n_queries_by_level=dict(common["n_queries_by_level"]),
        max_cardinality_by_level=dict(common["max_cardinality_by_level"]),
        target_policy="complete_only",
    )
    assert (
        capacity["production_training_allowed"]
        and capacity["query_overflow_count"]
        == capacity["cardinality_overflow_count"]
        == 0
    )
    p["capacity_admission"].update(
        dataset_index=str(index_path.relative_to(ROOT)),
        dataset_index_sha256=binding(index_path)["sha256"],
        reports_by_arm={a["role"]: copy.deepcopy(capacity) for a in p["arms"]},
    )
    target = ROOT / "configs/reconstruction/ht_reconstruction_phase60_20260925.json"
    assert not target.exists()
    target.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    print("PASS Phase60 preregistration, fixed train70000/validation100000")


if __name__ == "__main__":
    main()
