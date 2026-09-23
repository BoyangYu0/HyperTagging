#!/usr/bin/env python3
"""Repeat intended objective-balance contrast with enforced validation eligibility."""

import json, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous = ROOT / "configs/reconstruction/ht_reconstruction_phase57_20260922.json"
    p = json.loads(previous.read_text())
    cohort_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase58_validation_cohort_20260923.json"
    )
    cohort = json.loads(cohort_path.read_text())
    evidence_path = (
        ROOT / "artifacts/codex/reconstruction_phase57_closeout_20260923.json"
    )
    evidence = json.loads(evidence_path.read_text())
    retained_path = (
        ROOT / "artifacts/codex/reconstruction_phase57_retained_metrics_20260923.json"
    )
    retained = json.loads(retained_path.read_text())
    assert (
        evidence["status"] == "DIAGNOSTIC_COMPLETE"
        and evidence["independent_validation"] is False
        and evidence["strict_selection_overlap"] == 100
        and evidence["metric_completeness"] == "COMPLETE"
    )
    assert (
        retained["all_returned_beam_candidates_checked"]
        and cohort["all_required_overlaps_zero"]
    )
    for key in (
        "parent_phase56",
        "phase56_closeout_basis",
        "phase56_retained_metric_basis",
    ):
        p.pop(key)
    p.update(
        study_id="phase58-corrected-pretraining-objective-balance-20260923",
        preregistration_version="hypertagging-reconstruction-phase58-preregistration-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question="With corrected selection eligibility and fresh strict validation, does lower late pretraining PID weight improve joint reconstruction quality at fixed data and compute? Phase57 is selection-contaminated and cannot establish that contrast independently.",
    )
    p["common_training_contract"]["seed"] = 20260925
    p["common_training_contract"]["balanced_level_replay_contract"]["seed"] = 20260925
    p["pretraining_refinement"]["config"]["seed"] = 20260925
    p["pretraining_refinement"]["comparison_limitation"] = (
        "Exact Phase56 selection SET reused and ranked with seed20260925; all other49000 validation events excluded before reconstruction selection. Fresh strict100; no independent cross-phase selection replication. Same intended0.4vs0.2 balance contrast, new seed."
    )
    p["parent_phase57"] = binding(previous)
    p["phase57_closeout_basis"] = {
        **binding(evidence_path),
        "classification": "selection_contaminated_diagnostics_no_promotion",
        "selected_next_factor": "corrected_validation_pretraining_balance_replication",
        "sealed_test_accessed": False,
    }
    p["phase57_retained_metric_basis"] = {
        **binding(retained_path),
        "version": retained["version"],
        "evaluator_revision": retained["evaluator_revision"],
    }
    p["validation_cohort"] = {
        **binding(cohort_path),
        **{
            k: cohort[k]
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
        "validation_role_events": 50000,
        "previously_used": 49875,
        "newly_reserved": 100,
        "reused_selection_events": 1000,
        "selection_events": 1000,
        "rollout_selection_events": 1000,
        "strict_events": 100,
        "reconstruction_excluded_events": 49000,
        "remaining_untouched_after_phase58": 25,
        "limitation": "Additional independent validation data needed before another full100-event study; do not raid sealed test.",
    }
    p["decision_rules"].update(
        pretraining="Repeat intended latePID0.4vs0.2 contrast at parent2 and unchanged threshold20fail; first correct selection eligibility. Phase57 numerical differences are in-sample diagnostics, not validation evidence.",
        stop_dose_tuning="No new dose choice from contaminated Phase57. One bounded corrected comparison only; no campaign chain.",
        dataset_size="Hold70000 training events; no controlled learning curve establishes growth benefit. Validation-data expansion is now a priority and is separate from training set size.",
        authority="User authorized Phase57 review, publication and next bounded training pair on2026-09-23. No promotion, automatic chain or sealed test.",
    )
    output = ROOT / "configs/reconstruction/ht_reconstruction_phase58_20260923.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
