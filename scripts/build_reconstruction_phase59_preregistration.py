"""Preregister a bounded stability pilot with the final untouched validation cohort."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous = ROOT / "configs/reconstruction/ht_reconstruction_phase58_20260923.json"
    p = json.loads(previous.read_text())
    ep = ROOT / "artifacts/codex/reconstruction_phase58_closeout_20260924.json"
    rp = ROOT / "artifacts/codex/reconstruction_phase58_retained_metrics_20260924.json"
    cp = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase59_validation_cohort_20260924.json"
    )
    e = json.loads(ep.read_text())
    ret = json.loads(rp.read_text())
    c = json.loads(cp.read_text())
    assert (
        e["status"] == "INCOMPLETE_COMPARISON"
        and e["metric_completeness"] == "COMPLETE_AVAILABLE_ARTIFACTS"
        and e["strict_selection_overlap"] == 0
        and ret["all_returned_beam_candidates_checked"]
    )
    for key in (
        "parent_phase57",
        "phase57_closeout_basis",
        "phase57_retained_metric_basis",
    ):
        p.pop(key)
    p.update(
        study_id="phase59-pretraining-stability-pilot-20260924",
        preregistration_version="hypertagging-reconstruction-phase59-preregistration-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question="Does late PID weight0.1 versus0.2 improve objective stability at fixed parent2, data and compute? Only25 fresh strict events remain: feasibility and descriptive reconstruction, not confirmatory quality evidence.",
    )
    for a, w in zip(p["arms"], (0.2, 0.1), strict=True):
        a["pretraining_leaf_pid_phase_weights"] = [1.0, 1.0, w, w]
        a["hypothesis"] = (
            f"Late PID weight{w} with fixed parent2; test stability without weakening fail guard."
        )
    p["common_training_contract"]["seed"] = 20260926
    p["common_training_contract"]["balanced_level_replay_contract"]["seed"] = 20260926
    p["pretraining_refinement"]["config"]["seed"] = 20260926
    p["pretraining_refinement"]["comparison_limitation"] = (
        "Reused1000 selection events; fresh strict25 only, beam20. Feasibility pilot, no cross-phase causal comparison or promotion."
    )
    p["parent_phase58"] = binding(previous)
    p["phase58_closeout_basis"] = {
        **binding(ep),
        "classification": "incomplete_comparison_no_promotion",
        "selected_next_factor": "lower_pid_stability_pilot",
        "sealed_test_accessed": False,
    }
    p["phase58_retained_metric_basis"] = {
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
        "validation_role_events": 50000,
        "previously_used": 49975,
        "newly_reserved": 25,
        "reused_selection_events": 1000,
        "selection_events": 1000,
        "rollout_selection_events": 1000,
        "strict_events": 25,
        "reconstruction_excluded_events": 49000,
        "remaining_untouched_after_phase59": 0,
        "limitation": "No further campaign without new independent validation data; sealed test remains closed.",
    }
    p["decision_rules"].update(
        pretraining="Assess complete execution and objective guard stability first. Lower latePID is not proven better for reconstruction; record every failure and all available metrics.",
        stop_dose_tuning="One bounded feasibility pair only. No automatic chain; independent validation expansion required before another campaign.",
        dataset_size="Hold70000 training events. Training growth has no controlled benefit established; prioritize fresh validation and objective stability.",
        authority="User authorized Phase58 review, publication and next bounded training pair on2026-09-24. No promotion or sealed test.",
    )
    p["evaluation_contract"]["max_events"] = (
        25 if "max_events" in p["evaluation_contract"] else 25
    )
    p["pilot_classification"] = "FEASIBILITY_ONLY_NOT_CONFIRMATORY"
    p["unchanged_gate_thresholds_are_diagnostic_on_25_events"] = True
    output = ROOT / "configs/reconstruction/ht_reconstruction_phase59_20260924.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
