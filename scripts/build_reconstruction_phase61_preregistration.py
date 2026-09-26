#!/usr/bin/env python3
"""Bind a shared early-PID stability repair to untouched Phase60 strict events."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def binding(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    prior = ROOT / "configs/reconstruction/ht_reconstruction_phase60_20260925.json"
    evidence = ROOT / "artifacts/codex/reconstruction_phase60_closeout_20260926.json"
    closeout = json.loads(evidence.read_text())
    assert closeout["status"] == "FAILED_PRETRAINING"
    assert closeout["metric_completeness"] == "ALL_AVAILABLE_EXPORTED"
    assert closeout["treatment_executed"] is False
    assert all(a["strict_events_scored"] == a["validation_metric_records"] == a["completed_reconstruction_steps"] == 0 for a in closeout["arms"].values())
    p = copy.deepcopy(json.loads(prior.read_text()))
    p.pop("parent_phase59")
    p.pop("phase59_closeout_basis")
    p.update(study_id="phase61-early-pid-stability-pilot-20260926", preregistration_version="hypertagging-reconstruction-phase61-preregistration-v1", created_at=datetime.now(timezone.utc).isoformat(), pilot_classification="EARLY_PID_STABILITY_PILOT", scientific_question="Can a shared early PID weight 0.5 permit the planned late-PID 0.2 versus 0.1 contrast to execute under the unchanged gradient-dominance guard? This is an adaptive stability pilot, not an independent replication of Phase59.")
    p["parent_phase60"] = binding(prior)
    p["phase60_closeout_basis"] = {**binding(evidence), "classification": "failed_before_treatment", "selected_next_factor": "shared_early_pid_repair", "sealed_test_accessed": False}
    for arm in p["arms"]:
        arm["pretraining_leaf_pid_phase_weights"][:2] = [0.5, 0.5]
        arm["hypothesis"] = "Shared early weight reduction is a stability intervention; late PID treatment remains unexecuted and unproven. Require downstream topology and forest evidence if both arms complete."
    p["pretraining_refinement"]["config"]["leaf_pid_phase_weights"] = [0.5, 0.5, 0.2, 0.2]
    p["pretraining_refinement"]["comparison_limitation"] = "Adaptively chosen shared early-PID schedule after a training-only failure. Same seed and reservation, no independent seed replication; no strict or validation scores were produced in Phase60."
    p["pretraining_refinement"]["validation_population"] = "same_reserved_1000_selection_events_no_phase60_validation_scores"
    p["scientific_source_boundary"]["limitation"] = "Original step81096 parameters only; fresh train-only normalization, optimizer, RNG, memory and schedules. Shared early PID0.5 is new relative to Phase60. Only late PID0.2 versus0.1 differs within Phase61. Same seed and validation reservation; no cross-phase causal quality claim."
    p["shared_early_pid_repair"] = {"previous_weights": [1.0, 1.0], "new_weights": [0.5, 0.5], "applies_to_both_arms": True, "observed_failure_step": 547, "observed_dominance_ratio": 28.020851045975533, "dominance_threshold_unchanged": 20.0, "failure_action_unchanged": "fail", "limitation": "At frozen gradients halving PID halves that pairwise ratio; changed training trajectories make this neither a pass guarantee nor a quality prediction."}
    budget = p["validation_budget"]
    budget["remaining_untouched_after_phase61"] = budget.pop("remaining_untouched_after_phase60")
    budget.update(new_independent_validation_events=0, newly_reserved=0, reused_selection_events=1000, selection_reservation_reused=True, strict_reservation_reused=True, prior_strict_events_scored=0, prior_validation_metric_records=0, limitation="Reuse the authenticated Phase60 reservation because neither arm reached validation or strict evaluation. Conservatively retain all1100 reservations;48900 remain unreserved. Do not claim this is a fresh independent seed replication.")
    p["decision_rules"].update(dataset_size="Hold70000 training events and the existing100000 validation universe. No further expansion is needed for this pilot.", pretraining="Prioritize early objective stability. Halve early PID in both arms, retain late0.2versus0.1 and unchanged dominance20/fail. Quality benefit remains unproven.", stop_dose_tuning="One bounded shared-repair pilot only; no automatic follow-on, longer run, promotion or sealed-test access.", authority="User authorized Phase60 review, publication and next trainings on2026-09-26.")
    target = ROOT / "configs/reconstruction/ht_reconstruction_phase61_20260926.json"
    assert not target.exists()
    target.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    print("PASS Phase61 preregistration; same data/seed/cohort, shared early PID0.5")


if __name__ == "__main__":
    main()
