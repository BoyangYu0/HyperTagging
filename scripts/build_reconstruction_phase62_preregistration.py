"""Reserve a fresh Phase62 cohort and bind the independent replication."""

from pathlib import Path
import json, hashlib, copy, sys, datetime

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / "src")]
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    uid_sequence_sha256,
    uid_set_sha256,
)

O = Path(__file__).parent
STUDY = "phase62-early-pid-independent-replication-20260927"


def bind(p):
    return {
        "path": str(p.relative_to(R)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def write(p, d):
    assert not p.exists()
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


prior = R / "configs/reconstruction/ht_reconstruction_phase61_20260926.json"
p = json.loads(prior.read_text())
oldpath = R / p["validation_cohort"]["path"]
old = json.loads(oldpath.read_text())
uni = R / old["source_bindings"]["validation_universe"]["path"]
universe = json.loads(uni.read_text())
older = R / old["source_bindings"]["previous_validation_universe"]["path"]
used = (
    set(json.loads(older.read_text())["event_uids"])
    | set(old["checkpoint_selection_event_uids"])
    | set(old["event_uids"])
)
assert len(used) == 51100
ledger = (
    R / "configs/reconstruction/ht_reconstruction_phase62_used_validation_20260927.json"
)
write(
    ledger,
    {
        "version": "validation-selection-ledger-v1",
        "event_uids": sorted(used),
        "event_uid_count": 51100,
        "event_uids_sha256": uid_set_sha256(used),
        "source_bindings": [bind(older), bind(oldpath)],
        "sealed_test_accessed": False,
        "policy": "All original50000 exhausted; Phase60 reservation1000+100 scored by Phase61. Never reuse for Phase62.",
    },
)
fresh = set(universe["event_uids"]) - used
assert len(fresh) == 48900
ordered = ranked(fresh, 20260928)
selected = ordered[:1000]
strict = ordered[1000:1100]
excluded = set(universe["event_uids"]) - set(selected)
c = copy.deepcopy(old)
c.pop("remaining_untouched_after_phase60")
c.update(
    manifest_version="hypertagging-reconstruction-phase62-cohort-v1",
    study_id=STUDY,
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    seed=20260928,
    selection_original_seed=20260928,
    strict_selection_seed=20260928,
    remaining_untouched_after_phase62=47800,
    historical_used_event_uid_count=51100,
    checkpoint_selection_event_uids=selected,
    checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
    evaluation_event_uids=strict,
    event_uids=strict,
    evaluation_event_uids_sha256=uid_sequence_sha256(strict),
    event_uids_sha256=uid_sequence_sha256(strict),
    validation_exclusion_event_uids_sha256=uid_set_sha256(excluded),
    selection_reuse_policy="No reuse:1000 selection plus100 strict from48900 unreserved UIDs.",
    overlap_audit={
        "strict_vs_history": len(set(strict) & used),
        "selection_vs_history": len(set(selected) & used),
        "strict_vs_selection": len(set(strict) & set(selected)),
        "selection_vs_exclusions": len(set(selected) & excluded),
    },
)
c["source_bindings"]["previous_validation_universe"] = bind(ledger)
c["source_bindings"]["phase61_scored_cohort"] = bind(oldpath)
cohort = (
    R
    / "configs/reconstruction/ht_reconstruction_phase62_validation_cohort_20260927.json"
)
write(cohort, c)
# Preserve all budgets, objective and scientific guards. Change only paired seed and validation allocation.
p.pop("parent_phase60")
p.pop("phase60_closeout_basis")
p["parent_phase61"] = bind(prior)
p.update(
    study_id=STUDY,
    preregistration_version="hypertagging-reconstruction-phase62-preregistration-v1",
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pilot_classification="INDEPENDENT_SEED_REPLICATION",
    scientific_question="Replicate Phase61 earlyPID0.5 and latePID0.2 versus0.1 at seed20260928 on untouched validation. Does either schedule reproducibly improve joint topology and coherent forests at fixed data and budgets?",
)
p["common_training_contract"]["seed"] = 20260928
p["common_training_contract"]["balanced_level_replay_contract"]["seed"] = 20260928
p["pretraining_refinement"]["config"]["seed"] = 20260928
p["pretraining_refinement"]["comparison_limitation"] = (
    "One independent training seed and fresh event cohort; same validation source pool. No cross-phase causal rate comparison or promotion."
)
p["pretraining_refinement"]["validation_population"] = (
    "fresh1000_selection_excluding_all51100_previously_scored_or_reserved"
)
p["scientific_source_boundary"]["limitation"] = (
    "Same step81096 initialization and earlyPID0.5 schedule as Phase61. Fresh seed/selection/strict cohort; no strict outcome tuning. Compare latePID0.2 versus0.1 within this pair only."
)
for a in p["arms"]:
    a["hypothesis"] = (
        "Neither late-PID schedule is established as superior. Require replicated joint topology and coherent-forest evidence; preserve stability failures."
    )
p["validation_budget"] = {
    "validation_role_events": 100000,
    "previously_used": 51100,
    "new_independent_validation_events": 0,
    "newly_reserved": 1100,
    "reused_selection_events": 0,
    "selection_reservation_reused": False,
    "strict_reservation_reused": False,
    "selection_events": 1000,
    "rollout_selection_events": 1000,
    "strict_events": 100,
    "reconstruction_excluded_events": 99000,
    "remaining_untouched_after_phase62": 47800,
    "limitation": "Independent event/seed replication within existing source pool; not a new source-domain generalization experiment.",
}
p["validation_cohort"] = {
    **p["validation_cohort"],
    **bind(cohort),
    "checkpoint_selection_event_uids_sha256": c[
        "checkpoint_selection_event_uids_sha256"
    ],
    "evaluation_event_uids_sha256": c["event_uids_sha256"],
}
p["phase61_closeout_basis"] = {
    **bind(R / "artifacts/codex/reconstruction_phase61_closeout_20260927.json"),
    "classification": "completed_adaptive_pilot_mixed_quality",
    "selected_next_factor": "independent_seed_replication",
    "sealed_test_accessed": False,
}
p.pop("phase59_retained_metric_basis")
p["phase61_retained_metric_basis"] = {
    **bind(R / "artifacts/codex/reconstruction_phase61_retained_metrics_20260927.json"),
    "version": "phase61-retained-tree-export-v1",
}
p["decision_rules"].update(
    dataset_size="Hold70000 training events; no controlled data-size benefit established. Existing independent validation capacity suffices.",
    pretraining="Replicate objective quality/stability, not longer duration or another dose. Same early0.5 and late0.2vs0.1.",
    stop_dose_tuning="Exactly one bounded independent pair; no automatic follow-on, promotion or sealed test.",
    authority="User authorized Phase61 review, publication and next training submissions on2026-09-27.",
)
p["runtime_variation_boundary"] = {
    "phase61_shared_prefix_identical": False,
    "first_logged_loss_divergence_step": 944,
    "different_step1094_tensors": 123,
    "step1094_tensor_count": 141,
    "policy": "Preserve native divergence. Replicate configured schedule/seed pairing without claiming common-random-number identity or isolating runtime error. No deterministic kernel or guard change.",
}
write(R / "configs/reconstruction/ht_reconstruction_phase62_20260927.json", p)
