"""Reserve a fresh, bounded corrected-supervision comparison after Phase62."""

from pathlib import Path
import copy, datetime, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    uid_sequence_sha256,
    uid_set_sha256,
)

STUDY = "phase63-legal-supervision-recovery-20260928"


def binding(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build(checkpoint):
    oldpath = ROOT / "configs/reconstruction/ht_reconstruction_phase62_20260927.json"
    prereg = json.loads(oldpath.read_text())
    oldcohortpath = ROOT / prereg["validation_cohort"]["path"]
    old = json.loads(oldcohortpath.read_text())
    historypath = ROOT / old["source_bindings"]["previous_validation_universe"]["path"]
    used = (
        set(json.loads(historypath.read_text())["event_uids"])
        | set(old["checkpoint_selection_event_uids"])
        | set(old["event_uids"])
    )
    assert len(used) == 52200
    universepath = ROOT / old["source_bindings"]["validation_universe"]["path"]
    universe = set(json.loads(universepath.read_text())["event_uids"])
    fresh = ranked(universe - used, 20260929)
    selected, strict = fresh[:1000], fresh[1000:1100]
    ledger = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase63_used_validation_20260928.json"
    )
    write(
        ledger,
        {
            "version": "validation-selection-ledger-v1",
            "event_uids": sorted(used),
            "event_uid_count": len(used),
            "event_uids_sha256": uid_set_sha256(used),
            "source_bindings": [binding(historypath), binding(oldcohortpath)],
            "sealed_test_accessed": False,
        },
    )
    cohort = copy.deepcopy(old)
    cohort.pop("remaining_untouched_after_phase62", None)
    cohort.update(
        manifest_version="hypertagging-reconstruction-phase63-cohort-v1",
        study_id=STUDY,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=20260929,
        selection_original_seed=20260929,
        strict_selection_seed=20260929,
        remaining_untouched_after_phase63=46700,
        historical_used_event_uid_count=52200,
        checkpoint_selection_event_uids=selected,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        validation_exclusion_event_uids_sha256=uid_set_sha256(universe - set(selected)),
        selection_reuse_policy="Fresh1000 selection and100 strict; all52200 previous reservations excluded.",
        overlap_audit={
            "strict_vs_history": len(set(strict) & used),
            "selection_vs_history": len(set(selected) & used),
            "strict_vs_selection": len(set(strict) & set(selected)),
        },
    )
    cohort["source_bindings"]["previous_validation_universe"] = binding(ledger)
    cohort["source_bindings"]["phase62_scored_cohort"] = binding(oldcohortpath)
    cohortpath = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase63_validation_cohort_20260928.json"
    )
    write(cohortpath, cohort)
    prereg.pop("parent_phase61", None)
    prereg.pop("phase61_closeout_basis", None)
    prereg.pop("phase61_retained_metric_basis", None)
    prereg.pop("shared_early_pid_repair", None)
    prereg["parent_phase62"] = binding(oldpath)
    prereg.update(
        study_id=STUDY,
        preregistration_version="hypertagging-reconstruction-phase63-preregistration-v1",
        created_at=cohort["created_at"],
        pilot_classification="CORRECTED_SUPERVISION_PILOT",
        scientific_question="After legal-daughter and confidence-mask repairs, does unmatched-slot object recovery improve nontrivial recursive reconstruction over masking-only supervision, at identical data, initialization and compute?",
    )
    common = prereg["common_training_contract"]
    common["seed"] = 20260929
    common["balanced_level_replay_contract"]["seed"] = 20260929
    # Confidence targets are repaired, but alternative-candidate calibration is not established.
    # Keep the historical scoring rule fixed within this exploratory pair.
    prereg["pretraining_refinement"] = {
        "mode": "reuse_frozen_phase62_control",
        "step": 2188,
        "checkpoint_for_reconstruction": "fixed_final_step_2188",
        "additional_pretraining_steps": 0,
    }
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_rel = (
        "runtime_inputs/reconstruction_phase62_20260928/pretraining-control-step2188.pt"
    )
    original = prereg["arms"][0]
    capacity = prereg["capacity_admission"]["reports_by_arm"][
        "pretraining_balance_control"
    ]
    prereg["arms"] = []
    for role, policy, weight in [
        ("corrected_recovery", "recovery_objective", 2.0),
        ("masked_only", "masked_representable_only", 0.0),
    ]:
        arm = {
            "role": role,
            "label": role,
            "checkpoint": checkpoint_rel,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": 2188,
            "hypothesis": "Object-only encouragement may not repair irreversibly missing subtrees; compare against masking-only without changing geometry or search.",
            "overrides": {
                **original["overrides"],
                "unrepresentable_target_policy": policy,
                "recovery_objective_weight": weight,
            },
        }
        prereg["arms"].append(arm)
    prereg["capacity_admission"]["reports_by_arm"] = {
        a["role"]: copy.deepcopy(capacity) for a in prereg["arms"]
    }
    prereg["phase62_closeout_basis"] = {
        **binding(
            ROOT / "artifacts/codex/reconstruction_phase62_closeout_20260928.json"
        ),
        "classification": "completed_replication_control_gates_only",
        "selected_next_factor": "legal_supervision_recovery_ablation",
        "sealed_test_accessed": False,
    }
    prereg["phase62_retained_metric_basis"] = {
        **binding(
            ROOT
            / "artifacts/codex/reconstruction_phase62_retained_metrics_20260928.json"
        ),
        "version": "phase62-retained-tree-export-v1",
    }
    prereg["validation_cohort"].update(binding(cohortpath))
    prereg["validation_cohort"].update(
        checkpoint_selection_event_uids_sha256=cohort[
            "checkpoint_selection_event_uids_sha256"
        ],
        evaluation_event_uids_sha256=cohort["event_uids_sha256"],
    )
    prereg["validation_budget"] = {
        **prereg["validation_budget"],
        "previously_used": 52200,
        "remaining_untouched_after_phase63": 46700,
    }
    prereg["validation_budget"].pop("remaining_untouched_after_phase62", None)
    prereg["decision_rules"].update(
        authority="User requested all fixes, Phase62 review and next-phase submission on2026-09-28.",
        pretraining="Reuse authenticated final Phase62 control encoder in both arms; no new pretraining.",
        stop_dose_tuning="One corrected-supervision comparison only; no PID dose sweep, automatic successor, promotion or sealed-test access.",
    )
    prereg["scientific_source_boundary"]["limitation"] = (
        "Shared legal-daughter, recovery-slot and confidence-supervision fixes; frozen Phase62 control encoder; fresh reconstruction optimizer and RNG. Historical comparisons are descriptive only."
    )
    prereg["runtime_variation_boundary"] = {
        "policy": "Same frozen pretraining checkpoint and reconstruction seed, intervention begins at first missing target. No common-prefix identity or causal small-effect claim; one exploratory seed."
    }
    prereg["shared_supervision_repairs"] = [
        "forest_pointer_eligibility_v2",
        "unmatched_recovery_slots_v2",
        "masked_constrained_candidate_confidence_v2",
    ]
    write(
        ROOT / "configs/reconstruction/ht_reconstruction_phase63_20260928.json", prereg
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    build(parser.parse_args().checkpoint)
