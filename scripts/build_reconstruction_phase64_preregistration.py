"""Freeze one train-calibrated radial-initialization comparison after Phase63."""

from pathlib import Path
import copy, datetime, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    uid_sequence_sha256,
    uid_set_sha256,
)

STUDY = "phase64-radial-reconditioning-20260930"


def binding(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build():
    oldpath = ROOT / "configs/reconstruction/ht_reconstruction_phase63_20260928.json"
    prereg = json.loads(oldpath.read_text())
    oldcohortpath = ROOT / prereg["validation_cohort"]["path"]
    old = json.loads(oldcohortpath.read_text())
    historypath = ROOT / old["source_bindings"]["previous_validation_universe"]["path"]
    used = (
        set(json.loads(historypath.read_text())["event_uids"])
        | set(old["checkpoint_selection_event_uids"])
        | set(old["event_uids"])
    )
    assert len(used) == 53300
    universepath = ROOT / old["source_bindings"]["validation_universe"]["path"]
    universe = set(json.loads(universepath.read_text())["event_uids"])
    fresh = ranked(universe - used, 20260930)
    selected, strict = fresh[:1000], fresh[1000:1100]
    ledger = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase64_used_validation_20260930.json"
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
    cohort.pop("remaining_untouched_after_phase63", None)
    cohort.update(
        manifest_version="hypertagging-reconstruction-phase64-cohort-v1",
        study_id=STUDY,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=20260930,
        selection_original_seed=20260930,
        strict_selection_seed=20260930,
        remaining_untouched_after_phase64=45600,
        historical_used_event_uid_count=53300,
        checkpoint_selection_event_uids=selected,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        validation_exclusion_event_uids_sha256=uid_set_sha256(universe - set(selected)),
        selection_reuse_policy="Fresh1000 selection and100 strict; all53300 previous reservations excluded.",
        evaluation_limitation="Exploratory single-seed study on fresh events within the existing source pool; no source-domain generalization or causal cross-phase claim.",
        overlap_audit={
            "strict_vs_history": len(set(strict) & used),
            "selection_vs_history": len(set(selected) & used),
            "strict_vs_selection": len(set(strict) & set(selected)),
        },
    )
    cohort["source_bindings"]["previous_validation_universe"] = binding(ledger)
    cohort["source_bindings"]["phase63_scored_cohort"] = binding(oldcohortpath)
    cohortpath = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase64_validation_cohort_20260930.json"
    )
    write(cohortpath, cohort)
    diagpath = ROOT / "artifacts/codex/phase64_radial_diagnostics_20260930.json"
    diag = json.loads(diagpath.read_text())
    scale = diag["intervention"]["scale"]
    assert 0 < scale < 1 and diag["intervention"]["calibration_only_train"]
    assert (
        len(diag["train_calibration_event_uids"]) == 128
        and len(diag["development_event_uids"]) == 32
    )
    for model in (
        "initial_train",
        "corrected_recovery_development",
        "masked_only_development",
    ):
        assert all(
            v["radial_saturated_fraction"] > 0.99
            for v in diag["models"][model].values()
        )
    for model in ("reconditioned_train", "reconditioned_development"):
        assert all(
            v["radial_saturated_fraction"] < 0.01 and v["radial_derivative_mean"] > 0.1
            for v in diag["models"][model].values()
        )
    prereg.update(
        study_id=STUDY,
        preregistration_version="hypertagging-reconstruction-phase64-preregistration-v1",
        created_at=cohort["created_at"],
        pilot_classification="RADIAL_RECONDITIONING_PILOT",
        scientific_question="Does a train-calibrated one-time hyperbolic projection rescale improve radial usability and nontrivial recursive reconstruction after equal refinement, with masking-only supervision fixed?",
    )
    for key in (
        "parent_phase62",
        "phase62_closeout_basis",
        "phase62_retained_metric_basis",
    ):
        prereg.pop(key, None)
    prereg["parent_phase63"] = binding(oldpath)
    common = prereg["common_training_contract"]
    common["seed"] = 20260930
    common["balanced_level_replay_contract"]["seed"] = 20260930
    common["unrepresentable_target_policy"] = "masked_representable_only"
    common["recovery_objective_weight"] = 0.0
    init = copy.deepcopy(prereg["arms"][1])
    capacity = copy.deepcopy(
        prereg["capacity_admission"]["reports_by_arm"]["masked_only"]
    )
    prereg["arms"] = []
    for role, factor in [("radial_control", 1.0), ("radial_reconditioned", scale)]:
        arm = copy.deepcopy(init)
        arm.update(
            role=role,
            label=role,
            projection_scale=factor,
            hypothesis="Restore radial gradients through a single train-calibrated projection initialization change; no geometry kernel or objective change.",
        )
        prereg["arms"].append(arm)
    prereg["capacity_admission"]["reports_by_arm"] = {
        a["role"]: copy.deepcopy(capacity) for a in prereg["arms"]
    }
    ref = copy.deepcopy(
        json.loads(
            (
                ROOT / "configs/reconstruction/ht_reconstruction_phase62_20260927.json"
            ).read_text()
        )["pretraining_refinement"]
    )
    ref["config"].update(
        seed=20260930,
        leaf_pid_phase_weights=[0.5, 0.5, 0.2, 0.2],
        best_metric="validation_phase_weighted_objective",
    )
    ref["parent_ranking_weight"] = 2.0
    ref["mode"] = "matched_parameter_only_projection_initialization"
    ref["additional_pretraining_steps"] = 2188
    prereg["pretraining_refinement"] = ref
    prereg["geometry_intervention"] = {
        **diag["intervention"],
        "diagnostic_binding": binding(diagpath),
        "admission": {
            "maximum_measured_saturated_fraction": 0.01,
            "minimum_measured_mean_radial_derivative": 0.1,
        },
        "one_time_initialization_only": True,
        "direction_preserved_for_fixed_projection_input": True,
        "upstream_context_feedback_changes_actual_postscale_norms": True,
        "optimizer_and_buffers_fresh": True,
    }
    closeout = ROOT / "artifacts/codex/reconstruction_phase63_closeout_20260930.json"
    prereg["phase63_closeout_basis"] = {
        **binding(closeout),
        "classification": "completed_pair_both_fail_original_gates",
        "selected_next_factor": "train_calibrated_projection_rescaling",
        "sealed_test_accessed": False,
    }
    prereg["phase63_retained_metric_basis"] = {
        **binding(closeout),
        "version": "phase63-closeout-v1",
    }
    prereg["validation_cohort"].update(
        binding(cohortpath),
        checkpoint_selection_event_uids_sha256=cohort[
            "checkpoint_selection_event_uids_sha256"
        ],
        evaluation_event_uids_sha256=cohort["event_uids_sha256"],
    )
    prereg["validation_budget"].pop("remaining_untouched_after_phase63", None)
    prereg["validation_budget"].update(
        previously_used=53300,
        remaining_untouched_after_phase64=45600,
        limitation="Fresh events within the existing source pool; one exploratory seed, not a new source-domain generalization experiment.",
    )
    prereg["decision_rules"].update(
        authority="User requested the next step on2026-09-30 after Phase63 closeout; one bounded paired study.",
        pretraining="Same Phase62 final control source; rescale projection weight/bias once in candidate, then equal2188 refinement updates and4376 masking-only reconstruction updates.",
        stop_dose_tuning="No loss-dose sweep, automatic successor, promotion or sealed test.",
        representation_beneficial="Require usable radial gradients after refinement and joint held-out improvements in original and retained topology, nontrivial/depth-stratified components, source precision/recall and nontrivial coherent forests; all original gates necessary. More radius variance alone is insufficient.",
    )
    prereg["scientific_source_boundary"]["limitation"] = (
        "One-time train-calibrated projection reconditioning versus unchanged initialization, equal refinement and masking-only reconstruction. Prior cross-phase comparisons descriptive; no inference kernel or loss-weight change within this pair."
    )
    prereg["runtime_variation_boundary"] = {
        "policy": "Same source parameters except the registered projection rescale; same seed/data/compute. No bitwise common-prefix or training-seed uncertainty claim; single exploratory pair."
    }
    write(
        ROOT / "configs/reconstruction/ht_reconstruction_phase64_20260930.json", prereg
    )


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
