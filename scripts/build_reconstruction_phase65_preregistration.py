"""Reserve one bounded frozen-versus-late transfer comparison on repaired geometry."""

from pathlib import Path
import copy, datetime, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    uid_sequence_sha256,
    uid_set_sha256,
)


def binding(p):
    return {
        "path": str(p.relative_to(ROOT)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def write(p, d):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build():
    oldpath = ROOT / "configs/reconstruction/ht_reconstruction_phase64_20260930.json"
    p = json.loads(oldpath.read_text())
    oldcp = ROOT / p["validation_cohort"]["path"]
    old = json.loads(oldcp.read_text())
    hist = ROOT / old["source_bindings"]["previous_validation_universe"]["path"]
    used = (
        set(json.loads(hist.read_text())["event_uids"])
        | set(old["event_uids"])
        | set(old["checkpoint_selection_event_uids"])
    )
    assert len(used) == 54400
    universe = set(
        json.loads(
            (ROOT / old["source_bindings"]["validation_universe"]["path"]).read_text()
        )["event_uids"]
    )
    fresh = ranked(universe - used, 20261001)
    selection, strict = fresh[:1000], fresh[1000:1100]
    ledger = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase65_used_validation_20261001.json"
    )
    write(
        ledger,
        {
            "version": "validation-selection-ledger-v1",
            "event_uids": sorted(used),
            "event_uid_count": len(used),
            "event_uids_sha256": uid_set_sha256(used),
            "source_bindings": [binding(hist), binding(oldcp)],
            "sealed_test_accessed": False,
        },
    )
    c = copy.deepcopy(old)
    c.pop("remaining_untouched_after_phase64", None)
    study = "phase65-unsaturated-transfer-20261001"
    c.update(
        manifest_version="hypertagging-reconstruction-phase65-cohort-v1",
        study_id=study,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=20261001,
        selection_original_seed=20261001,
        strict_selection_seed=20261001,
        remaining_untouched_after_phase65=44500,
        historical_used_event_uid_count=54400,
        checkpoint_selection_event_uids=selection,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        validation_exclusion_event_uids_sha256=uid_set_sha256(
            universe - set(selection)
        ),
        selection_reuse_policy="Fresh 1000 selection and 100 strict; excludes all 54400 previous reservations.",
        overlap_audit={
            "strict_vs_history": len(set(strict) & used),
            "selection_vs_history": len(set(selection) & used),
            "strict_vs_selection": len(set(strict) & set(selection)),
        },
    )
    c["source_bindings"]["previous_validation_universe"] = binding(ledger)
    c["source_bindings"]["phase64_scored_cohort"] = binding(oldcp)
    cp = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase65_validation_cohort_20261001.json"
    )
    write(cp, c)
    p.update(
        study_id=study,
        preregistration_version="hypertagging-reconstruction-phase65-preregistration-v1",
        created_at=c["created_at"],
        pilot_classification="UNSATURATED_TRANSFER_PILOT",
        scientific_question="Does freezing the shared unsaturated Phase64 refined encoder preserve or improve nontrivial recursive quality relative to late adaptation at matched reconstruction compute?",
    )
    for key in (
        "parent_phase63",
        "phase63_closeout_basis",
        "phase63_retained_metric_basis",
        "geometry_intervention",
    ):
        p.pop(key, None)
    p["parent_phase64"] = binding(oldpath)
    common = p["common_training_contract"]
    common["seed"] = 20261001
    common["balanced_level_replay_contract"]["seed"] = 20261001
    source = Path(
        "/project/agkuhr/users/boyang/data/HyperTagging_artifacts/phase64_review_20260930/training-source/artifacts/runs/ht-reconstruction-phase64-20260930/radial_reconditioned/16773576/pretraining/checkpoint.pt"
    )
    checkpoint = "runtime_inputs/reconstruction_phase65_20261001/pretraining-reconditioned-step2188.pt"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    template = copy.deepcopy(p["arms"][0])
    capacity = copy.deepcopy(
        p["capacity_admission"]["reports_by_arm"]["radial_control"]
    )
    p["arms"] = []
    for role, freeze in [("late_adaptation", 2188), ("frozen_encoder", 4376)]:
        arm = copy.deepcopy(template)
        arm.pop("projection_scale", None)
        arm.update(
            role=role,
            label=role,
            checkpoint=checkpoint,
            checkpoint_sha256=sha,
            checkpoint_step=2188,
            hypothesis="Freeze usable geometry to test representation transfer versus late task adaptation, not a promoted encoder or demonstrated quality benefit.",
        )
        arm["overrides"]["freeze_pretrained_encoder_steps"] = freeze
        p["arms"].append(arm)
    p["capacity_admission"]["reports_by_arm"] = {
        a["role"]: copy.deepcopy(capacity) for a in p["arms"]
    }
    p["pretraining_refinement"] = {
        "mode": "reuse_frozen_phase64_reconditioned",
        "step": 2188,
        "checkpoint_for_reconstruction": "fixed_final_step_2188",
        "additional_pretraining_steps": 0,
    }
    close = ROOT / "artifacts/codex/reconstruction_phase64_closeout_20261001.json"
    p["phase64_closeout_basis"] = {
        **binding(close),
        "classification": "completed_pair_both_fail_original_gates",
        "selected_next_factor": "frozen_versus_late_unsaturated_transfer",
        "sealed_test_accessed": False,
    }
    p["phase64_retained_metric_basis"] = {
        **binding(close),
        "version": "phase64-closeout-v1",
    }
    radial = ROOT / "artifacts/codex/phase64_posttraining_radial_20261001.json"
    d = json.loads(radial.read_text())
    assert all(
        v["radial_saturated_fraction"] < 0.01 and v["radial_derivative_mean"] > 0.1
        for k, views in d["models"].items()
        if k.startswith("radial_reconditioned")
        for v in views.values()
    )
    p["geometry_admission"] = {
        **binding(radial),
        "maximum_saturated_fraction": 0.01,
        "minimum_mean_radial_derivative": 0.1,
        "calibration_or_tuning_on_strict": False,
    }
    p["validation_cohort"].update(
        binding(cp),
        checkpoint_selection_event_uids_sha256=c[
            "checkpoint_selection_event_uids_sha256"
        ],
        evaluation_event_uids_sha256=c["event_uids_sha256"],
    )
    p["validation_budget"].pop("remaining_untouched_after_phase64", None)
    p["validation_budget"].update(
        previously_used=54400, remaining_untouched_after_phase65=44500
    )
    p["decision_rules"].update(
        authority="User explicitly authorized one bounded next campaign on 2026-10-01.",
        pretraining="Reuse one authenticated Phase64 reconditioned final refinement checkpoint; zero additional pretraining in both arms.",
        stop_dose_tuning="Exactly two reconstruction jobs. No automatic successor, sweep, promotion, sealed test or longer budget.",
        representation_beneficial="All original gates plus joint nontrivial LCAG/source and coherent forest evidence; report depth>=2 endpoints and paired uncertainty. No benefit claim from geometry alone.",
    )
    p["scientific_source_boundary"]["limitation"] = (
        "Single-seed frozen-versus-late transfer diagnostic under repaired geometry; no data-size or pretraining-duration causal conclusion."
    )
    p["runtime_variation_boundary"]["policy"] = (
        "Same authenticated pretrained encoder and fresh decoder seed; no bitwise shared-prefix claim; conditional event uncertainty excludes training-seed variation."
    )
    write(ROOT / "configs/reconstruction/ht_reconstruction_phase65_20261001.json", p)


if __name__ == "__main__":
    build()
