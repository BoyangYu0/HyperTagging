"""Exercise actual UID selector before any GPU work, including the Phase57 failure."""

import copy
import json
import hashlib
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase59 import (
    validation_exclusions,
    preflight_selection,
    validate_seed_contract,
    validate_closeout_basis,
    ARM_ROLES,
)
from scripts.run_phase59_pretraining import (
    validate_pretraining_contract,
    refinement_config,
)

ROOT = Path(__file__).resolve().parents[1]


def cohort():
    return json.loads(
        (
            ROOT
            / "configs/reconstruction/ht_reconstruction_phase59_validation_cohort_20260924.json"
        ).read_text()
    )


def prereg():
    return json.loads(
        (
            ROOT / "configs/reconstruction/ht_reconstruction_phase59_20260924.json"
        ).read_text()
    )


def contract(p, a):
    r = copy.deepcopy(p["pretraining_refinement"])
    r["parent_ranking_weight"] = 2.0
    r["config"]["leaf_pid_phase_weights"] = a["pretraining_leaf_pid_phase_weights"]
    return {"arm_role": a["role"], "pretraining_refinement": r}


def test_real_selector_exactly_matches_preregistered_sequence_and_excludes_every_other_uid():
    c = cohort()
    ex = validation_exclusions(c)
    r = preflight_selection(
        c,
        {"max_validation_events": 1000, "seed": 20260926, "scientific_mode": True},
        ex,
    )
    assert r == {
        "status": "PASS",
        "selection_events": 1000,
        "strict_overlap": 0,
        "excluded_events": 49000,
        "before_any_training": True,
    }
    assert (
        set(c["event_uids"]) <= set(ex) and c["remaining_untouched_after_phase59"] == 0
    )


def test_phase57_exclusion_strategy_is_rejected_before_training():
    from scripts.run_reconstruction_phase57 import validation_exclusions as previous

    c = cohort()
    old = json.loads(
        (ROOT / c["source_bindings"]["phase57_cohort"]["path"]).read_text()
    )
    bad = previous(old)
    with pytest.raises(RuntimeError, match="preflight selector"):
        preflight_selection(
            c,
            {"max_validation_events": 1000, "seed": 20260926, "scientific_mode": True},
            bad,
        )


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "seed",
        "limit",
        "strict_history",
        "selection_set",
        "exclusion_count",
        "mode",
    ],
)
def test_bad_cohort_or_actual_selector_parameters_fail(change):
    c = cohort()
    cfg = {"seed": 20260926, "max_validation_events": 1000, "scientific_mode": True}
    ex = validation_exclusions(c)
    if change == "mode":
        cfg["scientific_mode"] = False
    if change == "order":
        c["checkpoint_selection_event_uids"].reverse()
    if change == "seed":
        cfg["seed"] = 20260924
    if change == "limit":
        cfg["max_validation_events"] = 999
    if change == "strict_history":
        c["event_uids"][0] = c["checkpoint_selection_event_uids"][0]
    if change == "selection_set":
        c["checkpoint_selection_event_uids"][0] = c["event_uids"][0]
    if change == "exclusion_count":
        c["validation_exclusion_event_uid_count"] = 48609
    with pytest.raises(RuntimeError):
        validation_exclusions(c)
        preflight_selection(c, cfg, ex)


def test_phase59_factor_isolation_and_bound_fresh_strict_cohort(tmp_path):
    p = prereg()
    c = cohort()
    validate_seed_contract(p["common_training_contract"], c["seed"])
    validate_closeout_basis(p)
    assert tuple(a["role"] for a in p["arms"]) == ARM_ROLES
    assert set(p["capacity_admission"]["reports_by_arm"]) == set(ARM_ROLES)
    assert all(
        r["production_training_allowed"]
        and r["query_overflow_count"] == r["cardinality_overflow_count"] == 0
        for r in p["capacity_admission"]["reports_by_arm"].values()
    )
    configs = []
    for a in p["arms"]:
        ctr = contract(p, a)
        validate_pretraining_contract(ctr, p)
        cfg = refinement_config(
            ctr,
            {"selection_manifest": "m", "dataset_index": "i", "checkpoint": "c"},
            c,
            tmp_path,
        )
        assert cfg.validation_event_uids == tuple(
            c["checkpoint_selection_event_uids"]
        ) and not set(cfg.validation_event_uids) & set(c["event_uids"])
        configs.append(ctr["pretraining_refinement"]["config"])
    assert configs[0].pop("leaf_pid_phase_weights") == [1, 1, 0.2, 0.2]
    assert configs[1].pop("leaf_pid_phase_weights") == [1, 1, 0.1, 0.1]
    assert configs[0] == configs[1] and configs[0]["objective_dominance_ratio"] == 20
    assert (
        p["common_training_contract"]["max_steps"] == 4376
        and p["data_binding"]["train_events"] == 70000
    )
    for k in (
        "phase58_closeout_basis",
        "phase58_retained_metric_basis",
        "validation_cohort",
    ):
        b = p[k]
        assert (
            hashlib.sha256((ROOT / b["path"]).read_bytes()).hexdigest() == b["sha256"]
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("objective_dominance_ratio", 21),
        ("pilot_objective_violation_action", "warn"),
        ("seed", 20260924),
        ("leaf_pid_phase_weights", [1, 1, 0.1, 0.1]),
    ],
)
def test_phase59_rejects_guard_and_factor_drift(key, value):
    p = prereg()
    c = contract(p, p["arms"][0])
    c["pretraining_refinement"]["config"][key] = value
    with pytest.raises(RuntimeError):
        validate_pretraining_contract(c, p)


def test_training_adapter_preserves_preflight_eligibility_and_seed(tmp_path):
    from scripts.run_reconstruction_phase35 import _training_config

    p = prereg()
    c = cohort()
    ex = validation_exclusions(c)
    runtime = {"selection_manifest": "m", "dataset_index": "i", "checkpoint": "c"}
    for arm in p["arms"]:
        cfg = _training_config(
            config={**p["common_training_contract"], **arm["overrides"]},
            runtime=runtime,
            training_output=tmp_path / arm["role"],
            validation_exclusions=ex,
        )
        assert cfg.scientific_mode is True and cfg.validation_enabled is True
        result = preflight_selection(
            c,
            {
                "max_validation_events": cfg.max_validation_events,
                "seed": cfg.seed,
                "scientific_mode": cfg.scientific_mode,
            },
            cfg.validation_excluded_event_uids,
        )
        assert (
            result["strict_overlap"] == 0
            and result["selection_events"] == cfg.rollout_validation_events == 1000
        )


def test_reuse_of_phase58_strict_events_rejected():
    c = cohort()
    previous = json.loads(
        (ROOT / c["source_bindings"]["phase58_cohort"]["path"]).read_text()
    )
    c["event_uids"][0] = previous["event_uids"][0]
    with pytest.raises(RuntimeError, match="untouched"):
        validation_exclusions(c)


def test_confirmatory_claim_or_extra_validation_budget_rejected():
    for key, value in [
        ("pilot_classification", "CONFIRMATORY"),
        ("validation_budget", {"strict_events": 100}),
    ]:
        p = prereg()
        p[key] = value
        with pytest.raises(RuntimeError, match="feasibility"):
            validate_closeout_basis(p)


def test_retained_evidence_accepts_native_phase58_and_rejects_diagnostic_phase57():
    from scripts.run_reconstruction_phase59 import validate_retained_basis

    p = prereg()
    binding = p["phase58_retained_metric_basis"]
    evidence = json.loads((ROOT / binding["path"]).read_text())
    validate_retained_basis(evidence, binding)
    evidence["version"] = "phase57-retained-tree-export-v1"
    with pytest.raises(RuntimeError, match="retained-tree"):
        validate_retained_basis(evidence, binding)
