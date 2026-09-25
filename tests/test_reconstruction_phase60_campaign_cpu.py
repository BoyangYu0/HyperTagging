"""Fresh-source isolation, exact selector behavior, and bounded Phase60 contracts."""

import copy
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase60 import (
    validation_exclusions,
    preflight_selection,
    validate_seed_contract,
    validate_closeout_basis,
    validate_retained_basis,
)
from scripts.run_phase60_pretraining import (
    validate_pretraining_contract,
    refinement_config,
)

ROOT = Path(__file__).resolve().parents[1]


def cohort():
    return json.loads(
        (
            ROOT
            / "configs/reconstruction/ht_reconstruction_phase60_validation_cohort_20260925.json"
        ).read_text()
    )


def prereg():
    return json.loads(
        (
            ROOT / "configs/reconstruction/ht_reconstruction_phase60_20260925.json"
        ).read_text()
    )


def test_exact_selector_excludes_every_other_event_before_training():
    c = cohort()
    exclusions = validation_exclusions(c)
    assert preflight_selection(
        c,
        {"max_validation_events": 1000, "seed": 20260927, "scientific_mode": True},
        exclusions,
    ) == {
        "status": "PASS",
        "selection_events": 1000,
        "strict_overlap": 0,
        "excluded_events": 99000,
        "before_any_training": True,
    }
    assert (
        c["permitted_selection_reuse_count"] == 0
        and c["remaining_untouched_after_phase60"] == 48900
    )


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "seed",
        "count",
        "observed_strict",
        "observed_selection",
        "overlap",
        "exclusion_hash",
        "scientific_mode",
    ],
)
def test_stale_or_contaminated_selection_rejected(change):
    c = cohort()
    cfg = {"max_validation_events": 1000, "seed": 20260927, "scientific_mode": True}
    old = json.loads(
        (
            ROOT / c["source_bindings"]["previous_validation_universe"]["path"]
        ).read_text()
    )["event_uids"]
    if change == "order":
        c["checkpoint_selection_event_uids"].reverse()
    if change == "seed":
        cfg["seed"] = 20260926
    if change == "count":
        cfg["max_validation_events"] = 999
    if change == "observed_strict":
        c["event_uids"][0] = old[0]
    if change == "observed_selection":
        c["checkpoint_selection_event_uids"][0] = old[0]
    if change == "overlap":
        c["event_uids"][0] = c["checkpoint_selection_event_uids"][0]
    if change == "exclusion_hash":
        c["validation_exclusion_event_uids_sha256"] = "0" * 64
    if change == "scientific_mode":
        cfg["scientific_mode"] = False
    with pytest.raises(RuntimeError):
        preflight_selection(c, cfg, validation_exclusions(c))


def test_expansion_preserves_every_old_role_and_training_payload():
    olddir = ROOT / "configs/training_selection/production_1m_20260812"
    newdir = ROOT / "configs/training_selection/phase60_validation_expansion_20260925"
    oldroles = json.loads((olddir / "roles.json").read_text())["entries"]
    newroles = json.loads((newdir / "roles.json").read_text())["entries"]
    lookup = {e["task_id"]: e for e in newroles}
    for e in oldroles:
        assert (
            lookup[e["task_id"]]["role"] == e["role"]
            and lookup[e["task_id"]]["source_file"] == e["source_file"]
        )
    oldsources = {e["source_file"] for e in oldroles}
    fresh = [
        e for e in newroles if e["task_id"] not in {r["task_id"] for r in oldroles}
    ]
    assert len(fresh) == 10 and all(
        e["role"] == "validation" and e["source_file"] not in oldsources for e in fresh
    )
    old = json.loads((olddir / "train_070k_phase40.json").read_text())
    new = json.loads((newdir / "train_070k.json").read_text())
    training = lambda d: {
        (e["task_id"], e["parquet_sha256_reference"])
        for e in d["entries"]
        if e["split"] == "train"
    }
    assert training(old) == training(new)
    assert new["selection_includes_test"] is False and not any(
        e["split"] == "test" for e in new["entries"]
    )


def test_factor_isolation_and_actual_refinement_config(tmp_path):
    p = prereg()
    c = cohort()
    validate_closeout_basis(p)
    validate_seed_contract(p["common_training_contract"], c["seed"])
    values = []
    for arm in p["arms"]:
        ref = copy.deepcopy(p["pretraining_refinement"])
        ref["parent_ranking_weight"] = 2.0
        ref["config"]["leaf_pid_phase_weights"] = arm[
            "pretraining_leaf_pid_phase_weights"
        ]
        contract = {"arm_role": arm["role"], "pretraining_refinement": ref}
        validate_pretraining_contract(contract, p)
        cfg = refinement_config(
            contract,
            {
                "selection_manifest": "selection.json",
                "dataset_index": "index.json",
                "checkpoint": "initial.pt",
            },
            c,
            tmp_path,
        )
        assert tuple(cfg.validation_event_uids) == tuple(
            c["checkpoint_selection_event_uids"]
        )
        assert (
            cfg.max_steps == 2188
            and cfg.objective_dominance_ratio == 20
            and cfg.pilot_objective_violation_action == "fail"
        )
        values.append(cfg.leaf_pid_phase_weights)
        bad = copy.deepcopy(contract)
        bad["pretraining_refinement"]["config"]["max_steps"] = 4376
        with pytest.raises(RuntimeError):
            validate_pretraining_contract(bad, p)
    assert values == [(1.0, 1.0, 0.2, 0.2), (1.0, 1.0, 0.1, 0.1)]
    bad = copy.deepcopy(p)
    bad["validation_budget"]["remaining_untouched_after_phase60"] = 0
    with pytest.raises(RuntimeError):
        validate_closeout_basis(bad)
    ret = json.loads((ROOT / p["phase59_retained_metric_basis"]["path"]).read_text())
    validate_retained_basis(ret, p["phase59_retained_metric_basis"])
    ret["all_returned_beam_candidates_checked"] = False
    with pytest.raises(RuntimeError):
        validate_retained_basis(ret, p["phase59_retained_metric_basis"])


def test_evaluation_uses_bound_size_for_strict_and_beam(tmp_path, monkeypatch):
    from scripts import run_reconstruction_phase60_full_decay as runner
    import hypertagging.evaluation.retained_tree_checks as checks

    calls = []
    monkeypatch.setattr(
        runner, "_legacy_run_evaluator", lambda **kw: calls.append(kw) or {}
    )
    monkeypatch.setattr(checks, "validate_retained_tree_report", lambda report: None)
    for n in (100, 20):
        m = tmp_path / f"{n}.json"
        m.write_text(json.dumps({"event_uid_count": n, "event_uids": list(range(n))}))
        kw = {
            "cohort_manifest": m,
            "contract": {"evaluation_contract": {"max_events": 100}},
        }
        if n == 20:
            kw["max_events"] = 20
        runner._run_evaluator(**kw)
        assert calls[-1]["max_events"] == n
        kw["max_events"] = 25
        with pytest.raises(RuntimeError):
            runner._run_evaluator(**kw)


@pytest.mark.parametrize("strict,beam", [(25, 20), (100, 25), (100, 0)])
def test_wrong_evaluator_counts_fail_before_training(strict, beam):
    from scripts.run_reconstruction_phase60 import validate_evaluation_population

    c = cohort()
    e = copy.deepcopy(prereg()["evaluation_contract"])
    validate_evaluation_population(e, c)
    e["max_events"] = strict
    e["beam_search"]["max_events"] = beam
    with pytest.raises(RuntimeError, match="population differs"):
        validate_evaluation_population(e, c)


def test_shared_capacity_repair_covers_all_new_targets_without_dropping_them():
    from hypertagging.data.capacity import production_capacity_report
    from hypertagging.training.model_config import MODEL_PRESETS

    p = prereg()
    c = p["common_training_contract"]
    index = json.loads((ROOT / p["data_binding"]["dataset_index"]).read_text())

    def capacity(limits):
        return production_capacity_report(
            index,
            global_n_queries=MODEL_PRESETS[c["model_preset"]].n_queries,
            global_max_cardinality=c["max_cardinality"],
            n_queries_by_level=dict(c["n_queries_by_level"]),
            max_cardinality_by_level=limits,
            target_policy="complete_only",
        )

    limits = dict(c["max_cardinality_by_level"])
    old = capacity({**limits, 5: 13})
    new = capacity(limits)
    assert (
        old["cardinality_overflow_count"] == 1
        and not old["production_training_allowed"]
    )
    assert (
        new["cardinality_overflow_count"] == new["query_overflow_count"] == 0
        and new["production_training_allowed"]
    )
    assert limits[5] == 14 and c["max_cardinality"] == 16
    assert all("max_cardinality_by_level" not in a["overrides"] for a in p["arms"])
