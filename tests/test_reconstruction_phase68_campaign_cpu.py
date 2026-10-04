"""Fresh-source isolation, exact selector behavior, and bounded Phase68 contracts."""

import copy
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase68 import (
    validation_exclusions,
    preflight_selection,
    validate_seed_contract,
    validate_closeout_basis,
)
from scripts.run_phase68_pretraining import (
    validate_pretraining_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def cohort():
    return json.loads(
        (
            ROOT
            / "configs/reconstruction/ht_reconstruction_phase68_validation_cohort_20261004.json"
        ).read_text()
    )


def prereg():
    return json.loads(
        (
            ROOT / "configs/reconstruction/ht_reconstruction_phase68_20261004.json"
        ).read_text()
    )


def test_exact_selector_excludes_every_other_event_before_training():
    c = cohort()
    exclusions = validation_exclusions(c)
    assert preflight_selection(
        c,
        {"max_validation_events": 1000, "seed": 20261004, "scientific_mode": True},
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
        and c["remaining_untouched_after_phase68"] == 41200
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
    cfg = {"max_validation_events": 1000, "seed": 20261004, "scientific_mode": True}
    old = json.loads(
        (
            ROOT / c["source_bindings"]["previous_validation_universe"]["path"]
        ).read_text()
    )["event_uids"]
    if change == "order":
        c["checkpoint_selection_event_uids"].reverse()
    if change == "seed":
        cfg["seed"] = 20260929
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


def test_factor_isolation_and_shared_frozen_encoder():
    p = prereg()
    validate_closeout_basis(p)
    validate_seed_contract(p["common_training_contract"], cohort()["seed"])
    configs = []
    for arm in p["arms"]:
        contract = {
            "arm_role": arm["role"],
            "pretraining_refinement": {
                **p["pretraining_refinement"],
            },
        }
        validate_pretraining_contract(contract, p)
        configs.append({**p["common_training_contract"], **arm["overrides"]})
    left, right = configs
    assert left.pop("scheduled_sampling_probability") == 0.5
    assert right.pop("scheduled_sampling_probability") == 0.5
    assert left["freeze_pretrained_encoder_steps"] == right["freeze_pretrained_encoder_steps"] == 2188
    assert left == right
    assert left["recovery_objective_weight"] == 0.0
    assert left["unrepresentable_target_policy"] == "masked_representable_only"


def test_evaluation_uses_bound_size_for_strict_and_beam(tmp_path, monkeypatch):
    from scripts import run_reconstruction_phase68_full_decay as runner
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
    from scripts.run_reconstruction_phase68 import validate_evaluation_population

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


def test_replication_preserves_science_and_excludes_scored_phase61():
    p = prereg()
    old = json.loads(
        (
            ROOT / "configs/reconstruction/ht_reconstruction_phase67_20261004.json"
        ).read_text()
    )
    for key in (
        "data_binding",
        "post_training_gates",
        "evaluation_contract",
        "execution_policy",
    ):
        assert p[key] == old[key]
    assert p["common_training_contract"] == old["common_training_contract"]
    assert p["pretraining_refinement"]["additional_pretraining_steps"] == 0
    assert p["pretraining_refinement"] != old["pretraining_refinement"]
    prior = json.loads((ROOT / old["validation_cohort"]["path"]).read_text())
    used = set(prior["event_uids"]) | set(prior["checkpoint_selection_event_uids"])
    c = cohort()
    assert not used & (set(c["event_uids"]) | set(c["checkpoint_selection_event_uids"]))
    for field in ("event_uids", "checkpoint_selection_event_uids"):
        bad = copy.deepcopy(c)
        bad[field][0] = next(iter(used))
        with pytest.raises(RuntimeError):
            validation_exclusions(bad)
    assert p["validation_budget"]["newly_reserved"] == 1100
    assert p["validation_budget"]["remaining_untouched_after_phase68"] == 41200


def test_completed_basis_rejects_failed_or_contaminated_evidence():
    from scripts.run_reconstruction_phase68 import validate_completed_basis

    p = prereg()
    d = json.loads((ROOT / p["phase67_closeout_basis"]["path"]).read_text())
    validate_completed_basis(d)
    for key, value in [
        ("status", "FAILED"),
        ("strict_selection_overlap", 1),
        ("sealed_test_accessed", True),
    ]:
        bad = copy.deepcopy(d)
        bad[key] = value
        with pytest.raises(RuntimeError):
            validate_completed_basis(bad)


@pytest.mark.parametrize("arm_index", [0, 1])
def test_real_trainer_admission_for_registered_masking_arms(
    tmp_path, monkeypatch, arm_index
):
    from scripts.run_reconstruction_phase35 import _training_config
    import hypertagging.training.reconstruction_trainer as trainer

    p = prereg()
    a = p["arms"][arm_index]
    config = {**p["common_training_contract"], **a["overrides"]}
    runtime = {
        "selection_manifest": "unused",
        "dataset_index": "unused",
        "checkpoint": "unused",
        "checkpoint_sha256": a["checkpoint_sha256"],
        "checkpoint_step": "2188",
    }
    c = _training_config(
        config=config,
        runtime=runtime,
        training_output=tmp_path / "out",
        validation_exclusions=validation_exclusions(cohort()),
    )
    from dataclasses import replace

    c = replace(c, device="cpu", mixed_precision=False)

    class ReachedData(Exception):
        pass

    def sentinel(*args, **kwargs):
        raise ReachedData

    monkeypatch.setattr(trainer, "build_real_data_module", sentinel)
    with pytest.raises(ReachedData):
        trainer.train_level_reconstruction(c)


def test_context_contrast_rejects_second_factor():
    from scripts.run_reconstruction_phase68 import validate_context_contrast
    p = prereg()
    validate_context_contrast(p)
    p["arms"][1]["overrides"]["auxiliary_teacher_weight"] = 1.0
    with pytest.raises(RuntimeError, match="second scientific factor"):
        validate_context_contrast(p)


def test_context_schedule_uses_full_replay_budget():
    from hypertagging.training.scheduled_sampling import TeacherForcingSchedule
    p = prereg()
    for arm in p["arms"]:
        probability = arm["overrides"]["scheduled_sampling_probability"]
        schedule = TeacherForcingSchedule(kind="linear", start_probability=1.0, end_probability=1.0-probability, duration_steps=2188)
        predicted = sum(64*(1-schedule.probability(s)) for s in range(4376))
        assert predicted == pytest.approx(p["exposure_expectations"][arm["role"]]["nominal_predicted_slots"])
        assert p["exposure_expectations"][arm["role"]]["replay_slots"] == 280064
