"""Bounded contracts and saved-count tests; never touch real scientific data."""

import copy
from types import SimpleNamespace

import pytest
import torch
from scripts import phase85_fresh_development as runner


def request():
    return dict(
        stage="development",
        arms=list(runner.ARMS),
        training_source_sha=runner.TRAINING_SOURCE,
        checkpoint_step=6000,
        threshold=0.5,
        optimizer_updates=0,
        sealed_test_access=False,
        maximum_events=600,
        training_pool_size=384,
        bootstrap_seed=20261010087,
        bootstrap_replicates=2000,
        plan={"path": "plan", "sha256": "plan"},
        checkpoints={a: {"path": a, "sha256": a} for a in runner.ARMS},
    )


def test_modes_and_controls():
    r = request()
    runner.validate_request(r, "smoke")
    with pytest.raises(ValueError):
        runner.validate_request(r, "evaluate")
    r.update(data_admission={"path": "data"}, admission={"path": "admitted"})
    runner.validate_request(r, "evaluate")
    with pytest.raises(ValueError):
        runner.validate_request(r, "smoke")


@pytest.mark.parametrize(
    "key,value",
    [
        ("stage", "primary"),
        ("threshold", 0.4),
        ("optimizer_updates", 1),
        ("checkpoint_step", 1500),
        ("sealed_test_access", True),
        ("training_pool_size", 1536),
        ("bootstrap_seed", 20261010085),
        ("maximum_events", 384),
        ("plan", None),
    ],
)
def test_wrong_contract(key, value):
    r = request()
    r[key] = value
    with pytest.raises(ValueError):
        runner.validate_request(r, "smoke")


def test_hash_and_duplicate_output(tmp_path):
    f = tmp_path / "file"
    f.write_text("original")
    b = runner.binding(f)
    assert runner.checked(b) == f
    f.write_text("changed")
    with pytest.raises(ValueError):
        runner.checked(b)
    out = tmp_path / "new"
    runner.claim_output(out)
    with pytest.raises(FileExistsError):
        runner.claim_output(out)


def test_checkpoint_guard():
    contract = {"settings": {"checkpoint": "fixed_final", "threshold": 0.5}}
    b = {"path": "contract", "sha256": "sha"}
    cp = dict(
        source_sha=runner.TRAINING_SOURCE,
        step=6000,
        contract=b,
        settings=contract["settings"],
        architecture=dict(context=128, hyperbolic=32, head=256, depth=4),
        connection_enabled=True,
        semantic_pair_supervision=True,
        resume_authorized=False,
    )
    runner.validate_checkpoint(cp, "connection_on", b, contract)
    for field, value in [
        ("connection_enabled", False),
        ("source_sha", "changed"),
        ("step", 1500),
        ("contract", {}),
        ("resume_authorized", True),
    ]:
        changed = copy.deepcopy(cp)
        changed[field] = value
        with pytest.raises(ValueError):
            runner.validate_checkpoint(changed, "connection_on", b, contract)


def test_rows_capacity_alias_and_category():
    rows = [dict(uid="one", category="charged", targets=torch.tensor([1, 2]))]
    runner.check_rows(rows, expected_uids=["one"])
    with pytest.raises(ValueError):
        runner.check_rows(rows * 2, expected_uids=["one"])
    with pytest.raises(ValueError):
        runner.check_rows(rows, expected_uids=["one"], per_category=100)
    rows[0]["targets"] = torch.zeros(257, dtype=torch.long)
    with pytest.raises(ValueError):
        runner.check_rows(rows, expected_uids=["one"])


def pair_fixture():
    good = dict(status="AVAILABLE", difference=0.1, interval95=[0.02, 0.18])
    safe = dict(status="AVAILABLE", difference=0, interval95=[0, 0])
    p = dict(
        paired_effects={
            "raw_exact_memberships": good.copy(),
            "accepted_exact_memberships": good.copy(),
            "continuum_accepted_events": safe.copy(),
        },
        b_event_background_fraction=safe.copy(),
        continuum_by_category={c: safe.copy() for c in runner.CATEGORIES[2:]},
        gaining_same_target_collisions=list("abcd"),
    )
    reports = {
        a: dict(
            aggregate=dict(
                counts=dict(
                    events=600,
                    nominal_b_trials=400,
                    unavailable_membership_trials=0,
                    accepted_source_conflicts=0,
                ),
                by_category={
                    c: dict(
                        raw_exact_memberships=5 + i, accepted_exact_memberships=4 + i
                    )
                    for c in runner.CATEGORIES[:2]
                },
            )
        )
        for i, a in enumerate(runner.ARMS)
    }
    return p, reports


def test_fresh_gate_requires_category_background_unknown_and_same_target():
    p, r = pair_fixture()
    assert runner.fresh_gate(p, r)["passed"]
    r["connection_on"]["aggregate"]["by_category"]["mixed"]["raw_exact_memberships"] = 0
    assert not runner.fresh_gate(p, r)["passed"]
    p, r = pair_fixture()
    p["gaining_same_target_collisions"] = list("abc")
    assert not runner.fresh_gate(p, r)["passed"]
    p, r = pair_fixture()
    r["connection_on"]["aggregate"]["counts"]["unavailable_membership_trials"] = 1
    assert not runner.fresh_gate(p, r)["passed"]
    p, r = pair_fixture()
    p["continuum_by_category"]["ccbar"]["interval95"] = [0, 0.051]
    assert not runner.fresh_gate(p, r)["passed"]


def test_correct_source_schema_and_fixed_seed():
    counts = dict(
        B_correct=2,
        B_nodes=4,
        B_to_unassigned=1,
        B_to_other_B=1,
        background_to_B=1,
        background_nodes=3,
        unknown_nodes=0,
        unknown_to_B=0,
        nodes=7,
    )
    rows = [
        dict(
            uid=str(i),
            category=c,
            stages={s: {"counts": counts.copy()} for s in ["proposal", "refinement"]},
        )
        for i, c in enumerate(["charged", "mixed"])
    ]

    def validate(c):
        if set(counts) - set(c):
            raise ValueError("Missing actual schema")

    checker = SimpleNamespace(validate_counts=validate)
    got = runner.fresh_source_errors(rows, rows, checker)
    assert got["refinement"]["source_recall"]["denominators"] == [8, 8]
    assert got["refinement"]["source_recall"]["difference"] == 0
    assert got["refinement"]["source_recall"]["seed"] == 20261010087
    assert got["refinement"]["source_precision"]["denominators"] == [8, 8]
    bad = copy.deepcopy(rows)
    bad[0]["stages"]["proposal"]["counts"].pop("B_nodes")
    with pytest.raises(ValueError):
        runner.fresh_source_errors(bad, bad, checker)
    bad = copy.deepcopy(rows)
    bad[0]["uid"] = "other"
    with pytest.raises(ValueError):
        runner.fresh_source_errors(rows, bad, checker)


def test_canonical_projection_is_truth_free_and_source_axis_exact():
    import importlib.util
    from pathlib import Path
    from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
    from hypertagging.reconstruction.hierarchical_inference import (
        project_schema_v4_fsps,
    )

    spec = importlib.util.spec_from_file_location(
        "fresh_native_fixture",
        Path(__file__).with_name("test_hierarchical_inference_cpu.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    full = module._normalized_native_v4_batch()
    policy = ReconstructionConstraintPolicy()
    a = runner.encode_normalized(full, "fixture", "charged", policy)
    projected = project_schema_v4_fsps(full)
    assert torch.equal(a["sources"], projected.batch["recursive_leaf_source_mask"][0])
    changed = {
        k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in full.items()
    }
    # Channel and retained truth PID labels are evaluation-only; generation must not see them.
    for key in ("b1_reconstructable_channel_ids", "b2_reconstructable_channel_ids"):
        if key in changed:
            changed[key].fill_(999)
    b = runner.encode_normalized(changed, "fixture", "charged", policy)
    assert a["detector"].keys() == b["detector"].keys()
    for key in a["detector"]:
        x, y = a["detector"][key], b["detector"][key]
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y), key
        else:
            assert x == y
    assert "supervision" not in a["detector"]


def test_resource_guard_and_observed_hooks_cleanup(monkeypatch):
    import time

    model = SimpleNamespace(encoder=torch.nn.Linear(2, 2))
    decoder = torch.nn.Linear(2, 2)
    with runner.observed_audit(model, decoder, time.monotonic()) as counts:
        decoder(model.encoder(torch.zeros(1, 2)))
    assert counts == {"encoder_calls": 1, "decoder_calls": 1}
    assert not decoder._forward_pre_hooks and not model.encoder._forward_pre_hooks
    with pytest.raises(ValueError):
        runner.resource_guard(time.monotonic() - 5401)


def test_restore_uses_trained_weights_and_rejects_normalizer_change(
    tmp_path, monkeypatch
):
    from scripts import phase85_pair_connection

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.runtime_feature_normalizer = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.encoder.weight.fill_(0)
                self.runtime_feature_normalizer.weight.fill_(1)

    def initialize(*args):
        return Model(), torch.nn.Linear(2, 2), None, None

    monkeypatch.setattr(phase85_pair_connection, "initialize", initialize)
    model = Model()
    decoder = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.encoder.weight.fill_(2)
        decoder.weight.fill_(3)
    contract = {"settings": {"threshold": 0.5}, "pretrain": "unused"}
    contract_path = tmp_path / "contract.json"
    runner.write(contract_path, contract)
    b = runner.binding(contract_path)
    cp = dict(
        source_sha=runner.TRAINING_SOURCE,
        step=6000,
        contract=b,
        settings=contract["settings"],
        architecture=dict(context=128, hyperbolic=32, head=256, depth=4),
        connection_enabled=True,
        semantic_pair_supervision=True,
        resume_authorized=False,
        normalizer=model.runtime_feature_normalizer,
        model_state_dict=model.state_dict(),
        decoder_state_dict=decoder.state_dict(),
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(cp, path)
    req = {
        "training_contract": b,
        "checkpoints": {"connection_on": runner.binding(path)},
    }
    original = {"runtime_normalizer": Model().runtime_feature_normalizer}

    def equal(a, b, name):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(name)
            for k in a:
                equal(a[k], b[k], name)
        elif not torch.equal(a, b):
            raise ValueError(name)

    legacy = SimpleNamespace(equal_tree=equal)
    loaded, head = runner.restore(req, original, "connection_on", legacy)
    assert torch.all(loaded.encoder.weight == 2) and torch.all(head.weight == 3)
    assert not loaded.training and not head.training
    assert not any(p.requires_grad for m in (loaded, head) for p in m.parameters())
    with torch.no_grad():
        original["runtime_normalizer"].weight.fill_(9)
    with pytest.raises(ValueError, match="normalizer"):
        runner.restore(req, original, "connection_on", legacy)
