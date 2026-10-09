import copy
import torch
import pytest
from scripts.diagnose_phase82_nonlinear_probe import (
    prepare_fit,
    new_probe,
    score_features,
    fit,
    risk,
)


def records():
    g = torch.Generator().manual_seed(82)
    return [
        {
            "uid": str(i),
            "partition": "fit" if i < 3 else "assessment",
            "encoder": torch.randn(6, 128, generator=g),
            "ij": torch.triu_indices(6, 6, 1),
            "target": torch.tensor([1, 1, 2, 2, 0, -1]),
        }
        for i in range(4)
    ]


def reference():
    return {"mean": torch.zeros(256), "scale": torch.ones(256)}


def test_fit_partition_excludes_assessment_labels_and_preserves_support():
    rows = records()
    x, y, w, c, ids = prepare_fit(rows, reference())
    assert ids == ["0", "1", "2"]
    assert len(x) == 18 and x.shape[1] == 256
    torch.testing.assert_close(w.sum(), torch.tensor(1.0))
    rows[-1]["target"] = object()
    other = prepare_fit(rows, reference())
    for a, b in zip((x, y, w), other[:3]):
        torch.testing.assert_close(a, b)
    bad = copy.deepcopy(rows)
    bad[0]["target"][0] = 3
    with pytest.raises(ValueError):
        prepare_fit(bad, reference())


def test_symmetric_scores_and_target_slot_renaming():
    r = records()[0]
    model = new_probe()
    ref = reference()
    a = score_features(model, r["encoder"], r["ij"], ref)
    b = score_features(model, r["encoder"], r["ij"].flip(0), ref)
    torch.testing.assert_close(a, b)
    rr = records()
    before = prepare_fit(rr, ref)
    for row in rr:
        row["target"] = torch.where(row["target"] > 0, 3 - row["target"], row["target"])
    after = prepare_fit(rr, ref)
    for x, y in zip(before[:3], after[:3]):
        torch.testing.assert_close(x, y)


def test_probe_gradients_and_finite_matched_fit_without_encoder_training():
    rows = records()
    ref = reference()
    model, a, stats = fit(rows, ref, updates=4, batch=8)
    null, b, other = fit(rows, ref, True, updates=4, batch=8)
    assert stats["initial_state_sha256"] == other["initial_state_sha256"]
    assert stats["event_class_order_sha256"] == other["event_class_order_sha256"]
    assert stats["parameters"] == 16513 and stats["optimizer_pair_presentations"] == 32
    assert all(
        not r["encoder"].requires_grad and r["encoder"].grad is None for r in rows
    )
    assert stats["final_risk_gradient"]["gradient_l2"] > 0
    assert any(not torch.equal(a["state"][k], b["state"][k]) for k in a["state"])


def test_exact_event_class_risk_gradient_matches_finite_difference():
    x, y, w, _, _ = prepare_fit(records(), reference())
    model = new_probe()
    out = risk(model, x, y, w, gradient=True)
    p = list(model.parameters())[-1]
    analytic = float(p.grad[0])
    initial = float(p[0])
    eps = 0.001
    with torch.no_grad():
        p[0] = initial + eps
    plus = risk(model, x, y, w)["risk"]
    with torch.no_grad():
        p[0] = initial - eps
    minus = risk(model, x, y, w)["risk"]
    assert abs((plus - minus) / (2 * eps) - analytic) < 1e-4
    assert out["risk"] > 0


def test_diagnostic_contract_rejects_scope_resources_and_budget_changes():
    from scripts.diagnose_phase82_nonlinear_probe import validate_contract

    c = dict(
        stage="training_role_diagnostic",
        updates=2048,
        batch_size=256,
        probe_parameters=16513,
        arms=["nonlinear_probe", "shuffled_nonlinear"],
        resources=dict(cpus=2, memory_gib=16, hours=1, gpus=0, requeue=False),
        scientific_model_updates=0,
        validation_events=0,
        automatic_successor=False,
    )
    validate_contract(c)
    for key, value in [
        ("updates", 4096),
        ("arms", ["nonlinear_probe"]),
        ("validation_events", 600),
        ("scientific_model_updates", 1),
        ("automatic_successor", True),
        ("resources", dict(cpus=2, memory_gib=32, hours=8, gpus=1, requeue=False)),
    ]:
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_contract(bad)
