import copy
import pytest
from scripts.phase84_exposure import (
    nested,
    exposure,
    plateau,
    validate,
    CATEGORIES,
    MILESTONES,
)
from scripts.phase76_development import settings


def rows():
    return [{"uid": f"{c}:{i}", "category": c} for c in CATEGORIES for i in range(256)]


def test_nested_balanced_stable_order_and_no_label_selection():
    rr = rows()
    small = nested(rr, 96)
    bridge = nested(rr, 384)
    assert set(x["uid"] for x in small) < set(x["uid"] for x in bridge)
    assert nested(rr, 1536) == rr
    assert all(sum(x["category"] == c for x in small) == 16 for c in CATEGORIES)
    changed = [{**r, "target": "arbitrary"} for r in rr]
    assert [r["uid"] for r in nested(changed, 96)] == [r["uid"] for r in small]
    assert nested(rr, 96) == [r for r in rr if r in small]
    bad = copy.deepcopy(rr)
    bad[0] = bad[1]
    with pytest.raises(ValueError):
        nested(bad, 96)
    with pytest.raises(ValueError):
        nested(rr, 100)


def test_exposure_actual_counts_and_prefix():
    rr = nested(rows(), 96)
    a = exposure(rr, 375, 7)
    b = exposure(rr, 1500, 7)
    assert (
        sum(a["per_uid"].values()) == 3000 and sum(b["per_category"].values()) == 12000
    )
    assert all(b["per_uid"][k] >= v for k, v in a["per_uid"].items())
    assert a == exposure(rr, 375, 7) and a["sequence_sha256"] != b["sequence_sha256"]


def test_zero_exact_is_never_plateau_futility_and_scope_fail_closed():
    z = {
        "mean_risk": {"member": 1.0},
        "native": {"counts": {"raw_exact_memberships": 0, "nominal_b_trials": 64}},
    }
    assert not plateau([z, z, z])
    a = copy.deepcopy(z)
    a["native"]["counts"]["raw_exact_memberships"] = 64
    assert plateau([a, a, a])
    c = {
        "kind": "phase84_exposure",
        "pool_size": 96,
        "milestones": list(MILESTONES),
        "settings": {**settings(), "downstream_updates": 6000},
        "resources": dict(cpus=2, memory_gib=32, hours=8, gpus=0, requeue=False),
        "heldout_events": 0,
        "automatic_successor": False,
        "sealed_test_access": False,
    }
    validate(c)
    for k, v in [
        ("pool_size", 6144),
        ("heldout_events", 600),
        ("automatic_successor", True),
    ]:
        bad = {**c, k: v}
        with pytest.raises(ValueError):
            validate(bad)
