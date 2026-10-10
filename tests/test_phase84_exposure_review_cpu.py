import copy
import pytest
from scripts.review_phase84_exposure import gates, paired_counts


def event(i, exact=0, bg=False):
    return {
        "uid": str(i),
        "category": "ccbar" if bg else "charged",
        "counts": {
            "nominal_b_trials": 0 if bg else 2,
            "continuum_events": int(bg),
            "continuum_accepted_events": 0,
            "raw_exact_memberships": exact,
            "accepted_exact_memberships": exact,
            "accepted_source_conflicts": 0,
        },
    }


def test_early_learning_is_trainability_not_incremental_gain():
    e = [event(i, 1) for i in range(4)] + [event(4, bg=True)]
    d = {"native": {"events": e}}
    r = gates(d, d, True)
    assert r["absolute_trainable"] and not r["incremental_exposure_gain"]
    assert not gates(d, d, False)["absolute_trainable"]
    b = copy.deepcopy(d)
    b["native"]["events"][-1]["counts"]["continuum_accepted_events"] = 1
    assert not gates(d, b, True)["absolute_trainable"]


def test_gains_are_distinct_collisions_and_paired_zeros_not_equivalence():
    a = [event(i) for i in range(4)] + [event(4, bg=True)]
    b = [event(i, 1) for i in range(4)] + [event(4, bg=True)]
    assert gates({"native": {"events": a}}, {"native": {"events": b}}, True)[
        "incremental_exposure_gain"
    ]
    z = paired_counts(a, a, 20)
    assert (
        z["raw_exact_memberships"]["all_zero_caution"]
        and z["raw_exact_memberships"]["denominator"] == 8
    )
    with pytest.raises(ValueError):
        paired_counts(a, b[:-1], 20)
