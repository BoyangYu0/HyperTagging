import pytest
from scripts.review_phase75_terminal import paired_effects


def rows(success=False):
    result = []
    for i, cat in enumerate(("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")):
        b = i < 2
        result.append(
            {
                "uid": str(i),
                "category": cat,
                "counts": {
                    "raw_exact_memberships": int(success and i == 0),
                    "accepted_exact_memberships": int(success and i == 0),
                    "nominal_b_trials": 2 if b else 0,
                    "continuum_accepted_events": int(success and i == 2),
                    "continuum_events": 0 if b else 1,
                },
            }
        )
    return result


def test_paired_successor_counts_keep_collisions_and_b_trials_separate():
    report = paired_effects(rows(), rows(True))
    r = report["accepted_exact_memberships"]
    assert (
        r["control"] == 0
        and r["projection"] == 1
        and r["denominator"] == 4
        and r["difference"] == 0.25
    )
    assert r["event_any_exact_binomial95"][1]["collisions"] == 2
    assert r["event_any_exact_binomial95"][0]["upper"] > 0
    assert report["continuum_accepted_events"]["denominator"] == 4


def test_pairing_rejects_reordered_identities():
    with pytest.raises(ValueError, match="identity"):
        paired_effects(rows(), list(reversed(rows())))
