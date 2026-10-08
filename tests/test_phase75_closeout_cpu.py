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
    assert r["paired_collisions"] == 2
    assert r["discordant_collision_counts"] == {
        "projection_greater": 1,
        "control_greater": 0,
        "equal": 1,
    }
    assert r["event_any_exact_binomial95"][1]["collisions"] == 2
    assert r["event_any_exact_binomial95"][0]["upper"] > 0
    assert report["continuum_accepted_events"]["denominator"] == 4


def test_pairing_rejects_reordered_identities():
    with pytest.raises(ValueError, match="identity"):
        paired_effects(rows(), list(reversed(rows())))


def test_sampling_accounting_authenticates_order_and_distinct_presentations():
    from scripts.review_phase75_terminal import sampling_accounting

    pool = [{"uid": str(i), "category": "charged"} for i in range(1536)]
    result = sampling_accounting(pool, 12000, 202610081)
    assert result["actual_distinct_presented"] == 1535
    assert result["not_presented"] == 1
    assert result["presentations_by_category"] == {"charged": 12000}
    changed = sampling_accounting(list(reversed(pool)), 12000, 202610081)
    assert result["data_order_sha256"] != changed["data_order_sha256"]
