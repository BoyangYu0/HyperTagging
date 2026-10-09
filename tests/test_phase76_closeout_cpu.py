import pytest
from scripts.review_phase76_terminal import paired_effects


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
        and r["intervention"] == 1
        and r["denominator"] == 4
        and r["difference"] == 0.25
    )
    assert r["paired_collisions"] == 2
    assert r["discordant_collision_counts"] == {
        "intervention_greater": 1,
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
    from scripts.review_phase76_terminal import sampling_accounting

    pool = [{"uid": str(i), "category": "charged"} for i in range(1536)]
    result = sampling_accounting(pool, 12000, 202610081)
    assert result["actual_distinct_presented"] == 1535
    assert result["not_presented"] == 1
    assert result["presentations_by_category"] == {"charged": 12000}
    changed = sampling_accounting(list(reversed(pool)), 12000, 202610081)
    assert result["data_order_sha256"] != changed["data_order_sha256"]


def test_continuum_source_membership_is_posthoc_and_unknown_is_unavailable():
    from scripts.review_phase76_terminal import continuum_membership
    from tests.test_full_decay_metrics_cpu import _full_tree
    from hypertagging.evaluation.full_decay_metrics import _tree_view

    tree = _full_tree()
    tree["node_mask"][0, 10] = False
    view = _tree_view(tree, 0, truth=True)
    sets = [
        view.source_set(i) if bool(view.active[i]) else frozenset()
        for i in range(len(view.active))
    ]
    row = {"full": tree, "supervision": {"node_sets": sets}}
    e = {
        "raw_groups": [list(sets[8]), list(sets[9])],
        "accepted_groups": [list(sets[8])],
    }
    r = continuum_membership(row, e)
    assert (
        r["component_trials"] == 2
        and r["raw_exact"] == 2
        and r["accepted_exact"] == 1
        and r["unavailable"] == 0
    )
    row["supervision"]["node_sets"][8] = frozenset()
    r = continuum_membership(row, e)
    assert r["unavailable"] == 1 and r["raw_exact"] == 1


def test_paired_source_errors_keep_micro_denominators_and_missing_support():
    from scripts.review_phase76_terminal import paired_source_errors

    def record(i, intersection, missing, extra):
        counts = dict(
            intersection=intersection,
            missing=missing,
            extra=extra,
            missing_to_unassigned=missing,
        )
        return dict(
            uid=str(i),
            category="charged",
            stages={k: {"counts": counts} for k in ("proposal", "refinement")},
        )

    a = [record(0, 1, 1, 1), record(1, 9, 0, 0)]
    b = [record(0, 2, 0, 0), record(1, 9, 0, 0)]
    r = paired_source_errors(a, b)["proposal"]["source_recall"]
    assert r["numerators"] == [10, 11] and r["denominators"] == [11, 11]
    assert r["paired_collisions"] == 2 and r["difference"] == pytest.approx(1 / 11)
    with pytest.raises(ValueError):
        paired_source_errors(a, list(reversed(b)))
    z = paired_source_errors([record(0, 0, 0, 0)], [record(0, 0, 0, 0)])
    assert z["refinement"]["source_precision"]["difference"] is None
