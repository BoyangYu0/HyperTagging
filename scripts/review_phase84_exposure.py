"""Post-hoc paired exposure accounting on fixed authenticated training identities."""

from collections import Counter
import random


def aggregate(events):
    out = Counter()
    for e in events:
        out.update(e["counts"])
    return dict(out)


def gates(early, final, eligible):
    a = {e["uid"]: e for e in early["native"]["events"]}
    b = {e["uid"]: e for e in final["native"]["events"]}
    if a.keys() != b.keys() or any(a[k]["category"] != b[k]["category"] for k in a):
        raise ValueError("Nonpaired events")
    ac = aggregate(a.values())
    bc = aggregate(b.values())
    bg_a = ac.get("continuum_accepted_events", 0) / max(
        1, ac.get("continuum_events", 0)
    )
    bg_b = bc.get("continuum_accepted_events", 0) / max(
        1, bc.get("continuum_events", 0)
    )
    guard = (
        eligible
        and bc.get("accepted_source_conflicts", 0) == 0
        and bg_b <= bg_a + 0.05
        and ac.get("continuum_events", 0) == bc.get("continuum_events", 0) > 0
    )
    correct = [
        k
        for k in b
        if b[k]["counts"].get("nominal_b_trials", 0) > 0
        and b[k]["counts"].get("raw_exact_memberships", 0) > 0
        and b[k]["counts"].get("accepted_exact_memberships", 0) > 0
    ]
    gain = [
        k
        for k in b
        if b[k]["counts"].get("nominal_b_trials", 0) > 0
        and all(
            b[k]["counts"].get(m, 0) > a[k]["counts"].get(m, 0)
            for m in ["raw_exact_memberships", "accepted_exact_memberships"]
        )
    ]
    return {
        "absolute_trainable": bool(guard and len(correct) >= 4),
        "incremental_exposure_gain": bool(
            guard
            and len(gain) >= 4
            and all(
                bc.get(m, 0) > ac.get(m, 0)
                for m in ["raw_exact_memberships", "accepted_exact_memberships"]
            )
        ),
        "correct_distinct_B_collisions": len(correct),
        "gaining_distinct_B_collisions": len(gain),
        "background_fraction_early": bg_a,
        "background_fraction_final": bg_b,
        "source_background_guard": bool(guard),
        "counts_early": ac,
        "counts_final": bc,
        "interpretation": "Exploratory training-role gate, not independent confirmation.5pp is diagnostic stability, not production acceptance.",
    }


def paired_counts(early, final, replicates=2000):
    a = {e["uid"]: e for e in early}
    b = {e["uid"]: e for e in final}
    if a.keys() != b.keys() or any(a[k]["category"] != b[k]["category"] for k in a):
        raise ValueError("Nonpaired events")
    groups = {
        c: [k for k in a if a[k]["category"] == c]
        for c in sorted({e["category"] for e in early})
    }
    result = {}
    for metric, denom in [
        ("raw_exact_memberships", "nominal_b_trials"),
        ("accepted_exact_memberships", "nominal_b_trials"),
        ("continuum_accepted_events", "continuum_events"),
    ]:
        n = sum(a[k]["counts"].get(denom, 0) for k in a)
        if n != sum(b[k]["counts"].get(denom, 0) for k in b):
            raise ValueError("Denominator changed")
        if not n:
            result[metric] = {"status": "UNAVAILABLE", "denominator": 0}
            continue
        d = {
            k: b[k]["counts"].get(metric, 0) - a[k]["counts"].get(metric, 0) for k in a
        }
        rng = random.Random(8409)
        boot = []
        for _ in range(replicates):
            drawn = [rng.choice(g) for g in groups.values() for _ in range(len(g))]
            nd = sum(a[k]["counts"].get(denom, 0) for k in drawn)
            if nd:
                boot.append(sum(d[k] for k in drawn) / nd)
        boot.sort()
        result[metric] = {
            "difference": sum(d.values()) / n,
            "denominator": n,
            "paired_events": len(a),
            "interval95": [
                boot[int(0.025 * (len(boot) - 1))],
                boot[int(0.975 * (len(boot) - 1))],
            ],
            "all_zero_caution": all(
                a[k]["counts"].get(metric, 0) == b[k]["counts"].get(metric, 0) == 0
                for k in a
            ),
        }
    return result
