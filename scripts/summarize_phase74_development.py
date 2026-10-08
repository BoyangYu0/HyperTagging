"""Read completed four-arm development results; no selection, promotion or jobs."""

import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    from hypertagging.training.capacity_development import ARMS

    records = {
        arm: json.loads((root / arm / "summary.json").read_text()) for arm in ARMS
    }
    if len({r["contract_sha256"] for r in records.values()}) != 1:
        raise ValueError("Mismatched campaign contracts")
    if (
        len({r["pretraining"]["data_order_sha256"] for r in records.values()}) != 1
        or len({r["downstream"]["data_order_sha256"] for r in records.values()}) != 1
    ):
        raise ValueError("Presentation histories differ")
    identities = [[e["uid"] for e in records[a]["heldout"]["events"]] for a in ARMS]
    if any(ids != identities[0] for ids in identities[1:]):
        raise ValueError("Held-out identities/order differ")
    rng = np.random.default_rng(202610081)
    report = {}
    for metric, denominator in [
        ("raw_exact_memberships", "nominal_b_trials"),
        ("accepted_exact_memberships", "nominal_b_trials"),
        ("continuum_accepted_events", "continuum_events"),
    ]:
        x = np.array(
            [
                [e["counts"][metric] for e in records[a]["heldout"]["events"]]
                for a in ARMS
            ],
            dtype=float,
        )
        n = np.array(
            [e["counts"][denominator] for e in records[ARMS[0]]["heldout"]["events"]],
            dtype=float,
        )
        eligible = n > 0
        x, n = x[:, eligible], n[eligible]

        def contrasts(r):
            return {
                "width": float((r[2] + r[3] - r[0] - r[1]) / 2),
                "objective": float((r[1] + r[3] - r[0] - r[2]) / 2),
                "interaction": float((r[3] - r[2]) - (r[1] - r[0])),
            }

        point = contrasts(x.sum(1) / n.sum())
        boot = []
        for _ in range(2000):
            sample = rng.integers(0, len(n), len(n))
            boot.append(contrasts(x[:, sample].sum(1) / n[sample].sum()))
        report[metric] = {
            "arms": {
                a: {
                    "numerator": int(x[i].sum()),
                    "denominator": int(n.sum()),
                    "any_success_collision_count": int((x[i] > 0).sum()),
                    "distinct_collisions": len(n),
                    "zero_success_event_rate_one_sided_95_upper": float(
                        1 - 0.05 ** (1 / len(n))
                    )
                    if not x[i].any()
                    else None,
                }
                for i, a in enumerate(ARMS)
            },
            "effects": {
                k: {
                    "estimate": v,
                    "paired_collision_bootstrap_95": np.quantile(
                        [b[k] for b in boot], [0.025, 0.975]
                    ).tolist(),
                }
                for k, v in point.items()
            },
            "uncertainty": "one seed; collision-paired bootstrap preserves correlated B trials; degenerate sparse bootstrap is not proof of zero population effect",
        }
    return {
        "stage": "development",
        "results": report,
        "primary_promotion": False,
        "automatic_successor": False,
        "tiny_gate": {
            a: records[a]["tiny"]["counts"]["raw_exact_memberships"]
            / records[a]["tiny"]["counts"]["nominal_b_trials"]
            >= 0.95
            for a in ARMS
        },
        "unavailable_hierarchy_metrics": records[ARMS[0]]["heldout"]["unavailable"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.output.open("x") as stream:
        json.dump(summarize(a.runs), stream, indent=2)
