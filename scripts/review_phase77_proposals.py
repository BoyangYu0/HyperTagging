"""Post-generation collision-paired training diagnostics, never an inference policy."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def paired_decision(rows, stage, repetitions=4000):
    """Resample whole collisions within category; B nodes are not iid trials."""
    categories = sorted({r["category"] for r in rows})
    keys = [
        "B_nodes",
        "background_nodes",
        "B_to_unassigned",
        "aggregate_B_to_unassigned",
        "background_to_B",
        "aggregate_background_to_B",
        "native_raw_exact",
        "aggregate_half_raw_exact",
        "available_B_targets",
    ]
    arrays = [
        np.array(
            [
                [r["stages"][stage]["counts"].get(k, 0) for k in keys]
                for r in rows
                if r["category"] == cat
            ],
            dtype=float,
        )
        for cat in categories
    ]

    def rates(total):
        b, bg, miss, amiss, fp, afp, exact, aexact, targets = total
        def ratio(n, d):
            return n / d if d else np.nan
        return np.array(
            [
                ratio(b - amiss, b) - ratio(b - miss, b),
                ratio(afp, bg) - ratio(fp, bg),
                ratio(aexact - exact, targets),
                ratio(b - amiss, b - amiss + afp) - ratio(b - miss, b - miss + fp),
            ]
        )

    total = sum(a.sum(0) for a in arrays)
    point = rates(total)
    rng = np.random.default_rng(7709)
    sampled = np.array(
        [
            rates(sum(a[rng.integers(len(a), size=len(a))].sum(0) for a in arrays))
            for _ in range(repetitions)
        ]
    )
    result = {}
    for i, key in enumerate(
        [
            "foreground_recall",
            "background_false_assignment_rate",
            "raw_exact_membership",
            "foreground_precision",
        ]
    ):
        finite = np.isfinite(sampled[:, i])
        result[key] = {
            "aggregate_minus_native": float(point[i])
            if np.isfinite(point[i])
            else None,
            "paired_collision_stratified_bootstrap95": np.quantile(
                sampled[finite, i], [0.025, 0.975]
            ).tolist()
            if finite.any()
            else None,
            "finite_replicates": int(finite.sum()),
        }
    return {
        "events": len(rows),
        "repetitions": repetitions,
        "seed": 7709,
        "effects": result,
        "interpretation": "Fixed training population, exploratory decision diagnostic; not independent confirmation, seed variability, or calibrated population confidence. Zero-exact bootstrap degeneracy is not equivalence.",
    }


def main(root):
    out = {
        "scope": "Training-only replay, paired collisions; zero new validation",
        "roles": {},
    }
    for role in ("train", "tiny"):
        with gzip.open(
            root / "diagnostic-v1" / f"{role}-joined-private.jsonl.gz", "rt"
        ) as f:
            rows = [json.loads(line) for line in f]
        out["roles"][role] = {
            stage: paired_decision(rows, stage) for stage in ("proposal", "refinement")
        }
    with (root / "decision-paired-diagnostics.json").open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
