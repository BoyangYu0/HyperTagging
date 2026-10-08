"""Lossless aggregate/history export; private event rows stay outside publication."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from scipy.stats import beta


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def package(review, parent, destination):
    value = json.loads(review.read_text())
    arms = list(value["arms"])
    summaries = {
        a: json.loads((parent / "campaign-v1/runs" / a / "summary.json").read_text())
        for a in arms
    }
    value["paired_membership_contingency"] = {}
    for i, left in enumerate(arms):
        for right in arms[i + 1 :]:
            table = Counter()
            for a, b in zip(
                summaries[left]["heldout"]["events"],
                summaries[right]["heldout"]["events"],
            ):
                assert a["uid"] == b["uid"]
                if a["category"] in ("charged", "mixed"):
                    table[
                        f"{int(a['counts']['accepted_exact_memberships'] > 0)}{int(b['counts']['accepted_exact_memberships'] > 0)}"
                    ] += 1
            value["paired_membership_contingency"][left + " versus " + right] = {
                "event_any_accepted_00_01_10_11": {
                    k: table[k] for k in ("00", "01", "10", "11")
                },
                "distinct_bbbar_collisions": 200,
            }
    for arm, record in value["arms"].items():
        for role, r in record["roles"].items():
            n = sum(
                x["processed"]
                for k, x in r["by_category"].items()
                if k in ("charged", "mixed")
            )
            k = sum(
                x["event_any_accepted"]
                for key, x in r["by_category"].items()
                if key in ("charged", "mixed")
            )
            r["event_any_accepted_exact_binomial95"] = {
                "successes": k,
                "collisions": n,
                "lower": float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
                "upper": float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
                "single_fitted_model_not_seed_uncertainty": True,
            }
        record["pretraining_initialization_sha256"] = json.loads(
            (parent / "campaign-v1/contract.json").read_text()
        )["initializations"][arm.split("-")[0]]["sha256"]
    value["source_sha"] = json.loads(
        (parent / "campaign-v1/contract.json").read_text()
    )["source_sha"]
    value["status"] = "COMPLETED_DEVELOPMENT_NO_PRIMARY_WINNER"
    value["coverage"] = {
        "train_subset_events": 1536,
        "original_training_corpus": 70000,
        "heldout_distinct_events": 600,
        "heldout_by_category": dict.fromkeys(
            ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"), 100
        ),
        "primary_requirement_met": False,
        "tiny_raw_gate_all_arms": True,
        "heldout_improvement_gate": "NOT_ESTABLISHED",
    }
    data = canonical(value)
    compressed = gzip.compress(data, mtime=0)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "phase74-review.json.gz").open("xb") as f:
        f.write(compressed)
    bind = {
        "version": "phase74-public-binding-v1",
        "file": "phase74-review.json.gz",
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "decoded_sha256": hashlib.sha256(data).hexdigest(),
        "decoded_bytes": len(data),
        "source_sha": value["source_sha"],
    }
    (destination / "binding.json").write_bytes(canonical(bind))
    print(bind)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    package(a.review, a.parent, a.output)
