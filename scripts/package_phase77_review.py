"""Publish aggregate training diagnostics with lossless, hash-bound delivery."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def load(p):
    return json.loads(p.read_text())


def package(root, previous, repository, destination):
    diagnostic = load(root / "diagnostic-v1/diagnostic-summary.json")
    assert diagnostic["validation_events_evaluated"] == 0
    assert diagnostic["new_training_jobs"] == 0
    prior = json.loads(
        gzip.decompress(
            (
                repository
                / "artifacts/codex/phase76_review_20261009/phase76-review.json.gz"
            ).read_bytes()
        )
    )
    value = {
        "version": "phase77-training-diagnosis-public-v1",
        "status": "COMPLETED",
        "stage": "training_only_diagnostic",
        "primary_eligible": False,
        "source_sha": load(root / "diagnostic-source-freeze.json")["source_sha"],
        "decision": load(root / "decision.json"),
        "populations": {},
        "compute": diagnostic["compute"],
        "diagnostic_workload_accounting": load(root / "compute-accounting.json"),
        "paired_decision_diagnostics": load(root / "decision-paired-diagnostics.json"),
        "training_history": load(root / "training-history-audit.json"),
        "historical_native_endpoints": {},
        "scope": "1536 training identities,24 tiny-fit views drawn from that pool,96 training-only gradient probes; zero validation evaluation or new training. Not independent confirmation.",
        "controls": prior["controls"],
        "unavailable": prior["arms"]["partial_context_on"]["unavailable"],
        "metric_definitions": {
            "background_to_B": "Known unassigned FSPs assigned either B slot, including B-bearing and continuum collisions; not accepted fake-B event count.",
            "oracle_foreground": "Truth-assisted exact foreground mask followed by model conditional B-slot argmax; not deployment or legal reachability.",
            "oracle_slot": "Truth-assisted correction of B-slot swaps among native foreground predictions, retaining native omissions and background extras; not deployment.",
            "aggregate_half": "One prespecified diagnostic: sum of two B probabilities>0.5, then B-slot argmax; not selected production threshold or accepted groups.",
            "loss_matching": "Native CE+soft-Jaccard+presence total-loss permutation. Hard-error diagnostic matching is separately reported; disagreement is not evidence of a bug.",
            "gradients": "Per-event normalized logit-gradient sums and squared norms, plus96 actual proposal-parameter gradient probes. They are not dataset parameter-gradient cancellation or FLOPs.",
            "uncertainty": "Category-stratified whole-collision bootstrap; training-only exploratory descriptive intervals. B/node trials correlated; zero-success intervals do not establish equivalence.",
        },
        "evidence_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "diagnostic-v1/diagnostic-summary.json",
                root / "decision.json",
                root / "diagnostic-design.json",
                root / "decision-paired-diagnostics.json",
                previous / "successor-campaign-v1/contract.json",
            ]
        },
    }
    # Prior settings are historical controls, not a new admitted training contract.
    value["controls"]["fresh_heldout_events"] = 0
    value["controls"]["scope"] = (
        "Existing model controls; Phase77 fits zero updates and designates zero heldout events."
    )
    value["compute"]["distinct_training_identities"] = 1536
    value["compute"]["tiny_views_reuse_training_identities"] = True
    for role, r in diagnostic["populations"].items():
        value["populations"][role] = {
            k: v
            for k, v in r.items()
            if k
            not in (
                "checkpoint",
                "reproduced_trace",
                "detached_trace",
                "joined_private",
            )
        }
        value["populations"][role]["bindings"] = {
            k: r[k]["sha256"]
            for k in (
                "checkpoint",
                "reproduced_trace",
                "detached_trace",
                "joined_private",
            )
        }
        value["historical_native_endpoints"][role] = prior["arms"][
            "partial_context_on"
        ]["roles"][role]
        assert r["max_probability_difference"] == 0
    value["historical_native_endpoint_scope"] = (
        "Reused immutable Phase76 train/tiny metrics, validated by exact output replay; no new heldout outcome read or inference. Retention guards unchanged. Physical hierarchy metrics remain unavailable."
    )
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    assert len(encoded) <= 5_000_000
    packed = gzip.compress(encoded, mtime=0)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "phase77-review.json.gz").write_bytes(packed)
    binding = {
        "version": "phase77-public-binding-v1",
        "file": "phase77-review.json.gz",
        "compressed_sha256": hashlib.sha256(packed).hexdigest(),
        "decoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "decoded_bytes": len(encoded),
        "source_sha": value["source_sha"],
    }
    (destination / "binding.json").write_text(
        json.dumps(binding, sort_keys=True) + "\n"
    )
    print(json.dumps(binding))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("root", "previous", "repository", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    package(a.root, a.previous, a.repository, a.output)
