"""Lossless bounded public delivery of all Phase78 supported diagnostic metrics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(root, repository, output):
    summary = json.loads((root / "diagnostic-v2/summary.json").read_text())
    prior = json.loads(
        gzip.decompress(
            (
                repository
                / "artifacts/codex/phase77_review_20261009/phase77-review.json.gz"
            ).read_bytes()
        )
    )
    value = {k: v for k, v in summary.items() if k != "contract"}
    value.update(
        version="phase78-partition-public-v1",
        status="COMPLETED",
        primary_eligible=False,
        source_sha=json.loads((root / "diagnostic-source-freeze-v2.json").read_text())[
            "source_sha"
        ],
        decision=json.loads((root / "decision.json").read_text()),
        design=json.loads((root / "diagnostic-design-v2.json").read_text()),
        historical_native_endpoints=prior["historical_native_endpoints"],
        unavailable=prior["unavailable"],
        historical_native_endpoint_scope="Immutable train/tiny endpoints reused after exact logits replay. No new validation outcomes read. Physical tree/pool/deep/closure unavailable.",
        metric_definitions={
            "pair_auc": "Within-event same-B versus cross-B on known B nodes and disjoint detector-source pairs. Event means and pooled pair ranks differ in weighting. Not exact source membership.",
            "background": "Known pair with at least one background node; same-B score>=0.5 is diagnostic contamination, not accepted fake-B event rate.",
            "probe_scope": "Frozen scientific model saw fit and assessment identities. Probe fitting uses384 B events; assessment uses128 B events. No independent validation.",
            "tiny_probe": "Main-fitted feature probe applied across different tiny checkpoint coordinates; descriptive transfer only, not tiny representation diagnostic.",
            "size": "Event contains at least one target in specified size bin; strata overlap.",
            "tie": "Conditional correct-count permutation ties include continuum events (no B support); subtract continuum to obtain B-event ties.",
            "source_support": "466 main and5 tiny shared-source pairs excluded; all events and native node/trial denominators retained.",
            "uncertainty": "2000 whole-event stratified bootstrap draws; training-role descriptive interval, not pair independence or fresh physics confirmation.",
        },
    )
    value["design"]["scope_limits"] = value["design"]["scope_limits"].replace(
        "checkpoints/normalization/constraints",
        "checkpoints, normalization and constraints",
    )
    value["design"]["correction"] = value["design"]["correction"].replace(
        "First diagnostic17109723", "First diagnostic attempt"
    )
    for role, r in value["populations"].items():
        r["bindings"] = {k: r.pop(k)["sha256"] for k in ("checkpoint", "trace")}
    value["evidence_hashes"] = {
        name.replace("/", "_").replace(".pt", "_weights"): digest(
            (root / name).read_bytes()
        )
        for name in (
            "diagnostic-design-v2.json",
            "diagnostic-v2/summary.json",
            "diagnostic-v2/probe-final.pt",
            "diagnostic-v2/probe-partition-private.json",
            "training-role-isolation.json",
        )
    }
    output.mkdir(parents=True, exist_ok=False)

    def write_bundle(name, obj):
        decoded = (
            json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        if len(decoded) > 5_000_000:
            raise ValueError("Decoded delivery cap exceeded")
        compressed = gzip.compress(decoded, mtime=0)
        (output / name).write_bytes(compressed)
        return {
            "file": name,
            "compressed_sha256": digest(compressed),
            "decoded_sha256": digest(decoded),
            "decoded_bytes": len(decoded),
        }

    events = json.loads(
        (root / "diagnostic-v2/train-event-metrics-private.json").read_text()
    )
    events = [{k: v for k, v in r.items() if k != "uid"} for r in events]
    value["event_parts"] = []
    for part in range(3):
        records = events[part * 512 : (part + 1) * 512]
        b = write_bundle(
            f"phase78-events-{part}.json.gz",
            {"role": "training", "start_event_index": part * 512, "events": records},
        )
        b["events"] = len(records)
        value["event_parts"].append(b)
    value["tiny_event_metrics"] = [
        {k: v for k, v in r.items() if k != "uid"}
        for r in json.loads(
            (root / "diagnostic-v2/tiny-event-metrics-private.json").read_text()
        )
    ]
    bind = write_bundle("phase78-review.json.gz", value)
    bind.update(version="phase78-public-binding-v1", source_sha=value["source_sha"])
    (output / "binding.json").write_text(json.dumps(bind, sort_keys=True) + "\n")
    print(json.dumps(bind))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "repository", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    package(a.root, a.repository, a.output)
