"""Lossless aggregate-only delivery of the single completed successor."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def package(root, destination):
    value = json.loads((root / "phase75-review-public.json").read_text())
    if (
        value["status"] != "COMPLETED"
        or value["stage"] != "development"
        or value["primary_eligible"]
        or set(value["arms"]) != {"joint", "project_conflicting_relation"}
    ):
        raise ValueError("Wrong completed development study")
    contract = json.loads((root / "successor-campaign-v1/contract.json").read_text())
    value["controls"] = {
        "context_width": 128,
        "hyperbolic_width": 32,
        "depth": 4,
        "common_head_width": 256,
        "encoder_initialization_sha256": contract["initial_checkpoint"]["sha256"],
        "settings": contract["settings"],
        "resources_per_arm": contract["resources"],
        "original_corpus_events": 70000,
        "train_subset_events": 1536,
        "fresh_heldout_events": 600,
        "primary_reservations": 0,
        "no_checkpoint_selection": True,
    }
    value["coverage"] = {
        "heldout_by_category": dict.fromkeys(
            ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"), 100
        ),
        "primary_requirement_met": False,
    }
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    packed = gzip.compress(data, mtime=0)
    if len(data) > 5_000_000:
        raise ValueError("Decoded publication bound exceeded")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "phase75-review.json.gz").open("xb") as f:
        f.write(packed)
    binding = {
        "version": "phase75-public-binding-v1",
        "file": "phase75-review.json.gz",
        "compressed_sha256": hashlib.sha256(packed).hexdigest(),
        "decoded_sha256": hashlib.sha256(data).hexdigest(),
        "decoded_bytes": len(data),
        "source_sha": value["source_sha"],
    }
    (destination / "binding.json").write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(binding)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    package(a.root, a.output)
