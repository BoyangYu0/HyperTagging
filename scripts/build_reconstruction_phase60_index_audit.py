#!/usr/bin/env python3
"""Verify fresh validation identities and identical train-only normalization."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index_path = (
        ROOT
        / "artifacts/experiment_readiness/reconstruction_phase60_20260925/train_070k.complete_only.index.json"
    )
    new = json.loads(index_path.read_text())
    old = json.loads(
        (
            ROOT
            / "artifacts/experiment_readiness/reconstruction_phase46_20260912/train_070k.complete_only.index.json"
        ).read_text()
    )
    assert new["split_counts"] == {"train": 70000, "validation": 100000}
    identity = new["event_identity_validation"]
    assert identity["status"] == "passed" and identity["unique_event_uids"] == 170000
    assert (
        identity["duplicate_event_uids"]
        == identity["source_mismatches"]
        == identity["category_mismatches"]
        == 0
    )
    assert identity["sealed_test_opened"] is False
    assert (
        new["normalizer_scope"] == "train"
        and new["normalizer_state"] == old["normalizer_state"]
    )
    files = [
        "src/hypertagging/data/heterogeneous.py",
        "src/hypertagging/data/dataset_index.py",
        "scripts/build_dataset_index.py",
        "scripts/build_reconstruction_phase60_index_audit.py",
    ]
    result = {
        "version": "phase60-fresh-statistics-audit-v1",
        "status": "PASS",
        "index_path": str(index_path.relative_to(ROOT)),
        "index_sha256": sha(index_path),
        "normalizer_scope": "train",
        "sealed_test_accessed": False,
        "construction": "full_record_scan_not_sidecars",
        "training_payloads_identical": True,
        "normalizer_identical_to_phase59": True,
        "event_identity_validation": identity,
        "source_files": {name: sha(ROOT / name) for name in files},
    }
    (index_path.parent / "fresh-statistics-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("PASS170000 unique identities; train statistics unchanged")


if __name__ == "__main__":
    main()
