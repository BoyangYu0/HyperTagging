#!/usr/bin/env python3
"""Add source-disjoint validation shards, preserving every historical source role."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from hypertagging.data.training_selection import (  # noqa: E402
    _inventory_entry,
    _role_entry,
    _rank,
    build_training_selection,
    write_hashed_manifest,
    load_hashed_manifest,
)

SEED = 20260925
QUOTAS = {
    "ccbar": 2,
    "charged": 1,
    "ddbar": 1,
    "mixed": 1,
    "ssbar": 1,
    "taupair": 2,
    "uubar": 2,
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    old = ROOT / "configs/training_selection/production_1m_20260812"
    dest = ROOT / "configs/training_selection/phase60_validation_expansion_20260925"
    dest.mkdir(parents=True, exist_ok=False)
    inventory = load_hashed_manifest(old / "inventory.json")
    roles = load_hashed_manifest(old / "roles.json")
    selection = load_hashed_manifest(old / "train_070k_phase40.json")
    data_root = Path(inventory["data_root"]).parent
    historical_sources = {e["source_file"] for e in inventory["entries"]}
    historical_tasks = {e["task_id"] for e in inventory["entries"]}
    candidates = []
    # Completion metadata only until fresh source roles are fixed. No test payloads.
    for marker in sorted(data_root.glob("*.parquet.complete")):
        d = json.loads(marker.read_text())
        if d["source_file"] in historical_sources or d["task_id"] in historical_tasks:
            continue
        candidates.append((marker, d))
    chosen = []
    for category, quota in QUOTAS.items():
        eligible = [(p, d) for p, d in candidates if d["physics_category"] == category]
        eligible.sort(
            key=lambda item: hashlib.sha256(
                f"{SEED}|{category}|{item[1]['task_id']}|{item[1]['source_file']}".encode()
            ).hexdigest()
        )
        assert len(eligible) >= quota
        chosen.extend(eligible[:quota])
    public_root = data_root
    data_root = Path(
        "/project/agkuhr/users/boyang/data/HyperTagging_artifacts/phase59_review_20260925/validation-data"
    )
    data_root.mkdir(exist_ok=False)
    entries = []
    for original in inventory["entries"]:
        entry = dict(original)
        entry.pop("inventory_entry_hash")
        entry["inventory_entry_hash"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        entries.append(entry)
    fresh = [_inventory_entry(p.with_suffix(""), public_root) for p, d in chosen]
    for entry in fresh:
        assert digest(public_root / entry["path"]) == entry["parquet_sha256_reference"]
        assert entry["schema_version"] == "direct-mdst-tree-v4"
        assert entry["source_git_commit"] in inventory["source_git_commits"]
    assert not historical_sources & {e["source_file"] for e in fresh}
    assert len({e["source_file"] for e in fresh}) == len(fresh) == 10
    entries.extend(fresh)
    inventory.update(
        data_root=str(data_root),
        entries=entries,
        shard_count=len(entries),
        event_count=sum(e["event_count"] for e in entries),
    )
    for key, field in [
        ("category_shard_counts", None),
        ("category_event_counts", "event_count"),
    ]:
        c = Counter()
        for e in entries:
            c[e["category"]] += e[field] if field else 1
        inventory[key] = dict(sorted(c.items()))
    write_hashed_manifest(inventory, dest / "inventory.json")
    inventory = load_hashed_manifest(dest / "inventory.json")
    by_task = {e["task_id"]: e for e in entries}
    old_roles = {e["task_id"]: e for e in roles["entries"]}
    assigned = [
        _role_entry(by_task[t], e["role"], e["rank"]) for t, e in old_roles.items()
    ]
    assigned.extend(
        _role_entry(e, "validation", _rank(SEED, e["category"], e)) for e in fresh
    )
    roles.update(inventory_hash=inventory["manifest_hash"], entries=assigned)
    roles["role_shard_counts"] = dict(Counter(e["role"] for e in assigned))
    roles["role_event_counts"] = {
        k: sum(e["event_count"] for e in assigned if e["role"] == k)
        for k in roles["role_shard_counts"]
    }
    counts = {}
    for e in assigned:
        counts.setdefault(e["category"], Counter())[e["role"]] += 1
    roles["category_role_shard_counts"] = {k: dict(v) for k, v in counts.items()}
    write_hashed_manifest(roles, dest / "roles.json")
    roles = load_hashed_manifest(dest / "roles.json")
    new = build_training_selection(
        inventory,
        roles,
        selection_name="train_070k_validation_100k_phase60",
        training_quotas=selection["training_category_shard_quotas"],
        include_test=False,
    )
    old_train = {
        e["task_id"]: e["parquet_sha256_reference"]
        for e in selection["entries"]
        if e["split"] == "train"
    }
    new_train = {
        e["task_id"]: e["parquet_sha256_reference"]
        for e in new["entries"]
        if e["split"] == "train"
    }
    assert old_train == new_train
    assert new["split_counts"] == {"train": 70000, "validation": 100000, "test": 0}
    # Flat, immutable read-input snapshot: only train/validation payloads, never test.
    for e in new["entries"]:
        original_root = (
            public_root / "HyperTagging_uni"
            if e["task_id"] in historical_tasks
            else public_root
        )
        for suffix in ("", ".metadata.json", ".complete"):
            name = e["path"] + suffix
            os.link(original_root / name, data_root / name)
    write_hashed_manifest(new, dest / "train_070k.json")
    audit = {
        "version": "phase60-validation-source-expansion-v1",
        "status": "SOURCE_ROLES_BOUND_UID_SCAN_REQUIRED",
        "train_events": 70000,
        "validation_events": 100000,
        "fresh_validation_events": 50000,
        "training_payloads_identical": True,
        "all_historical_roles_preserved": True,
        "new_sources_disjoint_from_entire_prior_1m_inventory": True,
        "sealed_test_accessed": False,
        "materialized_roles": ["train", "validation"],
        "materialized_shards": len(new["entries"]),
        "catalog_note": "Historical source-role catalog is preserved; only selected train/validation payloads are materialized in the flat snapshot. Sealed test is not materialized or opened.",
        "selection_seed": SEED,
        "category_shard_quotas": QUOTAS,
        "fresh_tasks": [e["task_id"] for e in fresh],
        "prior_inventory_sha256": digest(old / "inventory.json"),
        "prior_roles_sha256": digest(old / "roles.json"),
        "new_shard_hashes_verified": {
            e["path"]: e["parquet_sha256_reference"] for e in fresh
        },
    }
    (dest / "expansion-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
