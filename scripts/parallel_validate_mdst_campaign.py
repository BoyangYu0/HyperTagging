#!/usr/bin/env python
"""Parallel equivalent of the exhaustive mDST campaign validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def validate_one(record: Mapping[str, Any]) -> tuple[int, dict[str, Any], list[str]]:
    from scripts import mdst_batch_production as production

    uids: list[str] = []
    kwargs = production._validation_kwargs(record)
    kwargs["uid_callback"] = uids.append
    result = production.validate_shard(Path(str(record["output_file"])), **kwargs)
    return int(record["task_id"]), result, uids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_publication_one(
    record: Mapping[str, Any],
) -> tuple[int, dict[str, Any], list[str]]:
    """Verify immutable publication hashes and scan only the UID column.

    The worker-produced result sidecar was written only after a full validate_shard
    pass. The completion marker binds both Parquet and metadata bytes, so repeating
    those hashes proves that the already validated bytes have not changed.
    """

    import pyarrow.parquet as parquet

    output = Path(str(record["output_file"]))
    metadata_path = Path(str(output) + ".metadata.json")
    marker_path = Path(str(output) + ".complete")
    result_path = Path(str(output) + ".result.json")
    for path in (output, metadata_path, marker_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if sha256(output) != marker.get("parquet_sha256"):
        raise ValueError(f"Parquet hash mismatch for task {record['task_id']}")
    if sha256(metadata_path) != marker.get("sidecar_sha256"):
        raise ValueError(f"metadata hash mismatch for task {record['task_id']}")
    direct_fields = (
        "task_id",
        "task_record_hash",
        "campaign_id",
        "campaign_config_digest",
        "campaign_stage",
        "schema_version",
        "feature_spec_hash",
        "model_feature_contract_hash",
        "source_git_commit",
        "source_git_tree",
        "source_state",
        "physics_category",
        "entry_start",
        "entry_stop_exclusive",
        "planned_events",
        "klm_training_scope",
        "production_readiness_report_sha256",
        "track_fit_policy",
    )
    for companion_name, companion in (
        ("metadata", metadata),
        ("marker", marker),
        ("result", result),
    ):
        for field in direct_fields:
            if field in companion and companion[field] != record.get(field):
                raise ValueError(
                    f"{companion_name} {field} mismatch for task {record['task_id']}"
                )
    if metadata.get("source_file") != record.get("input_file"):
        raise ValueError(f"metadata source mismatch for task {record['task_id']}")
    if marker.get("source_file") != record.get("input_file"):
        raise ValueError(f"marker source mismatch for task {record['task_id']}")
    if result.get("source_file") != record.get("input_file"):
        raise ValueError(f"result source mismatch for task {record['task_id']}")
    expected_events = int(record["planned_events"])
    if metadata.get("event_count") != expected_events:
        raise ValueError(f"metadata event count mismatch for task {record['task_id']}")
    if marker.get("event_count") != expected_events:
        raise ValueError(f"marker event count mismatch for task {record['task_id']}")
    if result.get("events") != expected_events:
        raise ValueError(f"result event count mismatch for task {record['task_id']}")
    if result.get("unique_event_uids") != expected_events:
        raise ValueError(f"result UID count mismatch for task {record['task_id']}")
    if result.get("classification") != "COMPLETE_VALID":
        raise ValueError(f"result classification mismatch for task {record['task_id']}")
    table = parquet.read_table(output, columns=["event_uid"])
    uids = table.column("event_uid").to_pylist()
    if table.num_rows != expected_events or len(set(uids)) != expected_events:
        raise ValueError(f"Parquet UID count mismatch for task {record['task_id']}")
    return int(record["task_id"]), result, uids


def quantiles_from_histogram(histogram: Counter[int]) -> dict[str, float]:
    total = sum(histogram.values())
    if not total:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(histogram)

    def value_at(index: int) -> int:
        cumulative = 0
        for value in ordered:
            cumulative += histogram[value]
            if index < cumulative:
                return value
        raise IndexError(index)

    output: dict[str, float] = {}
    for label, fraction in (("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99)):
        position = fraction * (total - 1)
        lower = int(position)
        upper = min(lower + 1, total - 1)
        weight = position - lower
        output[label] = value_at(lower) * (1.0 - weight) + value_at(upper) * weight
    output["max"] = float(ordered[-1])
    return output


def validate_manifest_contract(
    records: list[dict[str, Any]], production: Any
) -> tuple[dict[str, Any], Counter[str]]:
    seen_task_ids: set[int] = set()
    seen_task_hashes: set[str] = set()
    by_file: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    planned_categories: Counter[str] = Counter()
    config_fields = (
        "schema_version",
        "pid_vocabulary_version",
        "feature_spec_hash",
        "charge_conjugate_normalization",
        "leaf_kinematics_mode",
        "track_fit_policy",
        "model_feature_contract_hash",
        "campaign_id",
        "campaign_config_digest",
        "source_git_commit",
        "source_git_tree",
        "source_state",
        "campaign_stage",
        "klm_training_scope",
        "production_readiness_report_sha256",
    )
    expected_config = {field: records[0].get(field) for field in config_fields}
    for record in records:
        task_id = int(record["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id {task_id}")
        seen_task_ids.add(task_id)
        contract_v2 = (
            record.get("manifest_schema_version") == production.MANIFEST_SCHEMA_VERSION
        )
        if contract_v2:
            task_hash = str(record.get("task_record_hash", ""))
            if task_hash != production.task_record_hash(record):
                raise ValueError(f"invalid task_record_hash for task {task_id}")
            if task_hash in seen_task_hashes:
                raise ValueError(f"duplicate task_record_hash for task {task_id}")
            seen_task_hashes.add(task_hash)
        for field, expected in expected_config.items():
            if record.get(field) != expected:
                raise ValueError(f"manifest scientific config mismatch for {field}")
        start = int(record["entry_start"])
        stop = int(record["entry_stop_exclusive"])
        if stop <= start or stop - start != int(record["planned_events"]):
            raise ValueError(f"invalid planned source range for task {task_id}")
        by_file[str(record["input_file"])].append((start, stop, task_id))
        planned_categories[str(record["physics_category"])] += int(
            record["planned_events"]
        )
        if not Path(str(record["output_file"])).is_file():
            raise ValueError(f"production is incomplete; missing task {task_id}")
    for input_file, ranges in by_file.items():
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if left[1] > right[0]:
                raise ValueError(
                    f"overlapping source entry ranges for {input_file}: {left} and {right}"
                )
    return expected_config, planned_categories


def validate_parallel(
    manifest: Path, source_root: Path, workers: int, mode: str
) -> dict[str, object]:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src"))
    from scripts import mdst_batch_production as production

    records = production.read_manifest(manifest)
    expected_config, planned_categories = validate_manifest_contract(records, production)
    produced_categories: Counter[str] = Counter()
    track_policies: Counter[str] = Counter()
    actual_leaf_modes: Counter[str] = Counter()
    b_roots: Counter[str] = Counter()
    node_histogram: Counter[int] = Counter()
    depth_histogram: Counter[int] = Counter()
    klm_nodes = 0
    klm_associated_ecl = 0
    incomplete_branches = 0
    output_bytes = 0
    klm_diagnostics: Counter[str] = Counter()
    klm_feature_availability: Counter[str] = Counter()
    klm_by_category: Counter[str] = Counter()
    total_events = 0
    completed = 0
    uid_digest = hashlib.sha256()
    database_file = tempfile.NamedTemporaryFile(
        prefix="hypertagging-parallel-uids-", suffix=".sqlite", delete=False
    )
    database_file.close()
    database_path = Path(database_file.name)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("CREATE TABLE event_uids (uid TEXT PRIMARY KEY)")
    context = mp.get_context("fork")
    worker_function = (
        validate_one if mode == "full" else validate_publication_one
    )
    try:
        with context.Pool(processes=workers) as pool:
            for task_id, result, uids in pool.imap(worker_function, records, chunksize=1):
                record = records[task_id]
                if int(record["task_id"]) != task_id:
                    raise ValueError(f"task order mismatch at {task_id}")
                try:
                    connection.executemany(
                        "INSERT INTO event_uids(uid) VALUES (?)",
                        ((uid,) for uid in uids),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"duplicate event_uid encountered while registering task {task_id}"
                    ) from error
                for uid in uids:
                    uid_digest.update(uid.encode("utf-8") + b"\0")
                completed += 1
                events = int(result["events"])
                total_events += events
                category = str(record["physics_category"])
                produced_categories[category] += events
                track_policies[str(record.get("track_fit_policy", "unknown"))] += events
                actual_leaf_modes.update(result.get("actual_leaf_mode_distribution", {}))
                b_roots.update(result.get("b_root_distribution", {}))
                node_histogram.update(int(value) for value in result.get("node_counts", []))
                depth_histogram.update(int(value) for value in result.get("max_depths", []))
                klm_nodes += int(result.get("klm_nodes", 0))
                klm_associated_ecl += int(result.get("klm_associated_ecl", 0))
                incomplete_branches += int(
                    result.get("incomplete_reconstructable_branches", 0)
                )
                output = Path(str(record["output_file"]))
                output_bytes += int(result.get("output_bytes", output.stat().st_size))
                shard_klm = dict(result.get("klm_diagnostics", {}))
                klm_feature_availability.update(
                    shard_klm.pop("klm_feature_availability", {})
                )
                klm_diagnostics.update(
                    {key: int(value) for key, value in shard_klm.items()}
                )
                klm_by_category[category] += int(result.get("klm_nodes", 0))
                if completed % 50 == 0 or completed == len(records):
                    print(
                        f"validated_shards={completed}/{len(records)} "
                        f"events={total_events}",
                        file=sys.stderr,
                        flush=True,
                    )
        planned = sum(int(record["planned_events"]) for record in records)
        if total_events != planned:
            raise ValueError(
                f"global event count mismatch: planned {planned}, found {total_events}"
            )
        unique_uid_count = int(
            connection.execute("SELECT COUNT(*) FROM event_uids").fetchone()[0]
        )
        if unique_uid_count != total_events:
            raise ValueError(
                f"global UID count mismatch: events {total_events}, UIDs {unique_uid_count}"
            )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    return {
        "tasks": len(records),
        "completed_shards": completed,
        "missing_shards": [],
        "planned_events": planned,
        "validated_events": total_events,
        "unique_event_uids": unique_uid_count,
        "event_uid_digest": uid_digest.hexdigest(),
        "global_uid_validation_passes": 1,
        "category_distribution": dict(sorted(produced_categories.items())),
        "planned_category_distribution": dict(sorted(planned_categories.items())),
        "track_fit_policy_distribution": dict(sorted(track_policies.items())),
        "actual_leaf_mode_distribution": dict(sorted(actual_leaf_modes.items())),
        "klm_node_distribution": {
            "klm_nodes": klm_nodes,
            "associated_with_ecl": klm_associated_ecl,
            "unmatched_or_unassociated": max(klm_nodes - klm_associated_ecl, 0),
        },
        "klm_diagnostics": {
            **dict(sorted(klm_diagnostics.items())),
            "klm_nodes_by_category": dict(sorted(klm_by_category.items())),
            "klm_feature_availability": dict(
                sorted(klm_feature_availability.items())
            ),
            "klm_nodes_pruned_or_unretained": max(
                int(klm_diagnostics.get("collected_klm_records", 0))
                - int(klm_diagnostics.get("retained_klm_nodes", 0)),
                0,
            ),
        },
        "b_root_distribution": dict(sorted(b_roots.items())),
        "incomplete_reconstructable_branches": incomplete_branches,
        "node_count_quantiles": quantiles_from_histogram(node_histogram),
        "maximum_depth_quantiles": quantiles_from_histogram(depth_histogram),
        "output_bytes": output_bytes,
        "output_bytes_per_event": output_bytes / max(total_events, 1),
        "all_completion_markers_valid": True,
        "klm_training_scope": expected_config.get("klm_training_scope", "unresolved"),
        **expected_config,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--mode", choices=("full", "publication"), default="full"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    result = validate_parallel(
        args.manifest.resolve(), args.source_root.resolve(), args.workers, args.mode
    )
    write_json_atomic(args.output.resolve(), result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
