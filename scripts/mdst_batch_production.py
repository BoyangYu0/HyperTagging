#!/usr/bin/env python
"""Plan and execute sharded direct-mDST preprocessing production."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import hashlib

import awkward as ak
import uproot

from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.basf2_mdst import (
    SUPPORTED_TRACK_FIT_POLICIES,
    TRACK_FIT_POLICY_MAX_P_VALUE_V1,
)
from hypertagging.preprocessing.schema_v3 import (
    SCHEMA_VERSION_V3,
    feature_spec_v3,
)
from hypertagging.preprocessing.schema_v4 import (
    SCHEMA_VERSION_V4,
    feature_spec_v4,
    iter_event_records_v4,
)


DEFAULT_INPUT_ROOT = Path(
    "/pnfs/desy.de/belle/local/belle/MC/release-08-03-00/"
    "DB00003335/MC16ri_run2"
)
DEFAULT_OUTPUT_ROOT = Path("/data/dust/user/boyangyu/hypertagging/production_10m")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "mdst_10m.jsonl"
DEFAULT_TARGET_EVENTS = 10_000_000
DEFAULT_EVENTS_PER_TASK = 5_000
DEFAULT_BASF2_PYTHON_SITE = Path("/data/dust/user/boyangyu/basf2_py38")


def _physics_category(path: Path) -> str:
    parts = path.parts
    try:
        mdst_index = parts.index("mdst")
    except ValueError:
        return "unknown"
    return parts[mdst_index - 1] if mdst_index > 0 else "unknown"


def discover_input_files(input_root: Path) -> list[Path]:
    """Return category-interleaved ROOT inputs for training-set diversity."""

    groups: dict[str, deque[Path]] = defaultdict(deque)
    for path in sorted(input_root.rglob("*.root")):
        groups[_physics_category(path)].append(path)
    if not groups:
        raise FileNotFoundError(f"No ROOT files found below {input_root}")

    interleaved: list[Path] = []
    categories = sorted(groups)
    while any(groups.values()):
        for category in categories:
            if groups[category]:
                interleaved.append(groups[category].popleft())
    return interleaved


def root_event_count(path: Path) -> int:
    """Read only the event-tree metadata from a Belle II ROOT file."""

    with uproot.open(path) as root_file:
        tree = root_file["tree"]
        return int(tree.num_entries)


def build_manifest_records(
    input_files: Iterable[Path],
    *,
    output_root: Path,
    target_events: int,
    events_per_task: int,
    schema_version: str = SCHEMA_VERSION_V4,
    charge_conjugate_normalization: bool = False,
    leaf_kinematics_mode: str = "raw_track_predicted_pid",
    track_fit_policy: str = TRACK_FIT_POLICY_MAX_P_VALUE_V1,
    git_commit: str = "unknown",
    event_buffer_size: int = 128,
    row_group_size: int = 128,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build exact, non-overlapping entry-range records up to the target."""

    if target_events <= 0:
        raise ValueError("target_events must be positive")
    if events_per_task <= 0:
        raise ValueError("events_per_task must be positive")
    if track_fit_policy not in SUPPORTED_TRACK_FIT_POLICIES:
        raise ValueError(f"unknown track_fit_policy: {track_fit_policy}")

    records: list[dict[str, object]] = []
    category_events: Counter[str] = Counter()
    planned_events = 0
    task_id = 0
    feature_hash = str(
        (
            feature_spec_v4()
            if schema_version == SCHEMA_VERSION_V4
            else feature_spec_v3()
        )["feature_spec_hash"]
    )
    for input_file in input_files:
        if planned_events >= target_events:
            break
        file_events = root_event_count(input_file)
        category = _physics_category(input_file)
        entry_start = 0
        while entry_start < file_events and planned_events < target_events:
            requested = min(events_per_task, target_events - planned_events)
            entry_stop = min(file_events, entry_start + requested)
            count = entry_stop - entry_start
            output_file = output_root / "shards" / f"mdst_{task_id:05d}.parquet"
            records.append(
                {
                    "task_id": task_id,
                    "input_file": str(input_file),
                    "physics_category": category,
                    "source_entries": file_events,
                    "entry_start": entry_start,
                    "entry_stop_exclusive": entry_stop,
                    "entry_sequence": f"{entry_start}:{entry_stop - 1}",
                    "planned_events": count,
                    "output_file": str(output_file),
                    "schema_version": schema_version,
                    "pid_vocabulary_version": PID_VOCABULARY_VERSION,
                    "charge_conjugate_normalization": bool(
                        charge_conjugate_normalization
                    ),
                    "leaf_kinematics_mode": leaf_kinematics_mode,
                    "track_fit_policy": track_fit_policy,
                    "feature_spec_hash": feature_hash,
                    "git_commit": git_commit,
                    "event_buffer_size": int(event_buffer_size),
                    "row_group_size": int(row_group_size),
                }
            )
            planned_events += count
            category_events[category] += count
            task_id += 1
            entry_start = entry_stop

    if planned_events < target_events:
        raise RuntimeError(
            f"Input contains only {planned_events:,} planned events, below target {target_events:,}"
        )
    return records, dict(sorted(category_events.items()))


def write_manifest(
    records: list[dict[str, object]],
    *,
    manifest: Path,
    input_root: Path,
    output_root: Path,
    target_events: int,
    events_per_task: int,
    category_events: dict[str, int],
    overwrite: bool,
) -> dict[str, object]:
    """Atomically write a JSONL task manifest and summary sidecar."""

    if manifest.exists() and not overwrite:
        raise FileExistsError(f"{manifest} exists; pass --overwrite to replace it")
    defaults = {
        "schema_version": SCHEMA_VERSION_V4,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "charge_conjugate_normalization": False,
        "leaf_kinematics_mode": "raw_track_predicted_pid",
        "track_fit_policy": TRACK_FIT_POLICY_MAX_P_VALUE_V1,
        "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
        "git_commit": "unknown",
    }
    for record in records:
        for key, value in defaults.items():
            record.setdefault(key, value)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    (output_root / "shards").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest.parent,
        prefix=f".{manifest.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        for record in records:
            temporary.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary_path, manifest)

    summary = {
        "manifest": str(manifest),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "target_events": target_events,
        "planned_events": sum(int(record["planned_events"]) for record in records),
        "events_per_task": events_per_task,
        "tasks": len(records),
        "category_events": category_events,
        "schema_version": records[0]["schema_version"] if records else SCHEMA_VERSION_V4,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": (
            records[0]["feature_spec_hash"]
            if records
            else feature_spec_v4()["feature_spec_hash"]
        ),
    }
    summary_path = manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_manifest_record(manifest: Path, task_id: int) -> dict[str, object]:
    """Read one task by stable manifest task id."""

    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if int(record["task_id"]) == task_id:
                record.setdefault(
                    "track_fit_policy", TRACK_FIT_POLICY_MAX_P_VALUE_V1
                )
                return record
    raise IndexError(f"task_id {task_id} is not present in {manifest}")


def validate_shard(
    path: Path,
    *,
    expected_events: int,
    expected_schema: str = SCHEMA_VERSION_V4,
    expected_feature_spec_hash: str | None = None,
    expected_pid_vocabulary_version: str | None = None,
    expected_leaf_kinematics_mode: str | None = None,
    expected_track_fit_policy: str | None = None,
    expected_charge_conjugate_normalization: bool | None = None,
    uid_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate schema and event count after one production task."""

    if expected_schema == SCHEMA_VERSION_V4:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if not sidecar.exists():
            raise ValueError(f"Missing schema-v4 metadata sidecar for {path}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        payload = {
            **metadata,
            "schema_version": metadata.get("schema_version"),
        }
        marker = path.with_suffix(path.suffix + ".complete")
        if not marker.exists():
            raise ValueError(f"Missing schema-v4 completion marker for {path}")
    else:
        payload = ak.to_list(ak.from_parquet(path))[0]
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Unexpected schema in {path}: {payload.get('schema_version')!r}")
    if (
        expected_feature_spec_hash is not None
        and payload.get("feature_spec_hash") != expected_feature_spec_hash
    ):
        raise ValueError(f"Feature-spec mismatch in {path}")
    if (
        expected_pid_vocabulary_version is not None
        and payload.get("pid_vocabulary_version") != expected_pid_vocabulary_version
    ):
        raise ValueError(f"PID-vocabulary mismatch in {path}")
    if (
        expected_leaf_kinematics_mode is not None
        and payload.get("requested_collection_mode") is not None
        and payload.get("requested_collection_mode") != expected_leaf_kinematics_mode
    ):
        raise ValueError(f"Leaf-kinematics-mode mismatch in {path}")
    if expected_schema == SCHEMA_VERSION_V4 and expected_leaf_kinematics_mode:
        actual_modes = {
            str(name): int(count)
            for name, count in payload.get("actual_leaf_mode_distribution", {}).items()
        }
        if expected_leaf_kinematics_mode == "raw_track_predicted_pid":
            if actual_modes.get("raw_track_predicted_pid", 0) <= 0:
                raise ValueError(f"Requested raw Tracks but output contains none in {path}")
        elif expected_leaf_kinematics_mode == "fixed_hypothesis_candidate":
            if actual_modes.get("fixed_hypothesis_candidate", 0) <= 0:
                raise ValueError(
                    f"Requested fixed-hypothesis candidates but output contains none in {path}"
                )
    if expected_schema == SCHEMA_VERSION_V4 and expected_track_fit_policy:
        actual_policy = payload.get("preprocessing_configuration", {}).get(
            "track_fit_policy"
        )
        if actual_policy != expected_track_fit_policy:
            raise ValueError(f"Track-fit-policy mismatch in {path}")
            if actual_modes.get("raw_track_predicted_pid", 0) > 0:
                raise ValueError(
                    f"Fixed-hypothesis production unexpectedly contains raw Tracks in {path}"
                )
    if (
        expected_charge_conjugate_normalization is not None
        and payload.get("charge_conjugate_normalization") is not None
        and bool(payload.get("charge_conjugate_normalization"))
        != bool(expected_charge_conjugate_normalization)
    ):
        raise ValueError(f"Charge-conjugate-normalization mismatch in {path}")
    if expected_schema == SCHEMA_VERSION_V4:
        actual_events = 0
        seen_uids: set[str] = set()
        for event in iter_event_records_v4(path):
            uid = str(event.get("event_uid", ""))
            if not uid:
                raise ValueError(f"Missing event_uid in {path}")
            if uid in seen_uids:
                raise ValueError(f"Duplicate event_uid within {path}")
            seen_uids.add(uid)
            if uid_callback is not None:
                uid_callback(uid)
            actual_events += 1
        unique_events = len(seen_uids)
    else:
        actual_events = len(payload["events"])
        event_uids = [event.get("event_uid", "") for event in payload["events"]]
        if any(not event_uid for event_uid in event_uids):
            raise ValueError(f"Missing event_uid in {path}")
        if len(set(event_uids)) != len(event_uids):
            raise ValueError(f"Duplicate event_uid within {path}")
        if uid_callback is not None:
            for uid in event_uids:
                uid_callback(str(uid))
        unique_events = len(set(event_uids))
    if actual_events != expected_events:
        raise ValueError(
            f"Event-count mismatch for {path}: expected {expected_events}, got {actual_events}"
        )
    return {
        "output_file": str(path),
        "events": actual_events,
        "unique_event_uids": unique_events,
        "schema_version": payload.get("schema_version"),
        "feature_spec_hash": payload.get("feature_spec_hash", ""),
        "pid_vocabulary_version": payload.get("pid_vocabulary_version", ""),
    }


def run_task(
    *,
    manifest: Path,
    task_id: int,
    repo_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    """Run one manifest task through basf2, validate it, and publish atomically."""

    record = read_manifest_record(manifest, task_id)
    output_file = Path(str(record["output_file"]))
    expected_events = int(record["planned_events"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        result = validate_shard(
            output_file,
            expected_events=expected_events,
            expected_schema=str(record["schema_version"]),
            expected_feature_spec_hash=str(record["feature_spec_hash"]),
            expected_pid_vocabulary_version=str(record["pid_vocabulary_version"]),
            expected_leaf_kinematics_mode=str(record["leaf_kinematics_mode"]),
            expected_track_fit_policy=str(record["track_fit_policy"]),
            expected_charge_conjugate_normalization=bool(
                record["charge_conjugate_normalization"]
            ),
        )
        result["status"] = "already-complete"
        result["task_id"] = task_id
        return result

    # A completed shard is advertised exclusively by its marker.  Invalidate
    # that publication before doing any overwrite work so a worker failure can
    # only leave an explicitly incomplete old/new shard, never a stale marker.
    if overwrite:
        output_file.with_suffix(output_file.suffix + ".complete").unlink(
            missing_ok=True
        )

    basf2 = shutil.which("basf2")
    if basf2 is None:
        raise RuntimeError(
            "basf2 is not on PATH. Source /cvmfs/belle.cern.ch/tools/b2setup "
            "release-08-03-00 before running a production task."
        )

    temporary_output = output_file.with_name(
        f".{output_file.stem}.partial-{os.getpid()}{output_file.suffix}"
    )
    command = [
        basf2,
        str(repo_root / "scripts" / "preprocess_mdst.py"),
        "--",
        "--input",
        str(record["input_file"]),
        "--output",
        str(temporary_output),
        "--entry-sequence",
        str(record["entry_sequence"]),
        "--schema-version",
        str(record["schema_version"]),
        "--overwrite",
    ]
    if bool(record.get("charge_conjugate_normalization", False)):
        command.append("--charge-conjugate-normalize-channels")
    command.extend(
        [
            "--leaf-kinematics-mode",
            str(record["leaf_kinematics_mode"]),
            "--track-fit-policy",
            str(record["track_fit_policy"]),
            "--event-buffer-size",
            str(record.get("event_buffer_size", 128)),
            "--row-group-size",
            str(record.get("row_group_size", 128)),
        ]
    )
    # ``basf2`` embeds Python 3.8, whereas the project venv uses Python 3.11.
    # Compiled wheels cannot be shared between them, so production has a small
    # Python-3.8 dependency target containing awkward/pyarrow/numpy.
    subprocess_environment = os.environ.copy()
    basf2_python_site = Path(
        subprocess_environment.get("BASF2_PYTHON_SITE", DEFAULT_BASF2_PYTHON_SITE)
    )
    if not basf2_python_site.is_dir():
        raise RuntimeError(
            f"Missing basf2 Python dependencies at {basf2_python_site}. "
            "Install the documented Python-3.8 production packages or set "
            "BASF2_PYTHON_SITE."
        )
    existing_pythonpath = subprocess_environment.get("PYTHONPATH")
    subprocess_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(basf2_python_site), existing_pythonpath) if path
    )
    # Keep the requested venv active for the planner/validator process, but do
    # not let basf2 mistake its Python-3.11 executable for the home of the
    # embedded Python-3.8 runtime.
    subprocess_environment.pop("VIRTUAL_ENV", None)
    subprocess_environment.pop("VIRTUAL_ENV_PROMPT", None)
    worker_bin = str(Path(sys.executable).parent.resolve())
    subprocess_environment["PATH"] = os.pathsep.join(
        path
        for path in subprocess_environment.get("PATH", "").split(os.pathsep)
        if path and str(Path(path).resolve()) != worker_bin
    )
    try:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            env=subprocess_environment,
        )
        result = validate_shard(
            temporary_output,
            expected_events=expected_events,
            expected_schema=str(record["schema_version"]),
            expected_feature_spec_hash=str(record["feature_spec_hash"]),
            expected_pid_vocabulary_version=str(record["pid_vocabulary_version"]),
            expected_leaf_kinematics_mode=str(record["leaf_kinematics_mode"]),
            expected_track_fit_policy=str(record["track_fit_policy"]),
            expected_charge_conjugate_normalization=bool(
                record["charge_conjugate_normalization"]
            ),
        )
        temporary_sidecar = temporary_output.with_suffix(
            temporary_output.suffix + ".metadata.json"
        )
        if temporary_sidecar.exists():
            os.replace(
                temporary_sidecar,
                output_file.with_suffix(output_file.suffix + ".metadata.json"),
            )
        os.replace(temporary_output, output_file)
        temporary_marker = temporary_output.with_suffix(
            temporary_output.suffix + ".complete"
        )
        if temporary_marker.exists():
            os.replace(
                temporary_marker,
                output_file.with_suffix(output_file.suffix + ".complete"),
            )
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
        temporary_sidecar = temporary_output.with_suffix(
            temporary_output.suffix + ".metadata.json"
        )
        if temporary_sidecar.exists():
            temporary_sidecar.unlink()
        temporary_output.with_suffix(
            temporary_output.suffix + ".complete"
        ).unlink(missing_ok=True)

    result.update(
        {
            "status": "completed",
            "task_id": task_id,
            "entry_sequence": record["entry_sequence"],
            "physics_category": record["physics_category"],
            "output_file": str(output_file),
        }
    )
    return result


def validate_production_manifest(manifest: Path) -> dict[str, object]:
    """Validate ranges, shards, scientific config, and global event UIDs."""

    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("production manifest is empty")
    for record in records:
        record.setdefault("track_fit_policy", TRACK_FIT_POLICY_MAX_P_VALUE_V1)
    by_file: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    seen_task_ids: set[int] = set()
    uid_database_file = tempfile.NamedTemporaryFile(
        prefix="hypertagging-uids-", suffix=".sqlite", delete=False
    )
    uid_database_file.close()
    uid_database_path = Path(uid_database_file.name)
    uid_database: sqlite3.Connection | None = None
    categories: Counter[str] = Counter()
    completed = 0
    missing: list[int] = []
    total_events = 0
    config_fields = (
        "schema_version",
        "pid_vocabulary_version",
        "feature_spec_hash",
        "charge_conjugate_normalization",
        "leaf_kinematics_mode",
        "track_fit_policy",
    )
    expected_config = {field: records[0].get(field) for field in config_fields}
    uid_digest = hashlib.sha256()
    try:
        uid_database = sqlite3.connect(uid_database_path)
        uid_database.execute("CREATE TABLE event_uids (uid TEXT PRIMARY KEY)")

        def register_uid(uid: str) -> None:
            assert uid_database is not None
            try:
                uid_database.execute("INSERT INTO event_uids(uid) VALUES (?)", (uid,))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate event_uid across shards: {uid}") from error
            uid_digest.update(uid.encode("utf-8") + b"\0")

        for record in records:
            task_id = int(record["task_id"])
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task_id {task_id}")
            seen_task_ids.add(task_id)
            for field, expected in expected_config.items():
                if record.get(field) != expected:
                    raise ValueError(f"manifest scientific config mismatch for {field}")
            start = int(record["entry_start"])
            stop = int(record["entry_stop_exclusive"])
            by_file[str(record["input_file"])].append((start, stop, task_id))
            categories[str(record["physics_category"])] += int(record["planned_events"])
            output = Path(str(record["output_file"]))
            if not output.exists():
                missing.append(task_id)
                continue
            result = validate_shard(
                output,
                expected_events=int(record["planned_events"]),
                expected_schema=str(record["schema_version"]),
                expected_feature_spec_hash=str(record["feature_spec_hash"]),
                expected_pid_vocabulary_version=str(record["pid_vocabulary_version"]),
                expected_leaf_kinematics_mode=str(record["leaf_kinematics_mode"]),
                expected_track_fit_policy=str(record["track_fit_policy"]),
                expected_charge_conjugate_normalization=bool(record["charge_conjugate_normalization"]),
                uid_callback=register_uid,
            )
            completed += 1
            total_events += int(result["events"])
        for input_file, ranges in by_file.items():
            ranges.sort()
            for left, right in zip(ranges, ranges[1:]):
                if left[1] > right[0]:
                    raise ValueError(
                        f"overlapping source entry ranges for {input_file}: {left} and {right}"
                    )
        planned = sum(int(record["planned_events"]) for record in records)
        if not missing and total_events != planned:
            raise ValueError(f"global event count mismatch: planned {planned}, found {total_events}")
        unique_uid_count = int(uid_database.execute("SELECT COUNT(*) FROM event_uids").fetchone()[0])
    finally:
        if uid_database is not None:
            uid_database.close()
        uid_database_path.unlink(missing_ok=True)
    return {
        "tasks": len(records),
        "completed_shards": completed,
        "missing_shards": missing,
        "planned_events": planned,
        "validated_events": total_events,
        "unique_event_uids": unique_uid_count,
        "event_uid_digest": uid_digest.hexdigest(),
        "global_uid_validation_passes": 1,
        "category_distribution": dict(sorted(categories.items())),
        **expected_config,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create an exact JSONL production manifest")
    plan.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--target-events", type=int, default=DEFAULT_TARGET_EVENTS)
    plan.add_argument("--events-per-task", type=int, default=DEFAULT_EVENTS_PER_TASK)
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--schema-version", default=SCHEMA_VERSION_V4)
    plan.add_argument("--charge-conjugate-normalization", action="store_true")
    plan.add_argument("--leaf-kinematics-mode", default="raw_track_predicted_pid")
    plan.add_argument(
        "--track-fit-policy",
        choices=SUPPORTED_TRACK_FIT_POLICIES,
        default=TRACK_FIT_POLICY_MAX_P_VALUE_V1,
    )
    plan.add_argument("--event-buffer-size", type=int, default=128)
    plan.add_argument("--row-group-size", type=int, default=128)

    task = subparsers.add_parser("run-task", help="Execute one manifest task")
    task.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    task.add_argument("--task-id", type=int, default=None)
    task.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    task.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate all produced shards globally")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        input_files = discover_input_files(args.input_root)
        git_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or "unknown"
        records, category_events = build_manifest_records(
            input_files,
            output_root=args.output_root,
            target_events=args.target_events,
            events_per_task=args.events_per_task,
            schema_version=args.schema_version,
            charge_conjugate_normalization=args.charge_conjugate_normalization,
            leaf_kinematics_mode=args.leaf_kinematics_mode,
            track_fit_policy=args.track_fit_policy,
            git_commit=git_commit,
            event_buffer_size=args.event_buffer_size,
            row_group_size=args.row_group_size,
        )
        summary = write_manifest(
            records,
            manifest=args.manifest,
            input_root=args.input_root,
            output_root=args.output_root,
            target_events=args.target_events,
            events_per_task=args.events_per_task,
            category_events=category_events,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "validate":
        print(json.dumps(validate_production_manifest(args.manifest), sort_keys=True))
        return 0

    task_id = args.task_id
    if task_id is None:
        array_task = os.environ.get("CONDOR_PROCESS_ID")
        if array_task is None:
            raise ValueError("--task-id is required outside an HTCondor job")
        task_id = int(array_task)
    result = run_task(
        manifest=args.manifest,
        task_id=task_id,
        repo_root=args.repo_root.resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
