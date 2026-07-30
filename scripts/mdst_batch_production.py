#!/usr/bin/env python
"""Plan and execute sharded direct-mDST preprocessing production."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import awkward as ak
import uproot


DEFAULT_INPUT_ROOT = Path(
    "/pnfs/desy.de/belle/local/belle/MC/release-08-03-00/"
    "DB00003335/MC16ri_run2"
)
DEFAULT_OUTPUT_ROOT = Path("/data/dust/user/boyangyu/hypertagging/production_10m")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "mdst_10m.jsonl"
DEFAULT_TARGET_EVENTS = 10_000_000
DEFAULT_EVENTS_PER_TASK = 25_000
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
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build exact, non-overlapping entry-range records up to the target."""

    if target_events <= 0:
        raise ValueError("target_events must be positive")
    if events_per_task <= 0:
        raise ValueError("events_per_task must be positive")

    records: list[dict[str, object]] = []
    category_events: Counter[str] = Counter()
    planned_events = 0
    task_id = 0
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
                return record
    raise IndexError(f"task_id {task_id} is not present in {manifest}")


def validate_shard(path: Path, *, expected_events: int) -> dict[str, object]:
    """Validate schema and event count after one production task."""

    payload = ak.to_list(ak.from_parquet(path))[0]
    if payload.get("schema_version") not in {"direct-mdst-tree-v1", "direct-mdst-tree-v2"}:
        raise ValueError(f"Unexpected schema in {path}: {payload.get('schema_version')!r}")
    actual_events = len(payload["events"])
    if actual_events != expected_events:
        raise ValueError(
            f"Event-count mismatch for {path}: expected {expected_events}, got {actual_events}"
        )
    event_uids = [event.get("event_uid", "") for event in payload["events"]]
    if any(not event_uid for event_uid in event_uids):
        raise ValueError(f"Missing event_uid in {path}")
    if len(set(event_uids)) != len(event_uids):
        raise ValueError(f"Duplicate event_uid within {path}")
    return {
        "output_file": str(path),
        "events": actual_events,
        "unique_event_uids": len(set(event_uids)),
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
        result = validate_shard(output_file, expected_events=expected_events)
        result["status"] = "already-complete"
        result["task_id"] = task_id
        return result

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
        "--overwrite",
    ]
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
        result = validate_shard(temporary_output, expected_events=expected_events)
        os.replace(temporary_output, output_file)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

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

    task = subparsers.add_parser("run-task", help="Execute one manifest task")
    task.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    task.add_argument("--task-id", type=int, default=None)
    task.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    task.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        input_files = discover_input_files(args.input_root)
        records, category_events = build_manifest_records(
            input_files,
            output_root=args.output_root,
            target_events=args.target_events,
            events_per_task=args.events_per_task,
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
