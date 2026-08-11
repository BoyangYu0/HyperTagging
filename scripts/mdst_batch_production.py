#!/usr/bin/env python
"""Plan and execute sharded direct-mDST preprocessing production."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import hashlib
import time
import traceback
from typing import Any, Mapping

import awkward as ak
import pyarrow.parquet as pq
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
    COMPLETION_MARKER_VERSION,
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
MANIFEST_SCHEMA_VERSION = "hypertagging-production-manifest-v2"
CAMPAIGN_CONFIG_VERSION = "hypertagging-campaign-config-v2"
FAILURE_SCHEMA_VERSION = "hypertagging-task-failure-v1"
RESULT_CLASSIFICATIONS = (
    "COMPLETE_VALID",
    "MISSING",
    "INCOMPLETE_NO_MARKER",
    "CORRUPT_HASH",
    "METADATA_MISMATCH",
    "PROVENANCE_MISMATCH",
    "EVENT_COUNT_MISMATCH",
)
PROVENANCE_FIELDS = (
    "campaign_id",
    "campaign_config_digest",
    "source_git_commit",
    "source_git_tree",
    "source_state",
    "task_record_hash",
    "task_id",
    "input_file",
    "input_file_size",
    "input_file_mtime_ns",
    "input_file_identity",
    "physics_category",
    "entry_start",
    "entry_stop_exclusive",
    "planned_events",
    "output_file",
    "schema_version",
    "pid_vocabulary_version",
    "feature_spec_hash",
    "model_feature_contract_hash",
    "leaf_kinematics_mode",
    "track_fit_policy",
    "charge_conjugate_normalization",
    "event_buffer_size",
    "row_group_size",
    "campaign_stage",
    "klm_training_scope",
)


class ShardValidationError(ValueError):
    """Validation failure carrying a stable operator-facing classification."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        if classification not in RESULT_CLASSIFICATIONS:
            raise ValueError(f"unknown shard classification: {classification}")
        self.classification = classification


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def task_record_hash(record: Mapping[str, Any]) -> str:
    """Hash one immutable task record, excluding only its self hash."""

    return _canonical_digest(
        {key: value for key, value in record.items() if key != "task_record_hash"}
    )


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=repo_root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def source_identity(repo_root: Path) -> dict[str, str]:
    """Return the exact commit/tree and clean state used to plan a campaign."""

    commit = _git_output(repo_root, "rev-parse", "HEAD")
    tree = _git_output(repo_root, "rev-parse", "HEAD^{tree}")
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "source_git_commit": commit,
        "source_git_tree": tree,
        "source_state": "dirty" if status else "clean",
    }


def verify_worker_source(repo_root: Path, record: Mapping[str, Any]) -> dict[str, str]:
    """Refuse mutable or mismatched worker checkouts before basf2 starts."""

    current = source_identity(repo_root)
    expected_commit = str(record.get("source_git_commit", ""))
    expected_tree = str(record.get("source_git_tree", ""))
    if str(record.get("source_state")) != "clean":
        raise RuntimeError("manifest was planned from a dirty source state")
    if current["source_state"] != "clean":
        raise RuntimeError("worker checkout is dirty; refusing mutable campaign source")
    if current["source_git_commit"] != expected_commit:
        raise RuntimeError(
            "worker source commit mismatch: "
            f"expected {expected_commit}, got {current['source_git_commit']}"
        )
    if current["source_git_tree"] != expected_tree:
        raise RuntimeError(
            "worker source tree mismatch: "
            f"expected {expected_tree}, got {current['source_git_tree']}"
        )
    return current


def _file_identity(path: Path, *, checksum: bool = False) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "input_file_size": int(stat.st_size),
        "input_file_mtime_ns": int(stat.st_mtime_ns),
        # st_dev is a client-local mount identifier and is not stable across
        # worker hosts for shared filesystems such as PNFS.  The manifest also
        # binds the absolute path, size, and nanosecond mtime; retain the inode
        # as a stable namespace identity without introducing a host-local field.
        "input_file_identity": (
            f"stat-v2:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
        ),
        "input_file_sha256": None,
    }
    if checksum:
        identity["input_file_sha256"] = _sha256_path(path)
    return identity


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    git_commit: str | None = None,
    source_git_commit: str | None = None,
    source_git_tree: str = "unknown",
    source_state: str = "dirty",
    campaign_id: str | None = None,
    campaign_config_digest: str | None = None,
    event_buffer_size: int = 128,
    row_group_size: int = 128,
    checksum_inputs: bool = False,
    campaign_stage: str = "custom",
    klm_training_scope: str = "unresolved",
    production_readiness_report_sha256: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build exact, non-overlapping entry-range records up to the target."""

    if target_events <= 0:
        raise ValueError("target_events must be positive")
    if events_per_task <= 0:
        raise ValueError("events_per_task must be positive")
    if track_fit_policy not in SUPPORTED_TRACK_FIT_POLICIES:
        raise ValueError(f"unknown track_fit_policy: {track_fit_policy}")
    if klm_training_scope not in {"included", "excluded_by_policy", "unresolved"}:
        raise ValueError(f"unknown klm_training_scope: {klm_training_scope}")

    input_files = list(input_files)
    resolved_commit = str(source_git_commit or git_commit or "unknown")
    spec = feature_spec_v4() if schema_version == SCHEMA_VERSION_V4 else feature_spec_v3()
    feature_hash = str(spec["feature_spec_hash"])
    model_contract_hash = str(spec.get("model_feature_contract_hash", "legacy"))
    input_identities: list[dict[str, Any]] = []
    for path in input_files:
        if path.exists():
            identity = _file_identity(path, checksum=checksum_inputs)
        else:
            identity = {
                "input_file_size": -1,
                "input_file_mtime_ns": -1,
                "input_file_identity": "missing-at-plan",
                "input_file_sha256": None,
            }
        input_identities.append({"input_file": str(path), **identity})
    config_payload = {
        "campaign_config_version": CAMPAIGN_CONFIG_VERSION,
        "source_git_commit": resolved_commit,
        "source_git_tree": source_git_tree,
        "source_state": source_state,
        "target_events": target_events,
        "events_per_task": events_per_task,
        "schema_version": schema_version,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": feature_hash,
        "model_feature_contract_hash": model_contract_hash,
        "charge_conjugate_normalization": bool(charge_conjugate_normalization),
        "leaf_kinematics_mode": leaf_kinematics_mode,
        "track_fit_policy": track_fit_policy,
        "event_buffer_size": int(event_buffer_size),
        "row_group_size": int(row_group_size),
        "campaign_stage": campaign_stage,
        "klm_training_scope": klm_training_scope,
        "production_readiness_report_sha256": production_readiness_report_sha256,
        "inputs": input_identities,
    }
    resolved_config_digest = campaign_config_digest or _canonical_digest(config_payload)
    resolved_campaign_id = campaign_id or (
        f"campaign-{resolved_config_digest[:12]}-{resolved_commit[:12]}"
    )
    records: list[dict[str, object]] = []
    category_events: Counter[str] = Counter()
    planned_events = 0
    task_id = 0
    identity_by_path = {entry["input_file"]: entry for entry in input_identities}
    file_event_counts = {path: root_event_count(path) for path in input_files}
    next_entry = {path: 0 for path in input_files}
    while planned_events < target_events:
        progress = False
        for input_file in input_files:
            if planned_events >= target_events:
                break
            file_events = file_event_counts[input_file]
            entry_start = next_entry[input_file]
            if entry_start >= file_events:
                continue
            progress = True
            category = _physics_category(input_file)
            requested = min(events_per_task, target_events - planned_events)
            entry_stop = min(file_events, entry_start + requested)
            count = entry_stop - entry_start
            output_file = (
                output_root
                / resolved_campaign_id
                / "shards"
                / f"mdst_{task_id:05d}.parquet"
            )
            identity = identity_by_path[str(input_file)]
            record: dict[str, object] = {
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "task_id": task_id,
                    "campaign_id": resolved_campaign_id,
                    "campaign_config_digest": resolved_config_digest,
                    "source_git_commit": resolved_commit,
                    "source_git_tree": source_git_tree,
                    "source_state": source_state,
                    "input_file": str(input_file),
                    "input_file_size": identity["input_file_size"],
                    "input_file_mtime_ns": identity["input_file_mtime_ns"],
                    "input_file_identity": identity["input_file_identity"],
                    "input_file_sha256": identity["input_file_sha256"],
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
                    "model_feature_contract_hash": model_contract_hash,
                    "git_commit": resolved_commit,
                    "event_buffer_size": int(event_buffer_size),
                    "row_group_size": int(row_group_size),
                    "campaign_stage": campaign_stage,
                    "klm_training_scope": klm_training_scope,
                    "production_readiness_report_sha256": production_readiness_report_sha256,
                }
            record["task_record_hash"] = task_record_hash(record)
            records.append(record)
            planned_events += count
            category_events[category] += count
            task_id += 1
            next_entry[input_file] = entry_stop
        if not progress:
            break

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
    spec = feature_spec_v4()
    defaults = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION_V4,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "charge_conjugate_normalization": False,
        "leaf_kinematics_mode": "raw_track_predicted_pid",
        "track_fit_policy": TRACK_FIT_POLICY_MAX_P_VALUE_V1,
        "feature_spec_hash": spec["feature_spec_hash"],
        "model_feature_contract_hash": spec["model_feature_contract_hash"],
        "git_commit": "unknown",
        "source_git_commit": "unknown",
        "source_git_tree": "unknown",
        "source_state": "dirty",
        "event_buffer_size": 128,
        "row_group_size": 128,
        "campaign_stage": "custom",
        "klm_training_scope": "unresolved",
        "production_readiness_report_sha256": None,
    }
    fallback_config = {
        "campaign_config_version": CAMPAIGN_CONFIG_VERSION,
        "target_events": target_events,
        "events_per_task": events_per_task,
        **defaults,
    }
    fallback_digest = _canonical_digest(fallback_config)
    for record in records:
        for key, value in defaults.items():
            record.setdefault(key, value)
        record.setdefault("campaign_config_digest", fallback_digest)
        record.setdefault("campaign_id", f"campaign-{fallback_digest[:12]}")
        input_path = Path(str(record.get("input_file", "")))
        identity = (
            _file_identity(input_path)
            if input_path.is_file()
            else {
                "input_file_size": -1,
                "input_file_mtime_ns": -1,
                "input_file_identity": "missing-at-plan",
                "input_file_sha256": None,
            }
        )
        for key, value in identity.items():
            record.setdefault(key, value)
        record["task_record_hash"] = task_record_hash(record)
    task_ids = [int(record["task_id"]) for record in records]
    task_hashes = [str(record["task_record_hash"]) for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest task IDs must be unique")
    if len(task_hashes) != len(set(task_hashes)):
        raise ValueError("manifest task hashes must be unique")
    campaign_ids = {str(record["campaign_id"]) for record in records}
    config_digests = {str(record["campaign_config_digest"]) for record in records}
    if len(campaign_ids) != 1 or len(config_digests) != 1:
        raise ValueError("manifest must contain exactly one campaign/config digest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if records:
        Path(str(records[0]["output_file"])).parent.mkdir(parents=True, exist_ok=True)
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
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": next(iter(campaign_ids), ""),
        "campaign_config_digest": next(iter(config_digests), ""),
        "source_git_commit": records[0]["source_git_commit"] if records else "unknown",
        "source_git_tree": records[0]["source_git_tree"] if records else "unknown",
        "source_state": records[0]["source_state"] if records else "dirty",
        "schema_version": records[0]["schema_version"] if records else SCHEMA_VERSION_V4,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": (
            records[0]["feature_spec_hash"]
            if records
            else feature_spec_v4()["feature_spec_hash"]
        ),
        "model_feature_contract_hash": (
            records[0]["model_feature_contract_hash"]
            if records
            else spec["model_feature_contract_hash"]
        ),
        "task_hash_digest": _canonical_digest(
            {str(record["task_id"]): record["task_record_hash"] for record in records}
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
                expected_hash = str(record.get("task_record_hash", ""))
                actual_hash = task_record_hash(record)
                if not expected_hash or expected_hash != actual_hash:
                    raise ShardValidationError(
                        "PROVENANCE_MISMATCH",
                        f"task_record_hash mismatch for task {task_id}",
                    )
                return record
    raise IndexError(f"task_id {task_id} is not present in {manifest}")


def _parquet_metadata(path: Path) -> dict[str, Any]:
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    output: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            output[key.decode("utf-8")] = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            output[key.decode("utf-8", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )
    return output


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    classification: str,
    label: str,
    path: Path,
) -> None:
    if actual != expected:
        raise ShardValidationError(
            classification,
            f"{label} mismatch in {path}: expected {expected!r}, got {actual!r}",
        )


def _validate_shard_or_raise(
    path: Path,
    *,
    expected_events: int,
    expected_schema: str = SCHEMA_VERSION_V4,
    expected_feature_spec_hash: str | None = None,
    expected_pid_vocabulary_version: str | None = None,
    expected_leaf_kinematics_mode: str | None = None,
    expected_track_fit_policy: str | None = None,
    expected_charge_conjugate_normalization: bool | None = None,
    expected_model_feature_contract_hash: str | None = None,
    expected_record: Mapping[str, Any] | None = None,
    uid_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate marker, hashes, metadata, provenance, range, and event rows."""

    if expected_schema == SCHEMA_VERSION_V4:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        existing = [candidate.exists() for candidate in (path, sidecar, marker)]
        if not any(existing):
            raise ShardValidationError("MISSING", f"No shard artifacts for {path}")
        if not marker.exists():
            raise ShardValidationError(
                "INCOMPLETE_NO_MARKER", f"Missing completion marker for {path}"
            )
        if not path.exists() or not sidecar.exists():
            raise ShardValidationError(
                "METADATA_MISMATCH",
                f"Completion marker exists without parquet and sidecar for {path}",
            )
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ShardValidationError(
                "METADATA_MISMATCH", f"Invalid marker/sidecar JSON for {path}: {error}"
            ) from error
        if marker_payload.get("marker_schema_version") != COMPLETION_MARKER_VERSION:
            raise ShardValidationError(
                "METADATA_MISMATCH", f"Unsupported completion-marker schema for {path}"
            )
        parquet_digest = str(marker_payload.get("parquet_sha256", ""))
        sidecar_digest = str(marker_payload.get("sidecar_sha256", ""))
        if (
            len(parquet_digest) != 64
            or parquet_digest != _sha256_path(path)
            or len(sidecar_digest) != 64
            or sidecar_digest != _sha256_path(sidecar)
        ):
            raise ShardValidationError(
                "CORRUPT_HASH", f"Completion-marker digest mismatch for {path}"
            )
        for field in (
            "schema_version",
            "event_count",
            "feature_spec_hash",
            "model_feature_contract_hash",
        ):
            _require_equal(
                marker_payload.get(field),
                metadata.get(field),
                classification="METADATA_MISMATCH",
                label=f"marker/sidecar {field}",
                path=path,
            )
        parquet_metadata = _parquet_metadata(path)
        payload = {
            **metadata,
            "schema_version": metadata.get("schema_version"),
        }
    else:
        if not path.exists():
            raise ShardValidationError("MISSING", f"Missing shard {path}")
        payload = ak.to_list(ak.from_parquet(path))[0]
    if payload.get("schema_version") != expected_schema:
        raise ShardValidationError(
            "METADATA_MISMATCH",
            f"Unexpected schema in {path}: {payload.get('schema_version')!r}",
        )
    if (
        expected_feature_spec_hash is not None
        and payload.get("feature_spec_hash") != expected_feature_spec_hash
    ):
        raise ShardValidationError("METADATA_MISMATCH", f"Feature-spec mismatch in {path}")
    if (
        expected_model_feature_contract_hash is not None
        and payload.get("model_feature_contract_hash")
        != expected_model_feature_contract_hash
    ):
        raise ShardValidationError(
            "METADATA_MISMATCH", f"Model-feature-contract mismatch in {path}"
        )
    if (
        expected_pid_vocabulary_version is not None
        and payload.get("pid_vocabulary_version") != expected_pid_vocabulary_version
    ):
        raise ShardValidationError("METADATA_MISMATCH", f"PID-vocabulary mismatch in {path}")
    if (
        expected_leaf_kinematics_mode is not None
        and payload.get("requested_collection_mode") is not None
        and payload.get("requested_collection_mode") != expected_leaf_kinematics_mode
    ):
        raise ShardValidationError("METADATA_MISMATCH", f"Leaf-kinematics-mode mismatch in {path}")
    if expected_schema == SCHEMA_VERSION_V4 and expected_leaf_kinematics_mode:
        actual_modes = {
            str(name): int(count)
            for name, count in payload.get("actual_leaf_mode_distribution", {}).items()
        }
        if expected_leaf_kinematics_mode == "raw_track_predicted_pid":
            if actual_modes.get("raw_track_predicted_pid", 0) <= 0:
                raise ShardValidationError("METADATA_MISMATCH", f"Requested raw Tracks but output contains none in {path}")
        elif expected_leaf_kinematics_mode == "fixed_hypothesis_candidate":
            if actual_modes.get("fixed_hypothesis_candidate", 0) <= 0:
                raise ShardValidationError(
                    "METADATA_MISMATCH",
                    f"Requested fixed-hypothesis candidates but output contains none in {path}"
                )
            if actual_modes.get("raw_track_predicted_pid", 0) > 0:
                raise ShardValidationError(
                    "METADATA_MISMATCH",
                    f"Fixed-hypothesis production unexpectedly contains raw Tracks in {path}",
                )
    if expected_schema == SCHEMA_VERSION_V4 and expected_track_fit_policy:
        actual_policy = payload.get("preprocessing_configuration", {}).get(
            "track_fit_policy"
        )
        if actual_policy != expected_track_fit_policy:
            raise ShardValidationError("METADATA_MISMATCH", f"Track-fit-policy mismatch in {path}")
    if (
        expected_charge_conjugate_normalization is not None
        and payload.get("charge_conjugate_normalization") is not None
        and bool(payload.get("charge_conjugate_normalization"))
        != bool(expected_charge_conjugate_normalization)
    ):
        raise ShardValidationError("METADATA_MISMATCH", f"Charge-conjugate-normalization mismatch in {path}")
    if expected_schema == SCHEMA_VERSION_V4 and expected_record is not None:
        provenance_pairs = {
            "campaign_id": expected_record.get("campaign_id"),
            "campaign_config_digest": expected_record.get("campaign_config_digest"),
            "source_git_commit": expected_record.get("source_git_commit"),
            "source_git_tree": expected_record.get("source_git_tree"),
            "source_state": expected_record.get("source_state"),
            "task_record_hash": expected_record.get("task_record_hash"),
            "task_id": expected_record.get("task_id"),
            "source_file": expected_record.get("input_file"),
            "source_file_size": expected_record.get("input_file_size"),
            "source_file_mtime_ns": expected_record.get("input_file_mtime_ns"),
            "source_file_identity": expected_record.get("input_file_identity"),
            "source_file_sha256": expected_record.get("input_file_sha256"),
            "entry_start": expected_record.get("entry_start"),
            "entry_stop_exclusive": expected_record.get("entry_stop_exclusive"),
            "planned_events": expected_record.get("planned_events"),
            "physics_category": expected_record.get("physics_category"),
            "output_file": expected_record.get("output_file"),
            "leaf_kinematics_mode": expected_record.get("leaf_kinematics_mode"),
            "track_fit_policy": expected_record.get("track_fit_policy"),
            "charge_conjugate_normalization": expected_record.get(
                "charge_conjugate_normalization"
            ),
            "event_buffer_size": expected_record.get("event_buffer_size"),
            "row_group_size": expected_record.get("row_group_size"),
            "campaign_stage": expected_record.get("campaign_stage"),
            "klm_training_scope": expected_record.get("klm_training_scope"),
            "production_readiness_report_sha256": expected_record.get(
                "production_readiness_report_sha256"
            ),
        }
        for field, expected in provenance_pairs.items():
            _require_equal(
                payload.get(field), expected,
                classification="PROVENANCE_MISMATCH", label=field, path=path,
            )
            _require_equal(
                marker_payload.get(field), expected,
                classification="PROVENANCE_MISMATCH", label=f"marker {field}", path=path,
            )
            _require_equal(
                parquet_metadata.get(field), expected,
                classification="PROVENANCE_MISMATCH", label=f"parquet {field}", path=path,
            )
    if expected_schema == SCHEMA_VERSION_V4:
        actual_events = 0
        seen_uids: set[str] = set()
        node_counts: list[int] = []
        max_depths: list[int] = []
        leaf_modes: Counter[str] = Counter()
        klm_nodes = 0
        klm_associated_ecl = 0
        incomplete_branches = 0
        root_distribution: Counter[str] = Counter()
        klm_diagnostics: Counter[str] = Counter()
        klm_feature_availability: Counter[str] = Counter()
        for event in iter_event_records_v4(path):
            uid = str(event.get("event_uid", ""))
            if not uid:
                raise ValueError(f"Missing event_uid in {path}")
            if uid in seen_uids:
                raise ValueError(f"Duplicate event_uid within {path}")
            seen_uids.add(uid)
            if uid_callback is not None:
                uid_callback(uid)
            nodes = list(event.get("nodes", []))
            node_by_id = {int(node["node_id"]): node for node in nodes}
            klm_descendant_memo: dict[int, bool] = {}

            def has_klm_descendant(node_id: int, visiting: set[int] | None = None) -> bool:
                if node_id in klm_descendant_memo:
                    return klm_descendant_memo[node_id]
                visiting = set() if visiting is None else visiting
                if node_id in visiting:
                    return False
                node = node_by_id[node_id]
                found = str(node.get("node_kind")) == "klm_cluster" or any(
                    has_klm_descendant(int(child), visiting | {node_id})
                    for child in node.get("daughter_ids", [])
                    if int(child) in node_by_id
                )
                klm_descendant_memo[node_id] = found
                return found
            node_counts.append(len(nodes))
            max_depths.append(max((int(node.get("level", 0)) for node in nodes), default=0))
            for node in nodes:
                leaf_modes[str(node.get("leaf_kinematics_mode", "unknown"))] += int(
                    not node.get("daughter_ids")
                )
                if str(node.get("node_kind")) == "klm_cluster":
                    klm_nodes += 1
                    klm_associated_ecl += int(bool(node.get("associated_reco_id")))
                    flags = set(node.get("flags", []))
                    klm_diagnostics["retained_klm_nodes"] += 1
                    klm_diagnostics["matched_reconstructed_klm_leaves"] += int(
                        "unmatched_reco" not in flags
                    )
                    klm_diagnostics["unmatched_klm_clusters"] += int(
                        "unmatched_reco" in flags
                    )
                    klm_diagnostics["klm_clusters_associated_with_ecl"] += int(
                        bool(node.get("associated_reco_id"))
                    )
                    for name, available in node.get("klm_availability", {}).items():
                        klm_feature_availability[f"{name}:{bool(available)}"] += 1
                if not node.get("daughter_ids") and abs(
                    int(node.get("raw_pdg", node.get("pdg", 0)))
                ) == 130:
                    klm_diagnostics["truth_kl_like_retained_leaves"] += 1
                if node.get("daughter_ids"):
                    complete = bool(node.get("recursive_reconstructable_complete", False))
                    klm_diagnostics["complete_branches_with_klm"] += int(complete)
                    klm_diagnostics["complete_branches_without_klm"] += int(
                        complete and not has_klm_descendant(int(node["node_id"]))
                    )
                incomplete_branches += int(
                    bool(node.get("daughter_ids"))
                    and not bool(node.get("recursive_reconstructable_complete", False))
                )
            if bool(event.get("b_root_discovery_fallback", False)):
                root_distribution["fallback"] += 1
            elif bool(event.get("b_root_discovery_valid", False)):
                root_distribution["strict"] += 1
            else:
                root_distribution["missing"] += 1
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
        raise ShardValidationError(
            "EVENT_COUNT_MISMATCH",
            f"Event-count mismatch for {path}: expected {expected_events}, got {actual_events}"
        )
    return {
        "classification": "COMPLETE_VALID",
        "output_file": str(path),
        "events": actual_events,
        "unique_event_uids": unique_events,
        "schema_version": payload.get("schema_version"),
        "feature_spec_hash": payload.get("feature_spec_hash", ""),
        "pid_vocabulary_version": payload.get("pid_vocabulary_version", ""),
        "model_feature_contract_hash": payload.get("model_feature_contract_hash", ""),
        "campaign_id": payload.get("campaign_id", ""),
        "campaign_config_digest": payload.get("campaign_config_digest", ""),
        "source_git_commit": payload.get("source_git_commit", payload.get("git_commit", "")),
        "task_record_hash": payload.get("task_record_hash", ""),
        "source_file": payload.get("source_file", ""),
        "entry_start": payload.get("entry_start"),
        "entry_stop_exclusive": payload.get("entry_stop_exclusive"),
        "actual_leaf_mode_distribution": (
            dict(sorted(leaf_modes.items())) if expected_schema == SCHEMA_VERSION_V4 else {}
        ),
        "klm_nodes": klm_nodes if expected_schema == SCHEMA_VERSION_V4 else 0,
        "klm_associated_ecl": klm_associated_ecl if expected_schema == SCHEMA_VERSION_V4 else 0,
        "incomplete_reconstructable_branches": (
            incomplete_branches if expected_schema == SCHEMA_VERSION_V4 else 0
        ),
        "b_root_distribution": (
            dict(sorted(root_distribution.items())) if expected_schema == SCHEMA_VERSION_V4 else {}
        ),
        "node_counts": node_counts if expected_schema == SCHEMA_VERSION_V4 else [],
        "max_depths": max_depths if expected_schema == SCHEMA_VERSION_V4 else [],
        "output_bytes": path.stat().st_size,
        "klm_diagnostics": (
            {
                **dict(sorted(klm_diagnostics.items())),
                "klm_feature_availability": dict(
                    sorted(klm_feature_availability.items())
                ),
                "collected_klm_records": int(
                    payload.get("preprocessing_configuration", {})
                    .get("collection", {})
                    .get("klm_records", 0)
                ),
            }
            if expected_schema == SCHEMA_VERSION_V4
            else {}
        ),
    }


def validate_shard(
    path: Path,
    **kwargs: Any,
) -> dict[str, object]:
    """Raise a classified error unless the shard is complete and fully valid."""

    return _validate_shard_or_raise(path, **kwargs)


def classify_shard(path: Path, **kwargs: Any) -> dict[str, object]:
    """Return an explicit shard classification without raising validation errors."""

    try:
        return _validate_shard_or_raise(path, **kwargs)
    except ShardValidationError as error:
        return {
            "output_file": str(path),
            "classification": error.classification,
            "error": str(error),
        }


def _validation_kwargs(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_events": int(record["planned_events"]),
        "expected_schema": str(record["schema_version"]),
        "expected_feature_spec_hash": str(record["feature_spec_hash"]),
        "expected_model_feature_contract_hash": str(
            record["model_feature_contract_hash"]
        ),
        "expected_pid_vocabulary_version": str(record["pid_vocabulary_version"]),
        "expected_leaf_kinematics_mode": str(record["leaf_kinematics_mode"]),
        "expected_track_fit_policy": str(record["track_fit_policy"]),
        "expected_charge_conjugate_normalization": bool(
            record["charge_conjugate_normalization"]
        ),
        "expected_record": record,
    }


def _task_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a manifest record onto persisted parquet/sidecar/marker names."""

    return {
        "campaign_id": record["campaign_id"],
        "campaign_config_digest": record["campaign_config_digest"],
        "source_git_commit": record["source_git_commit"],
        "source_git_tree": record["source_git_tree"],
        "source_state": record["source_state"],
        "task_record_hash": record["task_record_hash"],
        "task_id": record["task_id"],
        "source_file": record["input_file"],
        "source_file_size": record["input_file_size"],
        "source_file_mtime_ns": record["input_file_mtime_ns"],
        "source_file_identity": record["input_file_identity"],
        "source_file_sha256": record.get("input_file_sha256"),
        "physics_category": record["physics_category"],
        "category": record["physics_category"],
        "entry_start": record["entry_start"],
        "entry_stop_exclusive": record["entry_stop_exclusive"],
        "planned_events": record["planned_events"],
        "output_file": record["output_file"],
        "schema_version": record["schema_version"],
        "pid_vocabulary_version": record["pid_vocabulary_version"],
        "feature_spec_hash": record["feature_spec_hash"],
        "model_feature_contract_hash": record["model_feature_contract_hash"],
        "leaf_kinematics_mode": record["leaf_kinematics_mode"],
        "track_fit_policy": record["track_fit_policy"],
        "charge_conjugate_normalization": record[
            "charge_conjugate_normalization"
        ],
        "event_buffer_size": record["event_buffer_size"],
        "row_group_size": record["row_group_size"],
        "campaign_stage": record["campaign_stage"],
        "klm_training_scope": record["klm_training_scope"],
        "production_readiness_report_sha256": record.get(
            "production_readiness_report_sha256"
        ),
    }


def _artifact_paths(output_file: Path) -> list[Path]:
    return [
        output_file,
        output_file.with_suffix(output_file.suffix + ".metadata.json"),
        output_file.with_suffix(output_file.suffix + ".complete"),
        output_file.with_suffix(output_file.suffix + ".result.json"),
        output_file.with_suffix(output_file.suffix + ".failure.json"),
    ]


def quarantine_task_artifacts(
    output_file: Path,
    *,
    record: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> Path | None:
    """Move inconsistent artifacts into a recoverable task-specific directory."""

    existing = [path for path in _artifact_paths(output_file) if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = (
        output_file.parent.parent
        / "quarantine"
        / f"task-{int(record['task_id']):05d}"
        / stamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, destination / path.name)
    (destination / "diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "hypertagging-quarantine-diagnostic-v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": record["task_id"],
                "task_record_hash": record["task_record_hash"],
                "source_file": record["input_file"],
                "entry_start": record["entry_start"],
                "entry_stop_exclusive": record["entry_stop_exclusive"],
                **dict(diagnostics),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def _write_failure(
    output_file: Path,
    *,
    record: Mapping[str, Any],
    error: BaseException,
    event_uid: str | None = None,
    event_index: int | None = None,
) -> Path:
    path = output_file.with_suffix(output_file.suffix + ".failure.json")
    payload = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "task_id": record["task_id"],
        "source_file": record["input_file"],
        "entry_start": record["entry_start"],
        "entry_stop_exclusive": record["entry_stop_exclusive"],
        "event_uid": event_uid,
        "event_index": event_index,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "source_git_commit": record["source_git_commit"],
        "task_record_hash": record["task_record_hash"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traceback": "".join(traceback.format_exception(error))[-12000:],
    }
    _write_json_atomic(path, payload)
    return path


def _verify_input_identity(record: Mapping[str, Any]) -> None:
    path = Path(str(record["input_file"]))
    if not path.is_file():
        raise FileNotFoundError(f"manifest input file is missing: {path}")
    current = _file_identity(path, checksum=bool(record.get("input_file_sha256")))
    for field in (
        "input_file_size",
        "input_file_mtime_ns",
        "input_file_identity",
        "input_file_sha256",
    ):
        if current.get(field) != record.get(field):
            raise RuntimeError(
                f"input identity mismatch for {path}: {field} changed from "
                f"{record.get(field)!r} to {current.get(field)!r}"
            )


def run_task(
    *,
    manifest: Path,
    task_id: int,
    repo_root: Path,
    overwrite: bool = False,
    destructive_overwrite: bool = False,
) -> dict[str, object]:
    """Run one manifest task through basf2, validate it, and publish atomically."""

    record = read_manifest_record(manifest, task_id)
    output_file = Path(str(record["output_file"]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if str(record["task_record_hash"]) != task_record_hash(record):
        raise ShardValidationError(
            "PROVENANCE_MISMATCH", f"task hash changed for task {task_id}"
        )
    try:
        verify_worker_source(repo_root, record)
        _verify_input_identity(record)
        if str(record.get("campaign_stage")) == "production_10m":
            if str(record.get("klm_training_scope")) == "unresolved":
                raise RuntimeError("10M worker refused: klm_training_scope is unresolved")
            if not record.get("production_readiness_report_sha256"):
                raise RuntimeError(
                    "10M worker refused: no representative canary readiness report is bound"
                )
    except Exception as error:
        _write_failure(output_file, record=record, error=error)
        raise
    expected_events = int(record["planned_events"])

    destructive = bool(overwrite or destructive_overwrite)
    existing = classify_shard(output_file, **_validation_kwargs(record))
    if existing["classification"] == "COMPLETE_VALID" and not destructive:
        result = dict(existing)
        result["status"] = "already-complete"
        result["task_id"] = task_id
        return result
    if existing["classification"] == "COMPLETE_VALID" and destructive:
        quarantine_task_artifacts(
            output_file,
            record=record,
            diagnostics={"reason": "explicit destructive overwrite", **existing},
        )
    elif existing["classification"] != "MISSING":
        quarantine_task_artifacts(output_file, record=record, diagnostics=existing)

    basf2 = shutil.which("basf2")
    if basf2 is None:
        error = RuntimeError(
            "basf2 is not on PATH. Source /cvmfs/belle.cern.ch/tools/b2setup "
            "release-08-03-00 before running a production task."
        )
        _write_failure(output_file, record=record, error=error)
        raise error

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
    provenance_file_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"hypertagging-task-{task_id}-",
        suffix=".json",
        delete=False,
    )
    provenance_path = Path(provenance_file_handle.name)
    json.dump(_task_provenance(record), provenance_file_handle, sort_keys=True)
    provenance_file_handle.write("\n")
    provenance_file_handle.close()
    command.extend(["--production-provenance-json", str(provenance_path)])
    if bool(record.get("charge_conjugate_normalization", False)):
        command.append("--charge-conjugate-normalize-channels")
    if str(record.get("klm_training_scope")) == "excluded_by_policy":
        command.append("--no-klm-clusters")
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
        error = RuntimeError(
            f"Missing basf2 Python dependencies at {basf2_python_site}. "
            "Install the documented Python-3.8 production packages or set "
            "BASF2_PYTHON_SITE."
        )
        _write_failure(output_file, record=record, error=error)
        provenance_path.unlink(missing_ok=True)
        raise error
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
    started = time.perf_counter()
    try:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            env=subprocess_environment,
        )
        validation_started = time.perf_counter()
        validate_shard(temporary_output, **_validation_kwargs(record))
        validation_seconds = time.perf_counter() - validation_started
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
        result = validate_shard(output_file, **_validation_kwargs(record))
    except Exception as error:
        _write_failure(output_file, record=record, error=error)
        raise
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
        provenance_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    result.update(
        {
            "status": "completed",
            "task_id": task_id,
            "entry_sequence": record["entry_sequence"],
            "physics_category": record["physics_category"],
            "output_file": str(output_file),
            "campaign_id": record["campaign_id"],
            "campaign_config_digest": record["campaign_config_digest"],
            "source_git_commit": record["source_git_commit"],
            "source_git_tree": record["source_git_tree"],
            "source_state": record["source_state"],
            "task_record_hash": record["task_record_hash"],
            "source_file": record["input_file"],
            "entry_start": record["entry_start"],
            "entry_stop_exclusive": record["entry_stop_exclusive"],
            "elapsed_seconds": elapsed,
            "events_per_second": expected_events / max(elapsed, 1e-12),
            "validation_seconds": validation_seconds,
            "peak_resident_memory_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    result_path = output_file.with_suffix(output_file.suffix + ".result.json")
    _write_json_atomic(result_path, result)
    return result


def read_manifest(manifest: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("production manifest is empty")
    for record in records:
        record.setdefault("track_fit_policy", TRACK_FIT_POLICY_MAX_P_VALUE_V1)
    return records


def _quantiles(values: Iterable[int]) -> dict[str, float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    output: dict[str, float] = {}
    for label, fraction in (("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99)):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        output[label] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    output["max"] = float(ordered[-1])
    return output


def validate_production_manifest(manifest: Path) -> dict[str, object]:
    """Validate every task/shard and exact global production invariants."""

    records = read_manifest(manifest)
    by_file: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    seen_task_ids: set[int] = set()
    seen_task_hashes: set[str] = set()
    uid_database_file = tempfile.NamedTemporaryFile(
        prefix="hypertagging-uids-", suffix=".sqlite", delete=False
    )
    uid_database_file.close()
    uid_database_path = Path(uid_database_file.name)
    uid_database: sqlite3.Connection | None = None
    planned_categories: Counter[str] = Counter()
    produced_categories: Counter[str] = Counter()
    track_policies: Counter[str] = Counter()
    actual_leaf_modes: Counter[str] = Counter()
    b_roots: Counter[str] = Counter()
    node_counts: list[int] = []
    max_depths: list[int] = []
    klm_nodes = 0
    klm_associated_ecl = 0
    incomplete_branches = 0
    output_bytes = 0
    klm_diagnostics: Counter[str] = Counter()
    klm_feature_availability: Counter[str] = Counter()
    klm_by_category: Counter[str] = Counter()
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
            task_hash = str(record.get("task_record_hash", ""))
            contract_v2 = record.get("manifest_schema_version") == MANIFEST_SCHEMA_VERSION
            if contract_v2:
                if task_hash != task_record_hash(record):
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
            planned_categories[str(record["physics_category"])] += int(record["planned_events"])
            output = Path(str(record["output_file"]))
            if not output.exists():
                missing.append(task_id)
                continue
            validation_kwargs: dict[str, Any] = {
                "expected_events": int(record["planned_events"]),
                "expected_schema": str(record["schema_version"]),
                "expected_feature_spec_hash": str(record["feature_spec_hash"]),
                "expected_pid_vocabulary_version": str(record["pid_vocabulary_version"]),
                "expected_leaf_kinematics_mode": str(record["leaf_kinematics_mode"]),
                "expected_track_fit_policy": str(record["track_fit_policy"]),
                "expected_charge_conjugate_normalization": bool(record["charge_conjugate_normalization"]),
                "uid_callback": register_uid,
            }
            if contract_v2:
                validation_kwargs.update(
                    expected_model_feature_contract_hash=str(record["model_feature_contract_hash"]),
                    expected_record=record,
                )
            result = validate_shard(output, **validation_kwargs)
            completed += 1
            total_events += int(result["events"])
            produced_categories[str(record["physics_category"])] += int(result["events"])
            track_policies[str(record.get("track_fit_policy", "unknown"))] += int(result["events"])
            actual_leaf_modes.update(result.get("actual_leaf_mode_distribution", {}))
            b_roots.update(result.get("b_root_distribution", {}))
            node_counts.extend(result.get("node_counts", []))
            max_depths.extend(result.get("max_depths", []))
            klm_nodes += int(result.get("klm_nodes", 0))
            klm_associated_ecl += int(result.get("klm_associated_ecl", 0))
            incomplete_branches += int(result.get("incomplete_reconstructable_branches", 0))
            output_bytes += int(result.get("output_bytes", output.stat().st_size))
            shard_klm = dict(result.get("klm_diagnostics", {}))
            klm_feature_availability.update(
                shard_klm.pop("klm_feature_availability", {})
            )
            klm_diagnostics.update(
                {key: int(value) for key, value in shard_klm.items()}
            )
            klm_by_category[str(record["physics_category"])] += int(
                result.get("klm_nodes", 0)
            )
        for input_file, ranges in by_file.items():
            ranges.sort()
            for left, right in zip(ranges, ranges[1:]):
                if left[1] > right[0]:
                    raise ValueError(
                        f"overlapping source entry ranges for {input_file}: {left} and {right}"
                    )
        planned = sum(int(record["planned_events"]) for record in records)
        if missing:
            raise ValueError(f"production is incomplete; missing tasks: {missing}")
        if total_events != planned:
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
        "node_count_quantiles": _quantiles(node_counts),
        "maximum_depth_quantiles": _quantiles(max_depths),
        "output_bytes": output_bytes,
        "output_bytes_per_event": output_bytes / max(total_events, 1),
        "all_completion_markers_valid": True,
        "klm_training_scope": expected_config.get("klm_training_scope", "unresolved"),
        **expected_config,
    }


def production_status(manifest: Path) -> dict[str, Any]:
    """Classify every task without treating incompleteness as an exception."""

    records = read_manifest(manifest)
    counts: Counter[str] = Counter()
    tasks: list[dict[str, Any]] = []
    for record in records:
        contract_v2 = record.get("manifest_schema_version") == MANIFEST_SCHEMA_VERSION
        kwargs: dict[str, Any]
        if contract_v2:
            valid_hash = str(record.get("task_record_hash", "")) == task_record_hash(record)
            if not valid_hash:
                result = {
                    "classification": "PROVENANCE_MISMATCH",
                    "error": "manifest task_record_hash mismatch",
                }
            else:
                result = classify_shard(Path(str(record["output_file"])), **_validation_kwargs(record))
        else:
            kwargs = {
                "expected_events": int(record["planned_events"]),
                "expected_schema": str(record["schema_version"]),
                "expected_feature_spec_hash": str(record["feature_spec_hash"]),
                "expected_pid_vocabulary_version": str(record["pid_vocabulary_version"]),
                "expected_leaf_kinematics_mode": str(record["leaf_kinematics_mode"]),
                "expected_track_fit_policy": str(record["track_fit_policy"]),
                "expected_charge_conjugate_normalization": bool(record["charge_conjugate_normalization"]),
            }
            result = classify_shard(Path(str(record["output_file"])), **kwargs)
        classification = str(result["classification"])
        counts[classification] += 1
        tasks.append({"task_id": int(record["task_id"]), **result})
    return {
        "manifest": str(manifest),
        "campaign_id": records[0].get("campaign_id", "legacy"),
        "campaign_config_digest": records[0].get("campaign_config_digest", "legacy"),
        "tasks": len(records),
        "classifications": dict(sorted(counts.items())),
        "complete": counts["COMPLETE_VALID"],
        "missing_or_invalid": len(records) - counts["COMPLETE_VALID"],
        "task_status": tasks,
    }


def list_missing_tasks(manifest: Path) -> list[int]:
    return [
        int(item["task_id"])
        for item in production_status(manifest)["task_status"]
        if item["classification"] != "COMPLETE_VALID"
    ]


def render_resubmit(
    manifest: Path,
    *,
    repo_root: Path,
    output: Path | None = None,
    submit: bool = False,
) -> str:
    """Render a targeted Condor queue; submit only after explicit opt-in."""

    task_ids = list_missing_tasks(manifest)
    queue_values = ",".join(str(value) for value in task_ids)
    campaign_root = manifest.resolve().parent.parent
    log_dir = campaign_root / "logs" / "condor"
    text = "\n".join(
        [
            "universe = vanilla",
            f"executable = {repo_root / 'scripts/condor/submit_mdst_production_10m.sh'}",
            "arguments = --worker $(task_id)",
            f"initialdir = {repo_root}",
            "getenv = True",
            "should_transfer_files = NO",
            (
                f'environment = "MANIFEST={manifest.resolve()} '
                f'REPO_ROOT={repo_root.resolve()} OUTPUT_ROOT={campaign_root}"'
            ),
            f"output = {log_dir}/mdst-$(ClusterId).$(task_id).out",
            f"error = {log_dir}/mdst-$(ClusterId).$(task_id).err",
            f"log = {log_dir}/mdst-$(ClusterId).log",
            (f"queue task_id in ({queue_values})" if task_ids else "# no missing tasks"),
            "",
        ]
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if submit:
        if output is None:
            raise ValueError("--submit requires --output so the reviewed description is stable")
        subprocess.run(("condor_submit", str(output)), check=True)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create an exact JSONL production manifest")
    plan.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--target-events", type=int, default=None)
    plan.add_argument("--events-per-task", type=int, default=DEFAULT_EVENTS_PER_TASK)
    plan.add_argument(
        "--campaign-profile",
        choices=("pilot", "canary", "production"),
        default="production",
        help="Defaults to 5k, 100k, or 10M planned events; --target-events overrides.",
    )
    plan.add_argument("--campaign-id", default=None)
    plan.add_argument("--replan", action="store_true")
    plan.add_argument("--checksum-inputs", action="store_true")
    plan.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
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
    plan.add_argument(
        "--klm-training-scope",
        choices=("included", "excluded_by_policy", "unresolved"),
        default="unresolved",
    )
    plan.add_argument(
        "--production-readiness-report",
        type=Path,
        default=None,
        help="Representative canary readiness JSON to bind before any 10M worker can run.",
    )

    task = subparsers.add_parser("run-task", help="Execute one manifest task")
    task.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    task.add_argument("--task-id", type=int, default=None)
    task.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    task.add_argument("--destructive-overwrite", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate all produced shards globally")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    status = subparsers.add_parser("status", help="Classify every planned task")
    status.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    missing = subparsers.add_parser("list-missing", help="List task IDs not COMPLETE_VALID")
    missing.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    resubmit = subparsers.add_parser(
        "render-resubmit", help="Render a targeted HTCondor description"
    )
    resubmit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    resubmit.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    resubmit.add_argument("--output", type=Path, default=None)
    resubmit.add_argument("--submit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        input_files = discover_input_files(args.input_root)
        identity = source_identity(args.repo_root.resolve())
        profile_events = {"pilot": 5_000, "canary": 100_000, "production": DEFAULT_TARGET_EVENTS}
        target_events = int(args.target_events or profile_events[args.campaign_profile])
        events_per_task = int(args.events_per_task)
        if args.campaign_profile == "pilot":
            events_per_task = min(events_per_task, max(250, target_events // 6))
        stage = {
            "pilot": "pilot",
            "canary": "canary_100k",
            "production": "production_10m",
        }[args.campaign_profile]
        readiness_digest = None
        if args.production_readiness_report is not None:
            readiness_digest = _sha256_path(args.production_readiness_report)
        records, category_events = build_manifest_records(
            input_files,
            output_root=args.output_root,
            target_events=target_events,
            events_per_task=events_per_task,
            schema_version=args.schema_version,
            charge_conjugate_normalization=args.charge_conjugate_normalization,
            leaf_kinematics_mode=args.leaf_kinematics_mode,
            track_fit_policy=args.track_fit_policy,
            **identity,
            campaign_id=args.campaign_id,
            event_buffer_size=args.event_buffer_size,
            row_group_size=args.row_group_size,
            checksum_inputs=args.checksum_inputs,
            campaign_stage=stage,
            klm_training_scope=args.klm_training_scope,
            production_readiness_report_sha256=readiness_digest,
        )
        summary = write_manifest(
            records,
            manifest=args.manifest,
            input_root=args.input_root,
            output_root=args.output_root,
            target_events=target_events,
            events_per_task=events_per_task,
            category_events=category_events,
            overwrite=args.replan,
        )
        summary["campaign_profile"] = args.campaign_profile
        summary["worker_source_eligible"] = identity["source_state"] == "clean"
        summary["klm_training_scope"] = args.klm_training_scope
        summary["launch_gate"] = (
            "NO_GO"
            if stage == "production_10m"
            and (args.klm_training_scope == "unresolved" or not readiness_digest)
            else "RENDERED_NOT_SUBMITTED"
        )
        _write_json_atomic(args.manifest.with_suffix(".summary.json"), summary)
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "validate":
        print(json.dumps(validate_production_manifest(args.manifest), sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(production_status(args.manifest), sort_keys=True))
        return 0
    if args.command == "list-missing":
        for task_id in list_missing_tasks(args.manifest):
            print(task_id)
        return 0
    if args.command == "render-resubmit":
        print(
            render_resubmit(
                args.manifest,
                repo_root=args.repo_root.resolve(),
                output=args.output,
                submit=args.submit,
            ),
            end="",
        )
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
        destructive_overwrite=args.destructive_overwrite,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
