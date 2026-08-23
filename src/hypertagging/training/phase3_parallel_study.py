"""Fail-closed coordination and receipt validation for the phase-3 study.

This module contains no Slurm calls and no scientific-training entry points.
It only validates the immutable study plan, coordinates bounded calibration
slots, and checks self-hashed calibration evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
STUDY_VERSION = "ht-pretraining-1m-phase3-parallel-study-authorization-v1"
RECEIPT_VERSION = "ht-pretraining-1m-phase3-gpu-calibration-receipt-v2"
AGGREGATION_VERSION = "ht-pretraining-1m-phase3-calibration-receipt-aggregation-v1"
MAX_CONCURRENT_CALIBRATION_JOBS = 4
OWNER = "sole-authorized-phase3-follow-up-programming-operator"
TOTAL_PRESENTATIONS = 1_730_048
RESUME_PRESENTATIONS = 865_024
REMAINING_PRESENTATIONS = 865_024
OBJECTIVE_DOMINANCE_LIMIT = 20.0
CHECKPOINT_SHA256 = (
    "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
)
EXPECTED_GRES = {
    "h100nvl": "gpu:h100nvl:1",
    "v100": "gpu:v100:1",
}
EXPECTED_PRECISION = {
    "h100nvl": {
        "amp_dtype": "bfloat16",
        "grad_scaler_enabled": False,
        "bf16_policy": "require_cuda_bf16_supported",
    },
    "v100": {
        "amp_dtype": "float16",
        "grad_scaler_enabled": True,
        "bf16_policy": "forbid_bf16_use_float16_gradscaler",
    },
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite")
    return number


def resolve_plan_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def tuple_key(entry: dict[str, Any]) -> str:
    policy = entry.get("precision_policy", {})
    return ":".join(
        (
            str(entry.get("profile", "")),
            str(entry.get("batch_size", "")),
            str(policy.get("amp_dtype", "")),
            str(policy.get("grad_scaler_enabled", "")),
        )
    )


def entry_by_id(plan: dict[str, Any], calibration_id: str) -> dict[str, Any]:
    matches = [
        entry
        for entry in plan.get("calibration_matrix", [])
        if entry.get("calibration_id") == calibration_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"calibration ID is not unique in the study plan: {calibration_id}")
    return matches[0]


def validate_study_plan(plan: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    if plan.get("artifact_version") != STUDY_VERSION:
        raise RuntimeError("unsupported phase-3 parallel-study plan version")
    if plan.get("owner") != OWNER or not isinstance(plan.get("owner"), str):
        raise RuntimeError("parallel-study ownership is missing or ambiguous")
    if plan.get("max_concurrent_calibration_jobs") != MAX_CONCURRENT_CALIBRATION_JOBS:
        raise RuntimeError("parallel-study max concurrency is not exactly four")
    if plan.get("calibration_type") != "non_production_train_role_only":
        raise RuntimeError("calibration type is not the bounded non-production role")
    if plan.get("production_policy", {}).get("default_resume_count") != 1:
        raise RuntimeError("default production resume count is not exactly one")
    if plan.get("production_policy", {}).get("duplicate_contracts_forbidden") is not True:
        raise RuntimeError("duplicate production contracts are not forbidden")
    matrix = plan.get("calibration_matrix")
    if not isinstance(matrix, list) or len(matrix) != MAX_CONCURRENT_CALIBRATION_JOBS:
        raise RuntimeError("the configured calibration matrix must contain exactly four tuples")

    ids: set[str] = set()
    tuple_keys: set[str] = set()
    tuple_hashes: set[str] = set()
    roots: list[Path] = []
    receipt_paths: set[Path] = set()
    copy_paths: set[Path] = set()
    for entry in matrix:
        if not isinstance(entry, dict):
            raise RuntimeError("calibration matrix contains a non-object entry")
        calibration_id = str(entry.get("calibration_id", ""))
        if not calibration_id or calibration_id in ids:
            raise RuntimeError("duplicate or missing immutable calibration ID")
        ids.add(calibration_id)
        key = tuple_key(entry)
        if not all(key.split(":")) or key in tuple_keys:
            raise RuntimeError("duplicate or ambiguous calibration profile tuple")
        tuple_keys.add(key)
        tuple_body = dict(entry)
        stored_tuple_hash = tuple_body.pop("tuple_sha256", None)
        if not isinstance(stored_tuple_hash, str) or stored_tuple_hash != canonical_hash(tuple_body):
            raise RuntimeError(f"calibration tuple hash mismatch: {calibration_id}")
        if stored_tuple_hash in tuple_hashes:
            raise RuntimeError("duplicate calibration tuple hash")
        tuple_hashes.add(stored_tuple_hash)
        profile = str(entry.get("profile", ""))
        batch_size = entry.get("batch_size")
        if profile not in EXPECTED_GRES or batch_size not in (32, 64):
            raise RuntimeError("calibration matrix contains an unsupported profile or batch")
        if entry.get("exact_gres") != EXPECTED_GRES[profile]:
            raise RuntimeError(f"calibration GRES is not exact for {calibration_id}")
        if entry.get("precision_policy") != EXPECTED_PRECISION[profile]:
            raise RuntimeError(f"precision/scaler policy mismatch for {calibration_id}")
        if not str(entry.get("hypothesis_id", "")) or not str(entry.get("hypothesis", "")):
            raise RuntimeError(f"hypothesis ownership is ambiguous for {calibration_id}")
        for field in ("checkpoint_copy_path", "output_root", "attempt_root", "metrics_path", "receipt_path"):
            if not entry.get(field):
                raise RuntimeError(f"{field} is missing for {calibration_id}")
        output_root = resolve_plan_path(entry["output_root"], root=root)
        attempt_root = resolve_plan_path(entry["attempt_root"], root=root)
        receipt_path = resolve_plan_path(entry["receipt_path"], root=root)
        copy_path = resolve_plan_path(entry["checkpoint_copy_path"], root=root)
        metric_path = resolve_plan_path(entry["metrics_path"], root=root)
        if _path_overlap(output_root, attempt_root):
            raise RuntimeError(f"output and attempt roots overlap for {calibration_id}")
        if receipt_path.parent != attempt_root or metric_path.parent != attempt_root:
            raise RuntimeError(f"receipt/metrics ownership is ambiguous for {calibration_id}")
        for other in roots:
            if _path_overlap(output_root, other) or _path_overlap(attempt_root, other):
                raise RuntimeError("calibration jobs share or nest output/attempt roots")
        roots.extend((output_root, attempt_root))
        if receipt_path in receipt_paths or metric_path in receipt_paths:
            raise RuntimeError("calibration jobs share receipt or metrics paths")
        receipt_paths.add(receipt_path)
        receipt_paths.add(metric_path)
        if copy_path in copy_paths:
            raise RuntimeError("calibration jobs share checkpoint-copy paths")
        copy_paths.add(copy_path)
        if entry.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError("calibration source checkpoint hash binding changed")

    expected_keys = {
        f"h100nvl:32:bfloat16:False",
        f"h100nvl:64:bfloat16:False",
        f"v100:32:float16:True",
        f"v100:64:float16:True",
    }
    if tuple_keys != expected_keys:
        raise RuntimeError("configured calibration matrix is not the exact H100/V100 32/64 set")
    if plan.get("required_receipt_policy", {}).get("mode") != "exact_configured_set":
        raise RuntimeError("receipt policy is not fail-closed exact-set aggregation")
    stored_plan_hash = plan.get("plan_sha256")
    if stored_plan_hash:
        body = dict(plan)
        body.pop("plan_sha256", None)
        if stored_plan_hash != canonical_hash(body):
            raise RuntimeError("parallel-study plan hash mismatch")
    return plan


def load_study_plan(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict):
        raise RuntimeError("parallel-study plan must be a JSON object")
    validate_study_plan(plan, root=root)
    return plan


def _coordination_paths(plan: dict[str, Any], *, root: Path = ROOT) -> tuple[Path, Path]:
    coordination = plan.get("coordination", {})
    coordination_root = resolve_plan_path(coordination.get("root", ""), root=root)
    if not coordination_root:
        raise RuntimeError("parallel-study coordination root is missing")
    return coordination_root / "registry.lock", coordination_root / "active.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid coordination registry: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"coordination registry is not an object: {path}")
    return value


@contextmanager
def _registry_lock(plan: dict[str, Any], *, root: Path = ROOT) -> Iterator[tuple[Path, Path]]:
    lock_path, state_path = _coordination_paths(plan, root=root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield lock_path, state_path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write_registry(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def calibration_slot(
    plan: dict[str, Any], calibration_id: str, *, owner: str, root: Path = ROOT
) -> Iterator[None]:
    entry = entry_by_id(plan, calibration_id)
    if owner != plan["owner"]:
        raise RuntimeError("calibration owner does not match the immutable study owner")
    with _registry_lock(plan, root=root) as (_lock_path, state_path):
        state = _read_json_object(state_path)
        active = state.get("active", {})
        if not isinstance(active, dict):
            raise RuntimeError("active calibration registry is malformed")
        if calibration_id in active:
            raise RuntimeError("duplicate active calibration ID")
        if len(active) >= MAX_CONCURRENT_CALIBRATION_JOBS:
            raise RuntimeError("maximum four concurrent calibration jobs exceeded")
        active[calibration_id] = {
            "calibration_id": calibration_id,
            "tuple_sha256": entry["tuple_sha256"],
            "owner": owner,
            "pid": os.getpid(),
            "started_at_unix": time.time(),
            "production_allowed": False,
        }
        _write_registry(state_path, {"artifact_version": STUDY_VERSION, "active": active})
    try:
        yield
    finally:
        with _registry_lock(plan, root=root) as (_lock_path, state_path):
            state = _read_json_object(state_path)
            active = state.get("active", {})
            if isinstance(active, dict):
                active.pop(calibration_id, None)
                if active:
                    _write_registry(
                        state_path, {"artifact_version": STUDY_VERSION, "active": active}
                    )
                elif state_path.exists():
                    state_path.unlink()


def active_calibrations(plan: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    with _registry_lock(plan, root=root) as (_lock_path, state_path):
        state = _read_json_object(state_path)
        active = state.get("active", {})
        if not isinstance(active, dict):
            raise RuntimeError("active calibration registry is malformed")
        return dict(active)


def assert_no_active_calibrations(plan: dict[str, Any], *, root: Path = ROOT) -> None:
    active = active_calibrations(plan, root=root)
    if active:
        raise RuntimeError("production action forbidden while calibration jobs are active")


def _read_metrics(path: Path) -> tuple[list[dict[str, Any]], float, float]:
    if not path.is_file():
        raise RuntimeError(f"pilot metrics file is missing: {path}")
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise RuntimeError("pilot metrics are not valid JSON lines") from error
    if not records or not all(isinstance(record, dict) for record in records):
        raise RuntimeError("pilot metrics are empty or malformed")
    throughputs: list[float] = []
    objective_ratios: list[float] = []
    objective_passes = 0
    for record in records:
        if record.get("split", "train") != "train":
            raise RuntimeError("calibration pilot emitted a non-train record")
        if any(str(key).lower().startswith("validation") for key in record):
            raise RuntimeError("calibration pilot must not access validation metrics")
        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise RuntimeError(f"pilot metric {key} is non-finite")
        for key in ("loss", "raw_gradient_norm", "learning_rate"):
            if key in record:
                _finite(record[key], f"pilot metric {key}")
        if "events_per_second" in record:
            throughput = _finite(record["events_per_second"], "train throughput")
            if throughput <= 0:
                raise RuntimeError("train throughput must be positive")
            throughputs.append(throughput)
        if "objective_preflight_pass" in record:
            if record["objective_preflight_pass"] is not True:
                raise RuntimeError("objective dominance preflight failed")
            objective_passes += 1
        if "objective_weighted_dominance_ratio" in record:
            ratio = _finite(
                record["objective_weighted_dominance_ratio"],
                "objective dominance ratio",
            )
            if ratio > OBJECTIVE_DOMINANCE_LIMIT:
                raise RuntimeError("objective dominance ratio exceeded fail-closed limit 20.0")
            objective_ratios.append(ratio)
    if not throughputs:
        raise RuntimeError("calibration metrics have no finite train throughput")
    if objective_passes == 0 or not objective_ratios:
        raise RuntimeError("calibration metrics lack a passing finite objective-dominance check")
    usable = throughputs[-min(3, len(throughputs)):]
    return records, sum(usable) / len(usable), max(objective_ratios)


def load_healthy_receipt(
    path: Path,
    plan: dict[str, Any],
    entry: dict[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[dict[str, Any], float, float]:
    expected_path = resolve_plan_path(entry["receipt_path"], root=root)
    if path.resolve() != expected_path:
        raise RuntimeError(f"receipt path does not match configured path: {path}")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load calibration receipt: {path}") from error
    if not isinstance(receipt, dict):
        raise RuntimeError("calibration receipt must be a JSON object")
    stored = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if stored != canonical_hash(body):
        raise RuntimeError(f"calibration receipt hash mismatch: {path}")
    if receipt.get("artifact_version") != RECEIPT_VERSION:
        raise RuntimeError("unsupported parallel calibration receipt version")
    for key, expected in (
        ("calibration_id", entry["calibration_id"]),
        ("tuple_sha256", entry["tuple_sha256"]),
        ("owner", plan["owner"]),
        ("hypothesis_id", entry["hypothesis_id"]),
        ("hypothesis", entry["hypothesis"]),
    ):
        if receipt.get(key) != expected:
            raise RuntimeError(f"calibration receipt binding mismatch for {key}")
    if receipt.get("terminal_state") != "healthy" or receipt.get("calibration_complete") is not True:
        raise RuntimeError("calibration receipt is not terminal healthy")
    if receipt.get("production_submission_performed") is not False:
        raise RuntimeError("calibration receipt reports a production submission")
    if receipt.get("production_allowed") is not False:
        raise RuntimeError("calibration receipt is not marked non-production")
    if (
        receipt.get("output_root") != entry["output_root"]
        or receipt.get("attempt_root") != entry["attempt_root"]
    ):
        raise RuntimeError("calibration output/attempt ownership binding changed")
    scientific = receipt.get("scientific_contract", {})
    if (
        scientific.get("resume_presentations") != RESUME_PRESENTATIONS
        or scientific.get("remaining_presentations") != REMAINING_PRESENTATIONS
        or scientific.get("objective_dominance_limit") != OBJECTIVE_DOMINANCE_LIMIT
        or scientific.get("submission_performed") is not False
    ):
        raise RuntimeError("calibration scientific contract binding changed")
    profile = receipt.get("profile", {})
    if (
        profile.get("name") != entry["profile"]
        or profile.get("exact_gres") != entry["exact_gres"]
        or profile.get("batch_size") != entry["batch_size"]
        or profile.get("precision_policy") != entry["precision_policy"]
    ):
        raise RuntimeError("calibration receipt profile/batch/precision binding changed")
    checkpoint = receipt.get("checkpoint_copy", {})
    if (
        checkpoint.get("source_path") != entry["source_checkpoint_path"]
        or
        checkpoint.get("source_sha256") != CHECKPOINT_SHA256
        or checkpoint.get("copy_sha256") != CHECKPOINT_SHA256
        or checkpoint.get("source_unchanged") is not True
        or resolve_plan_path(checkpoint.get("copy_path", ""), root=root)
        != resolve_plan_path(entry["checkpoint_copy_path"], root=root)
    ):
        raise RuntimeError("calibration checkpoint immutability binding failed")
    loadability = checkpoint.get("loadability", {})
    if (
        loadability.get("loadable") is not True
        or loadability.get("finite_tensors") is not True
        or int(loadability.get("tensor_count", 0)) <= 0
        or int(loadability.get("tensor_numel", 0)) <= 0
    ):
        raise RuntimeError("calibration checkpoint loadability/finiteness evidence is missing")
    pilot = receipt.get("pilot", {})
    if (
        pilot.get("executed") is not True
        or pilot.get("role") != "train"
        or pilot.get("validation_access") != "forbidden"
        or pilot.get("sealed_test_access") != "forbidden"
        or pilot.get("stress_access") != "forbidden"
        or pilot.get("max_steps") != 256
        or pilot.get("max_seconds") != 900
        or pilot.get("expected_learning_rate") != 0.0005
    ):
        raise RuntimeError("calibration pilot role isolation failed")
    metrics_path = resolve_plan_path(pilot.get("metrics_path", ""), root=root)
    if metrics_path != resolve_plan_path(entry["metrics_path"], root=root):
        raise RuntimeError("calibration metrics path is not configured")
    records, throughput, objective_ratio = _read_metrics(metrics_path)
    if pilot.get("record_count") != len(records):
        raise RuntimeError("calibration receipt record count is stale")
    if _finite(receipt.get("queue_delay_seconds"), "queue delay") < 0:
        raise RuntimeError("queue delay must be non-negative")
    fixture = receipt.get("fixture_probe", {})
    for key in ("fixture_batches_per_second", "fixture_peak_memory_mib"):
        if _finite(fixture.get(key), f"fixture {key}") <= 0:
            raise RuntimeError(f"fixture {key} must be positive")
    return receipt, throughput, objective_ratio


def claim_production_contract(
    plan: dict[str, Any],
    *,
    identity: str,
    output_path: Path,
    root: Path = ROOT,
) -> None:
    assert_no_active_calibrations(plan, root=root)
    if not identity or not output_path:
        raise RuntimeError("production contract identity/ownership is ambiguous")
    coordination = plan["coordination"]
    registry_path = resolve_plan_path(
        coordination.get("production_contract_registry", ""), root=root
    )
    with _registry_lock(plan, root=root):
        registry = _read_json_object(registry_path)
        contracts = registry.get("contracts", {})
        if not isinstance(contracts, dict):
            raise RuntimeError("production contract registry is malformed")
        if identity in contracts:
            raise RuntimeError("duplicate production contract identity")
        output_key = str(output_path.resolve())
        if any(value.get("output_path") == output_key for value in contracts.values() if isinstance(value, dict)):
            raise RuntimeError("duplicate production contract output path")
        contracts[identity] = {
            "identity": identity,
            "output_path": output_key,
            "owner": plan["owner"],
            "claimed_at_unix": time.time(),
        }
        _write_registry(registry_path, {"artifact_version": STUDY_VERSION, "contracts": contracts})


__all__ = [
    "AGGREGATION_VERSION",
    "CHECKPOINT_SHA256",
    "EXPECTED_GRES",
    "EXPECTED_PRECISION",
    "MAX_CONCURRENT_CALIBRATION_JOBS",
    "OBJECTIVE_DOMINANCE_LIMIT",
    "OWNER",
    "RECEIPT_VERSION",
    "ROOT",
    "STUDY_VERSION",
    "active_calibrations",
    "assert_no_active_calibrations",
    "calibration_slot",
    "canonical_hash",
    "claim_production_contract",
    "entry_by_id",
    "file_sha256",
    "load_healthy_receipt",
    "load_study_plan",
    "resolve_plan_path",
    "tuple_key",
    "validate_study_plan",
]
