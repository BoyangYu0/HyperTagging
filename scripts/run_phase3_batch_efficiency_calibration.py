#!/usr/bin/env python
"""Run one bounded, train-role-only phase-3 GPU calibration.

This command is intentionally interactive/in-allocation only.  It never calls
Slurm and never submits a job.  The later Spark operator must invoke it once
for H100 NVL and then once for V100, with separate output directories.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.device_profiles import get_device_profile  # noqa: E402
from hypertagging.training.presentation_progress import (  # noqa: E402
    PHASE3_RESUME_PRESENTATIONS,
    VIRTUAL_STEP_PRESENTATIONS,
    validate_batch_profile,
)

CHECKPOINT = ROOT / (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CHECKPOINT_SHA256 = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
CALIBRATION_VERSION = "ht-pretraining-1m-phase3-gpu-calibration-receipt-v1"
FORBIDDEN_TOKENS = ("sealed", "stress", "srun", "salloc", "sbatch", "scancel", "requeue")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_fresh_exact_allocation(gres: str) -> dict[str, Any]:
    expected = str(gres)
    observed = (
        os.environ.get("HT_PHASE3_ALLOCATION_GRES")
        or os.environ.get("SLURM_JOB_GRES")
        or os.environ.get("SLURM_TRES_PER_NODE")
        or ""
    )
    if observed != expected:
        raise RuntimeError(
            f"fresh exact-GRES preflight requires {expected!r}; observed {observed!r}"
        )
    token = os.environ.get("HT_PHASE3_FRESH_PREFLIGHT_TOKEN", "")
    if not token or token.lower() in {"stale", "reuse", "false"}:
        raise RuntimeError("fresh in-allocation preflight token is missing or stale")
    if os.environ.get("HT_PHASE3_CALIBRATION_ACTIVE") != "1":
        raise RuntimeError("calibration requires HT_PHASE3_CALIBRATION_ACTIVE=1")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for the fresh GPU preflight")
    query = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if query.returncode != 0 or not query.stdout.strip():
        raise RuntimeError("fresh nvidia-smi preflight failed")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("calibration requires exactly one visible GPU")
    return {
        "exact_gres": expected,
        "preflight_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "nvidia_smi_row": rows[0],
        "observed_at_unix": time.time(),
    }


def _assert_profile_device(profile_name: str, preflight: dict[str, Any]) -> None:
    row = str(preflight["nvidia_smi_row"]).lower()
    expected = "h100" if profile_name == "h100nvl" else "v100"
    if expected not in row:
        raise RuntimeError(f"preflight GPU name does not match {profile_name}: {row}")


def _run_fixture_probe(batch_size: int, profile_name: str) -> dict[str, Any]:
    """Bounded synthetic allocation/throughput probe; no scientific training."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU env
        raise RuntimeError("GPU calibration requires torch in the frozen environment") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("calibration requires exactly one CUDA device")
    device = torch.device("cuda")
    if profile_name == "h100nvl" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("H100 profile requires validated CUDA BF16 support")
    if profile_name == "v100" and torch.cuda.is_bf16_supported():
        # BF16 availability is not a reason to use it on the V100 profile.
        pass
    dtype = torch.bfloat16 if profile_name == "h100nvl" else torch.float16
    torch.cuda.reset_peak_memory_stats(device)
    width = 1024
    left = torch.randn((batch_size, width), device=device, dtype=dtype)
    right = torch.randn((width, width), device=device, dtype=dtype)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(4):
        left @ right
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)
    peak = int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
    del left, right
    return {
        "fixture": "synthetic_matmul_no_training_v1",
        "batch_size": batch_size,
        "dtype": str(dtype).removeprefix("torch."),
        "fixture_iterations": 4,
        "fixture_batches_per_second": 4.0 / elapsed,
        "fixture_peak_memory_mib": peak,
    }


def _assert_command_safe(command: list[str]) -> None:
    joined = " ".join(command).lower()
    for token in FORBIDDEN_TOKENS:
        if token in joined:
            raise RuntimeError(f"calibration command contains forbidden token {token!r}")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"pilot metrics file is missing: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError("pilot metrics file is empty")
    for record in records:
        if str(record.get("split", "train")) != "train":
            raise RuntimeError("train-role-only calibration emitted a non-train record")
        if any(str(key).lower().startswith("validation") for key in record):
            raise RuntimeError("calibration pilot must not tune against validation metrics")
        for key in ("loss", "raw_gradient_norm", "learning_rate"):
            if key in record and not isinstance(record[key], (int, float)):
                raise RuntimeError(f"pilot metric {key} is not numeric")
            if key in record and not (float("-inf") < float(record[key]) < float("inf")):
                raise RuntimeError(f"pilot metric {key} is non-finite")
        if "objective_preflight_pass" in record and not bool(record["objective_preflight_pass"]):
            raise RuntimeError("objective dominance preflight failed")
        if "objective_weighted_dominance_ratio" in record and float(
            record["objective_weighted_dominance_ratio"]
        ) > 20.0:
            raise RuntimeError("objective dominance ratio exceeded fail-closed limit 20.0")
    return records


def _copy_checkpoint(destination: Path) -> dict[str, Any]:
    if not CHECKPOINT.is_file() or _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("immutable source checkpoint hash/path gate failed")
    if destination.exists():
        raise RuntimeError("calibration checkpoint copy already exists; refusing overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT, destination)
    if _sha256(destination) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint copy hash mismatch")
    return {
        "source": str(CHECKPOINT.relative_to(ROOT)),
        "copy": str(destination.relative_to(ROOT)),
        "sha256": CHECKPOINT_SHA256,
        "source_unchanged": _sha256(CHECKPOINT) == CHECKPOINT_SHA256,
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another phase-3 calibration is active; run sequentially") from error
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("h100nvl", "v100"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-copy", type=Path, required=True)
    parser.add_argument("--pilot-metrics", type=Path, default=None)
    parser.add_argument("--expected-learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--pilot-command",
        nargs=argparse.REMAINDER,
        help="optional train-role-only pilot command; execute only in the later Spark session",
    )
    parser.add_argument("--execute-pilot", action="store_true")
    args = parser.parse_args(argv)
    profile = get_device_profile(args.profile)
    validate_batch_profile(
        args.batch_size,
        total_presentations=1_730_048,
        milestone_presentations=tuple(
            value * VIRTUAL_STEP_PRESENTATIONS
            for value in (13516, 27032, 40548, 54064, 67580, 81096, 94612, 108128)
        ),
    )
    if args.output.exists():
        raise RuntimeError("receipt output exists; immutable calibration receipts cannot be overwritten")
    with _exclusive_lock(args.output.parent / ".phase3-calibration.lock"):
        preflight = _require_fresh_exact_allocation(profile.gres)
        _assert_profile_device(args.profile, preflight)
        fixture = _run_fixture_probe(args.batch_size, args.profile)
        checkpoint = _copy_checkpoint(args.checkpoint_copy)
        command = list(args.pilot_command or [])
        if command and command[0] == "--":
            command = command[1:]
        if args.execute_pilot:
            if not command:
                raise RuntimeError("--execute-pilot requires --pilot-command")
            _assert_command_safe(command)
            if args.pilot_metrics is None:
                raise RuntimeError("--execute-pilot requires --pilot-metrics")
            env = dict(os.environ)
            env.update(
                {
                    "HT_PHASE3_ROLE": "train",
                    "HT_PHASE3_VALIDATION_ACCESS": "forbidden",
                    "HT_PHASE3_SEALED_TEST_ACCESS": "forbidden",
                    "HT_PHASE3_STRESS_ACCESS": "forbidden",
                    "HT_PHASE3_CHECKPOINT_COPY": str(args.checkpoint_copy),
                }
            )
            completed = subprocess.run(command, env=env, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"train-role stability pilot failed: {completed.returncode}")
            records = _read_metrics(args.pilot_metrics)
        else:
            records = []
        receipt: dict[str, Any] = {
            "artifact_version": CALIBRATION_VERSION,
            "profile": profile.contract(),
            "allocation_preflight": preflight,
            "fixture_probe": fixture,
            "checkpoint_copy": checkpoint,
            "pilot": {
                "executed": bool(args.execute_pilot),
                "role": "train",
                "validation_access": "forbidden",
                "sealed_test_access": "forbidden",
                "stress_access": "forbidden",
                "metrics_path": str(args.pilot_metrics) if args.pilot_metrics else None,
                "record_count": len(records),
                "expected_learning_rate": args.expected_learning_rate,
            },
            "scientific_contract": {
                "resume_presentations": PHASE3_RESUME_PRESENTATIONS,
                "remaining_presentations": 865_024,
                "objective_dominance_limit": 20.0,
                "submission_performed": False,
            },
            "calibration_complete": bool(args.execute_pilot and records),
            "created_at_unix": time.time(),
        }
        receipt["receipt_sha256"] = _canonical_hash(receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.partial")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
