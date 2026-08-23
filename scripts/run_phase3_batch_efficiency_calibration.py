#!/usr/bin/env python
"""Run one bounded, train-role-only phase-3 calibration in an allocation.

This command never calls a scheduler or submits a job.  The later operator
invokes four distinct instances, one per immutable matrix tuple.  The shared
coordination registry admits at most four active instances and marks every
instance non-production until its self-hashed receipt is terminal healthy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.phase3_parallel_study import (  # noqa: E402
    CHECKPOINT_SHA256,
    OBJECTIVE_DOMINANCE_LIMIT,
    RECEIPT_VERSION,
    calibration_slot,
    canonical_hash,
    entry_by_id,
    file_sha256,
    load_study_plan,
    resolve_plan_path,
)
from hypertagging.training.presentation_progress import (  # noqa: E402
    PHASE3_RESUME_PRESENTATIONS,
    VIRTUAL_STEP_PRESENTATIONS,
    validate_batch_profile,
)


DEFAULT_PLAN = ROOT / "configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json"
MAX_PILOT_STEPS = 256
MAX_PILOT_SECONDS = 900
FORBIDDEN_TOKENS = (
    "sealed",
    "stress",
    "validation",
    "srun",
    "salloc",
    "sbatch",
    "scancel",
    "requeue",
    "--device cpu",
    "device=cpu",
    "--scientific-mode",
)
# The shared receipt validator enforces the exact fail-closed message:
# objective dominance ratio exceeded fail-closed limit 20.0


def _assert_command_safe(command: list[str]) -> None:
    joined = " ".join(command).lower()
    for token in FORBIDDEN_TOKENS:
        if token in joined:
            raise RuntimeError(f"calibration command contains forbidden token {token!r}")


def _require_fresh_exact_allocation(gres: str) -> dict[str, Any]:
    observed = (
        os.environ.get("HT_PHASE3_ALLOCATION_GRES")
        or os.environ.get("SLURM_JOB_GRES")
        or os.environ.get("SLURM_TRES_PER_NODE")
        or ""
    )
    if observed != gres:
        raise RuntimeError(
            f"fresh exact-GRES preflight requires {gres!r}; observed {observed!r}"
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
        "exact_gres": gres,
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
    """Run only a bounded synthetic CUDA allocation/throughput probe."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU env
        raise RuntimeError("GPU calibration requires torch in the frozen environment") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("calibration requires exactly one CUDA device")
    device = torch.device("cuda")
    if profile_name == "h100nvl" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("H100 profile requires validated CUDA BF16 support")
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


def _copy_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file() or file_sha256(source) != CHECKPOINT_SHA256:
        raise RuntimeError("immutable source checkpoint hash/path gate failed")
    if destination.exists():
        raise RuntimeError("calibration checkpoint copy already exists; refusing overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_hash = file_sha256(destination)
    if copied_hash != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint copy hash mismatch")
    return {
        "source_path": str(source.relative_to(ROOT)),
        "copy_path": str(destination.relative_to(ROOT)),
        "source_sha256": CHECKPOINT_SHA256,
        "copy_sha256": copied_hash,
        "source_unchanged": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _assert_new_paths(paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("calibration output, attempt, metrics, copy, or receipt paths collide")
    if any(path.exists() for path in resolved):
        raise RuntimeError("calibration output/attempt path exists; refusing overwrite")


def _assert_cli_matches_plan(args: argparse.Namespace, entry: dict[str, Any], plan: dict[str, Any]) -> None:
    policy = entry["precision_policy"]
    expected_scaler = "enabled" if policy["grad_scaler_enabled"] else "disabled"
    checks = {
        "profile": entry["profile"],
        "batch_size": entry["batch_size"],
        "amp_dtype": policy["amp_dtype"],
        "grad_scaler": expected_scaler,
        "owner": plan["owner"],
    }
    for key, expected in checks.items():
        if getattr(args, key) != expected:
            raise RuntimeError(f"{key} does not match immutable calibration tuple")
    path_checks = {
        "checkpoint_copy": entry["checkpoint_copy_path"],
        "output_root": entry["output_root"],
        "attempt_root": entry["attempt_root"],
        "pilot_metrics": entry["metrics_path"],
        "receipt": entry["receipt_path"],
    }
    for argument, configured in path_checks.items():
        if Path(getattr(args, argument)).resolve() != resolve_plan_path(configured, root=ROOT):
            raise RuntimeError(f"{argument} does not match immutable calibration tuple")
    if args.queue_delay_seconds < 0 or not float(args.queue_delay_seconds) < float("inf"):
        raise RuntimeError("queue delay must be finite and non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--profile", choices=("h100nvl", "v100"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), required=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--grad-scaler", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--checkpoint-copy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--pilot-metrics", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--queue-delay-seconds", type=float, required=True)
    parser.add_argument("--execute-pilot", action="store_true")
    parser.add_argument(
        "--pilot-command",
        nargs=argparse.REMAINDER,
        help="bounded train-role-only pilot command, executed only inside this allocation",
    )
    args = parser.parse_args(argv)
    plan = load_study_plan(args.study_plan, root=ROOT)
    entry = entry_by_id(plan, args.calibration_id)
    _assert_cli_matches_plan(args, entry, plan)
    command = list(args.pilot_command or [])
    if command and command[0] == "--":
        command = command[1:]
    if args.execute_pilot and not command:
        raise RuntimeError("--execute-pilot requires --pilot-command")
    if args.execute_pilot:
        _assert_command_safe(command)
    elif command:
        raise RuntimeError("a pilot command requires --execute-pilot")
    if args.pilot_metrics.resolve().parent != args.attempt_root.resolve():
        raise RuntimeError("pilot metrics must be owned by the configured attempt root")
    paths = [
        args.output_root,
        args.attempt_root,
        args.checkpoint_copy,
        args.pilot_metrics,
        args.receipt,
    ]
    _assert_new_paths(paths)
    validate_batch_profile(
        args.batch_size,
        total_presentations=1_730_048,
        milestone_presentations=tuple(
            value * VIRTUAL_STEP_PRESENTATIONS
            for value in (13516, 27032, 40548, 54064, 67580, 81096, 94612, 108128)
        ),
    )
    with calibration_slot(plan, args.calibration_id, owner=args.owner, root=ROOT):
        preflight = _require_fresh_exact_allocation(entry["exact_gres"])
        _assert_profile_device(args.profile, preflight)
        args.output_root.mkdir(parents=True, exist_ok=False)
        args.attempt_root.mkdir(parents=True, exist_ok=False)
        fixture = _run_fixture_probe(args.batch_size, args.profile)
        checkpoint_source = resolve_plan_path(entry["source_checkpoint_path"], root=ROOT)
        checkpoint = _copy_checkpoint(checkpoint_source, args.checkpoint_copy)
        records: list[dict[str, Any]] = []
        throughput = None
        objective_ratio = None
        if args.execute_pilot:
            env = dict(os.environ)
            env.update(
                {
                    "HT_PHASE3_ROLE": "train",
                    "HT_PHASE3_NONPRODUCTION": "1",
                    "HT_PHASE3_VALIDATION_ACCESS": "forbidden",
                    "HT_PHASE3_SEALED_TEST_ACCESS": "forbidden",
                    "HT_PHASE3_STRESS_ACCESS": "forbidden",
                    "HT_PHASE3_CALIBRATION_ID": args.calibration_id,
                    "HT_PHASE3_OWNER": args.owner,
                    "HT_PHASE3_CHECKPOINT_COPY": str(args.checkpoint_copy),
                    "HT_PHASE3_PILOT_MAX_STEPS": str(MAX_PILOT_STEPS),
                    "HT_PHASE3_PILOT_MAX_SECONDS": str(MAX_PILOT_SECONDS),
                }
            )
            try:
                completed = subprocess.run(
                    command,
                    env=env,
                    cwd=ROOT,
                    check=False,
                    timeout=MAX_PILOT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("bounded train-role stability pilot exceeded 900 seconds") from error
            if completed.returncode != 0:
                raise RuntimeError(f"train-role stability pilot failed: {completed.returncode}")
            from hypertagging.training.phase3_parallel_study import _read_metrics

            records, throughput, objective_ratio = _read_metrics(args.pilot_metrics)
        if file_sha256(checkpoint_source) != CHECKPOINT_SHA256:
            raise RuntimeError("source checkpoint changed during calibration")
        if not args.execute_pilot:
            terminal_state = "incomplete"
        else:
            terminal_state = "healthy"
        receipt: dict[str, Any] = {
            "artifact_version": RECEIPT_VERSION,
            "calibration_id": args.calibration_id,
            "tuple_sha256": entry["tuple_sha256"],
            "owner": args.owner,
            "hypothesis_id": entry["hypothesis_id"],
            "hypothesis": entry["hypothesis"],
            "profile": {
                "name": entry["profile"],
                "exact_gres": entry["exact_gres"],
                "batch_size": entry["batch_size"],
                "precision_policy": entry["precision_policy"],
            },
            "allocation_preflight": preflight,
            "fixture_probe": fixture,
            "checkpoint_copy": checkpoint,
            "output_root": entry["output_root"],
            "attempt_root": entry["attempt_root"],
            "queue_delay_seconds": args.queue_delay_seconds,
            "pilot": {
                "executed": bool(args.execute_pilot),
                "role": "train",
                "validation_access": "forbidden",
                "sealed_test_access": "forbidden",
                "stress_access": "forbidden",
                "metrics_path": entry["metrics_path"],
                "record_count": len(records),
                "max_steps": MAX_PILOT_STEPS,
                "max_seconds": MAX_PILOT_SECONDS,
                "expected_learning_rate": 0.0005,
                "command_sha256": canonical_hash(command) if command else None,
                "throughput_events_per_second": throughput,
                "objective_weighted_dominance_ratio": objective_ratio,
            },
            "scientific_contract": {
                "resume_presentations": PHASE3_RESUME_PRESENTATIONS,
                "remaining_presentations": 865_024,
                "objective_dominance_limit": OBJECTIVE_DOMINANCE_LIMIT,
                "submission_performed": False,
                "production_submission_performed": False,
            },
            "production_allowed": False,
            "production_submission_performed": False,
            "calibration_complete": bool(args.execute_pilot and records),
            "terminal_state": terminal_state,
            "created_at_unix": time.time(),
        }
        receipt["receipt_sha256"] = canonical_hash(receipt)
        _write_json(args.receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
