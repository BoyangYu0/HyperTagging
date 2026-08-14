#!/usr/bin/env python3
"""Write an atomic, canonically hashed Slurm attempt receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--wrapper-status", type=Path, required=True)
    parser.add_argument("--batch-exit-status", type=int, required=True)
    parser.add_argument("--terminal-stage", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args()

    wrapper: dict[str, object] | None = None
    if args.wrapper_status.is_file():
        wrapper = json.loads(args.wrapper_status.read_text(encoding="utf-8"))

    artifacts: dict[str, dict[str, object]] = {}
    candidates = {
        "job_contract": args.contract,
        "stage_log": args.stage_log,
        "wrapper_status": args.wrapper_status,
        "allocation": args.attempt_root / "allocation.txt",
        "gpu_query": args.attempt_root / "gpu-query.csv",
        "gpu_compute_apps": args.attempt_root / "gpu-compute-apps.csv",
        "gpu_pmon": args.attempt_root / "gpu-pmon.txt",
        "gpu_preflight": args.attempt_root / "gpu-preflight.json",
        "gpu_telemetry": args.attempt_root / "gpu-telemetry.jsonl",
        "gpu_telemetry_summary": args.attempt_root / "gpu-telemetry-summary.json",
    }
    if args.run_root is not None:
        candidates.update(
            {
                "checkpoint": args.run_root / "checkpoint.pt",
                "metrics": args.run_root / "metrics.jsonl",
                "signal_checkpoint": args.run_root / "signal-checkpoint.pt",
            }
        )
    for name, path in candidates.items():
        if path.is_file():
            artifacts[name] = _artifact(path)

    trainer_status = wrapper.get("trainer_status") if wrapper is not None else None
    action = wrapper.get("action") if wrapper is not None else "trainer_not_started"
    telemetry: dict[str, object] | None = None
    telemetry_summary = args.attempt_root / "gpu-telemetry-summary.json"
    if telemetry_summary.is_file():
        telemetry_payload = json.loads(telemetry_summary.read_text(encoding="utf-8"))
        telemetry = {
            "telemetry_version": telemetry_payload.get("telemetry_version"),
            "status": telemetry_payload.get("status"),
            "started_at": telemetry_payload.get("started_at"),
            "completed_at": telemetry_payload.get("completed_at"),
            "interval_seconds": telemetry_payload.get("interval_seconds"),
            "sample_count": telemetry_payload.get("sample_count"),
            "peak_memory_used_mib": telemetry_payload.get("peak_memory_used_mib"),
            "peak_gpu_utilization_percent": telemetry_payload.get(
                "peak_gpu_utilization_percent"
            ),
            "peak_temperature_c": telemetry_payload.get("peak_temperature_c"),
            "error": telemetry_payload.get("error"),
        }
    telemetry_complete = bool(
        telemetry
        and telemetry.get("status") == "completed"
        and int(telemetry.get("sample_count", 0)) > 0
    )
    success = (
        args.batch_exit_status == 0
        and trainer_status == 0
        and action == "trainer_exit"
        and args.terminal_stage == "trainer_complete"
        and telemetry_complete
    )
    receipt: dict[str, object] = {
        "receipt_version": "hypertagging-slurm-attempt-v2",
        "status": "completed" if success else "failed_or_nonterminal",
        "terminal_stage": args.terminal_stage,
        "batch_exit_status": args.batch_exit_status,
        "trainer_status": trainer_status,
        "wrapper": wrapper,
        "started_at": args.started_at,
        "completed_at": args.completed_at,
        "gpu_telemetry": telemetry,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "artifacts": artifacts,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
