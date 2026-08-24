#!/usr/bin/env python3
"""Write a hashed terminal receipt for a reconstruction Slurm attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--wrapper-status", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-exit-status", type=int, required=True)
    parser.add_argument("--terminal-stage", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    wrapper = (
        json.loads(args.wrapper_status.read_text(encoding="utf-8"))
        if args.wrapper_status.is_file()
        else {}
    )
    result_path = args.run_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    telemetry_path = args.attempt_root / "gpu-telemetry.jsonl"
    telemetry_summary_path = args.attempt_root / "gpu-telemetry-summary.json"
    telemetry = (
        json.loads(telemetry_summary_path.read_text(encoding="utf-8"))
        if telemetry_summary_path.is_file()
        else {}
    )
    config = contract.get("config", {})
    healthy_result = (
        isinstance(result, dict)
        and result.get("status") == "completed"
        and result.get("optimizer_steps") == config.get("max_steps")
        and result.get("source_checkpoint", {}).get("unchanged") is True
        and result.get("data", {}).get("split_counts", {}).get("test", 1) == 0
        and result.get("first_20_optimizer_steps", {}).get("passed") is True
        and finite(result)
    )
    terminal_success = (
        args.batch_exit_status == 0
        and args.terminal_stage == "trainer_complete"
        and wrapper.get("action") == "trainer_exit"
        and wrapper.get("wrapper_status") == 0
        and healthy_result
        and telemetry.get("status") == "completed"
        and int(telemetry.get("sample_count", 0)) > 0
    )
    if contract.get("mode") == "calibration" and result is not None:
        terminal_success = terminal_success and float(result.get("elapsed_seconds", 1e99)) <= int(
            contract.get("max_wall_seconds", 900)
        )

    candidates = {
        "job_contract": args.contract,
        "stage_log": args.stage_log,
        "wrapper_status": args.wrapper_status,
        "allocation": args.attempt_root / "allocation.txt",
        "environment": args.attempt_root / "environment.txt",
        "nvidia_smi": args.attempt_root / "nvidia-smi.txt",
        "gpu_query": args.attempt_root / "gpu-query.csv",
        "gpu_preflight": args.attempt_root / "gpu-preflight.json",
        "gpu_telemetry": telemetry_path,
        "gpu_telemetry_summary": telemetry_summary_path,
    }
    if result_path.is_file():
        candidates["result"] = result_path
    training_root = args.run_root / "training"
    for name, path in (
        ("metrics", training_root / "metrics.jsonl"),
        ("checkpoint", training_root / "checkpoint.pt"),
        ("latest", training_root / "latest.pt"),
        ("signal_checkpoint", training_root / "signal-checkpoint.pt"),
        ("split_manifest", training_root / "split_manifest.json"),
    ):
        candidates[name] = path
    artifacts = {name: artifact(path) for name, path in candidates.items() if path.is_file()}
    receipt: dict[str, Any] = {
        "receipt_version": "hypertagging-reconstruction-fullscale-attempt-v1",
        "status": "completed" if terminal_success else "failed_or_nonterminal",
        "terminal_stage": args.terminal_stage,
        "batch_exit_status": args.batch_exit_status,
        "wrapper": wrapper,
        "mode": contract.get("mode"),
        "experiment": contract.get("experiment"),
        "contract_sha256": contract.get("contract_sha256"),
        "source_checkpoint": {
            "path": contract.get("checkpoint"),
            "step": contract.get("checkpoint_step"),
            "sha256": contract.get("checkpoint_sha256"),
            "unchanged": bool(result and result.get("source_checkpoint", {}).get("unchanged")),
        },
        "precision": {
            "gres": contract.get("gres"),
            "amp_dtype": config.get("amp_dtype"),
            "grad_scaler_enabled": config.get("grad_scaler_enabled"),
        },
        "first_20_optimizer_steps": (
            result.get("first_20_optimizer_steps") if isinstance(result, dict) else None
        ),
        "metrics": result.get("metrics") if isinstance(result, dict) else None,
        "data": result.get("data") if isinstance(result, dict) else None,
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
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
