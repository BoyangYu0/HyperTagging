#!/usr/bin/env python3
"""Write a hashed terminal receipt for a read-only pretraining validation job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--telemetry-summary", type=Path, required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args()
    candidates = {
        "contract": args.contract,
        "stage_log": args.stage_log,
        "result": args.result,
        "gpu_preflight": args.gpu_preflight,
        "gpu_telemetry_summary": args.telemetry_summary,
    }
    artifacts = {
        name: _artifact(path) for name, path in candidates.items() if path.is_file()
    }
    result = (
        json.loads(args.result.read_text(encoding="utf-8"))
        if args.result.is_file()
        else {}
    )
    success = (
        args.exit_status == 0
        and result.get("status") == "completed"
        and result.get("optimizer_steps") == 0
        and result.get("checkpoint", {}).get("unchanged") is True
    )
    receipt = {
        "receipt_version": "hypertagging-pretraining-validation-attempt-v1",
        "status": "completed" if success else "failed_or_nonterminal",
        "exit_status": args.exit_status,
        "started_at": args.started_at,
        "completed_at": args.completed_at,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
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
