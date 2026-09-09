#!/usr/bin/env python3
"""Write a canonical terminal receipt for one phase-40r1 Slurm attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--terminal-stage", required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    completed = args.exit_status == 0 and args.terminal_stage == "full_decay_complete"
    payload = {
        "receipt_version": "hypertagging-reconstruction-phase40r1-attempt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if completed else "failed",
        "study_id": contract["study_id"],
        "task_id": contract["task_id"],
        "arm_role": contract["arm_role"],
        "contract_sha256": contract["contract_sha256"],
        "slurm_job_id": str(__import__("os").environ.get("SLURM_JOB_ID", "")),
        "started_at": args.started_at,
        "terminal_stage": args.terminal_stage,
        "exit_status": args.exit_status,
        "sealed_test_accessed": False,
        "automatic_promotion": False,
        "longer_run_authorized": False,
        "promotion_authorized": False,
        "sealed_test_request_authorized": False,
        "artifacts": {
            "stage_log": artifact(args.stage_log),
            "training_result": artifact(args.run_root / "result.json"),
            "full_decay_gates": artifact(args.run_root / "full-decay-gates.json"),
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
