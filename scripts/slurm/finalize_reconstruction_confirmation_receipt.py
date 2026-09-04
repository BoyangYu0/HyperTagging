#!/usr/bin/env python3
"""Finalize a paired reconstruction confirmation task receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--telemetry-summary", type=Path, required=True)
    parser.add_argument("--batch-exit-status", type=int, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8")) if args.result.is_file() else {}
    telemetry = json.loads(args.telemetry_summary.read_text(encoding="utf-8")) if args.telemetry_summary.is_file() else {}
    healthy = (
        args.batch_exit_status == 0
        and result.get("status") == "completed"
        and result.get("training_performed") is False
        and result.get("sealed_test_role_access") == "forbidden"
        and result.get("promotion_authorized") is False
        and result.get("checkpoints_unchanged") is True
        and result.get("selection", {}).get("original_rollout_cohort_disjoint") is True
        and telemetry.get("status") == "completed"
    )
    candidates = {
        "contract": args.contract,
        "result": args.result,
        "gpu_telemetry": args.telemetry,
        "gpu_telemetry_summary": args.telemetry_summary,
    }
    receipt = {
        "receipt_version": "hypertagging-reconstruction-paired-confirmation-attempt-v1",
        "status": "completed" if healthy else "failed_or_nonterminal",
        "batch_exit_status": args.batch_exit_status,
        "study_id": contract.get("study_id"),
        "task_id": contract.get("task_id"),
        "contract_sha256": contract.get("contract_sha256"),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        },
        "sealed_test_role_access": "forbidden",
        "promotion_authorized": False,
        "training_performed": False,
        "gpu_telemetry": telemetry,
        "artifacts": {name: artifact(path) for name, path in candidates.items() if path.is_file()},
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
