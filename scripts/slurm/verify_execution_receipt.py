#!/usr/bin/env python3
"""Fail closed on a Slurm attempt receipt or any referenced artifact drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(path: Path, *, require_completed: bool = False) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt_version = payload.get("receipt_version")
    if receipt_version not in {
        "hypertagging-slurm-attempt-v1",
        "hypertagging-slurm-attempt-v2",
    }:
        raise RuntimeError("unsupported Slurm attempt receipt")
    stored = str(payload.get("receipt_sha256", ""))
    canonical = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("Slurm attempt receipt hash mismatch")
    for name, evidence in payload.get("artifacts", {}).items():
        artifact = Path(str(evidence["path"]))
        if not artifact.is_absolute() or not artifact.is_file():
            raise RuntimeError(f"receipt artifact is absent or non-absolute: {name}")
        if artifact.stat().st_size != int(evidence["bytes"]):
            raise RuntimeError(f"receipt artifact size changed: {name}")
        if _sha256(artifact) != evidence["sha256"]:
            raise RuntimeError(f"receipt artifact hash changed: {name}")
    if require_completed:
        wrapper = payload.get("wrapper") or {}
        if (
            payload.get("status") != "completed"
            or payload.get("terminal_stage") != "trainer_complete"
            or payload.get("batch_exit_status") != 0
            or payload.get("trainer_status") != 0
            or wrapper.get("action") != "trainer_exit"
            or wrapper.get("wrapper_status") != 0
        ):
            raise RuntimeError("Slurm attempt receipt does not prove normal completion")
        required = {"job_contract", "allocation", "gpu_preflight", "checkpoint", "metrics"}
        missing = required - set(payload.get("artifacts", {}))
        if missing:
            raise RuntimeError(
                f"completed Slurm attempt lacks required evidence: {sorted(missing)}"
            )
        if receipt_version == "hypertagging-slurm-attempt-v2":
            telemetry = payload.get("gpu_telemetry") or {}
            telemetry_artifacts = {"gpu_telemetry", "gpu_telemetry_summary"}
            missing_telemetry = telemetry_artifacts - set(payload.get("artifacts", {}))
            if (
                missing_telemetry
                or telemetry.get("status") != "completed"
                or int(telemetry.get("sample_count", 0)) <= 0
            ):
                raise RuntimeError(
                    "completed Slurm attempt lacks healthy periodic GPU telemetry"
                )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()
    payload = verify_receipt(args.receipt.resolve(), require_completed=args.require_completed)
    print(
        json.dumps(
            {
                "receipt_verified": True,
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
                "job_id": payload.get("slurm", {}).get("job_id"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
