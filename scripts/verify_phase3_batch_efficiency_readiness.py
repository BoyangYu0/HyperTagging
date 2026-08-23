#!/usr/bin/env python
"""Verify the programming-phase readiness artifact and its fail-closed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_version") != "ht-pretraining-1m-phase3-batch-efficiency-readiness-v1":
        raise RuntimeError("unsupported batch-efficiency readiness version")
    stored_hash = payload.get("readiness_hash")
    body = dict(payload)
    body.pop("readiness_hash", None)
    actual_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored_hash != actual_hash:
        raise RuntimeError("readiness artifact hash mismatch")
    contract = payload["scientific_contract"]
    if contract["total_presentations"] != 1_730_048:
        raise RuntimeError("total presentations changed")
    if contract["resume_presentations"] != 865_024 or contract["remaining_presentations"] != 865_024:
        raise RuntimeError("resume/remaining presentation boundary changed")
    if contract["virtual_step_unit_presentations"] != 16:
        raise RuntimeError("virtual-step unit changed")
    if contract["objective_dominance_limit"] != 20.0:
        raise RuntimeError("objective dominance fail-closed limit changed")
    if contract["sealed_test_role_access"] != "forbidden" or contract["stress_payload_access"] != "forbidden":
        raise RuntimeError("payload isolation weakened")
    calibration = payload["calibration"]
    if calibration["calibration_order"] != ["h100nvl", "v100"] or calibration["sequential_only"] is not True:
        raise RuntimeError("calibration order/serialization gate changed")
    submission = payload["submission"]
    if submission["production_submission_authorized"] is not False:
        raise RuntimeError("programming readiness must not authorize production")
    if submission["submission_performed"] is not False or submission["job_count"] != 1:
        raise RuntimeError("submission uniqueness/status gate changed")
    if payload["renderer_verifier"]["structural_scientific_slurm_submission_allowed"] is not False:
        raise RuntimeError("structural provenance gate changed")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        default=Path("artifacts/codex/ht_pretraining_1m_batch_efficiency_readiness_20260823.json"),
        nargs="?",
    )
    args = parser.parse_args(argv)
    verify(args.path)
    print(json.dumps({"verified": True, "submission_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
