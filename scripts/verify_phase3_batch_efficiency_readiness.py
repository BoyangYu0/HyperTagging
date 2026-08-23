#!/usr/bin/env python
"""Verify the programming-phase readiness artifact and its fail-closed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.phase3_parallel_study import (  # noqa: E402
    MAX_CONCURRENT_CALIBRATION_JOBS,
    assert_no_active_calibrations,
    file_sha256,
    load_study_plan,
    resolve_plan_path,
)


def _verify_legacy(path: Path) -> dict[str, object]:
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


def _verify_parallel(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_version") != "ht-pretraining-1m-phase3-parallel-study-readiness-v1":
        raise RuntimeError("unsupported parallel-study readiness version")
    body = dict(payload)
    stored_hash = body.pop("readiness_hash", None)
    actual_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored_hash != actual_hash:
        raise RuntimeError("parallel-study readiness artifact hash mismatch")
    plan_path = resolve_plan_path(payload["study_plan"], root=ROOT)
    plan = load_study_plan(plan_path, root=ROOT)
    if payload.get("study_plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("parallel-study readiness plan hash mismatch")
    calibration = payload["calibration"]
    if calibration.get("gpu_calibration_completed") is not False:
        raise RuntimeError("GPU calibration must remain incomplete in programming readiness")
    if calibration.get("max_concurrent_jobs") != MAX_CONCURRENT_CALIBRATION_JOBS:
        raise RuntimeError("parallel-study max concurrency changed")
    if calibration.get("configured_calibration_count") != 4:
        raise RuntimeError("parallel-study configured matrix count changed")
    if calibration.get("sequential_only") is not False:
        raise RuntimeError("parallel-study readiness retains sequential-only gating")
    if calibration.get("receipt_policy") != "exact_configured_set":
        raise RuntimeError("parallel-study receipt policy is not exact-set fail-closed")
    if calibration.get("production_during_active_calibration") != "forbidden":
        raise RuntimeError("production-active-calibration gate changed")
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
    submission = payload["submission"]
    if (
        submission.get("production_submission_authorized") is not False
        or submission.get("submission_performed") is not False
        or submission.get("job_count") != 1
        or submission.get("default_production_resume_count") != 1
    ):
        raise RuntimeError("parallel-study production gate changed")
    if payload["renderer_verifier"]["structural_scientific_slurm_submission_allowed"] is not False:
        raise RuntimeError("structural provenance gate changed")
    assert_no_active_calibrations(plan, root=ROOT)
    return payload


def verify(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    version = payload.get("artifact_version")
    if version == "ht-pretraining-1m-phase3-batch-efficiency-readiness-v1":
        return _verify_legacy(path)
    return _verify_parallel(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        default=Path("artifacts/codex/ht_pretraining_1m_batch_efficiency_readiness_parallel_study_20260823.json"),
        nargs="?",
    )
    args = parser.parse_args(argv)
    verify(args.path)
    print(json.dumps({"verified": True, "submission_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
