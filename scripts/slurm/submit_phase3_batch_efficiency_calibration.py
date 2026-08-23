#!/usr/bin/env python
"""Submit one immutable phase-3 calibration tuple through the tracked adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.phase3_parallel_study import (  # noqa: E402
    entry_by_id,
    load_study_plan,
    resolve_plan_path,
)


DEFAULT_PLAN = ROOT / "configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json"
DEFAULT_GPU_ENVIRONMENT = "/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"
SBATCH = "/opt/slurm/bin/sbatch"


def _entry_paths(entry: dict[str, Any]) -> dict[str, Path]:
    tuple_root = resolve_plan_path(entry["attempt_root"], root=ROOT).parent
    return {
        "tuple_root": tuple_root,
        "output_root": resolve_plan_path(entry["output_root"], root=ROOT),
        "attempt_root": resolve_plan_path(entry["attempt_root"], root=ROOT),
        "checkpoint_copy": resolve_plan_path(entry["checkpoint_copy_path"], root=ROOT),
        "metrics": resolve_plan_path(entry["metrics_path"], root=ROOT),
        "receipt": resolve_plan_path(entry["receipt_path"], root=ROOT),
    }


def build_sbatch_command(
    entry: dict[str, Any],
    *,
    plan_path: Path,
    gpu_environment: str,
    submitted_epoch: int,
    token: str,
) -> tuple[list[str], Path, Path]:
    paths = _entry_paths(entry)
    calibration_id = str(entry["calibration_id"])
    log_root = paths["tuple_root"] / "slurm"
    output = log_root / "stdout-%j.log"
    error = log_root / "stderr-%j.log"
    policy = entry["precision_policy"]
    export = {
        "HT_PHASE3_CALIBRATION_ACTIVE": "1",
        "HT_PHASE3_EXPECTED_GRES": entry["exact_gres"],
        "HT_PHASE3_SUBMIT_EPOCH": str(submitted_epoch),
        "HT_PHASE3_FRESH_PREFLIGHT_TOKEN": token,
        "HT_PHASE3_PLAN": str(plan_path.resolve()),
        "HT_PHASE3_OWNER": "sole-authorized-phase3-follow-up-programming-operator",
        "HT_PHASE3_CALIBRATION_ID": calibration_id,
        "HT_PHASE3_PROFILE": entry["profile"],
        "HT_PHASE3_BATCH_SIZE": str(entry["batch_size"]),
        "HT_PHASE3_AMP_DTYPE": policy["amp_dtype"],
        "HT_PHASE3_GRAD_SCALER": "enabled" if policy["grad_scaler_enabled"] else "disabled",
        "HT_PHASE3_CHECKPOINT_COPY": str(paths["checkpoint_copy"]),
        "HT_PHASE3_OUTPUT_ROOT": str(paths["output_root"]),
        "HT_PHASE3_ATTEMPT_ROOT": str(paths["attempt_root"]),
        "HT_PHASE3_METRICS": str(paths["metrics"]),
        "HT_PHASE3_RECEIPT": str(paths["receipt"]),
        "HT_PHASE3_LOG_ROOT": str(log_root),
        "HT_PHASE3_GPU_ENVIRONMENT": gpu_environment,
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export.items())
    command = [
        SBATCH,
        "--account=others",
        "--partition=inter",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=32G",
        "--time=00:25:00",
        "--no-requeue",
        f"--gres={entry['exact_gres']}",
        f"--job-name={calibration_id}",
        f"--output={output}",
        f"--error={error}",
        f"--export={export_arg}",
        str(ROOT / "scripts/slurm/run_phase3_batch_efficiency_calibration.sbatch"),
    ]
    return command, output, error


def _job_name_is_present(calibration_id: str) -> bool:
    for command in (
        ["/opt/slurm/bin/squeue", "-h", "-n", calibration_id, "-o", "%j"],
        [
            "/opt/slurm/bin/sacct",
            "--allocations",
            f"--name={calibration_id}",
            "--starttime=2026-08-23T00:00:00",
            "--format=JobNameRaw",
            "--noheader",
        ],
    ):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0 and any(line.strip() == calibration_id for line in result.stdout.splitlines()):
            return True
    return False


def submit(entry: dict[str, Any], *, plan_path: Path, gpu_environment: str) -> dict[str, Any]:
    paths = _entry_paths(entry)
    manifest = paths["tuple_root"] / "slurm" / "submission.json"
    if manifest.exists() or any(paths[key].exists() for key in ("output_root", "attempt_root", "checkpoint_copy", "metrics", "receipt")):
        raise RuntimeError("calibration tuple already has an artifact or submission manifest; refusing duplicate")
    calibration_id = str(entry["calibration_id"])
    if _job_name_is_present(calibration_id):
        raise RuntimeError("calibration job name already exists in Slurm history; refusing duplicate")
    paths["tuple_root"].mkdir(parents=True, exist_ok=False)
    (paths["tuple_root"] / "slurm").mkdir(exist_ok=False)
    submitted_epoch = int(time.time())
    token = secrets.token_hex(32)
    command, output, error = build_sbatch_command(
        entry,
        plan_path=plan_path,
        gpu_environment=gpu_environment,
        submitted_epoch=submitted_epoch,
        token=token,
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    match = re.search(r"Submitted batch job ([0-9]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"sbatch returned no job ID: {result.stdout.strip()}")
    payload = {
        "artifact_version": "ht-pretraining-1m-phase3-calibration-submission-v1",
        "calibration_id": calibration_id,
        "tuple_sha256": entry["tuple_sha256"],
        "exact_gres": entry["exact_gres"],
        "slurm_job_id": match.group(1),
        "submitted_epoch": submitted_epoch,
        "partition": "inter",
        "stdout_pattern": str(output),
        "stderr_pattern": str(error),
        "adapter": "scripts/slurm/run_phase3_batch_efficiency_calibration.sbatch",
        "gpu_environment": gpu_environment,
        "submission_command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    payload["submission_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--gpu-environment", default=DEFAULT_GPU_ENVIRONMENT)
    args = parser.parse_args(argv)
    plan = load_study_plan(args.study_plan, root=ROOT)
    entry = entry_by_id(plan, args.calibration_id)
    payload = submit(entry, plan_path=args.study_plan.resolve(), gpu_environment=args.gpu_environment)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
