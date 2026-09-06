#!/usr/bin/env python3
"""Atomically submit the two preregistered phase-38 Slurm tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reconstruction_phase35_evaluation_cohort import atomic_json  # noqa: E402
from scripts.run_reconstruction_phase38 import ARM_ROLES, verify_contract  # noqa: E402


WRAPPER = ROOT / "scripts/slurm/run_reconstruction_phase38.sbatch"


def run(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def receipt_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    value = dict(payload)
    value["receipt_sha256"] = receipt_hash(value)
    atomic_json(path, value)


def scheduler_record(job_id: str) -> str:
    return run(["/opt/slurm/bin/scontrol", "show", "job", "-dd", job_id])


def state(record: str) -> str:
    return record.split("JobState=", 1)[1].split()[0].split("+", 1)[0]


def submission_command(contract: dict[str, Any], path: Path) -> list[str]:
    resources = contract["resources"]
    return [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--hold",
        "--account=others",
        "--partition=inter",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        f"--gres={resources['gres']}",
        "--no-requeue",
        "--export=NONE",
        f"--job-name={contract['task_id']}",
        f"--comment=phase38:{contract['contract_sha256']}",
        str(WRAPPER.resolve(strict=True)),
        str(path.resolve(strict=True)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.contract) != 2:
        raise RuntimeError("phase38 requires exactly two contracts")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite phase38 submission receipt")

    verified = []
    for candidate in args.contract:
        path = candidate.resolve(strict=True)
        contract, _runtime = verify_contract(path)
        verified.append((path, contract))
    verified.sort(key=lambda item: ARM_ROLES.index(item[1]["arm_role"]))
    if tuple(item[1]["arm_role"] for item in verified) != ARM_ROLES:
        raise RuntimeError("phase38 contract arm set changed")
    if len({item[1]["expected_git_sha"] for item in verified}) != 1:
        raise RuntimeError("phase38 contracts do not share one source revision")

    planned = [
        {
            "arm_role": contract["arm_role"],
            "task_id": contract["task_id"],
            "contract": str(path.relative_to(ROOT)),
            "contract_sha256": contract["contract_sha256"],
            "submission_argv": submission_command(contract, path),
        }
        for path, contract in verified
    ]
    created = datetime.now(timezone.utc).isoformat()
    jobs: list[dict[str, Any]] = []
    job_ids: list[str] = []
    error: str | None = None

    def payload(status: str) -> dict[str, Any]:
        return {
            "receipt_version": "hypertagging-reconstruction-phase38-submission-v1",
            "created_at": created,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "study_id": verified[0][1]["study_id"],
            "git_sha": verified[0][1]["expected_git_sha"],
            "git_tag": verified[0][1]["expected_git_tag"],
            "planned_jobs": planned,
            "jobs": jobs,
            "submitted_job_ids": job_ids,
            "atomic_held_release": True,
            "automatic_promotion": False,
            "longer_run_authorized": False,
            "promotion_authorized": False,
            "sealed_test_accessed": False,
            "sealed_test_request_authorized": False,
            "error": error,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(output, payload("preparing_held_submissions"))
    try:
        for (path, contract), plan in zip(verified, planned):
            raw = run(plan["submission_argv"])
            job_id = raw.split(";", 1)[0].strip()
            if not job_id.isdigit():
                raise RuntimeError(f"invalid phase38 Slurm job ID: {raw!r}")
            job_ids.append(job_id)
            write_receipt(output, payload("held_job_id_received"))
            record = scheduler_record(job_id)
            if state(record) != "PENDING" or "Reason=JobHeldUser" not in record:
                raise RuntimeError(f"phase38 job {job_id} was not held")
            if f"Comment=phase38:{contract['contract_sha256']}" not in record:
                raise RuntimeError(f"phase38 job {job_id} comment mismatch")
            jobs.append(
                {
                    "arm_role": contract["arm_role"],
                    "task_id": contract["task_id"],
                    "job_id": job_id,
                    "contract_sha256": contract["contract_sha256"],
                    "scheduler_state_while_held": "PENDING",
                    "scheduler_record_while_held": record,
                }
            )
            write_receipt(output, payload("held_job_verified"))
        write_receipt(output, payload("prepared_held"))
        run(["/opt/slurm/bin/scontrol", "release", ",".join(job_ids)])
        for job in jobs:
            record = scheduler_record(job["job_id"])
            current = state(record)
            if current not in {"PENDING", "CONFIGURING", "RUNNING"}:
                raise RuntimeError(
                    f"phase38 job {job['job_id']} has unexpected state {current}"
                )
            job["scheduler_state_after_release"] = current
            job["scheduler_record_after_release"] = record
        write_receipt(output, payload("submitted"))
    except BaseException as failure:
        error = str(failure)
        for job_id in job_ids:
            subprocess.run(
                ["/opt/slurm/bin/scancel", job_id], cwd=ROOT, check=False, timeout=30
            )
        write_receipt(output, payload("submission_failed_jobs_cancelled"))
        raise
    print(
        json.dumps(
            {"status": "submitted", "job_ids": job_ids, "receipt": str(output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
