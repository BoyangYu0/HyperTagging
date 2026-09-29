#!/usr/bin/env python3
"""Submit the preregistered Phase63 pair or an explicit zero-step arm repair."""

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
from scripts.run_reconstruction_phase63 import ARM_ROLES, verify_contract  # noqa: E402


WRAPPER = ROOT / "scripts/slurm/run_reconstruction_phase63.sbatch"


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
        f"--comment=phase63:{contract['contract_sha256']}",
        str(WRAPPER.resolve(strict=True)),
        str(path.resolve(strict=True)),
    ]


def validate_failed_arm_retry(contract: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    """Admit only the recorded zero-step failure with an admission-only source repair."""
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("receipt_sha256") != receipt_hash(receipt)
            or receipt.get("slurm_job_id") != "16742233"
            or receipt.get("arm_role") != "masked_only"
            or receipt.get("status") != "failed" or receipt.get("exit_status") != 1
            or receipt.get("terminal_stage") != "trainer"
            or receipt.get("sealed_test_accessed") is not False
            or receipt.get("artifacts", {}).get("training_result") is not None
            or receipt.get("artifacts", {}).get("full_decay_gates") is not None):
        raise RuntimeError("retry requires the authenticated Phase63 zero-step failure")
    original_root = receipt_path.parents[6]
    expected = original_root / "artifacts/slurm/reconstruction-phase63/jobs/16742233/attempt-00/receipt.json"
    if receipt_path != expected:
        raise RuntimeError("retry receipt is outside its native evidence layout")
    old = json.loads((receipt_path.parent / "submitted-contract.json").read_text())
    from scripts.run_reconstruction_phase63 import canonical_contract_hash
    if canonical_contract_hash(old) != old["contract_sha256"] or old["contract_sha256"] != receipt["contract_sha256"]:
        raise RuntimeError("retry original contract hash mismatch")
    for key in ("arm_role", "study_id", "config", "data", "cohort", "checkpoint",
                "checkpoint_sha256", "checkpoint_step", "preregistration",
                "pretraining_refinement", "evaluation_contract", "post_training_gates",
                "resources", "output_root", "gpu_environment"):
        if contract[key] != old[key]:
            raise RuntimeError(f"retry changed the preregistered {key}")
    old_run = original_root / old["output_root"] / "16742233"
    if (old_run / "training").exists() or (old_run / "result.json").exists():
        raise RuntimeError("retry is not a zero-training restart")
    error_path = original_root / "artifacts/slurm/phase63-masked_only-20260928-16742233.err"
    if "ValueError: recovery_objective_weight must be finite and positive" not in error_path.read_text():
        raise RuntimeError("retry does not match the diagnosed admission failure")
    old_sha = old["expected_git_sha"]
    trainer = "src/hypertagging/training/reconstruction_trainer.py"
    changed = run(["git", "diff", "--name-only", old_sha, "HEAD", "--", "src"])
    if changed.splitlines() != [trainer]:
        raise RuntimeError("retry changed scientific code beyond the admission repair")
    before = run(["git", "show", f"{old_sha}:{trainer}"])
    after = (ROOT / trainer).read_text().strip()
    expected_after = before.replace(
        "or config.recovery_objective_weight <= 0",
        'or config.recovery_objective_weight < 0\n        or (config.unrepresentable_target_policy == "recovery_objective"\n            and config.recovery_objective_weight == 0)', 1,
    ).replace('raise ValueError("recovery_objective_weight must be finite and positive")',
              'raise ValueError("recovery_objective_weight must be finite and nonnegative, and positive when recovery_objective is active")', 1)
    if after != expected_after:
        raise RuntimeError("retry source repair is not the reviewed admission-only change")
    scheduler = run(["/opt/slurm/bin/sacct", "-j", "16742233", "-X", "-n", "-P",
                     "--format=JobID,State,ExitCode"])
    if scheduler.strip() != "16742233|FAILED|1:0":
        raise RuntimeError("retry scheduler failure evidence differs")
    return {"original_job_id": "16742233", "receipt_path": str(receipt_path),
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "original_source_sha": old_sha, "scientific_change": "inactive_weight_admission_only",
            "optimizer_steps_before_retry": 0, "automatic_retry": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-failed-receipt", type=Path,
                        help="Explicit continuation: retry only the Phase63 arm that failed before training")
    args = parser.parse_args()
    if len(args.contract) != (1 if args.retry_failed_receipt else 2):
        raise RuntimeError("phase63 requires two contracts, or one masked-only contract for an explicit retry")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite phase63 submission receipt")

    verified = []
    for candidate in args.contract:
        path = candidate.resolve(strict=True)
        contract, _runtime = verify_contract(path)
        verified.append((path, contract))
    verified.sort(key=lambda item: ARM_ROLES.index(item[1]["arm_role"]))
    expected_roles = ("masked_only",) if args.retry_failed_receipt else ARM_ROLES
    if tuple(item[1]["arm_role"] for item in verified) != expected_roles:
        raise RuntimeError("phase63 contract arm set changed")
    retry = None
    if args.retry_failed_receipt:
        retry = validate_failed_arm_retry(verified[0][1], args.retry_failed_receipt)
    if len({item[1]["expected_git_sha"] for item in verified}) != 1:
        raise RuntimeError("phase63 contracts do not share one source revision")

    queued = run(
        [
            "/opt/slurm/bin/squeue",
            "--noheader",
            "--user",
            __import__("getpass").getuser(),
            "--format=%j|%k",
        ]
    )
    if any(
        c["task_id"] in queued or "phase63:" + c["contract_sha256"] in queued
        for _, c in verified
    ):
        raise RuntimeError("Duplicate Phase63 scheduler job detected before submission")

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
            "receipt_version": "hypertagging-reconstruction-phase63-submission-v1",
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
            "explicit_failed_arm_continuation": retry,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(output, payload("preparing_held_submissions"))
    try:
        for (path, contract), plan in zip(verified, planned):
            raw = run(plan["submission_argv"])
            job_id = raw.split(";", 1)[0].strip()
            if not job_id.isdigit():
                raise RuntimeError(f"invalid phase63 Slurm job ID: {raw!r}")
            job_ids.append(job_id)
            write_receipt(output, payload("held_job_id_received"))
            record = scheduler_record(job_id)
            if state(record) != "PENDING" or "Reason=JobHeldUser" not in record:
                raise RuntimeError(f"phase63 job {job_id} was not held")
            if f"Comment=phase63:{contract['contract_sha256']}" not in record:
                raise RuntimeError(f"phase63 job {job_id} comment mismatch")
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
                    f"phase63 job {job['job_id']} has unexpected state {current}"
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
