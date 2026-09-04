#!/usr/bin/env python3
"""Validate, submit, and receipt the complete phase-35 training campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_reconstruction_phase35_contract import (  # noqa: E402
    CAMPAIGN_SUBMISSION_RECEIPT,
    PHASE35_ARM_ROLES,
    PHASE35_CAMPAIGN_PERMISSIONS,
    canonical_contract_hash,
    git,
    phase35_contract_relative_path,
    phase35_submission_command,
    safe_repo_output_path,
    slurm_job,
    verify_contract,
    verify_live_slurm_record,
)

EXPECTED_ARMS = set(PHASE35_ARM_ROLES)
SUBMISSION_RECEIPT = CAMPAIGN_SUBMISSION_RECEIPT


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def slurm_submit_line(job_id: str) -> str:
    for _ in range(5):
        result = subprocess.run(
            (
                "/opt/slurm/bin/sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobIDRaw,SubmitLine",
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            for row in result.stdout.splitlines():
                record_job_id, separator, submit_line = row.partition("|")
                if separator and record_job_id == job_id and submit_line.strip():
                    return submit_line.strip()
        time.sleep(1)
    raise RuntimeError(f"Slurm accounting has no submit line for job {job_id}")


def verify_submission_line(submit_line: str, expected: list[str]) -> None:
    actual = shlex.split(submit_line)
    if actual != expected:
        raise RuntimeError(
            f"Slurm submit line differs from authorized argv: "
            f"actual={actual!r} expected={expected!r}"
        )


def submission_command(
    contract: dict[str, Any], contract_path: Path
) -> list[str]:
    return phase35_submission_command(contract, contract_path)


def _with_receipt_hash(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt.pop("receipt_sha256", None)
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return receipt


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_json(path: Path, payload: dict[str, Any]) -> None:
    """Exclusively reserve and durably publish a receipt before submission."""

    safe_repo_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_repo_output_path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _with_receipt_hash(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace an already-reserved receipt in the same directory."""

    safe_repo_output_path(path, require_file=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(
                json.dumps(
                    _with_receipt_hash(payload),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def scheduler_state(record: str) -> str:
    marker = "JobState="
    if marker not in record:
        return "UNKNOWN"
    return record.split(marker, 1)[1].split()[0].split("+", 1)[0]


def cancellation_state(job_id: str) -> str:
    result = subprocess.run(
        (
            "/opt/slurm/bin/sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,State",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return "ACCOUNTING_UNAVAILABLE"
    for row in result.stdout.splitlines():
        record_job_id, separator, state = row.partition("|")
        if separator and record_job_id == job_id and state.strip():
            return state.strip().split()[0].split("+", 1)[0]
    return "ACCOUNTING_MISSING"


def cancel_and_confirm(job_ids: list[str]) -> dict[str, str]:
    for job_id in job_ids:
        subprocess.run(
            ("/opt/slurm/bin/scancel", job_id),
            cwd=ROOT,
            check=False,
            timeout=30,
        )
    terminal = {
        "CANCELLED",
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
    states = {job_id: cancellation_state(job_id) for job_id in job_ids}
    for _ in range(10):
        if all(state in terminal for state in states.values()):
            break
        time.sleep(1)
        states = {job_id: cancellation_state(job_id) for job_id in job_ids}
    return states


def jobs_with_comment(comment: str) -> list[str]:
    result = subprocess.run(
        (
            "/opt/slurm/bin/squeue",
            "--user",
            pwd.getpwuid(os.getuid()).pw_name,
            "-h",
            "-o",
            "%i|%k",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not recover phase35 jobs by comment: {result.stderr.strip()}"
        )
    return sorted(
        {
            job_id.strip()
            for row in result.stdout.splitlines()
            for job_id, separator, value in (row.partition("|"),)
            if separator and value.strip() == comment and job_id.strip().isdigit()
        }
    )


class SubmissionInterrupted(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.contract) != 3:
        raise RuntimeError("phase35 campaign requires exactly three contracts")
    output = safe_repo_output_path(
        args.output,
        expected_relative=SUBMISSION_RECEIPT,
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite submission receipt: {output}")

    verified: list[tuple[Path, dict[str, Any]]] = []
    for candidate in args.contract:
        path = safe_repo_output_path(candidate, require_file=True)
        contract, _ = verify_contract(path, require_slurm=False)
        expected_contract_relative = phase35_contract_relative_path(
            str(contract["arm_role"])
        )
        safe_repo_output_path(
            path,
            expected_relative=expected_contract_relative,
            require_file=True,
        )
        if canonical_contract_hash(contract) != contract["contract_sha256"]:
            raise RuntimeError("phase35 contract hash changed before submission")
        verified.append((path, contract))
    roles = {contract["arm_role"] for _, contract in verified}
    if roles != EXPECTED_ARMS:
        raise RuntimeError(f"phase35 campaign arm set changed: {roles}")
    sha_values = {contract["expected_git_sha"] for _, contract in verified}
    tag_values = {contract["expected_git_tag"] for _, contract in verified}
    receipt_values = {
        (
            contract["test_receipt"]["path"],
            contract["test_receipt"]["sha256"],
        )
        for _, contract in verified
    }
    if len(sha_values) != 1 or len(tag_values) != 1 or len(receipt_values) != 1:
        raise RuntimeError("phase35 contracts do not share one tested source")

    ordered_verified = sorted(
        verified, key=lambda item: item[1]["arm_role"]
    )
    planned_jobs = [
        {
            "arm_role": contract["arm_role"],
            "task_id": contract["task_id"],
            "contract": str(path.relative_to(ROOT)),
            "contract_sha256": contract["contract_sha256"],
            "scheduler_comment": f"phase35:{contract['contract_sha256']}",
        }
        for path, contract in ordered_verified
    ]
    for planned in planned_jobs:
        existing = jobs_with_comment(str(planned["scheduler_comment"]))
        if existing:
            raise RuntimeError(
                "refusing duplicate phase35 submission; scheduler comment already "
                f"exists on jobs {existing}"
            )
    safe_repo_output_path("artifacts/slurm")
    (ROOT / "artifacts/slurm").mkdir(parents=True, exist_ok=True)
    safe_repo_output_path("artifacts/slurm")

    created_at = datetime.now(timezone.utc).isoformat()
    submissions: list[dict[str, Any]] = []
    submitted_ids: list[str] = []
    release_argv: list[str] | None = None
    release_records: dict[str, str] = {}
    cancellation_states: dict[str, str] = {}
    error_message: str | None = None
    active_arm: str | None = None

    def receipt(status: str) -> dict[str, Any]:
        return {
            "receipt_version": (
                "hypertagging-reconstruction-phase35-submission-v1"
            ),
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "study_id": verified[0][1]["study_id"],
            "git_sha": next(iter(sha_values)),
            "git_tag": next(iter(tag_values)),
            "test_receipt": {
                "path": next(iter(receipt_values))[0],
                "sha256": next(iter(receipt_values))[1],
            },
            "permissions": dict(PHASE35_CAMPAIGN_PERMISSIONS),
            "sealed_test_accessed": False,
            "automatic_promotion": False,
            "atomic_campaign_release": True,
            "jobs_initially_submitted_held": True,
            "planned_jobs": planned_jobs,
            "active_submission_arm": active_arm,
            "jobs": submissions,
            "submitted_job_ids": submitted_ids,
            "release_argv": release_argv,
            "scheduler_records_after_release": release_records,
            "cancellation_states": cancellation_states,
            "error": error_message,
        }

    # Reserve and fsync the receipt before the first scheduler mutation. If
    # this fails, no job has been submitted.
    reserve_json(output, receipt("preparing_held_submissions"))
    status = "preparing_held_submissions"

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt_submission(signum: int, _frame: object) -> None:
        raise SubmissionInterrupted(
            f"phase35 submission interrupted by signal {signum}"
        )

    for signum in previous_handlers:
        signal.signal(signum, interrupt_submission)
    try:
        for path, contract in ordered_verified:
            active_arm = str(contract["arm_role"])
            atomic_json(output, receipt("submitting_held_arm"))
            command = submission_command(contract, path)
            raw_job_id = run(command)
            job_id = raw_job_id.split(";", 1)[0].strip()
            if not job_id.isdigit():
                raise RuntimeError(
                    f"sbatch returned an invalid job ID: {raw_job_id!r}"
                )
            submitted_ids.append(job_id)
            # Journal the returned scheduler identity before slower accounting
            # and allocation-contract checks.
            atomic_json(output, receipt("held_job_id_received"))
            submit_line = slurm_submit_line(job_id)
            verify_submission_line(submit_line, command)
            scheduler_record = slurm_job(job_id)
            verify_live_slurm_record(
                scheduler_record,
                contract=contract,
                contract_path=path,
            )
            if (
                scheduler_state(scheduler_record) != "PENDING"
                or "Reason=JobHeldUser" not in scheduler_record
            ):
                raise RuntimeError(
                    f"phase35 job {job_id} was not retained on the user hold"
                )
            submissions.append(
                {
                    "arm_role": contract["arm_role"],
                    "task_id": contract["task_id"],
                    "job_id": job_id,
                    "contract": str(path.relative_to(ROOT)),
                    "contract_sha256": contract["contract_sha256"],
                    "submission_argv": command,
                    "scheduler_submit_line": submit_line,
                    "scheduler_record_while_held": scheduler_record,
                    "scheduler_state_while_held": scheduler_state(
                        scheduler_record
                    ),
                    "requeue": 0,
                    "restarts": 0,
                }
            )
            active_arm = None
            atomic_json(output, receipt("held_job_verified"))
        status = "prepared_held"
        atomic_json(output, receipt(status))
        release_argv = [
            "/opt/slurm/bin/scontrol",
            "release",
            ",".join(submitted_ids),
        ]
        # Persist the exact, complete release authorization before asking Slurm
        # to make any task runnable.  Runtime verification rejects every earlier
        # preparation state, while this state remains recoverable after SIGKILL.
        status = "release_in_progress"
        atomic_json(output, receipt(status))
        run(release_argv)
        allowed_released_states = {"PENDING", "CONFIGURING", "RUNNING"}
        for (path, contract), submission in zip(
            ordered_verified,
            submissions,
            strict=True,
        ):
            job_id = str(submission["job_id"])
            scheduler_record = slurm_job(job_id)
            verify_live_slurm_record(
                scheduler_record,
                contract=contract,
                contract_path=path,
            )
            state = scheduler_state(scheduler_record)
            if state not in allowed_released_states:
                raise RuntimeError(
                    f"phase35 job {job_id} entered unexpected state after "
                    f"release: {state}"
                )
            if "Reason=JobHeld" in scheduler_record:
                raise RuntimeError(
                    f"phase35 job {job_id} remained held after campaign release"
                )
            release_records[job_id] = scheduler_record
            submission["scheduler_record_after_release"] = scheduler_record
            submission["scheduler_state_after_release"] = state
        status = "submitted"
        atomic_json(output, receipt(status))
    except BaseException as error:
        # Cleanup is a critical section: a second interactive signal must not
        # interrupt recovery, cancellation, or the terminal receipt update.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        error_message = str(error)
        # An sbatch timeout or interruption can create a held job before its ID
        # reaches stdout. Recover every planned identity by its unique comment.
        recovery_errors: list[str] = []
        for planned in planned_jobs:
            try:
                for recovered in jobs_with_comment(
                    str(planned["scheduler_comment"])
                ):
                    if recovered not in submitted_ids:
                        submitted_ids.append(recovered)
            except Exception as recovery_error:
                recovery_errors.append(str(recovery_error))
        if recovery_errors:
            error_message += f"; recovery_errors={recovery_errors}"
        cancellation_states.update(cancel_and_confirm(submitted_ids))
        terminal_states = {
            "CANCELLED",
            "COMPLETED",
            "FAILED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        cancelled_or_terminal = all(
            state in terminal_states for state in cancellation_states.values()
        )
        status = (
            "submission_failed_no_jobs_created"
            if not submitted_ids
            else (
                "submission_failed_jobs_terminal_after_cancel"
                if cancelled_or_terminal
                else "submission_failed_cancellation_unconfirmed"
            )
        )
        atomic_json(output, receipt(status))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if status != "submitted":
        raise RuntimeError(error_message or "phase35 submission failed")
    print(
        json.dumps(
            {
                "status": status,
                "job_ids": submitted_ids,
                "receipt": str(output),
                "git_sha": git("rev-parse", "HEAD"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
