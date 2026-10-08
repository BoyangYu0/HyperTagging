"""Atomic one-shot four-arm submission; render-only unless --submit is explicit."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import sha, write  # noqa: E402
from scripts.run_phase74_development import validate_contract, verify_bindings  # noqa: E402


def acquire_submission_lock(path, contract_hash):
    with path.open("x") as stream:
        json.dump(
            {
                "contract_sha256": contract_hash,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())


def submit(path, execute=False):
    c = json.loads(path.read_text())
    validate_contract(c)
    source = Path(c["source_root"])
    verify_bindings(c, source)
    admission = json.loads((path.parent / "admission.json").read_text())
    if admission["status"] != "PASS" or admission["contract_sha256"] != sha(path):
        raise ValueError("Runtime admission mismatch")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Frozen source revision mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    ).strip():
        raise ValueError("Frozen source not clean")
    queue = subprocess.check_output(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%T"],
        text=True,
    )
    if any("phase74-train-" in line for line in queue.splitlines()):
        raise RuntimeError("Concurrent Phase74 campaign already active")
    base = path.parent
    if (base / "submission-lock.json").exists() or any(base.glob("*-submission.json")):
        raise RuntimeError("Campaign submission already attempted; never duplicate")
    commands = []
    for arm in c["arms"]:
        commands.append(
            [
                "sbatch",
                "--parsable",
                "--export=NIL",
                "--chdir=" + str(source),
                "--job-name=phase74-train-" + arm,
                "--output=" + str(base / (arm + "-%j.log")),
                str(source / "scripts/slurm/run_phase74_development.sbatch"),
                str(path),
                arm,
            ]
        )
    if not execute:
        return {"status": "ELIGIBLE_NOT_SUBMITTED", "commands": commands}
    acquire_submission_lock(base / "submission-lock.json", sha(path))
    receipts = []
    for arm, cmd in zip(c["arms"], commands):
        job = subprocess.check_output(cmd, text=True).strip()
        if not job.split(";")[0].isdigit():
            raise RuntimeError("Unrecognized scheduler acceptance; do not resubmit")
        receipt = {
            "arm": arm,
            "job_id": job,
            "command": cmd,
            "source_sha": c["source_sha"],
            "contract_sha256": sha(path),
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "development",
        }
        write(base / (arm + "-submission.json"), receipt)
        receipts.append(receipt)
        print(arm, job, flush=True)
    return {"status": "SUBMITTED", "receipts": receipts}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    print(json.dumps(submit(a.contract.resolve(), a.submit), indent=2))
