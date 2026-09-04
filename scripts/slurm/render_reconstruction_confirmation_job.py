#!/usr/bin/env python3
"""Render an authorized paired reconstruction confirmation Slurm task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = "configs/reconstruction/ht_reconstruction_phase34_confirmation_20260904.json"
CONTRACT_VERSION = "hypertagging-reconstruction-phase34-confirmation-contract-v1"
SOURCE_FILES = (
    PREREGISTRATION,
    "artifacts/codex/reconstruction_phase34_comparison_no_promotion_20260904.json",
    "scripts/evaluate_reconstruction_checkpoint_pair.py",
    "scripts/slurm/render_reconstruction_confirmation_job.py",
    "scripts/slurm/verify_reconstruction_confirmation_contract.py",
    "scripts/slurm/run_reconstruction_confirmation.sbatch",
    "scripts/slurm/finalize_reconstruction_confirmation_receipt.py",
    "scripts/slurm/preflight_gpu_environment.py",
    "scripts/slurm/monitor_gpu_telemetry.py",
    "scripts/diagnose_reconstruction_query_activation.py",
    "src/hypertagging/training/fixed_validation.py",
    "src/hypertagging/training/checkpointing.py",
    "src/hypertagging/training/data_module.py",
    "src/hypertagging/reconstruction/level_rollout.py",
    "src/hypertagging/evaluation/hierarchical_metrics.py",
    "environment/gpu/runtime-contract.json",
    "environment/gpu/requirements-cu126.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {command!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    return path


def hashed(value: str) -> dict[str, str]:
    path = repo_path(value)
    return {"path": value, "sha256": sha256(path)}


def live_slurm() -> dict[str, Any]:
    snapshot = command_output(
        ("/opt/slurm/bin/sinfo", "-h", "-p", "inter", "-N", "-o", "%N|%T|%G")
    )
    usable = [
        line for line in snapshot.splitlines()
        if "gpu:h100nvl:" in line
        and not any(state in line.lower() for state in ("down", "drain", "fail", "maint"))
    ]
    if not usable:
        raise RuntimeError("no live H100 NVL node is available in partition inter")
    return {
        "version": command_output(("/opt/slurm/bin/sbatch", "--version")),
        "sinfo_snapshot": snapshot.splitlines(),
        "selected_exact_gres": "gpu:h100nvl:1",
    }


def verify_test_receipt(path: Path, *, expected_git_sha: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("git_sha") != expected_git_sha:
        raise RuntimeError("test receipt is not passed for the expected Git SHA")
    tests = receipt.get("tests")
    if not isinstance(tests, list) or not tests or not all(
        item.get("passed") is True and item.get("exit_code") == 0 for item in tests
    ):
        raise RuntimeError("test receipt does not contain an all-passing test set")
    if receipt.get("sealed_test_accessed") is not False:
        raise RuntimeError("test receipt does not preserve the sealed-test denial")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("confirmation_a", "confirmation_b"), required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-env", type=Path,
        default=Path("/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a confirmation contract")
    head = command_output(("git", "rev-parse", "HEAD"))
    if head != args.expected_git_sha:
        raise RuntimeError("expected confirmation Git SHA is not checked out")
    if command_output(("git", "rev-list", "-n", "1", args.expected_git_tag)) != head:
        raise RuntimeError("expected confirmation tag does not resolve to HEAD")
    if command_output(("git", "status", "--porcelain", "--untracked-files=no")):
        raise RuntimeError("tracked worktree must be clean before contract rendering")
    if not (args.gpu_env / "bin/python").is_file():
        raise RuntimeError("GPU environment is unavailable")
    test_receipt_path = args.test_receipt.resolve(strict=True)
    test_receipt_path.relative_to(ROOT.resolve())
    verify_test_receipt(test_receipt_path, expected_git_sha=head)
    preregistration = json.loads(repo_path(PREREGISTRATION).read_text(encoding="utf-8"))
    cohorts = {item["role"]: item for item in preregistration["cohorts"]}
    cohort = cohorts[args.cohort]
    task_id = f"phase34-paired-{args.cohort}-20260904"
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "study_id": preregistration["study_id"],
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_git_sha": head,
        "expected_git_tag": args.expected_git_tag,
        "gpu_environment": str(args.gpu_env),
        "device": "cuda",
        "preregistration": hashed(PREREGISTRATION),
        "prior_decision": preregistration["prior_decision"],
        "arms": preregistration["arms"],
        "data": preregistration["data_binding"],
        "cohort": cohort,
        "analysis": preregistration["analysis"],
        "resources": preregistration["execution_policy"],
        "output_root": f"artifacts/runs/ht-reconstruction-phase34-confirmation-20260904/{args.cohort}",
        "test_receipt": {
            "path": str(test_receipt_path.relative_to(ROOT)),
            "sha256": sha256(test_receipt_path),
        },
        "authorization": {
            "basis": "explicit_user_operator_instruction_2026-09-04",
            "validation_payload_access_authorized": True,
            "execution_authorized": True,
            "scheduler_authorized": True,
            "submission_authorized": True,
            "scientific_validation_authorized": True,
            "training_authorized": False,
            "checkpoint_mutation_authorized": False,
            "sealed_test_access_authorized": False,
            "promotion_authorized": False,
        },
        "submission_authorized": True,
        "submission_performed": False,
        "live_slurm": live_slurm(),
    }
    paths = [*SOURCE_FILES, str(test_receipt_path.relative_to(ROOT))]
    paths.extend(str(arm["checkpoint"]) for arm in contract["arms"])
    paths.extend((str(contract["data"]["selection_manifest"]), str(contract["data"]["dataset_index"])))
    contract["hashed_inputs"] = [hashed(path) for path in dict.fromkeys(paths)]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    contract["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    command = [
        "/opt/slurm/bin/sbatch",
        "--account=others",
        "--partition=inter",
        "--gres=gpu:h100nvl:1",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=06:00:00",
        "--no-requeue",
        "--export=NIL",
        f"--job-name={task_id}",
        "scripts/slurm/run_reconstruction_confirmation.sbatch",
        str(args.output.resolve()),
    ]
    print(json.dumps({
        "contract": str(args.output),
        "contract_sha256": contract["contract_sha256"],
        "command": command,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
