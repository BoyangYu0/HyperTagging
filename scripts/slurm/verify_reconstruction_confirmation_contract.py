#!/usr/bin/env python3
"""Fail-closed verification for paired reconstruction confirmation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "hypertagging-reconstruction-phase34-confirmation-contract-v1"
PREREGISTRATION = "configs/reconstruction/ht_reconstruction_phase34_confirmation_20260904.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str, *, suffix: str | None = None) -> Path:
    path = (ROOT / value).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    if suffix is not None and path.suffix != suffix:
        raise RuntimeError(f"unexpected input suffix for {value}")
    return path


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def slurm_job(job_id: str) -> str:
    result = subprocess.run(
        ("/opt/slurm/bin/scontrol", "show", "job", "-o", job_id),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def verify_contract_hash(contract: dict[str, Any]) -> str:
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError("confirmation contract hash mismatch")
    return actual


def verify_hashed_inputs(items: list[dict[str, str]]) -> None:
    if not items:
        raise RuntimeError("confirmation contract has no hashed inputs")
    seen: set[str] = set()
    for item in items:
        value = str(item.get("path", ""))
        if value in seen:
            raise RuntimeError("duplicate hashed input")
        seen.add(value)
        if sha256(repo_path(value)) != item.get("sha256"):
            raise RuntimeError(f"hashed input changed: {value}")


def verify_contract(path: Path, *, require_slurm: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract_hash = verify_contract_hash(contract)
    verify_hashed_inputs(list(contract.get("hashed_inputs", [])))
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("unsupported confirmation contract version")
    prereg = json.loads(repo_path(PREREGISTRATION, suffix=".json").read_text(encoding="utf-8"))
    if contract.get("study_id") != prereg.get("study_id"):
        raise RuntimeError("confirmation study ID changed")
    if contract.get("prior_decision") != prereg.get("prior_decision"):
        raise RuntimeError("prior no-promotion decision binding changed")
    prior = contract["prior_decision"]
    prior_path = repo_path(str(prior["path"]), suffix=".json")
    if sha256(prior_path) != prior["sha256"]:
        raise RuntimeError("prior no-promotion decision hash changed")
    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
    if (
        prior_payload.get("decision", {}).get("verdict") != "NO_PROMOTION"
        or prior_payload.get("decision", {}).get("promotion_authorized") is not False
        or prior_payload.get("decision", {}).get("sealed_test_authorized") is not False
    ):
        raise RuntimeError("prior decision does not preserve no-promotion/test denial")
    declared_arms = {arm["role"]: arm for arm in prereg["arms"]}
    if {arm["role"]: arm for arm in contract.get("arms", [])} != declared_arms:
        raise RuntimeError("confirmation arm binding changed")
    for arm in contract["arms"]:
        checkpoint = repo_path(str(arm["checkpoint"]), suffix=".pt")
        if sha256(checkpoint) != arm["checkpoint_sha256"]:
            raise RuntimeError("confirmation checkpoint hash changed")
    if contract.get("data") != prereg.get("data_binding"):
        raise RuntimeError("confirmation data binding changed")
    data = contract["data"]
    if data.get("role") != "validation" or data.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("confirmation data role is not validation-only")
    if sha256(repo_path(data["selection_manifest"], suffix=".json")) != data["selection_manifest_sha256"]:
        raise RuntimeError("confirmation selection hash changed")
    if sha256(repo_path(data["dataset_index"], suffix=".json")) != data["dataset_index_sha256"]:
        raise RuntimeError("confirmation index hash changed")
    cohorts = {item["role"]: item for item in prereg["cohorts"]}
    cohort = contract.get("cohort")
    if not isinstance(cohort, dict) or cohorts.get(cohort.get("role")) != cohort:
        raise RuntimeError("confirmation cohort is not preregistered")
    if int(cohort["event_count"]) != 1000 or int(cohort["rank_offset"]) not in {0, 1000}:
        raise RuntimeError("confirmation cohort size/offset changed")
    if contract.get("analysis") != prereg.get("analysis"):
        raise RuntimeError("confirmation analysis contract changed")
    if contract["analysis"].get("automatic_promotion") is not False:
        raise RuntimeError("confirmation cannot authorize automatic promotion")
    if contract.get("resources") != prereg.get("execution_policy"):
        raise RuntimeError("confirmation resource policy changed")
    resources = contract["resources"]
    if (
        resources.get("gres") != "gpu:h100nvl:1"
        or resources.get("cpus_per_task") != 8
        or resources.get("memory") != "32G"
        or resources.get("time") != "06:00:00"
        or resources.get("requeue") is not False
        or resources.get("maximum_restarts") != 0
    ):
        raise RuntimeError("confirmation resource policy is unsafe or unsupported")
    authorization = contract.get("authorization", {})
    required_true = (
        "validation_payload_access_authorized",
        "execution_authorized",
        "scheduler_authorized",
        "submission_authorized",
        "scientific_validation_authorized",
    )
    required_false = (
        "training_authorized",
        "checkpoint_mutation_authorized",
        "sealed_test_access_authorized",
        "promotion_authorized",
    )
    if any(authorization.get(key) is not True for key in required_true):
        raise RuntimeError("required confirmation authorization is missing")
    if any(authorization.get(key) is not False for key in required_false):
        raise RuntimeError("forbidden confirmation authority was enabled")
    if contract.get("submission_authorized") is not True:
        raise RuntimeError("confirmation submission is not authorized")
    receipt_binding = contract.get("test_receipt", {})
    test_receipt = json.loads(repo_path(receipt_binding.get("path", ""), suffix=".json").read_text())
    if sha256(repo_path(receipt_binding["path"])) != receipt_binding.get("sha256"):
        raise RuntimeError("test receipt hash changed")
    if test_receipt.get("status") != "passed" or test_receipt.get("sealed_test_accessed") is not False:
        raise RuntimeError("test receipt does not authorize confirmation")
    expected_sha = str(contract.get("expected_git_sha", ""))
    expected_tag = str(contract.get("expected_git_tag", ""))
    if not HEX40.fullmatch(expected_sha) or git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("confirmation source Git SHA mismatch")
    if git("rev-list", "-n", "1", expected_tag) != expected_sha:
        raise RuntimeError("confirmation implementation tag mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is dirty")
    if test_receipt.get("git_sha") != expected_sha:
        raise RuntimeError("test receipt Git SHA mismatch")
    gpu_environment = Path(str(contract.get("gpu_environment", "")))
    if not gpu_environment.is_absolute() or not (gpu_environment / "bin/python").is_file():
        raise RuntimeError("GPU environment is unavailable")
    if contract.get("device") != "cuda":
        raise RuntimeError("confirmation device changed")
    output_root = (ROOT / str(contract.get("output_root", ""))).resolve()
    allowed = (ROOT / "artifacts/runs/ht-reconstruction-phase34-confirmation-20260904").resolve()
    if allowed not in output_root.parents:
        raise RuntimeError("confirmation output root is outside the study namespace")
    if require_slurm:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            raise RuntimeError("confirmation execution requires Slurm")
        scheduler_record = slurm_job(job_id)
        if " Requeue=0 " not in f" {scheduler_record} ":
            raise RuntimeError("confirmation task must disable Slurm requeue")
        if " Restarts=0 " not in f" {scheduler_record} ":
            raise RuntimeError("confirmation task cannot execute after a restart")
    runtime = {
        "contract_sha256": contract_hash,
        "gpu_environment": str(gpu_environment),
        "output_root": str(output_root),
        "task_id": str(contract["task_id"]),
        "expected_git_sha": expected_sha,
    }
    return contract, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--shell-output", type=Path, required=True)
    args = parser.parse_args()
    _, runtime = verify_contract(args.contract.resolve(strict=True))
    args.shell_output.write_text(
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
