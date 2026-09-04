#!/usr/bin/env python3
"""Render one authorized phase-35 reconstruction training job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.slurm.verify_reconstruction_phase35_contract import (  # noqa: E402
    CONTRACT_VERSION,
    CAMPAIGN_SUBMISSION_RECEIPT,
    IMPLEMENTATION_TAG,
    OUTPUT_NAMESPACE,
    PHASE35_SOURCE_FILES,
    PHASE35_TEST_RECEIPT,
    PREREGISTRATION,
    REPAIR_PREREGISTRATION,
    ROOT,
    STUDY_ID,
    load_evaluation_cohort,
    load_preregistration,
    load_repair_preregistration,
    load_validation_exclusions,
    phase35_contract_relative_path,
    phase35_task_id,
    repo_path,
    resolved_arm_config,
    safe_repo_output_path,
    sha256,
    verify_clean_worktree,
    verify_test_receipt,
)


SOURCE_FILES = PHASE35_SOURCE_FILES


def command_output(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {command!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def hashed(value: str) -> dict[str, str]:
    path = repo_path(value)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def live_slurm() -> dict[str, Any]:
    snapshot = command_output(
        ("/opt/slurm/bin/sinfo", "-h", "-p", "inter", "-N", "-o", "%N|%T|%G")
    )
    usable = [
        line
        for line in snapshot.splitlines()
        if "gpu:h100nvl:" in line
        and not any(
            state in line.lower()
            for state in ("down", "drain", "fail", "maint")
        )
    ]
    if not usable:
        raise RuntimeError("no live H100 NVL node exists in partition inter")
    return {
        "version": command_output(("/opt/slurm/bin/sbatch", "--version")),
        "sinfo_snapshot": snapshot.splitlines(),
        "selected_exact_gres": "gpu:h100nvl:1",
        "capacity_note": (
            "node liveness is verified here; queue admission and allocation "
            "are verified by Slurm and the runtime contract"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-env",
        type=Path,
        default=Path(
            "/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"
        ),
    )
    args = parser.parse_args()
    expected_output_relative = phase35_contract_relative_path(args.arm)
    output = safe_repo_output_path(
        args.output,
        expected_relative=expected_output_relative,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite phase35 contract: {output}")
    head = command_output(("git", "rev-parse", "HEAD"))
    if head != args.expected_git_sha:
        raise RuntimeError("expected phase35 Git SHA is not checked out")
    if args.expected_git_tag != IMPLEMENTATION_TAG:
        raise RuntimeError("unexpected phase35 repair implementation tag")
    if command_output(("git", "rev-list", "-n", "1", args.expected_git_tag)) != head:
        raise RuntimeError("expected phase35 tag does not resolve to HEAD")
    verify_clean_worktree()

    preregistration = load_preregistration()
    repair_preregistration = load_repair_preregistration()
    runtime_binding = preregistration["runtime_binding"]
    if (
        str(args.gpu_env) != runtime_binding["gpu_environment"]
        or sha256(args.gpu_env / "bin/python")
        != runtime_binding["python_sha256"]
        or sha256(args.gpu_env / "pyvenv.cfg")
        != runtime_binding["pyvenv_cfg_sha256"]
    ):
        raise RuntimeError("phase35 exact GPU environment binding changed")
    config = resolved_arm_config(preregistration, args.arm)
    exclusions = load_validation_exclusions(preregistration)
    if len(exclusions) != preregistration["validation_exclusion"]["event_uid_count"]:
        raise RuntimeError("phase35 validation exclusion count changed")
    load_evaluation_cohort(preregistration, exclusions=exclusions)
    source = preregistration["source_checkpoint"]
    checkpoint = repo_path(source["path"], suffix=".pt")
    if sha256(checkpoint) != source["sha256"]:
        raise RuntimeError("phase35 source checkpoint changed")
    baseline = preregistration["baseline_reconstruction_checkpoint"]
    baseline_checkpoint = repo_path(baseline["path"], suffix=".pt")
    if sha256(baseline_checkpoint) != baseline["sha256"]:
        raise RuntimeError("phase35 baseline reconstruction checkpoint changed")
    data = preregistration["data_binding"]
    if (
        sha256(repo_path(data["selection_manifest"]))
        != data["selection_manifest_sha256"]
        or sha256(repo_path(data["dataset_index"]))
        != data["dataset_index_sha256"]
    ):
        raise RuntimeError("phase35 data binding changed")
    receipt = safe_repo_output_path(
        args.test_receipt,
        expected_relative=PHASE35_TEST_RECEIPT,
        require_file=True,
    )
    verify_test_receipt(
        receipt,
        expected_git_sha=head,
        expected_git_tag=args.expected_git_tag,
    )
    task_id = phase35_task_id(args.arm)
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "study_id": STUDY_ID,
        "task_id": task_id,
        "mode": "production",
        "experiment": task_id,
        "arm_role": args.arm,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_git_sha": head,
        "expected_git_tag": args.expected_git_tag,
        "gpu_environment": str(args.gpu_env),
        "device": "cuda",
        "preregistration": hashed(PREREGISTRATION),
        "repair_preregistration": hashed(REPAIR_PREREGISTRATION),
        "checkpoint": source["path"],
        "checkpoint_sha256": source["sha256"],
        "checkpoint_step": source["step"],
        "data": data,
        "config": config,
        "resources": repair_preregistration["execution_policy"],
        "validation_exclusion": preregistration["validation_exclusion"],
        "evaluation_cohort": preregistration["evaluation_cohort"],
        "hierarchy_supervision": repair_preregistration[
            "hierarchy_supervision"
        ],
        "sealed_test_role_access": "forbidden",
        "source_checkpoint_mutation": "forbidden",
        "automatic_promotion": False,
        "output_root": f"{OUTPUT_NAMESPACE}/{args.arm}",
        "test_receipt": {
            "path": str(receipt.relative_to(ROOT)),
            "sha256": sha256(receipt),
        },
        "authorization": {
            "basis": "explicit_user_operator_instruction_2026-09-04",
            "training_authorized": True,
            "validation_payload_access_authorized": True,
            "execution_authorized": True,
            "scheduler_authorized": True,
            "submission_authorized": True,
            "scientific_validation_authorized": True,
            "source_checkpoint_mutation_authorized": False,
            "sealed_test_access_authorized": False,
            "promotion_authorized": False,
        },
        "submission_authorized": True,
        "submission_performed": False,
        "campaign_submission_receipt": CAMPAIGN_SUBMISSION_RECEIPT,
        "repair_lineage": {
            "repair_preregistration": hashed(REPAIR_PREREGISTRATION),
            "parent_study_id": repair_preregistration["parent_study_id"],
            "parent_git_sha": repair_preregistration["parent_git_sha"],
            "parent_git_tag": repair_preregistration["parent_git_tag"],
            "clean_restart_required": True,
            "restart_source_checkpoint": repair_preregistration[
                "restart_source_checkpoint"
            ],
            "resume_from_failed_attempt_authorized": False,
            "reuse_failed_output_authorized": False,
            "excluded_completed_arm_role": (
                repair_preregistration["excluded_completed_arm_role"]
            ),
            "excluded_completed_arm_rerun_authorized": False,
        },
        "live_slurm": live_slurm(),
    }
    paths = [
        *SOURCE_FILES,
        source["path"],
        baseline["path"],
        data["selection_manifest"],
        data["dataset_index"],
        preregistration["validation_exclusion"]["manifest"],
        preregistration["evaluation_cohort"]["manifest"],
        preregistration["diagnostic_basis"]["full_decay_report"]["path"],
        preregistration["diagnostic_basis"]["lineage_receipt"]["path"],
        str(receipt.relative_to(ROOT)),
        repair_preregistration["parent_campaign_submission_receipt"]["path"],
    ]
    for historical_attempt in repair_preregistration["historical_attempts"]:
        paths.extend(
            historical_attempt[key]["path"]
            for key in ("contract", "attempt_receipt", "stderr")
        )
    paths.extend(
        report["path"]
        for report in repair_preregistration["step500_diagnostics"]["reports"]
    )
    exclusion_manifest = json.loads(
        repo_path(preregistration["validation_exclusion"]["manifest"]).read_text()
    )
    paths.extend(str(item["path"]) for item in exclusion_manifest["sources"])
    contract["hashed_inputs"] = [
        hashed(path) for path in dict.fromkeys(paths)
    ]
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "contract": str(output),
                "contract_sha256": contract["contract_sha256"],
                "direct_sbatch_forbidden": True,
                "campaign_submitter": (
                    "scripts/slurm/submit_reconstruction_phase35_campaign.py"
                ),
                "campaign_submission_receipt": CAMPAIGN_SUBMISSION_RECEIPT,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
