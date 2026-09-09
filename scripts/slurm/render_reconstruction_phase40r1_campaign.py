#!/usr/bin/env python3
"""Render the tested, immutable two-arm phase-40r1 job contracts."""

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

from scripts.build_reconstruction_phase35_evaluation_cohort import (  # noqa: E402
    atomic_json,
    sha256,
)
from scripts.run_reconstruction_phase40r1 import (  # noqa: E402
    CONTRACT_VERSION,
    STUDY_ID,
    canonical_contract_hash,
)


PREREGISTRATION = ROOT / "configs/reconstruction/ht_reconstruction_phase40r1_20260908.json"
SOURCE_CHECKPOINT = ROOT / "runtime_inputs/reconstruction_phase34_20260904/checkpoint-step-81096.pt"
GPU_ENVIRONMENT = "/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"
SOURCE_FILES = (
    "configs/level_reconstruction.yaml",
    "configs/training_selection/production_1m_20260812/train_070k_phase40.json",
    "artifacts/experiment_readiness/reconstruction_phase40_20260907/train_070k.complete_only.index.json",
    "scripts/evaluate_full_decay.py",
    "scripts/build_reconstruction_phase40_data.py",
    "scripts/build_reconstruction_phase40r1_preregistration.py",
    "scripts/build_reconstruction_phase40_validation_cohort.py",
    "scripts/run_reconstruction_phase35.py",
    "scripts/run_reconstruction_phase35_full_decay.py",
    "scripts/run_reconstruction_phase40r1.py",
    "scripts/run_reconstruction_phase40r1_full_decay.py",
    "scripts/slurm/finalize_reconstruction_phase40r1_receipt.py",
    "scripts/slurm/render_reconstruction_phase40r1_campaign.py",
    "scripts/slurm/run_reconstruction_phase40r1.sbatch",
    "scripts/slurm/submit_reconstruction_phase40r1_campaign.py",
    "scripts/train_level_reconstruction.py",
    "src/hypertagging/reconstruction/hierarchical_inference.py",
    "src/hypertagging/reconstruction/level_rollout.py",
    "src/hypertagging/training/checkpoint_selection.py",
    "src/hypertagging/training/reconstruction_trainer.py",
    "tests/test_confidence_calibration_cpu.py",
    "tests/test_reconstruction_phase40r1_campaign_cpu.py",
)


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git failed: {args}")
    return result.stdout.strip()


def receipt_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--contract-output-dir", type=Path, required=True)
    parser.add_argument("--passed-test-count", type=int, required=True)
    parser.add_argument("--test-command", required=True)
    args = parser.parse_args()
    head = git("rev-parse", "HEAD")
    if git("rev-list", "-n", "1", args.expected_git_tag) != head:
        raise RuntimeError("phase40r1 tag does not bind current HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("phase40r1 tracked worktree must be clean")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    cohort_binding = prereg["untouched_validation_cohort"]
    cohort_path = ROOT / cohort_binding["path"]
    if sha256(cohort_path) != cohort_binding["sha256"]:
        raise RuntimeError("phase40r1 cohort changed after preregistration")

    permissions = dict(prereg["authority"])
    test_receipt = args.test_receipt.resolve()
    if test_receipt.exists():
        raise RuntimeError("refusing to overwrite phase40r1 test receipt")
    test_payload = {
        "receipt_version": "hypertagging-reconstruction-phase40r1-test-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "git_sha": head,
        "git_tag": args.expected_git_tag,
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "sha256": sha256(PREREGISTRATION),
        },
        "test_command": args.test_command,
        "passed_test_count": args.passed_test_count,
        "static_checks": [
            "Python compilation",
            "ruff --select F",
            "bash syntax",
            "JSON syntax",
            "git diff whitespace",
            "tracked worktree cleanliness",
        ],
        "permissions": permissions,
        "sealed_test_accessed": False,
    }
    test_payload["receipt_sha256"] = receipt_hash(test_payload)
    atomic_json(test_receipt, test_payload)

    output_dir = args.contract_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hashed_inputs = [
        {"path": path, "sha256": sha256(ROOT / path)} for path in SOURCE_FILES
    ]
    hashed_inputs.extend(
        [
            {"path": str(PREREGISTRATION.relative_to(ROOT)), "sha256": sha256(PREREGISTRATION)},
            {"path": cohort_binding["path"], "sha256": cohort_binding["sha256"]},
            {
                "path": prereg["data_binding"]["selection_manifest"],
                "sha256": prereg["data_binding"]["selection_manifest_sha256"],
            },
            {
                "path": prereg["data_binding"]["dataset_index"],
                "sha256": prereg["data_binding"]["dataset_index_sha256"],
            },
            {
                "path": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
                "sha256": sha256(SOURCE_CHECKPOINT),
            },
            {
                "path": str(test_receipt.relative_to(ROOT)),
                "sha256": sha256(test_receipt),
            },
        ]
    )
    outputs = []
    for arm in prereg["arms"]:
        role = arm["role"]
        config = {**prereg["common_training_contract"], **arm["overrides"]}
        contract = {
            "contract_version": CONTRACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "study_id": STUDY_ID,
            "task_id": f"phase40r1-{role}-20260908",
            "arm_role": role,
            "expected_git_sha": head,
            "expected_git_tag": args.expected_git_tag,
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "sha256": sha256(PREREGISTRATION),
            },
            "test_receipt": {
                "path": str(test_receipt.relative_to(ROOT)),
                "sha256": sha256(test_receipt),
                "receipt_sha256": test_payload["receipt_sha256"],
            },
            "cohort": {
                "path": cohort_binding["path"],
                "sha256": cohort_binding["sha256"],
            },
            "checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "checkpoint_step": 81096,
            "data": prereg["data_binding"],
            "config": config,
            "post_training_gates": prereg["post_training_gates"],
            "evaluation_contract": prereg["evaluation_contract"],
            "gpu_environment": GPU_ENVIRONMENT,
            "resources": prereg["execution_policy"],
            "output_root": f"artifacts/runs/ht-reconstruction-phase40r1-20260908/{role}",
            "hashed_inputs": hashed_inputs,
            "submission_authorized": True,
            "submission_performed": False,
            "source_checkpoint_mutation": "forbidden",
            "automatic_promotion": False,
            "longer_run_authorized": False,
            "promotion_authorized": False,
            "sealed_test_role_access": "forbidden",
            "sealed_test_request_authorized": False,
        }
        contract["contract_sha256"] = canonical_contract_hash(contract)
        output = output_dir / f"phase40r1_{role}_contract_20260908.json"
        if output.exists():
            raise RuntimeError(f"refusing to overwrite phase40r1 contract: {output}")
        atomic_json(output, contract)
        outputs.append(
            {
                "arm_role": role,
                "path": str(output),
                "sha256": sha256(output),
                "contract_sha256": contract["contract_sha256"],
            }
        )
    print(json.dumps({"git_sha": head, "contracts": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
