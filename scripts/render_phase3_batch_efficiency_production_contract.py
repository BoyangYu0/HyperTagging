#!/usr/bin/env python
"""Render one post-calibration production contract without submitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CHECKPOINT_SHA256 = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
PARENT_AUTHORIZATION = (
    "artifacts/codex/ht_pretraining_1m_phase3_execution_authorization_20260823.json"
)
PARENT_AUTHORIZATION_SHA256 = "1af20420655a95aa7ce0a3d1ad4a6e357c7fe45510c3f8bafaf80ad3fdbb7991"
PARENT_AUTHORIZATION_CANONICAL_SHA256 = "c952524ce32b1c504cc6210cc8bc540bb6180a928c145fe29109b89b3fe3b5e3"
TOTAL_PRESENTATIONS = 1_730_048
MILESTONES = (13_516, 27_032, 40_548, 54_064, 67_580, 81_096, 94_612, 108_128)


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument(
        "--expected-git-tag",
        default="ht-pretraining-1m-phase3-batch-efficiency-implementation-20260823",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="ht-pretrain-1m-phase3-selected-20260823")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("production contract exists; refusing overwrite")
    selection = json.loads(args.selection.read_text())
    stored = selection.get("selection_sha256")
    body = dict(selection)
    body.pop("selection_sha256", None)
    if stored != _hash(body):
        raise RuntimeError("selection manifest hash mismatch")
    if selection.get("production_submission_authorized") is not True:
        raise RuntimeError("production rendering requires a new authorized calibration evidence artifact")
    if selection.get("submission_performed") is not False or selection.get("job_count") != 1:
        raise RuntimeError("selection does not describe exactly one unsubmitted production job")
    selected = str(selection.get("selected_profile", ""))
    if selected not in {"h100nvl", "v100"}:
        raise RuntimeError("selection profile is invalid")
    gres = "gpu:h100nvl:1" if selected == "h100nvl" else "gpu:v100:1"
    train_config = (
        "configs/slurm/pretrain_1m_phase3_batch_efficiency_h100nvl_20260823.yaml"
        if selected == "h100nvl"
        else "configs/slurm/pretrain_1m_phase3_batch_efficiency_v100_20260823.yaml"
    )
    data_selection = ROOT / "configs/training_selection/production_1m_20260812/train_865k.json"
    dataset_index = ROOT / "artifacts/experiment_readiness/production_1m_20260812/train_865k/train_865k.complete_only.index.json"
    config_path = ROOT / train_config
    contract: dict[str, Any] = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "batch_efficiency_contract_version": "ht-pretraining-1m-phase3-batch-efficiency-production-v1",
        "batch_efficiency_production": True,
        "implementation_git_sha": args.expected_git_sha,
        "expected_git_sha": args.expected_git_sha,
        "expected_git_tag": args.expected_git_tag,
        "experiment": args.experiment,
        "mode": "scientific",
        "verification_scope": "batch_efficiency_authorized_execution_with_provenance_exception",
        "submission_authorized": True,
        "export_policy": "NIL",
        "gpu_environment": "/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1",
        "seed": 20260812,
        "max_restarts": 2,
        "job_count": 1,
        "gres": gres,
        "selected_profile": selected,
        "train_config": train_config,
        "dataset_index": str(dataset_index.relative_to(ROOT)),
        "selection_manifest": str(data_selection.relative_to(ROOT)),
        "resume_checkpoint": CHECKPOINT,
        "resume_checkpoint_step": 54064,
        "resume_checkpoint_sha256": CHECKPOINT_SHA256,
        "total_presentations": TOTAL_PRESENTATIONS,
        "resume_presentations": 865_024,
        "remaining_presentations": 865_024,
        "virtual_step_presentations": 16,
        "validation_events": 5000,
        "validation_milestones_virtual_steps": list(MILESTONES),
        "validation_checkpoint_milestones_not_duplicated": list(MILESTONES[:4]),
        "objective_dominance_limit": 20.0,
        "late_phase_leaf_pid_weight": 0.4,
        "sealed_test_role_access": "forbidden",
        "stress_payload_access": "forbidden",
        "scientific_slurm_submission_allowed": False,
        "scientific_submission_blockers": [
            "missing source commit f4e54df23b5c60115e475c5d68df4651899d678e",
            "missing source tree b6e3a4118b960e3a4676a61af9601438d56cef96",
        ],
        "provenance_status": {
            "scientific_slurm_submission_allowed": False,
            "blockers": [
                "missing source commit f4e54df23b5c60115e475c5d68df4651899d678e",
                "missing source tree b6e3a4118b960e3a4676a61af9601438d56cef96",
            ],
        },
        "operator_authorization_parent": True,
        "parent_operator_authorization_artifact": PARENT_AUTHORIZATION,
        "parent_operator_authorization_sha256": PARENT_AUTHORIZATION_SHA256,
        "parent_operator_authorization_canonical_sha256": PARENT_AUTHORIZATION_CANONICAL_SHA256,
        "production_submission_authorized": True,
        "submission_performed": False,
        "calibration_selection_evidence": str(args.selection.resolve()),
        "hashed_inputs": [
            {"path": str(args.selection.resolve()), "sha256": _file_hash(args.selection)},
            {"path": str((ROOT / train_config).resolve()), "sha256": _file_hash(config_path)},
            {"path": str(data_selection.resolve()), "sha256": _file_hash(data_selection)},
            {"path": str(dataset_index.resolve()), "sha256": _file_hash(dataset_index)},
            {"path": str((ROOT / CHECKPOINT).resolve()), "sha256": CHECKPOINT_SHA256},
            {"path": str((ROOT / PARENT_AUTHORIZATION).resolve()), "sha256": PARENT_AUTHORIZATION_SHA256},
        ],
        "exactly_one_submission_command_required": True,
    }
    contract["contract_sha256"] = _hash(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
