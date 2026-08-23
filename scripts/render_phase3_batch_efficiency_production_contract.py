#!/usr/bin/env python
"""Render one immutable post-calibration production contract, never submit it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.phase3_parallel_study import (  # noqa: E402
    CHECKPOINT_SHA256,
    TOTAL_PRESENTATIONS,
    assert_no_active_calibrations,
    canonical_hash,
    claim_production_contract,
    file_sha256,
    load_study_plan,
    resolve_plan_path,
)

PARENT_AUTHORIZATION = (
    "artifacts/codex/ht_pretraining_1m_phase3_execution_authorization_20260823.json"
)
PARENT_AUTHORIZATION_SHA256 = "1af20420655a95aa7ce0a3d1ad4a6e357c7fe45510c3f8bafaf80ad3fdbb7991"
PARENT_AUTHORIZATION_CANONICAL_SHA256 = "c952524ce32b1c504cc6210cc8bc540bb6180a928c145fe29109b89b3fe3b5e3"
MILESTONES = (13_516, 27_032, 40_548, 54_064, 67_580, 81_096, 94_612, 108_128)


def _hash(payload: dict[str, Any]) -> str:
    return canonical_hash(payload)


def _verify_failed_production_job(job_id: str, expected_job_name: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "/opt/slurm/bin/sacct",
            "-X",
            "-P",
            "-n",
            "-j",
            job_id,
            "-o",
            "JobID,JobName,State,ExitCode",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot verify the failed production source job")
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 4 and fields[0] == job_id and fields[1] == expected_job_name:
            if fields[2] == "FAILED" and fields[3] != "0:0":
                return {
                    "job_id": fields[0],
                    "job_name": fields[1],
                    "state": fields[2],
                    "exit_code": fields[3],
                }
    raise RuntimeError("production retry source is not proven terminally failed")


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
    parser.add_argument("--retry-of-job-id")
    parser.add_argument("--retry-reason")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("production contract exists; refusing overwrite")
    selection = json.loads(args.selection.read_text())
    stored = selection.get("selection_sha256")
    body = dict(selection)
    body.pop("selection_sha256", None)
    if stored != _hash(body):
        raise RuntimeError("selection manifest hash mismatch")
    if selection.get("artifact_version") != "ht-pretraining-1m-phase3-parallel-selection-v1":
        raise RuntimeError("selection is not the parallel-study selection version")
    if selection.get("production_submission_authorized") is not True:
        raise RuntimeError("production rendering requires explicit post-calibration authorization")
    if (
        selection.get("submission_performed") is not False
        or selection.get("job_count") != 1
        or selection.get("production_resume_policy") != "default_exactly_one"
        or selection.get("production_variant_count") != 1
        or selection.get("duplicate_production_contracts_forbidden") is not True
    ):
        raise RuntimeError("selection does not describe exactly one authorized production resume")
    variants = selection.get("production_variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise RuntimeError("production continuation was not explicitly enumerated exactly once")
    variant = variants[0]
    if variant.get("duplicate_of") is not None or variant.get("scientifically_distinct") is not True:
        raise RuntimeError("production continuation is duplicate or not scientifically enumerated")
    plan_path = resolve_plan_path(selection.get("study_plan", ""), root=ROOT)
    plan = load_study_plan(plan_path, root=ROOT)
    if selection.get("study_plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("selection study-plan hash mismatch")
    assert_no_active_calibrations(plan, root=ROOT)
    if selection.get("calibration_active_count") != 0:
        raise RuntimeError("selection was produced while calibration was active")
    receipt_aggregation_path = resolve_plan_path(
        selection.get("receipt_aggregation", ""), root=ROOT
    )
    if selection.get("receipt_aggregation_sha256") != file_sha256(receipt_aggregation_path):
        raise RuntimeError("selection receipt-aggregation hash mismatch")
    selected_id = str(selection.get("selected_calibration_id", ""))
    candidate = selection.get("candidates", {}).get(selected_id)
    if not isinstance(candidate, dict):
        raise RuntimeError("selected calibration candidate is missing")
    selected = str(selection.get("selected_profile", ""))
    batch_size = selection.get("selected_batch_size")
    if selected not in {"h100nvl", "v100"} or batch_size not in {32, 64}:
        raise RuntimeError("selection profile/batch is invalid")
    gres = "gpu:h100nvl:1" if selected == "h100nvl" else "gpu:v100:1"
    if candidate.get("exact_gres") != gres or candidate.get("batch_size") != batch_size:
        raise RuntimeError("selected candidate GRES or batch does not bind to the selection")
    if variant.get("selected_calibration_id") != selected_id:
        raise RuntimeError("production variant selects a different calibration")
    train_config = str(candidate.get("production_config", ""))
    config_path = resolve_plan_path(train_config, root=ROOT)
    if not config_path.is_file():
        raise RuntimeError("selected production config is not tracked")
    data_selection = ROOT / "configs/training_selection/production_1m_20260812/train_865k.json"
    dataset_index = ROOT / "artifacts/experiment_readiness/production_1m_20260812/train_865k/train_865k.complete_only.index.json"
    checkpoint = (
        "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
        "15933802/checkpoint-step-54064.pt"
    )
    production_identity = str(variant.get("production_contract_identity_sha256", ""))
    expected_identity = _hash(
        {
            "variant_id": variant["production_variant_id"],
            "selected_calibration_id": selected_id,
            "tuple_sha256": candidate["tuple_sha256"],
            "parent_implementation_commit": plan["parent_implementation_commit"],
            "resume_presentations": 865_024,
        }
    )
    if production_identity != expected_identity:
        raise RuntimeError("production contract identity hash mismatch")
    retry_source = None
    if args.retry_of_job_id is not None:
        if not args.retry_of_job_id.isdigit() or not args.retry_reason:
            raise RuntimeError("production retry requires a numeric failed job ID and reason")
        retry_source = _verify_failed_production_job(
            args.retry_of_job_id,
            f"ht3-production-resume-{variant['production_variant_id'].removeprefix('ht3-production-resume-')}",
        )
    contract: dict[str, Any] = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "batch_efficiency_contract_version": "ht-pretraining-1m-phase3-batch-efficiency-production-v2",
        "parallel_study_authorization_version": plan["artifact_version"],
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
        "batch_size": batch_size,
        "precision_policy": candidate["precision_policy"],
        "selected_calibration_id": selected_id,
        "production_variant_id": variant["production_variant_id"],
        "production_contract_identity_sha256": production_identity,
        "train_config": train_config,
        "dataset_index": str(dataset_index.relative_to(ROOT)),
        "selection_manifest": "configs/training_selection/production_1m_20260812/train_865k.json",
        "resume_checkpoint": checkpoint,
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
        "parallel_study_plan": (
            str(plan_path.relative_to(ROOT))
            if plan_path.is_relative_to(ROOT)
            else str(plan_path)
        ),
        "calibration_receipt_aggregation": selection["receipt_aggregation"],
        "calibration_active_count": 0,
        "hashed_inputs": [
            {"path": str(args.selection.resolve()), "sha256": file_sha256(args.selection)},
            {"path": str(receipt_aggregation_path), "sha256": file_sha256(receipt_aggregation_path)},
            {"path": str(config_path), "sha256": file_sha256(config_path)},
            {"path": str(data_selection.resolve()), "sha256": file_sha256(data_selection)},
            {"path": str(dataset_index.resolve()), "sha256": file_sha256(dataset_index)},
            {"path": str((ROOT / checkpoint).resolve()), "sha256": CHECKPOINT_SHA256},
            {"path": str((ROOT / PARENT_AUTHORIZATION).resolve()), "sha256": PARENT_AUTHORIZATION_SHA256},
        ],
        "exactly_one_submission_command_required": True,
        "duplicate_production_contracts_forbidden": True,
    }
    if retry_source is not None:
        contract["retry_of_slurm_job"] = retry_source
        contract["retry_reason"] = args.retry_reason
    claim_production_contract(
        plan,
        identity=production_identity,
        output_path=args.output,
        allow_existing_identity=retry_source is not None,
        root=ROOT,
    )
    contract["contract_sha256"] = _hash(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
