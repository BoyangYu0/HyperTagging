#!/usr/bin/env python
"""Render and validate exact one-GPU sbatch commands without submitting them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DIAGNOSTIC_CONFIGS = (
    "configs/slurm/pretrain_diagnostic.yaml",
    "configs/slurm/pretrain_diagnostic_small_candidate.yaml",
)
SCIENTIFIC_CONFIGS = (
    "configs/slurm/pretrain_035k_scientific.yaml",
    "configs/slurm/pretrain_035k_h100_rerun_20260815.yaml",
    "configs/slurm/pretrain_1m_h100_20260821.yaml",
    "configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml",
)

ACCELERATOR_PRIORITY = (
    "gpu:h200nvl:1",
    "gpu:h100nvl:1",
    "gpu:v100:1",
)

OPERATOR_AUTHORIZATION_DATE = "2026-08-21"
OPERATOR_AUTHORIZATION_SOURCE = "interactive_user_instruction"
OPERATOR_AUTHORIZATION_SCOPE = (
    "exactly_one_production_1m_pretraining_job_on_gpu:h100nvl:1"
)
EXPECTED_MISSING_PROVENANCE_COMMIT = (
    "f4e54df23b5c60115e475c5d68df4651899d678e"
)
EXPECTED_MISSING_PROVENANCE_TREE = "b6e3a4118b960e3a4676a61af9601438d56cef96"

PHASE3_EXECUTION_AUTHORIZATION_ARTIFACT = (
    "artifacts/codex/ht_pretraining_1m_phase3_execution_authorization_20260823.json"
)
PHASE3_AUTHORIZATION_PARENT_CONTRACT = (
    "artifacts/slurm/ht-pretrain-1m-phase3-recovery-20260823.operator-authorized.job-contract.json"
)

from hypertagging.utils.gpu_safety import (  # noqa: E402
    ALLOWED_SLURM_GRES,
    load_local_microtest_completion_receipt,
)


def _run(command: tuple[str, ...]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"read-only Slurm command failed: {command!r}")
    return result.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_live_slurm(
    gres: str, *, require_highest_priority: bool = True
) -> dict[str, Any]:
    version = _run(("/opt/slurm/bin/sbatch", "--version")).strip()
    help_text = _run(("/opt/slurm/bin/sbatch", "--help"))
    required_flags = (
        "--account",
        "--partition",
        "--gres",
        "--signal",
        "--requeue",
        "--export",
    )
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(f"site sbatch lacks required flags: {missing}")
    # Slurm 23.02 at this site has no sbatch --test-only. Do not invent or call it.
    test_only_supported = "--test-only" in help_text
    sinfo = _run(("/opt/slurm/bin/sinfo", "-h", "-p", "inter", "-N", "-o", "%N|%T|%G"))
    gres_name = gres.removeprefix("gpu:").removesuffix(":1")
    sinfo_lines = sinfo.splitlines()
    if not any(f"gpu:{gres_name}:" in line for line in sinfo_lines):
        raise RuntimeError(f"exact GRES {gres!r} is not advertised in partition inter")
    usable_lines = [
        line
        for line in sinfo_lines
        if not any(
            state in line.lower()
            for state in ("down", "drain", "fail", "maint", "unknown")
        )
    ]
    usable_priority = [
        candidate
        for candidate in ACCELERATOR_PRIORITY
        if any(
            f"gpu:{candidate.removeprefix('gpu:').removesuffix(':1')}:" in line
            for line in usable_lines
        )
    ]
    if not usable_priority or (
        require_highest_priority and usable_priority[0] != gres
    ):
        raise RuntimeError(
            f"selected GRES {gres!r} violates live accelerator priority; "
            f"usable priority is {usable_priority}"
        )
    partition = _run(("/opt/slurm/bin/scontrol", "show", "partition", "inter"))
    if "PartitionName=inter" not in partition or "State=UP" not in partition:
        raise RuntimeError("partition inter is absent or not up")
    assoc = _run(
        (
            "/opt/slurm/bin/sacctmgr",
            "-nP",
            "show",
            "assoc",
            f"user={getpass.getuser()}",
            "account=others",
            "format=User,Account,Partition",
        )
    )
    if "|others|" not in assoc:
        raise RuntimeError("live Slurm association does not confirm account others")
    return {
        "version": version,
        "account": "others",
        "partition": "inter",
        "exact_gres": gres,
        "sbatch_test_only_supported": test_only_supported,
        "required_flags_verified": list(required_flags),
        "sinfo_snapshot": sinfo_lines,
        "user_requested_accelerator_priority": list(ACCELERATOR_PRIORITY),
        "usable_priority": usable_priority,
        "selection_reason": (
            (
                f"selected {gres} as the explicit authorized full-scale target; "
                "the exact GRES is live in partition inter"
            )
            if not require_highest_priority
            else (
                f"selected {gres} as the highest-priority exact usable GRES "
                "advertised by the live inter partition"
            )
        ),
    }


def _hashed_input(path: str | Path) -> dict[str, str]:
    resolved = _resolve(path)
    return {"path": str(path), "sha256": _sha256(resolved)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnostic", "scientific"), required=True)
    parser.add_argument("--gres", choices=ALLOWED_SLURM_GRES, required=True)
    parser.add_argument(
        "--gpu-env",
        default="/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1",
    )
    parser.add_argument("--expected-git-sha", default=None)
    parser.add_argument("--expected-git-tag")
    parser.add_argument("--experiment")
    parser.add_argument(
        "--diagnostic-config",
        choices=DIAGNOSTIC_CONFIGS,
        default=DIAGNOSTIC_CONFIGS[0],
    )
    parser.add_argument(
        "--scientific-config",
        choices=SCIENTIFIC_CONFIGS,
        default=SCIENTIFIC_CONFIGS[0],
    )
    parser.add_argument(
        "--fullscale",
        action="store_true",
        help="render the production-1m H100 contract",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--local-admission-receipt", type=Path)
    parser.add_argument("--local-completion-receipt", type=Path)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="repository-local checkpoint for the initial scientific attempt",
    )
    parser.add_argument(
        "--resume-checkpoint-step",
        type=int,
        help="exact optimizer step recorded by the initial scientific checkpoint",
    )
    parser.add_argument(
        "--resume-checkpoint-sha256",
        help="sha256 of the exact initial scientific checkpoint",
    )
    parser.add_argument(
        "--blocked-no-submit",
        action="store_true",
        help=(
            "render an exact scientific contract and command that the allocation "
            "prologue will refuse to execute while provenance blockers remain"
        ),
    )
    parser.add_argument(
        "--user-authorized-scientific-submit",
        action="store_true",
        help=(
            "record explicit interactive user authorization to execute scientific "
            "training while retaining unresolved provenance blockers"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocked_no_submit and args.mode != "scientific":
        raise RuntimeError("--blocked-no-submit is only valid for scientific contracts")
    if args.fullscale and args.mode != "scientific":
        raise RuntimeError("--fullscale requires scientific mode")
    if args.fullscale and not (
        args.blocked_no_submit or args.user_authorized_scientific_submit
    ):
        raise RuntimeError(
            "full-scale rendering requires blocked mode or explicit operator authorization"
        )
    if args.fullscale and args.gres != "gpu:h100nvl:1":
        raise RuntimeError("full-scale production is authorized only on gpu:h100nvl:1")
    if args.user_authorized_scientific_submit and args.mode != "scientific":
        raise RuntimeError(
            "--user-authorized-scientific-submit is only valid for scientific contracts"
        )
    if args.blocked_no_submit and args.user_authorized_scientific_submit:
        raise RuntimeError("scientific submission cannot be both blocked and authorized")
    if args.seed < 0:
        raise RuntimeError("seed must be non-negative")
    if not 0 <= args.max_restarts <= 10:
        raise RuntimeError("max_restarts must be between 0 and 10")

    admission_receipt = (
        _resolve(args.local_admission_receipt).resolve()
        if args.local_admission_receipt is not None
        else None
    )
    completion_receipt = (
        _resolve(args.local_completion_receipt).resolve()
        if args.local_completion_receipt is not None
        else None
    )
    if (admission_receipt is None) != (completion_receipt is None):
        raise RuntimeError("local microtest evidence requires both receipt paths")
    if admission_receipt is not None and completion_receipt is not None:
        load_local_microtest_completion_receipt(
            completion_receipt,
            admission_path=admission_receipt,
        )
    resume_checkpoint: Path | None = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = _resolve(args.resume_checkpoint).resolve()
        try:
            resume_checkpoint.relative_to(ROOT.resolve())
        except ValueError as error:
            raise RuntimeError("resume checkpoint must remain inside the repository") from error
        if not resume_checkpoint.is_file() or resume_checkpoint.suffix != ".pt":
            raise RuntimeError("resume checkpoint must be an existing .pt file")
        if args.mode != "scientific":
            raise RuntimeError("resume checkpoint rendering is scientific-only")
        if args.fullscale and (
            args.resume_checkpoint_step != 54064
            or args.resume_checkpoint_sha256
            != "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
        ):
            raise RuntimeError(
                "production-1m recovery must bind checkpoint step 54064 and its "
                "immutable sha256"
            )
        if args.resume_checkpoint_sha256 != _sha256(resume_checkpoint):
            raise RuntimeError("resume checkpoint sha256 does not match the file")
    elif args.resume_checkpoint_step is not None or args.resume_checkpoint_sha256:
        raise RuntimeError(
            "resume checkpoint step and sha256 require --resume-checkpoint"
        )

    expected_sha = args.expected_git_sha or _run(("git", "rev-parse", "HEAD")).strip()
    provenance_result: dict[str, Any] | None = None
    if args.mode == "scientific":
        if admission_receipt is None or completion_receipt is None:
            raise RuntimeError(
                "scientific render requires admitted and successful completion receipts"
            )
        status_command = [
            sys.executable,
            str(ROOT / "scripts/validate_training_provenance.py"),
        ]
        if not args.blocked_no_submit and not args.user_authorized_scientific_submit:
            status_command.append("--require-scientific-slurm-ready")
        status = subprocess.run(
            status_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if status.returncode != 0:
            raise RuntimeError(
                "scientific render blocked by provenance/readiness status"
            )
        provenance_result = json.loads(status.stdout)
        if args.blocked_no_submit and provenance_result.get(
            "scientific_slurm_submission_allowed"
        ):
            raise RuntimeError(
                "blocked no-submit render requires an active scientific blocker"
            )
        if args.fullscale:
            blockers = provenance_result.get("blockers", [])
            if provenance_result.get("scientific_slurm_submission_allowed") is not False:
                raise RuntimeError(
                    "full-scale contract requires the structural provenance gate to remain false"
                )
            if not any(
                EXPECTED_MISSING_PROVENANCE_COMMIT in str(blocker)
                and EXPECTED_MISSING_PROVENANCE_TREE in str(blocker)
                for blocker in blockers
            ):
                raise RuntimeError(
                    "full-scale contract lacks the exact unresolved provenance blocker"
                )
        if (
            not args.blocked_no_submit
            and not args.user_authorized_scientific_submit
            and not args.expected_git_tag
        ):
            raise RuntimeError(
                "scientific render requires an immutable expected Git tag"
            )
        if not (Path(args.gpu_env) / "bin/python").is_file():
            raise RuntimeError("frozen GPU environment has not been installed")

    live = (
        validate_live_slurm(args.gres)
        if not args.fullscale
        else validate_live_slurm(args.gres, require_highest_priority=False)
    )
    config = (
        args.diagnostic_config
        if args.mode == "diagnostic"
        else args.scientific_config
    )
    if args.fullscale:
        import yaml

        config_payload = yaml.safe_load(
            _resolve(config).read_text(encoding="utf-8")
        )
        dataset_index = str(config_payload["dataset_index"])
        selection_manifest = str(config_payload["data"])
    else:
        config_payload = {}
        dataset_index = (
            "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
            "train_035k.complete_only.index.json"
        )
        selection_manifest = (
            "configs/training_selection/production_1m_20260812/train_035k.json"
        )
    hashed_paths = (
        selection_manifest,
        dataset_index,
        config,
        "environment/gpu/requirements-cu126.lock",
        "environment/gpu/runtime-contract.json",
        "environment/gpu/environment-lock.sha256",
        "configs/training_selection/production_1m_20260812/training_readiness.json",
        "configs/training_selection/production_1m_20260812/provenance_status.json",
    )
    if args.fullscale and resume_checkpoint is not None:
        hashed_paths = hashed_paths + (
            "artifacts/slurm/ht-pretrain-production-1m-h100-20260821.operator-authorized.job-contract.json",
            PHASE3_EXECUTION_AUTHORIZATION_ARTIFACT,
        )
    hashed_inputs = [_hashed_input(path) for path in hashed_paths]
    if admission_receipt is not None and completion_receipt is not None:
        hashed_inputs.extend(
            (_hashed_input(admission_receipt), _hashed_input(completion_receipt))
        )
    if resume_checkpoint is not None:
        hashed_inputs.append(_hashed_input(resume_checkpoint))
    contract: dict[str, Any] = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "mode": args.mode,
        "label": (
            "NON-SCIENTIFIC DIAGNOSTIC; NO SCIENTIFIC CLAIMS"
            if args.mode == "diagnostic"
            else (
                "SCIENTIFIC PRODUCTION 1M PRETRAINING"
                if args.fullscale
                else "SCIENTIFIC 35K PRETRAINING"
            )
        ),
        "account": "others",
        "partition": "inter",
        "gres": args.gres,
        "export_policy": "NIL",
        "expected_git_sha": expected_sha,
        "expected_git_tag": args.expected_git_tag,
        "gpu_environment": args.gpu_env,
        "gpu_environment_present_at_render": (
            Path(args.gpu_env) / "bin/python"
        ).is_file(),
        "train_config": config,
        "experiment": args.experiment
        or (
            "ht-nonsci-diag"
            if args.mode == "diagnostic"
            else (
                "ht-pretrain-production-1m-phase3-recovery-20260823"
                if args.fullscale and resume_checkpoint is not None
                else (
                    "ht-pretrain-production-1m-h100-20260821"
                    if args.fullscale
                    else "ht-pretrain-035k"
                )
            )
        ),
        "seed": args.seed,
        "max_restarts": args.max_restarts,
        "fullscale": bool(args.fullscale),
        "initialization_policy": (
            "exact_resume_from_checkpoint"
            if args.fullscale and resume_checkpoint is not None
            else "from_scratch"
            if args.fullscale
            else (
                "resume_checkpoint"
                if resume_checkpoint is not None
                else "from_scratch"
            )
        ),
        "partition_max_time": "2-00:00:00" if args.fullscale else None,
        "checkpoint_resume_policy": (
            {
                "signal": "USR1",
                "signal_seconds_before_limit": 300,
                "checkpoint_at_optimizer_boundary": True,
                "pending_validation_serialized": True,
                "requeue_uses_signal_checkpoint": True,
                "bounded_max_restarts": args.max_restarts,
                "no_silent_restart": True,
                "no_double_counting": True,
            }
            if args.fullscale
            else None
        ),
        "resource_contract": {
            "gres": args.gres,
            "cpus_per_task": 8,
            "memory": "64G",
            "requested_time": "2-00:00:00" if args.fullscale else "12:00:00",
            "partition": "inter",
        },
        "output_contract": {
            "run_root_template": "artifacts/runs/{experiment}/{seed}/{slurm_job_id}",
            "attempt_root_template": "artifacts/slurm/jobs/{slurm_job_id}/attempt-{restart_count:02d}",
            "contract_copy": "provenance/job-contract.json",
            "required_attempt_receipt": "receipt.json",
            "required_metrics": "metrics.jsonl",
            "required_checkpoint": "checkpoint.pt",
            "required_signal_checkpoint": "signal-checkpoint.pt",
            "no_silent_overwrite": True,
        },
        "dataset_index": dataset_index,
        "selection_manifest": selection_manifest,
        "sealed_test_role_access": "forbidden",
        "hashed_inputs": hashed_inputs,
        "live_slurm": live,
        "local_admission_receipt": (
            str(admission_receipt) if admission_receipt is not None else None
        ),
        "local_completion_receipt": (
            str(completion_receipt) if completion_receipt is not None else None
        ),
        "resume_checkpoint": (
            str(resume_checkpoint.relative_to(ROOT.resolve()))
            if resume_checkpoint is not None
            else None
        ),
        "submission_performed": False,
        "submission_authorized": not args.blocked_no_submit,
        "execution_authorization": {
            "execution_authorized": not args.blocked_no_submit,
            "basis": (
                "operator_provenance_exception"
                if args.fullscale and args.user_authorized_scientific_submit
                else (
                    "blocked_no_submit"
                    if args.blocked_no_submit
                    else "scientific_provenance_gate"
                )
            ),
        },
        "verification_scope": (
            "blocked_no_submit"
            if args.blocked_no_submit
            else (
                "operator_authorized_execution_with_provenance_exception"
                if args.fullscale and args.user_authorized_scientific_submit
                else "user_authorized_execution_with_recorded_provenance_limitations"
                if args.user_authorized_scientific_submit
                else "execution"
            )
        ),
        "scientific_submission_blockers": (
            provenance_result.get("blockers", [])
            if args.mode == "scientific"
            and (args.blocked_no_submit or args.user_authorized_scientific_submit)
            else []
        ),
        "provenance_status": provenance_result,
        "provenance_validation": (
            {
                "status": provenance_result.get("status"),
                "scientific_slurm_submission_allowed": provenance_result.get(
                    "scientific_slurm_submission_allowed"
                ),
                "blockers": provenance_result.get("blockers", []),
                "expected_missing_source_commit": EXPECTED_MISSING_PROVENANCE_COMMIT,
                "expected_missing_source_tree": EXPECTED_MISSING_PROVENANCE_TREE,
                "execution_authorization_does_not_modify_validator": True,
            }
            if args.mode == "scientific"
            else None
        ),
        "user_submission_authorization": (
            {
                "authorized": True,
                "recorded_at": (
                    OPERATOR_AUTHORIZATION_DATE
                    if args.fullscale
                    else datetime.now(timezone.utc).isoformat()
                ),
                "authorization_date": (
                    OPERATOR_AUTHORIZATION_DATE if args.fullscale else None
                ),
                "source": OPERATOR_AUTHORIZATION_SOURCE,
                "scope": (
                    OPERATOR_AUTHORIZATION_SCOPE
                    if args.fullscale
                    else "single_35k_small_candidate_scientific_submission"
                ),
                "job_count": 1 if args.fullscale else None,
                "gres": args.gres if args.fullscale else None,
                "provenance_limitations_must_remain_recorded": True,
            }
            if args.user_authorized_scientific_submit
            else None
        ),
        "accelerator_selection": {
            "priority": list(ACCELERATOR_PRIORITY),
            "selected": args.gres,
            "live_usable_priority": live.get("usable_priority", []),
            "reason": live.get("selection_reason", "live exact-GRES validation passed"),
        },
        "operator_provenance_exception": None,
    }
    if args.fullscale and resume_checkpoint is not None:
        contract["resume_checkpoint_sha256"] = args.resume_checkpoint_sha256
        # The recovery contract is rendered from the implementation commit and
        # may then be committed together with its immutable contract/report
        # artifacts.  The verifier requires the final worktree to descend from
        # this exact source commit; it never permits an unrelated source.
        contract["implementation_git_sha"] = expected_sha
        contract["artifact_git_sha"] = expected_sha
        contract["recovery_lineage"] = {
            "kind": "production_1m_phase3_exact_resume_v1",
            "historical_job_id": "15933802",
            "historical_experiment": "ht-pretrain-production-1m-h100-20260821",
            "historical_commit": "93b71c5d7c1bc20181640aafb4e918abb9267362",
            "historical_tag": "ht-pretraining-production-1m-h100-operator-authorized-20260821",
            "historical_contract": {
                "path": "artifacts/slurm/ht-pretrain-production-1m-h100-20260821.operator-authorized.job-contract.json",
                "canonical_sha256": "2af2b8fc51c7f1bceb26e5013c822967316a0f2b1d09671eb8b10fc0e8fd3406",
                "file_sha256": "8dfa6b2320c8992e69c68f7d570bcb0e562306b928be57c2ece0c8f8626f5a0d",
            },
            "source_checkpoint": str(
                resume_checkpoint.relative_to(ROOT.resolve())
            ),
            "source_checkpoint_step": args.resume_checkpoint_step,
            "source_checkpoint_sha256": args.resume_checkpoint_sha256,
            "source_checkpoint_bytes": resume_checkpoint.stat().st_size,
            "source_checkpoint_unchanged": True,
            "historical_attempt_root": "artifacts/slurm/jobs/15933802/attempt-00",
            "replacement_experiment": contract["experiment"],
            "replacement_attempt_root_must_not_be": (
                "artifacts/slurm/jobs/15933802/attempt-00"
            ),
        }
    if args.fullscale:
        validation_events = int(config_payload["validation_events"])
        batch_size = int(config_payload["batch_size"])
        contract.update(
            {
                "expected_optimizer_steps": int(config_payload["max_steps"]),
                "expected_train_events": 865000,
                "expected_validation_events": validation_events,
                "expected_presentations": int(config_payload["max_steps"])
                * batch_size,
                "expected_two_pool_presentations": 2 * 865000,
                "presentation_excess_over_two_pools": 48,
                "expected_curriculum_phase_steps": list(
                    config_payload["curriculum_phase_steps"]
                ),
                "expected_validation_interval_steps": int(
                    config_payload["validate_every"]
                ),
                "expected_checkpoint_interval_steps": int(
                    config_payload["checkpoint_every"]
                ),
                "expected_validation_batches": int(
                    config_payload["validation_batches"]
                ),
                "validation_final_partial_batch_events": validation_events % batch_size,
                "learning_rate_schedule": {
                    "type": "linear_warmup_cosine_decay",
                    "total_steps": int(config_payload["lr_schedule_total_steps"]),
                    "learning_rate": float(config_payload["learning_rate"]),
                    "warmup_fraction": float(config_payload["warmup_fraction"]),
                    "warmup_steps": 10000,
                    "max_warmup_steps": int(config_payload["max_warmup_steps"]),
                    "min_lr_ratio": float(config_payload["min_lr_ratio"]),
                    "minimum_learning_rate": float(config_payload["learning_rate"])
                    * float(config_payload["min_lr_ratio"]),
                },
                "fullscale_gpu_preflight": {
                    "status": "pending_in_allocation",
                    "required_gres": "gpu:h100nvl:1",
                    "required_runtime_lock": "environment/gpu/requirements-cu126.lock",
                    "submission_blocked_until_pass": True,
                },
                "expected_validation_checkpoint_milestones": [
                    {
                        "kind": kind,
                        "step": step,
                    }
                    for step in (
                        13516,
                        27032,
                        40548,
                        54064,
                        67580,
                        81096,
                        94612,
                        108128,
                    )
                    for kind in ("validation", "checkpoint")
                ],
                "stage_gate_override": {
                    "status": "operator_directed_fullscale_advancement",
                    "overrides_repository_plan": "100k/250k promotion sequence",
                    "technical_and_scientific_gates_preserved": True,
                    "limitation": (
                        "This override does not waive provenance, data-role, UID/source, "
                        "finite-gradient, objective, checkpoint, runtime, or contract gates; "
                        "the separate operator provenance exception authorizes only this "
                        "exact single production-1M H100 execution scope and does not alter "
                        "the validator or the missing object/tree."
                    ),
                },
            }
        )
        if args.user_authorized_scientific_submit:
            limitation = str(provenance_result["blockers"][0])
            contract["operator_provenance_exception"] = {
                "status": "explicit_operator_authorized_exception",
                "authorization_date": OPERATOR_AUTHORIZATION_DATE,
                "source": OPERATOR_AUTHORIZATION_SOURCE,
                "scope": OPERATOR_AUTHORIZATION_SCOPE,
                "job_count": 1,
                "gres": "gpu:h100nvl:1",
                "execution_authorized": True,
                "structural_provenance_validation": {
                    "status": provenance_result["status"],
                    "scientific_slurm_submission_allowed": False,
                    "validator_unchanged": True,
                    "missing_source_commit": EXPECTED_MISSING_PROVENANCE_COMMIT,
                    "missing_source_tree": EXPECTED_MISSING_PROVENANCE_TREE,
                },
                "limitation": limitation,
            }
    if args.fullscale and resume_checkpoint is not None:
        contract.update(
            {
                "execution_authorization_artifact": PHASE3_EXECUTION_AUTHORIZATION_ARTIFACT,
                "authorization_artifact_sha256": _sha256(
                    _resolve(PHASE3_EXECUTION_AUTHORIZATION_ARTIFACT)
                ),
                "authorization_parent_contract": PHASE3_AUTHORIZATION_PARENT_CONTRACT,
                "authorization_parent_contract_file_sha256": _sha256(
                    _resolve(PHASE3_AUTHORIZATION_PARENT_CONTRACT)
                ),
                "fresh_in_allocation_preflight_required": True,
                "gpu_pilot_completed": False,
            }
        )
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/slurm").mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)

    walltime = (
        "00:30:00"
        if args.mode == "diagnostic"
        else ("2-00:00:00" if args.fullscale else "12:00:00")
    )
    job_name = str(contract["experiment"])
    command = [
        "/opt/slurm/bin/sbatch",
        "--account=others",
        "--partition=inter",
        f"--gres={args.gres}",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={walltime}",
        "--signal=B:USR1@300",
        "--requeue",
        f"--job-name={job_name}",
        "--export=NIL",
        "scripts/slurm/train_one_gpu.sbatch",
        str(args.output.resolve()),
    ]
    print(
        json.dumps({"contract": str(args.output), "sbatch_command": command}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
