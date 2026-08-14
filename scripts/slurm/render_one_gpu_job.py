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

ACCELERATOR_PRIORITY = (
    "gpu:h200nvl:1",
    "gpu:h100nvl:1",
    "gpu:v100:1",
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


def validate_live_slurm(gres: str) -> dict[str, Any]:
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
    if not usable_priority or usable_priority[0] != gres:
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
            f"selected {gres} as the highest-priority exact usable GRES "
            "advertised by the live inter partition"
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
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--local-admission-receipt", type=Path)
    parser.add_argument("--local-completion-receipt", type=Path)
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

    expected_sha = args.expected_git_sha or _run(("git", "rev-parse", "HEAD")).strip()
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

    live = validate_live_slurm(args.gres)
    config = (
        args.diagnostic_config
        if args.mode == "diagnostic"
        else "configs/slurm/pretrain_035k_scientific.yaml"
    )
    dataset_index = (
        "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
        "train_035k.complete_only.index.json"
    )
    hashed_paths = (
        "configs/training_selection/production_1m_20260812/train_035k.json",
        dataset_index,
        config,
        "environment/gpu/requirements-cu126.lock",
        "environment/gpu/runtime-contract.json",
        "environment/gpu/environment-lock.sha256",
        "configs/training_selection/production_1m_20260812/training_readiness.json",
        "configs/training_selection/production_1m_20260812/provenance_status.json",
    )
    hashed_inputs = [_hashed_input(path) for path in hashed_paths]
    if admission_receipt is not None and completion_receipt is not None:
        hashed_inputs.extend(
            (_hashed_input(admission_receipt), _hashed_input(completion_receipt))
        )
    contract: dict[str, Any] = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "mode": args.mode,
        "label": (
            "NON-SCIENTIFIC DIAGNOSTIC; NO SCIENTIFIC CLAIMS"
            if args.mode == "diagnostic"
            else "SCIENTIFIC 35K PRETRAINING"
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
        or ("ht-nonsci-diag" if args.mode == "diagnostic" else "ht-pretrain-035k"),
        "seed": args.seed,
        "max_restarts": args.max_restarts,
        "dataset_index": dataset_index,
        "selection_manifest": hashed_paths[0],
        "sealed_test_role_access": "forbidden",
        "hashed_inputs": hashed_inputs,
        "live_slurm": live,
        "local_admission_receipt": (
            str(admission_receipt) if admission_receipt is not None else None
        ),
        "local_completion_receipt": (
            str(completion_receipt) if completion_receipt is not None else None
        ),
        "submission_performed": False,
        "submission_authorized": not args.blocked_no_submit,
        "verification_scope": (
            "blocked_no_submit"
            if args.blocked_no_submit
            else (
                "user_authorized_execution_with_recorded_provenance_limitations"
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
        "user_submission_authorization": (
            {
                "authorized": True,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source": "interactive_user_instruction",
                "scope": "single_35k_small_candidate_scientific_submission",
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
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/slurm").mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)

    walltime = "00:30:00" if args.mode == "diagnostic" else "12:00:00"
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
