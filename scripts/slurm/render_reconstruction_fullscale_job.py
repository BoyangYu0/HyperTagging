#!/usr/bin/env python3
"""Render a hash-bound reconstruction calibration or production Slurm job."""

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
PREREGISTRATION = (
    "configs/reconstruction/ht_reconstruction_transfer_preregistration_20260824.json"
)
CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CHECKPOINT_SHA256 = (
    "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
)
SELECTION = "configs/training_selection/production_1m_20260812/train_035k.json"
INDEX = (
    "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
    "train_035k.complete_only.index.json"
)
SOURCE_FILES = (
    "schemas/ht_reconstruction_fullscale_v1.schema.json",
    PREREGISTRATION,
    "configs/level_reconstruction.yaml",
    "scripts/slurm/verify_reconstruction_fullscale_contract.py",
    "scripts/slurm/render_reconstruction_fullscale_job.py",
    "scripts/slurm/run_reconstruction_fullscale.sbatch",
    "scripts/slurm/finalize_reconstruction_fullscale_receipt.py",
    "scripts/run_reconstruction_fullscale.py",
    "scripts/slurm/preflight_gpu_environment.py",
    "scripts/slurm/monitor_gpu_telemetry.py",
    "scripts/slurm/run_with_bounded_requeue.sh",
    "src/hypertagging/training/reconstruction_trainer.py",
    "src/hypertagging/training/pretrained_transfer.py",
    "src/hypertagging/training/checkpointing.py",
    "src/hypertagging/training/data_module.py",
    "src/hypertagging/training/learning_rate.py",
    "src/hypertagging/models/level_autoregressive.py",
    "src/hypertagging/losses/level_reconstruction.py",
    "environment/gpu/runtime-contract.json",
    "environment/gpu/requirements-cu126.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"read-only command failed: {command!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    return path


def hashed(value: str) -> dict[str, str]:
    path = repo_path(value)
    return {"path": value, "sha256": sha256(path)}


def live_slurm(gres: str) -> dict[str, Any]:
    snapshot = run(("/opt/slurm/bin/sinfo", "-h", "-p", "inter", "-N", "-o", "%N|%T|%G"))
    gpu_type = gres.split(":")[1]
    usable = [
        line
        for line in snapshot.splitlines()
        if f"gpu:{gpu_type}:" in line
        and not any(state in line.lower() for state in ("down", "drain", "fail", "maint"))
    ]
    if not usable:
        raise RuntimeError(f"exact GRES {gres} is not live in partition inter")
    return {
        "version": run(("/opt/slurm/bin/sbatch", "--version")),
        "sinfo_snapshot": snapshot.splitlines(),
        "selected_exact_gres": gres,
    }


def build_config(mode: str, gres: str, batch_size: int) -> dict[str, Any]:
    max_steps = 128 if mode == "calibration" else 70016 // batch_size
    if mode == "production" and max_steps * batch_size != 70016:
        raise RuntimeError("production batch size must divide the 70016 presentation target")
    return {
        "max_steps": max_steps,
        "lr_schedule_total_steps": max_steps,
        "presentations_target": max_steps * batch_size,
        "learning_rate": 0.001,
        "warmup_fraction": 0.05,
        "warmup_steps": None,
        "max_warmup_steps": 10000,
        "min_lr_ratio": 0.0,
        "batch_size": batch_size,
        "seed": 20260824,
        "checkpoint_every": max_steps if mode == "calibration" else 500,
        "validate_every": 129 if mode == "calibration" else 100,
        "rollout_validate_every": 129 if mode == "calibration" else 500,
        "max_validation_events": 2000,
        "rollout_validation_events": 1000,
        "validation_batch_size": 4,
        "validation_enabled": mode == "production",
        "target_policy": "complete_only",
        "scheduled_sampling_probability": 0.25,
        "scheduled_sampling_schedule": "linear",
        "scheduled_sampling_duration_steps": 1000,
        "freeze_pretrained_encoder_steps": max_steps,
        "transfer_leaf_pid_head": True,
        "freeze_leaf_pid_head_steps": max_steps,
        "model_preset": "small_candidate",
        "max_cardinality": 16,
        "max_cardinality_by_level": [[1, 16], [2, 16], [3, 16]],
        "object_positive_weight": 12.0,
        "pointer_positive_weight": 16.0,
        "best_metric": "predicted_edge_f1",
        "best_mode": "max",
        "initial_state_policy": "upsilon4s",
        "rollout_pid_kinematics_mode": "soft_decision_hard_construction",
        "rollout_pid_temperature": 0.5,
        "mixed_precision": True,
        "amp_dtype": "bfloat16" if gres == "gpu:h100nvl:1" else "float16",
        "grad_scaler_enabled": gres == "gpu:v100:1",
        "scientific_mode": mode == "production",
        "log_every": 10,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "production"), required=True)
    parser.add_argument("--gres", choices=("gpu:h100nvl:1", "gpu:v100:1"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-env", type=Path,
        default=Path("/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite reconstruction contract")
    if run(("git", "rev-parse", "HEAD")) != args.expected_git_sha:
        raise RuntimeError("expected reconstruction source SHA is not checked out")
    if run(("git", "rev-list", "-n", "1", args.expected_git_tag)) != args.expected_git_sha:
        raise RuntimeError("expected reconstruction tag does not resolve to source SHA")
    if run(("git", "status", "--porcelain", "--untracked-files=no")):
        raise RuntimeError("tracked worktree must be clean before contract rendering")
    if not (args.gpu_env / "bin/python").is_file():
        raise RuntimeError("GPU environment is unavailable")
    checkpoint = repo_path(CHECKPOINT)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("authorized source checkpoint hash changed")
    if args.batch_size not in {32, 64}:
        raise RuntimeError("calibration/production batch must be one of the measured ladder values 32/64")
    config = build_config(args.mode, args.gres, args.batch_size)
    output_root = (
        f"artifacts/runs/ht-reconstruction-calibration-20260824/{args.experiment}"
        if args.mode == "calibration"
        else "artifacts/runs/ht-reconstruction-transfer-fullscale-20260824"
    )
    contract: dict[str, Any] = {
        "contract_version": "hypertagging-reconstruction-fullscale-v1",
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "account": "others",
        "partition": "inter",
        "gres": args.gres,
        "gpu_environment": str(args.gpu_env),
        "expected_git_sha": args.expected_git_sha,
        "expected_git_tag": args.expected_git_tag,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_step": 54064,
        "selection_manifest": SELECTION,
        "dataset_index": INDEX,
        "training_role": "train",
        "evaluation_role": "validation",
        "validation_access_policy": "forbidden_during_calibration" if args.mode == "calibration" else "validation_only",
        "sealed_test_role_access": "forbidden",
        "source_checkpoint_mutation": "forbidden",
        "provenance_validation": {
            "transfer_classification": "exploratory_reconstruction_transfer",
            "pretraining_success_gate_passed": False,
            "intended_pretraining_contract": "1m_not_completed",
            "exception": "authorized_step_54064_is_the_advanced_immutable_loadable_finite_checkpoint",
        },
        "preregistration": hashed(PREREGISTRATION),
        "config": config,
        "resources": {
            "cpus": 8,
            "memory": "64G",
            "time": "00:15:00" if args.mode == "calibration" else "12:00:00",
            "max_restarts": 0 if args.mode == "calibration" else 1,
            "signal": None if args.mode == "calibration" else "B:USR1@300",
        },
        "max_wall_seconds": 900 if args.mode == "calibration" else 43200,
        "output_root": output_root,
        "hashed_inputs": [hashed(SELECTION), hashed(INDEX), *[hashed(path) for path in SOURCE_FILES]],
        "live_slurm": live_slurm(args.gres),
        "submission_authorized": True,
        "submission_performed": False,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    command = [
        "sbatch", "--account=others", "--partition=inter", f"--gres={args.gres}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
        f"--time={contract['resources']['time']}", "--export=NIL",
        f"--job-name={args.experiment}",
    ]
    if args.mode == "production":
        command.extend(["--signal=B:USR1@300", "--requeue"])
    command.extend(["scripts/slurm/run_reconstruction_fullscale.sbatch", str(args.output.resolve())])
    print(json.dumps({"contract": str(args.output), "contract_sha256": contract["contract_sha256"], "command": command}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
