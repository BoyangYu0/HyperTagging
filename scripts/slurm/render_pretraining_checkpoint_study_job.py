#!/usr/bin/env python3
"""Render, but never submit, a fixed-validation checkpoint-comparison job."""

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
sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_pretraining_checkpoint_study_contract import (  # noqa: E402
    ALLOWED_GRES,
    ALLOWED_STEPS,
    CONTRACT_VERSION,
)


def _run(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"read-only command failed: {command!r}: {result.stderr}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("checkpoint-study input must remain inside the repository") from error
    return resolved


def _hashed(path: Path) -> dict[str, str]:
    resolved = _repo_path(path)
    return {"path": str(resolved.relative_to(ROOT)), "sha256": _sha256(resolved)}


def _live_slurm(gres: str) -> dict[str, Any]:
    snapshot = _run(("/opt/slurm/bin/sinfo", "-h", "-p", "inter", "-N", "-o", "%N|%T|%G"))
    gpu_type = gres.split(":")[1]
    if not any(
        f"gpu:{gpu_type}:" in line
        and not any(state in line.lower() for state in ("down", "drain", "fail", "maint"))
        for line in snapshot.splitlines()
    ):
        raise RuntimeError(f"exact GRES {gres} is not live in partition inter")
    return {
        "version": _run(("/opt/slurm/bin/sbatch", "--version")),
        "sinfo_snapshot": snapshot.splitlines(),
        "selected_exact_gres": gres,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gres", choices=sorted(ALLOWED_GRES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, choices=sorted(ALLOWED_STEPS), required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-env",
        type=Path,
        default=Path("/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite an existing checkpoint-study contract")
    if _run(("git", "rev-parse", "HEAD")) != args.expected_git_sha:
        raise RuntimeError("expected checkpoint-study source SHA is not checked out")
    if _run(("git", "rev-list", "-n", "1", args.expected_git_tag)) != args.expected_git_sha:
        raise RuntimeError("expected checkpoint-study tag does not resolve to source SHA")
    if _run(("git", "status", "--porcelain", "--untracked-files=no")):
        raise RuntimeError("tracked worktree must be clean before contract rendering")
    checkpoint = _repo_path(args.checkpoint)
    selection = _repo_path(args.selection_manifest)
    dataset_index = _repo_path(args.dataset_index)
    if checkpoint.name != f"checkpoint-step-{args.checkpoint_step}.pt":
        raise RuntimeError("checkpoint filename and requested step disagree")
    if not (args.gpu_env / "bin/python").is_file():
        raise RuntimeError("GPU environment is unavailable")
    source_files = (
        Path("scripts/evaluate_hyperbolic_pretrain.py"),
        Path("scripts/slurm/evaluate_pretraining_checkpoint_study.sbatch"),
        Path("scripts/slurm/verify_pretraining_checkpoint_study_contract.py"),
        Path("src/hypertagging/evaluation/pretraining_validation.py"),
        Path("src/hypertagging/training/fixed_validation.py"),
        Path("src/hypertagging/training/pretrain_trainer.py"),
        Path("environment/gpu/requirements-cu126.lock"),
        Path("environment/gpu/runtime-contract.json"),
    )
    role = {2188: "phase_2_anchor", 3282: "selected_best", 4376: "terminal"}[
        args.checkpoint_step
    ]
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "mode": "read_only_pretraining_checkpoint_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "account": "others",
        "partition": "inter",
        "gres": args.gres,
        "gpu_environment": str(args.gpu_env),
        "expected_git_sha": args.expected_git_sha,
        "expected_git_tag": args.expected_git_tag,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": args.checkpoint_step,
        "comparison_role": role,
        "selection_manifest": str(selection.relative_to(ROOT)),
        "dataset_index": str(dataset_index.relative_to(ROOT)),
        "evaluation_role": "validation",
        "validation_events": 2000,
        "selection_strategy": "manifest_validation_role_uid_hash",
        "sealed_test_role_access": "forbidden",
        "optimizer_steps": 0,
        "checkpoint_mutation": "forbidden",
        "training_artifact_writes": "forbidden",
        "evaluation_output_base": (
            f"artifacts/studies/pretraining-checkpoint-comparison/{args.experiment}"
        ),
        "resources": {"cpus": 8, "memory": "64G", "time": "02:00:00"},
        "hashed_inputs": [
            _hashed(selection),
            _hashed(dataset_index),
            *[_hashed(path) for path in source_files],
        ],
        "live_slurm": _live_slurm(args.gres),
        "submission_authorized": True,
        "submission_performed": False,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    command = (
        "sbatch",
        "--account=others",
        "--partition=inter",
        f"--gres={args.gres}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=02:00:00",
        "--export=NIL",
        f"--job-name={args.experiment}",
        "scripts/slurm/evaluate_pretraining_checkpoint_study.sbatch",
        str(args.output.resolve()),
    )
    print(json.dumps({"contract": str(args.output), "command": list(command)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
