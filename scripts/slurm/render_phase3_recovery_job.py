#!/usr/bin/env python
"""Render the immutable production-1M phase-3 recovery contract.

This wrapper fixes the recovery inputs and delegates to the generic renderer.
It only performs read-only scheduler inspection and writes the requested
contract; it never submits, cancels, requeues, or mutates a Slurm job.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CHECKPOINT_SHA256 = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
IMPLEMENTATION_TAG = "ht-pretraining-1m-phase3-recovery-implementation-v2-20260823"
CONFIG = "configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml"
EXPERIMENT = "ht-pretrain-1m-phase3-recovery-20260823"
ADMISSION = (
    "artifacts/experiment_readiness/production_1m_20260812/"
    "v100_nonfinite_resume_smoke_20260815_02/admission.json"
)
COMPLETION = (
    "artifacts/experiment_readiness/production_1m_20260812/"
    "v100_nonfinite_resume_smoke_20260815_02/completion.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument(
        "--expected-git-tag",
        default=IMPLEMENTATION_TAG,
        help="immutable annotated tag for the implementation commit",
    )
    parser.add_argument(
        "--gpu-env",
        default="/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/slurm/ht-pretrain-1m-phase3-recovery-20260823.operator-authorized.job-contract.json",
    )
    args = parser.parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts/slurm/render_one_gpu_job.py"),
        "--mode",
        "scientific",
        "--gres",
        "gpu:h100nvl:1",
        "--gpu-env",
        args.gpu_env,
        "--expected-git-sha",
        args.expected_git_sha,
        "--expected-git-tag",
        args.expected_git_tag,
        "--scientific-config",
        CONFIG,
        "--fullscale",
        "--seed",
        "20260812",
        "--max-restarts",
        "2",
        "--local-admission-receipt",
        str(ROOT / ADMISSION),
        "--local-completion-receipt",
        str(ROOT / COMPLETION),
        "--resume-checkpoint",
        CHECKPOINT,
        "--resume-checkpoint-step",
        "54064",
        "--resume-checkpoint-sha256",
        CHECKPOINT_SHA256,
        "--user-authorized-scientific-submit",
        "--experiment",
        EXPERIMENT,
        "--output",
        str(args.output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
