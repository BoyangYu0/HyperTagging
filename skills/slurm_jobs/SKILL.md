# SLURM jobs for HyperTagging

## Purpose

How to create, review, test, and submit SLURM jobs for this repository.

## Rules

- Search existing submissions before creating new ones.
- Preserve cluster-specific conventions.
- Never run real training locally on GPU.
- Local GPU is only for tiny smoke tests after `squeue` and `nvidia-smi` checks.
- Real training and real datasets only via `sbatch`/SLURM.
- Do not call `sbatch` automatically unless explicitly instructed.
- Always provide dry-run render commands first.
- Always log git commit, config, environment, and GPU state.
- Prefer short walltime for debug jobs.
- Make account and partition configurable.
- Write logs to `logs/slurm`.
- Checkpoint to `outputs/checkpoints` or a configured path.

## Workflow

1. Inspect configs and scripts.
2. Render the job script.
3. Run `shellcheck` if available.
4. Run a CPU dry-run.
5. Optionally run a local tiny GPU smoke test after the safety guard.
6. Submit with `sbatch` only when explicitly requested.
7. Monitor with `squeue --me`.
8. Inspect logs and checkpoints.

## Template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=<job>
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
squeue --me
nvidia-smi
uv --cache-dir /tmp/uv-cache run python <script> --device cuda
```

## Commands

```bash
squeue --me
sbatch job.sh
scancel <jobid>
nvidia-smi
uv --cache-dir /tmp/uv-cache run pytest
python scripts/slurm/render_slurm_job.py --dry-run --command '<command>'
```
