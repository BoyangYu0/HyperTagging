#!/usr/bin/env bash
#SBATCH --account=others
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?CPU allocation required}"
: "${SLURM_ARRAY_TASK_ID:?Array task required}"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]
[[ "$#" == "2" && "$1" == /* && "$2" == /* ]]
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
cd "$1"
exec /project/agkuhr/users/boyang/envs/hypertagging-ci-20260909/bin/python \
  scripts/diagnose_search_survival.py run --plan "$2" --task-id "$SLURM_ARRAY_TASK_ID"
