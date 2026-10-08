#!/bin/bash
#SBATCH --account=others
#SBATCH --partition=agkuhr
#SBATCH --nodelist=gar-cn-etp02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# == 2 ]] || exit 64
: "${SLURM_JOB_ID:?Slurm required}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP LD_PRELOAD
python_exe=/project/agkuhr/users/boyang/envs/hypertagging-ci-20260909/bin/python
out="$($python_exe -c 'import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1]))["output_root"])/sys.argv[2])' "$1" "$2")"
exec "$python_exe" scripts/run_phase73_membership_development.py --plan "$1" --arm "$2" --output "$out"
