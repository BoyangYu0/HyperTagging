#!/bin/bash
#SBATCH --account=others
#SBATCH --partition=agkuhr
#SBATCH --nodelist=gar-cn-etp02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# == 2 ]] || exit 64
: "${SLURM_JOB_ID:?Slurm required}"
export PATH=/project/agkuhr/users/boyang/envs/hypertagging-ci-20260909/bin:/usr/bin:/bin
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONSTARTUP LD_PRELOAD
contract="$1"
arm="$2"
finalize() {
  status=$?
  trap - EXIT
  python - "$contract" "$arm" "$SLURM_JOB_ID" "$status" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);c=json.loads(p.read_text())
record={'job_id':sys.argv[3],'exit_status':int(sys.argv[4]),'arm':sys.argv[2],
        'source_sha':c['source_sha'],'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'automatic_successor':False,'status':'COMPLETED' if sys.argv[4]=='0' else 'FAILED'}
with (p.parent/(sys.argv[2]+'-exit.json')).open('x') as f:json.dump(record,f,indent=2)
PY
  exit "$status"
}
trap finalize EXIT
python scripts/phase81_pair_supervision.py run --contract "$contract" --arm "$arm"
