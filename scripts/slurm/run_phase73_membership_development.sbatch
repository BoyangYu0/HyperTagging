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
contract="$1"
arm="$2"
finalize() {
  status=$?
  trap - EXIT
  "$python_exe" - "$contract" "$arm" "$SLURM_JOB_ID" "$status" <<'PYRECEIPT'
import datetime,json,pathlib,sys
contract=pathlib.Path(sys.argv[1]);c=json.loads(contract.read_text());arm=sys.argv[2]
record={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':sys.argv[3],'arm':arm,'exit_status':int(sys.argv[4]),'source_sha':c['source_sha'],'status':'COMPLETED' if sys.argv[4]=='0' else 'FAILED','automatic_successor':False}
with (contract.parent/(arm+'-terminal.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
PYRECEIPT
  exit "$status"
}
trap finalize EXIT
"$python_exe" scripts/run_phase73_membership_development.py --plan "$1" --arm "$2" --output "$out"
