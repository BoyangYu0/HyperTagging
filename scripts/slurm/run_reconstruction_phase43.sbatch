#!/usr/bin/env bash
#SBATCH --account=others
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --no-requeue
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_reconstruction_phase43.sbatch /absolute/path/to/contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?phase43 entrypoint requires Slurm}"
: "${SLURM_SUBMIT_DIR:?Slurm submit directory is missing}"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || {
  echo "phase43 refuses restarted allocations" >&2
  exit 76
}

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
cd "${repo_root}"
contract_source="$1"
[[ "${contract_source}" = /* ]] || {
  echo "phase43 contract path must be absolute" >&2
  exit 64
}

evidence_root="artifacts/slurm/reconstruction-phase43/jobs/${SLURM_JOB_ID}"
[[ ! -e "${evidence_root}" && ! -L "${evidence_root}" ]] || {
  echo "refusing to reuse ${evidence_root}" >&2
  exit 73
}
mkdir -p "${evidence_root}/attempt-00"
attempt_root="${evidence_root}/attempt-00"
contract="${attempt_root}/submitted-contract.json"
stage_log="${attempt_root}/stages.log"
receipt="${attempt_root}/receipt.json"
started_at="$(date --utc --iso-8601=seconds)"
terminal_stage=bootstrap
run_root=""

stage() {
  terminal_stage="$1"
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" | tee -a "${stage_log}"
}

finalize() {
  status=$?
  trap - EXIT
  set +e
  stage "${terminal_stage}" "batch_exit_${status}"
  /usr/bin/python3 scripts/slurm/finalize_reconstruction_phase43_receipt.py \
    --receipt "${receipt}" \
    --contract "${contract}" \
    --stage-log "${stage_log}" \
    --run-root "${run_root:-${attempt_root}}" \
    --exit-status "${status}" \
    --terminal-stage "${terminal_stage}" \
    --started-at "${started_at}" || true
  exit "${status}"
}
trap finalize EXIT

stage bootstrap started
cp -- "${contract_source}" "${contract}"
chmod 0444 "${contract}"
gpu_environment="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu_environment"])' "${contract}")"
output_root="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "${contract}")"
"${gpu_environment}/bin/python" -c \
  'import pathlib,sys; from scripts.run_reconstruction_phase43 import verify_contract; verify_contract(pathlib.Path(sys.argv[1]))' \
  "${contract}"
stage contract_verification passed

date --utc --iso-8601=seconds > "${attempt_root}/allocation.txt"
hostname --fqdn >> "${attempt_root}/allocation.txt"
/opt/slurm/bin/scontrol show job -dd "${SLURM_JOB_ID}" >> "${attempt_root}/allocation.txt"
nvidia-smi -q > "${attempt_root}/nvidia-smi.txt"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres gpu:h100nvl:1 > "${attempt_root}/gpu-preflight.json"
stage gpu_preflight passed

run_root="${output_root}/${SLURM_JOB_ID}"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || {
  echo "refusing to reuse ${run_root}" >&2
  exit 73
}
mkdir -p "${run_root}/provenance"
cp "${contract}" "${run_root}/provenance/submitted-contract.json"
cp "${attempt_root}/allocation.txt" "${run_root}/provenance/allocation.txt"
cp "${attempt_root}/gpu-preflight.json" "${run_root}/provenance/gpu-preflight.json"

stage trainer started
"${gpu_environment}/bin/python" scripts/run_reconstruction_phase43.py \
  --contract "${contract}" \
  --output "${run_root}/result.json" \
  --training-output-dir "${run_root}/training"
stage trainer completed

stage full_decay_gates started
"${gpu_environment}/bin/python" scripts/run_reconstruction_phase43_full_decay.py \
  --contract "${contract}" \
  --training-result "${run_root}/result.json" \
  --training-dir "${run_root}/training" \
  --output-dir "${run_root}/full-decay-reports" \
  --output "${run_root}/full-decay-gates.json"
terminal_stage=full_decay_complete
stage full_decay_complete completed
exit 0
