#!/usr/bin/env bash
#SBATCH --account=others
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_reconstruction_confirmation.sbatch /absolute/path/to/contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?confirmation entrypoint requires Slurm}"
: "${SLURM_SUBMIT_DIR:?Slurm submit directory is missing}"

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
cd "${repo_root}"
contract="$1"
[[ "${contract}" = /* ]] || { echo "contract path must be absolute" >&2; exit 64; }
attempt_root="artifacts/slurm/reconstruction-confirmation/jobs/${SLURM_JOB_ID}"
mkdir -p "${attempt_root}"
result="${attempt_root}/result.unavailable.json"
receipt="${attempt_root}/receipt.json"
telemetry="${attempt_root}/gpu-telemetry.jsonl"
telemetry_summary="${attempt_root}/gpu-telemetry-summary.json"
telemetry_pid=""

finalize() {
  batch_status=$?
  trap - EXIT
  set +e
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
  fi
  if [[ -n "${telemetry_pid}" ]]; then
    wait "${telemetry_pid}" || true
  fi
  /usr/bin/python3 scripts/slurm/finalize_reconstruction_confirmation_receipt.py \
    --contract "${contract}" --result "${result}" --receipt "${receipt}" \
    --telemetry "${telemetry}" --telemetry-summary "${telemetry_summary}" \
    --batch-exit-status "${batch_status}"
  exit "${batch_status}"
}
trap finalize EXIT

runtime_file="$(mktemp "${attempt_root}/.runtime.XXXXXX")"
/usr/bin/python3 scripts/slurm/verify_reconstruction_confirmation_contract.py \
  "${contract}" --shell-output "${runtime_file}"
# shellcheck disable=SC1090
source "${runtime_file}"
rm -f -- "${runtime_file}"

run_root="${output_root}/${SLURM_JOB_ID}"
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse ${run_root}" >&2; exit 73; }
mkdir -p "${run_root}/provenance"
cp "${contract}" "${run_root}/provenance/job-contract.json"
/opt/slurm/bin/scontrol show job -dd "${SLURM_JOB_ID}" > "${run_root}/provenance/allocation.txt"
nvidia-smi -q > "${run_root}/provenance/nvidia-smi.txt"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres gpu:h100nvl:1 > "${run_root}/provenance/gpu-preflight.json"

"${gpu_environment}/bin/python" scripts/slurm/monitor_gpu_telemetry.py \
  --output "${telemetry}" --summary "${telemetry_summary}" --interval-seconds 15 &
telemetry_pid=$!
result="${run_root}/result.json"

"${gpu_environment}/bin/python" scripts/evaluate_reconstruction_checkpoint_pair.py \
  --contract "${contract}" --output "${result}"
