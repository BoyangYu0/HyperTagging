#!/usr/bin/env bash
#SBATCH --account=others
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: train_one_gpu.sbatch /absolute/path/to/job-contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?this entrypoint may only run inside Slurm}"

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
mkdir -p artifacts/slurm
job_contract="$1"
[[ "${job_contract}" = /* ]] || {
  echo "job contract path must be absolute" >&2
  exit 64
}
runtime_dir="$(mktemp -d "artifacts/slurm/.verified-runtime.${SLURM_JOB_ID}.XXXXXX")"
runtime_file="${runtime_dir}/runtime.sh"
cleanup_runtime() {
  rm -f -- "${runtime_file}"
  rmdir -- "${runtime_dir}" 2>/dev/null || true
}
trap cleanup_runtime EXIT

/usr/bin/python3 scripts/slurm/verify_job_contract.py \
  "${job_contract}" --shell-output "${runtime_file}"
# shellcheck disable=SC1090
source "${runtime_file}"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres "${expected_gres}"

run_root="artifacts/runs/${experiment}/${seed}/${SLURM_JOB_ID}"
mkdir -p "${run_root}/checkpoints" "${run_root}/provenance"
cp "${job_contract}" "${run_root}/provenance/job-contract.json"
nvidia-smi -q > "${run_root}/provenance/nvidia-smi.txt"
env | sort > "${run_root}/provenance/environment.txt"

resume_args=()
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
  signal_checkpoint="${run_root}/signal-checkpoint.pt"
  [[ -f "${signal_checkpoint}" ]] || {
    echo "requeued job is missing its signal checkpoint" >&2
    exit 77
  }
  resume_args=(--resume "${signal_checkpoint}")
fi

cleanup_runtime
trap - EXIT
exec scripts/slurm/run_with_bounded_requeue.sh --max-restarts "${max_restarts}" -- \
  "${gpu_environment}/bin/python" scripts/train_hyperbolic_pretrain.py \
  --config "${train_config}" \
  --expected-gres "${expected_gres}" \
  --seed "${seed}" \
  --output-dir "${run_root}" \
  "${resume_args[@]}"
