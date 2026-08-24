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
  echo "usage: run_reconstruction_fullscale.sbatch /absolute/path/to/job-contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?this entrypoint may only run inside Slurm}"
: "${SLURM_SUBMIT_DIR:?Slurm did not provide an absolute submit directory}"
[[ "${SLURM_SUBMIT_DIR}" = /* ]] || { echo "submit directory must be absolute" >&2; exit 64; }

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
cd "${repo_root}"
mkdir -p artifacts/slurm
job_contract="$1"
[[ "${job_contract}" = /* ]] || { echo "contract path must be absolute" >&2; exit 64; }
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "${restart_count}" =~ ^[0-9]+$ ]] || { echo "invalid restart count" >&2; exit 76; }

attempt_root="artifacts/slurm/reconstruction-fullscale/jobs/${SLURM_JOB_ID}/attempt-$(printf '%02d' "${restart_count}")"
mkdir -p "${attempt_root}"
stage_log="${attempt_root}/stages.log"
wrapper_status_file="${attempt_root}/wrapper-status.json"
receipt_file="${attempt_root}/receipt.json"
runtime_dir="$(mktemp -d "${attempt_root}/.verified-runtime.XXXXXX")"
runtime_file="${runtime_dir}/runtime.sh"
started_at="$(date --utc --iso-8601=seconds)"
terminal_stage=bootstrap
run_root=""
telemetry_pid=""

log_stage() {
  terminal_stage="$1"
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" | tee -a "${stage_log}"
}

stop_telemetry() {
  [[ -n "${telemetry_pid}" ]] || return 0
  if kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
  fi
  wait "${telemetry_pid}" || true
  telemetry_pid=""
}

finalize() {
  batch_status=$?
  trap - EXIT
  set +e
  stop_telemetry
  log_stage "${terminal_stage}" "batch_exit_${batch_status}"
  /usr/bin/python3 scripts/slurm/finalize_reconstruction_fullscale_receipt.py \
    --receipt "${receipt_file}" \
    --contract "${job_contract}" \
    --attempt-root "${attempt_root}" \
    --stage-log "${stage_log}" \
    --wrapper-status "${wrapper_status_file}" \
    --run-root "${run_root}" \
    --batch-exit-status "${batch_status}" \
    --terminal-stage "${terminal_stage}" \
    --started-at "${started_at}" \
    --completed-at "$(date --utc --iso-8601=seconds)"
  exit "${batch_status}"
}
trap finalize EXIT

log_stage bootstrap started
cp "${job_contract}" "${attempt_root}/submitted-job-contract.json"
/usr/bin/python3 scripts/slurm/verify_reconstruction_fullscale_contract.py \
  "${job_contract}" --shell-output "${runtime_file}"
# shellcheck disable=SC1090
source "${runtime_file}"
rm -f -- "${runtime_file}"
rmdir -- "${runtime_dir}"
log_stage contract_verification passed

date --utc --iso-8601=seconds > "${attempt_root}/allocation.txt"
hostname --fqdn >> "${attempt_root}/allocation.txt"
/opt/slurm/bin/scontrol show job -dd "${SLURM_JOB_ID}" >> "${attempt_root}/allocation.txt"
nvidia-smi -q > "${attempt_root}/nvidia-smi.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits > "${attempt_root}/gpu-query.csv"
env | sort > "${attempt_root}/environment.txt"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres "${expected_gres}" | tee "${attempt_root}/gpu-preflight.json"
log_stage gpu_preflight passed

run_root="${output_root}/${SLURM_JOB_ID}"
if [[ "${restart_count}" == "0" && -e "${run_root}" ]]; then
  echo "refusing to reuse reconstruction output root ${run_root}" >&2
  exit 73
fi
mkdir -p "${run_root}/provenance"
cp "${job_contract}" "${run_root}/provenance/job-contract.json"
cp "${attempt_root}/allocation.txt" "${run_root}/provenance/allocation.txt"
cp "${attempt_root}/nvidia-smi.txt" "${run_root}/provenance/nvidia-smi.txt"
cp "${attempt_root}/gpu-preflight.json" "${run_root}/provenance/gpu-preflight.json"
cp "${attempt_root}/environment.txt" "${run_root}/provenance/environment.txt"
printf '%s\n' "${attempt_root}" > "${run_root}/provenance/current-attempt.txt"

telemetry_file="${attempt_root}/gpu-telemetry.jsonl"
telemetry_summary="${attempt_root}/gpu-telemetry-summary.json"
/usr/bin/python3 scripts/slurm/monitor_gpu_telemetry.py \
  --output "${telemetry_file}" --summary "${telemetry_summary}" --interval-seconds 15 &
telemetry_pid=$!

resume_args=()
if [[ "${restart_count}" -gt 0 ]]; then
  signal_checkpoint="${run_root}/training/signal-checkpoint.pt"
  [[ -f "${signal_checkpoint}" ]] || { echo "requeue missing signal checkpoint" >&2; exit 77; }
  resume_args=(--resume "${signal_checkpoint}")
fi

log_stage trainer started
set +e
scripts/slurm/run_with_bounded_requeue.sh \
  --max-restarts "${max_restarts}" \
  --status-file "${wrapper_status_file}" -- \
  "${gpu_environment}/bin/python" scripts/run_reconstruction_fullscale.py \
  --contract "${job_contract}" \
  --output "${run_root}/result.json" \
  --training-output-dir "${run_root}/training" \
  "${resume_args[@]}"
trainer_status=$?
set -e
if (( trainer_status != 0 )); then
  terminal_stage=trainer_failed
  exit "${trainer_status}"
fi
stop_telemetry
log_stage gpu_runtime_telemetry completed
if grep -q '"action":"requeue_requested"' "${wrapper_status_file}"; then
  terminal_stage=requeue_requested
else
  terminal_stage=trainer_complete
fi
exit 0
