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
  echo "usage: run_reconstruction_phase35.sbatch /absolute/path/to/job-contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?this entrypoint may only run inside Slurm}"
: "${SLURM_SUBMIT_DIR:?Slurm did not provide an absolute submit directory}"
[[ "${SLURM_SUBMIT_DIR}" = /* ]] || {
  echo "submit directory must be absolute" >&2
  exit 64
}
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || {
  echo "phase35 training refuses restarted allocations" >&2
  exit 76
}

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
cd "${repo_root}"
for namespace_component in \
  artifacts \
  artifacts/slurm \
  artifacts/slurm/reconstruction-phase35 \
  artifacts/slurm/reconstruction-phase35/jobs
do
  [[ ! -L "${namespace_component}" ]] || {
    echo "phase35 runtime namespace contains a symlink: ${namespace_component}" >&2
    exit 73
  }
done
mkdir -p artifacts/slurm/reconstruction-phase35/jobs
[[ "$(realpath -e artifacts/slurm/reconstruction-phase35/jobs)" == \
  "${repo_root}/artifacts/slurm/reconstruction-phase35/jobs" ]] || {
  echo "phase35 runtime namespace escaped the repository" >&2
  exit 73
}
original_job_contract="$1"
[[ "${original_job_contract}" = /* ]] || {
  echo "contract path must be absolute" >&2
  exit 64
}
job_evidence_root="artifacts/slurm/reconstruction-phase35/jobs/${SLURM_JOB_ID}"
if [[ -e "${job_evidence_root}" || -L "${job_evidence_root}" ]]; then
  echo "refusing to reuse phase35 job evidence root ${job_evidence_root}" >&2
  exit 73
fi
mkdir "${job_evidence_root}"
attempt_root="${job_evidence_root}/attempt-00"
mkdir "${attempt_root}"
stage_log="${attempt_root}/stages.log"
wrapper_status_file="${attempt_root}/wrapper-status.json"
receipt_file="${attempt_root}/receipt.json"
job_contract="${attempt_root}/submitted-job-contract.json"
runtime_dir="$(mktemp -d "${attempt_root}/.verified-runtime.XXXXXX")"
runtime_file="${runtime_dir}/runtime.sh"
started_at="$(date --utc --iso-8601=seconds)"
terminal_stage=bootstrap
run_root=""
telemetry_pid=""
telemetry_stop_file="${attempt_root}/telemetry-stop-authorized.txt"
python_cache_root="${attempt_root}/python-cache"
export PYTHONPYCACHEPREFIX="${python_cache_root}"

log_stage() {
  terminal_stage="$1"
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" | tee -a "${stage_log}"
}

stop_telemetry() {
  [[ -n "${telemetry_pid}" ]] || return 0
  if kill -0 "${telemetry_pid}" 2>/dev/null; then
    printf '%s\n' wrapper_authorized_stop_v1 > "${telemetry_stop_file}"
    sync -f "${telemetry_stop_file}"
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
  finalizer_status=0
  /usr/bin/python3 scripts/slurm/finalize_reconstruction_fullscale_receipt.py --receipt "${receipt_file}" --contract "${job_contract}" --attempt-root "${attempt_root}" --stage-log "${stage_log}" --wrapper-status "${wrapper_status_file}" --run-root "${run_root}" --batch-exit-status "${batch_status}" --terminal-stage "${terminal_stage}" --started-at "${started_at}" --completed-at "$(date --utc --iso-8601=seconds)" || finalizer_status=$?
  if (( batch_status == 0 )); then
    if (( finalizer_status != 0 )); then
      echo "phase35 terminal evidence finalization failed" >&2
      exit "${finalizer_status}"
    fi
    [[ -s "${receipt_file}" ]] || {
      echo "phase35 terminal receipt is missing" >&2
      exit 78
    }
    /usr/bin/python3 scripts/slurm/verify_reconstruction_phase35_contract.py \
      --attempt-receipt "${receipt_file}" \
      --expected-contract-sha256 "${contract_sha256}" || {
        echo "phase35 terminal receipt is not canonically healthy" >&2
        exit 79
      }
  fi
  exit "${batch_status}"
}
trap finalize EXIT

log_stage bootstrap started
cp -- "${original_job_contract}" "${job_contract}"
chmod 0444 "${job_contract}"
/usr/bin/python3 scripts/slurm/verify_reconstruction_phase35_contract.py "${job_contract}" --shell-output "${runtime_file}"
# shellcheck disable=SC1090
source "${runtime_file}"
rm -f -- "${runtime_file}"
rmdir -- "${runtime_dir}"
log_stage contract_and_live_allocation_verification passed

date --utc --iso-8601=seconds > "${attempt_root}/allocation.txt"
hostname --fqdn >> "${attempt_root}/allocation.txt"
/opt/slurm/bin/scontrol show job -dd "${SLURM_JOB_ID}" >> "${attempt_root}/allocation.txt"
nvidia-smi -q > "${attempt_root}/nvidia-smi.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits > "${attempt_root}/gpu-query.csv"
env | sort > "${attempt_root}/environment.txt"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py --expected-gres "${expected_gres}" | tee "${attempt_root}/gpu-preflight.json"
log_stage gpu_preflight passed

run_root="${output_root}/${SLURM_JOB_ID}"
mkdir -p "${output_root}"
[[ ! -L "${output_root}" && "$(realpath -e "${output_root}")" == "${output_root}" ]] || {
  echo "phase35 arm output namespace escaped its contract path" >&2
  exit 73
}
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
  echo "refusing to reuse phase35 output root ${run_root}" >&2
  exit 73
fi
mkdir "${run_root}"
mkdir "${run_root}/provenance"
cp "${job_contract}" "${run_root}/provenance/job-contract.json"
cp "${attempt_root}/allocation.txt" "${run_root}/provenance/allocation.txt"
cp "${attempt_root}/nvidia-smi.txt" "${run_root}/provenance/nvidia-smi.txt"
cp "${attempt_root}/gpu-preflight.json" "${run_root}/provenance/gpu-preflight.json"
cp "${attempt_root}/environment.txt" "${run_root}/provenance/environment.txt"
printf '%s\n' "${attempt_root}" > "${run_root}/provenance/current-attempt.txt"

telemetry_file="${attempt_root}/gpu-telemetry.jsonl"
telemetry_summary="${attempt_root}/gpu-telemetry-summary.json"
/usr/bin/python3 scripts/slurm/monitor_gpu_telemetry.py \
  --output "${telemetry_file}" \
  --summary "${telemetry_summary}" \
  --interval-seconds 15 \
  --authorized-stop-file "${telemetry_stop_file}" &
telemetry_pid=$!

log_stage trainer started
set +e
scripts/slurm/run_with_bounded_requeue.sh --max-restarts 0 --status-file "${wrapper_status_file}" -- "${gpu_environment}/bin/python" scripts/run_reconstruction_phase35.py --contract "${job_contract}" --output "${run_root}/result.json" --training-output-dir "${run_root}/training"
trainer_status=$?
set -e
if (( trainer_status != 0 )); then
  terminal_stage=trainer_failed
  exit "${trainer_status}"
fi
kill -0 "${telemetry_pid}" 2>/dev/null || {
  echo "phase35 GPU telemetry exited before the trainer completed" >&2
  terminal_stage=telemetry_failed
  exit 80
}
stop_telemetry
log_stage gpu_runtime_telemetry completed
log_stage strict_full_decay_comparison started
"${gpu_environment}/bin/python" scripts/run_reconstruction_phase35_full_decay.py \
  --contract "${job_contract}" \
  --training-result "${run_root}/result.json" \
  --training-dir "${run_root}/training" \
  --output-dir "${run_root}/full-decay-reports" \
  --output "${run_root}/full-decay-comparison.json"
log_stage strict_full_decay_comparison completed
terminal_stage=full_decay_complete
exit 0
