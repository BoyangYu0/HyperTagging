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

: "${SLURM_SUBMIT_DIR:?Slurm did not provide an absolute submit directory}"
[[ "${SLURM_SUBMIT_DIR}" = /* ]] || {
  echo "Slurm submit directory must be absolute" >&2
  exit 64
}
repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
[[ -f "${repo_root}/scripts/slurm/verify_job_contract.py" ]] || {
  echo "Slurm submit directory is not the HyperTagging repository" >&2
  exit 64
}
cd "${repo_root}"
mkdir -p artifacts/slurm
job_contract="$1"
[[ "${job_contract}" = /* ]] || {
  echo "job contract path must be absolute" >&2
  exit 64
}
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "${restart_count}" =~ ^[0-9]+$ ]] || {
  echo "invalid Slurm restart counter" >&2
  exit 76
}
attempt_root="artifacts/slurm/jobs/${SLURM_JOB_ID}/attempt-$(printf '%02d' "${restart_count}")"
mkdir -p "${attempt_root}"
stage_log="${attempt_root}/stages.log"
wrapper_status_file="${attempt_root}/wrapper-status.json"
receipt_file="${attempt_root}/receipt.json"
started_at="$(date --utc --iso-8601=seconds)"
terminal_stage=bootstrap
run_root=""
log_stage() {
  terminal_stage="$1"
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" \
    | tee -a "${stage_log}"
}
finalize_attempt() {
  local batch_status=$?
  trap - EXIT
  set +e
  log_stage "${terminal_stage}" "batch_exit_${batch_status}"
  receipt_args=(
    --receipt "${receipt_file}"
    --contract "${job_contract}"
    --attempt-root "${attempt_root}"
    --stage-log "${stage_log}"
    --wrapper-status "${wrapper_status_file}"
    --batch-exit-status "${batch_status}"
    --terminal-stage "${terminal_stage}"
    --started-at "${started_at}"
    --completed-at "$(date --utc --iso-8601=seconds)"
  )
  if [[ -n "${run_root}" ]]; then
    receipt_args+=(--run-root "${run_root}")
  fi
  /usr/bin/python3 scripts/slurm/finalize_execution_receipt.py "${receipt_args[@]}"
  receipt_status=$?
  if (( receipt_status != 0 )); then
    echo "failed to write Slurm attempt receipt" >&2
    if (( batch_status == 0 )); then
      batch_status=78
    fi
  fi
  exit "${batch_status}"
}
trap finalize_attempt EXIT
log_stage bootstrap started
cp "${job_contract}" "${attempt_root}/submitted-job-contract.json"
runtime_dir="$(mktemp -d "${attempt_root}/.verified-runtime.XXXXXX")"
runtime_file="${runtime_dir}/runtime.sh"
cleanup_runtime() {
  rm -f -- "${runtime_file}"
  rmdir -- "${runtime_dir}" 2>/dev/null || true
}

/usr/bin/python3 scripts/slurm/verify_job_contract.py \
  "${job_contract}" --shell-output "${runtime_file}"
log_stage contract_verification passed
# shellcheck disable=SC1090
source "${runtime_file}"
log_stage allocation_telemetry started
date --utc --iso-8601=seconds > "${attempt_root}/allocation.txt"
hostname --fqdn >> "${attempt_root}/allocation.txt"
/opt/slurm/bin/scontrol show job -dd "${SLURM_JOB_ID}" >> "${attempt_root}/allocation.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits \
  > "${attempt_root}/gpu-query.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits \
  > "${attempt_root}/gpu-compute-apps.csv"
nvidia-smi pmon -c 1 > "${attempt_root}/gpu-pmon.txt"
nvidia-smi -q > "${attempt_root}/nvidia-smi.txt"
env | sort > "${attempt_root}/environment.txt"
log_stage allocation_telemetry complete
log_stage gpu_preflight started
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres "${expected_gres}" | tee "${attempt_root}/gpu-preflight.json"
log_stage gpu_preflight passed

run_root="artifacts/runs/${experiment}/${seed}/${SLURM_JOB_ID}"
mkdir -p "${run_root}/checkpoints" "${run_root}/provenance"
cp "${job_contract}" "${run_root}/provenance/job-contract.json"
cp "${attempt_root}/nvidia-smi.txt" "${run_root}/provenance/nvidia-smi.txt"
cp "${attempt_root}/environment.txt" "${run_root}/provenance/environment.txt"
printf '%s\n' "${attempt_root}" > "${run_root}/provenance/current-attempt.txt"

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
trap finalize_attempt EXIT
log_stage trainer started
set +e
scripts/slurm/run_with_bounded_requeue.sh \
  --max-restarts "${max_restarts}" \
  --status-file "${wrapper_status_file}" -- \
  "${gpu_environment}/bin/python" scripts/train_hyperbolic_pretrain.py \
  --config "${train_config}" \
  --expected-gres "${expected_gres}" \
  --seed "${seed}" \
  --output-dir "${run_root}" \
  "${resume_args[@]}"
trainer_wrapper_status=$?
set -e
if (( trainer_wrapper_status != 0 )); then
  terminal_stage=trainer_failed
  exit "${trainer_wrapper_status}"
fi
if [[ -f "${wrapper_status_file}" ]] \
  && grep -q '"action":"requeue_requested"' "${wrapper_status_file}"; then
  terminal_stage=requeue_requested
else
  terminal_stage=trainer_complete
fi
exit 0
