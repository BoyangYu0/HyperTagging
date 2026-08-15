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
  echo "usage: evaluate_pretraining_checkpoint_study.sbatch /absolute/path/to/job-contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?this entrypoint may only run inside Slurm}"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || {
  echo "read-only checkpoint comparison does not permit restart/requeue" >&2
  exit 76
}

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

: "${SLURM_SUBMIT_DIR:?Slurm did not provide an absolute submit directory}"
repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
cd "${repo_root}"
job_contract="$1"
[[ "${job_contract}" = /* ]] || {
  echo "job contract path must be absolute" >&2
  exit 64
}

attempt_root="artifacts/slurm/studies/pretraining-checkpoint-comparison/jobs/${SLURM_JOB_ID}/attempt-00"
mkdir -p "${attempt_root}"
stage_log="${attempt_root}/stages.log"
receipt="${attempt_root}/receipt.json"
gpu_preflight="${attempt_root}/gpu-preflight.json"
telemetry_file="${attempt_root}/gpu-telemetry.jsonl"
telemetry_summary="${attempt_root}/gpu-telemetry-summary.json"
runtime_dir="$(mktemp -d "${attempt_root}/.verified-runtime.XXXXXX")"
runtime_file="${runtime_dir}/runtime.sh"
started_at="$(date --utc --iso-8601=seconds)"
telemetry_pid=""
result_file="${attempt_root}/result-not-started.json"

log_stage() {
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" \
    | tee -a "${stage_log}"
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
  status=$?
  trap - EXIT
  set +e
  stop_telemetry
  log_stage evaluation "exit_${status}"
  /usr/bin/python3 scripts/slurm/finalize_pretraining_validation_receipt.py \
    --receipt "${receipt}" \
    --contract "${job_contract}" \
    --stage-log "${stage_log}" \
    --result "${result_file}" \
    --gpu-preflight "${gpu_preflight}" \
    --telemetry-summary "${telemetry_summary}" \
    --exit-status "${status}" \
    --started-at "${started_at}" \
    --completed-at "$(date --utc --iso-8601=seconds)"
  exit "${status}"
}
trap finalize EXIT

log_stage bootstrap started
cp "${job_contract}" "${attempt_root}/submitted-job-contract.json"
/usr/bin/python3 scripts/slurm/verify_pretraining_checkpoint_study_contract.py \
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
env | sort > "${attempt_root}/environment.txt"
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres "${expected_gres}" | tee "${gpu_preflight}"
log_stage gpu_preflight passed

evaluation_root="${evaluation_output_base}/${SLURM_JOB_ID}"
[[ ! -e "${evaluation_root}" ]] || {
  echo "refusing to reuse checkpoint-study output root ${evaluation_root}" >&2
  exit 73
}
mkdir -p "${evaluation_root}/provenance"
cp "${job_contract}" "${evaluation_root}/provenance/job-contract.json"
cp "${attempt_root}/allocation.txt" "${evaluation_root}/provenance/allocation.txt"
cp "${attempt_root}/nvidia-smi.txt" "${evaluation_root}/provenance/nvidia-smi.txt"
cp "${gpu_preflight}" "${evaluation_root}/provenance/gpu-preflight.json"
result_file="${evaluation_root}/result.json"

/usr/bin/python3 scripts/slurm/monitor_gpu_telemetry.py \
  --output "${telemetry_file}" \
  --summary "${telemetry_summary}" \
  --interval-seconds 15 &
telemetry_pid=$!
log_stage evaluation started
"${gpu_environment}/bin/python" scripts/evaluate_hyperbolic_pretrain.py \
  --checkpoint "${checkpoint}" \
  --data "${selection_manifest}" \
  --dataset-index "${dataset_index}" \
  --output "${result_file}" \
  --device cuda \
  --expected-checkpoint-sha256 "${checkpoint_sha256}" \
  --expected-source-git-sha "${expected_git_sha}" \
  --expected-checkpoint-step "${checkpoint_step}" \
  --allow-comparison-checkpoint
stop_telemetry
log_stage evaluation completed
