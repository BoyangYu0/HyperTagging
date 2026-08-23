#!/usr/bin/env bash
# Fail-closed Slurm adapter for one immutable phase-3 calibration tuple.
# Submission must supply the exact typed GRES and all HT_PHASE3_* bindings.
# The Python calibration harness remains the authority for tuple/path/receipt
# validation; this adapter only proves the allocation and invokes it in the
# frozen CUDA environment.

set -euo pipefail

: "${SLURM_JOB_ID:?this entrypoint may only run inside Slurm}"
: "${SLURM_SUBMIT_DIR:?Slurm did not provide an absolute submit directory}"
: "${HT_PHASE3_CALIBRATION_ACTIVE:?missing calibration activation marker}"
: "${HT_PHASE3_EXPECTED_GRES:?missing exact expected GRES}"
: "${HT_PHASE3_SUBMIT_EPOCH:?missing fresh submission timestamp}"
: "${HT_PHASE3_FRESH_PREFLIGHT_TOKEN:?missing fresh preflight token}"
: "${HT_PHASE3_PLAN:?missing immutable study plan}"
: "${HT_PHASE3_OWNER:?missing immutable owner}"
: "${HT_PHASE3_CALIBRATION_ID:?missing immutable calibration ID}"
: "${HT_PHASE3_PROFILE:?missing calibration profile}"
: "${HT_PHASE3_BATCH_SIZE:?missing calibration batch size}"
: "${HT_PHASE3_AMP_DTYPE:?missing calibration precision}"
: "${HT_PHASE3_GRAD_SCALER:?missing calibration scaler policy}"
: "${HT_PHASE3_CHECKPOINT_COPY:?missing checkpoint-copy path}"
: "${HT_PHASE3_OUTPUT_ROOT:?missing output-root path}"
: "${HT_PHASE3_ATTEMPT_ROOT:?missing attempt-root path}"
: "${HT_PHASE3_METRICS:?missing metrics path}"
: "${HT_PHASE3_RECEIPT:?missing receipt path}"
: "${HT_PHASE3_LOG_ROOT:?missing Slurm evidence root}"

[[ "${HT_PHASE3_CALIBRATION_ACTIVE}" = 1 ]] || {
  echo "calibration activation marker is not exactly 1" >&2
  exit 64
}
[[ "${HT_PHASE3_SUBMIT_EPOCH}" =~ ^[0-9]+$ ]] || {
  echo "submission timestamp must be an integer Unix epoch" >&2
  exit 64
}
[[ "${HT_PHASE3_EXPECTED_GRES}" =~ ^gpu:(h100nvl|v100):1$ ]] || {
  echo "expected GRES is not one exact authorized calibration GRES" >&2
  exit 64
}

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
[[ -f "${repo_root}/scripts/run_phase3_batch_efficiency_calibration.py" ]] || {
  echo "Slurm submit directory is not the HyperTagging repository" >&2
  exit 64
}
cd "${repo_root}"

gpu_environment="${HT_PHASE3_GPU_ENVIRONMENT:-/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1}"
[[ -x "${gpu_environment}/bin/python" ]] || {
  echo "verified frozen GPU Python is unavailable: ${gpu_environment}" >&2
  exit 78
}

export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_AUDIT LD_DEBUG

job_gres="${SLURM_JOB_GRES:-}"
tres_gres="${SLURM_TRES_PER_NODE:-}"
[[ -n "${job_gres}" || -n "${tres_gres}" ]] || {
  echo "Slurm did not expose an allocated typed GRES" >&2
  exit 78
}
if [[ -n "${job_gres}" && "${job_gres}" != "${HT_PHASE3_EXPECTED_GRES}" ]]; then
  echo "SLURM_JOB_GRES does not match the exact requested GRES" >&2
  exit 78
fi
if [[ -n "${tres_gres}" && "${tres_gres}" != "gres:${HT_PHASE3_EXPECTED_GRES}" && "${tres_gres}" != "${HT_PHASE3_EXPECTED_GRES}" ]]; then
  echo "SLURM_TRES_PER_NODE does not match the exact requested GRES" >&2
  exit 78
fi

allocation_epoch="$(date +%s)"
(( allocation_epoch >= HT_PHASE3_SUBMIT_EPOCH )) || {
  echo "allocation clock precedes submission clock; queue delay is not measurable" >&2
  exit 78
}
queue_delay_seconds=$((allocation_epoch - HT_PHASE3_SUBMIT_EPOCH))

mkdir -p "${HT_PHASE3_LOG_ROOT}"
echo "calibration_id=${HT_PHASE3_CALIBRATION_ID:-missing} slurm_job_id=${SLURM_JOB_ID} expected_gres=${HT_PHASE3_EXPECTED_GRES} queue_delay_seconds=${queue_delay_seconds}" >&2

# This is the fresh exact-GRES in-allocation attestation.  It verifies the
# Slurm job record, one visible GPU, model/GRES binding, CUDA build, and the
# frozen runtime lock before the calibration harness starts.
"${gpu_environment}/bin/python" scripts/slurm/preflight_gpu_environment.py \
  --expected-gres "${HT_PHASE3_EXPECTED_GRES}" \
  | tee "${HT_PHASE3_LOG_ROOT}/preflight-${SLURM_JOB_ID}.json"

export HT_PHASE3_ALLOCATION_GRES="${HT_PHASE3_EXPECTED_GRES}"
export HT_PHASE3_CALIBRATION_ACTIVE=1

"${gpu_environment}/bin/python" scripts/run_phase3_batch_efficiency_calibration.py \
  --study-plan "${HT_PHASE3_PLAN}" \
  --calibration-id "${HT_PHASE3_CALIBRATION_ID}" \
  --owner "${HT_PHASE3_OWNER}" \
  --profile "${HT_PHASE3_PROFILE}" \
  --batch-size "${HT_PHASE3_BATCH_SIZE}" \
  --amp-dtype "${HT_PHASE3_AMP_DTYPE}" \
  --grad-scaler "${HT_PHASE3_GRAD_SCALER}" \
  --checkpoint-copy "${HT_PHASE3_CHECKPOINT_COPY}" \
  --output-root "${HT_PHASE3_OUTPUT_ROOT}" \
  --attempt-root "${HT_PHASE3_ATTEMPT_ROOT}" \
  --pilot-metrics "${HT_PHASE3_METRICS}" \
  --receipt "${HT_PHASE3_RECEIPT}" \
  --queue-delay-seconds "${queue_delay_seconds}" \
  --execute-pilot \
  --pilot-command -- \
  "${gpu_environment}/bin/python" scripts/run_phase3_batch_efficiency_stability_pilot.py \
    --profile "${HT_PHASE3_PROFILE}" \
    --batch-size "${HT_PHASE3_BATCH_SIZE}" \
    --checkpoint-copy "${HT_PHASE3_CHECKPOINT_COPY}" \
    --metrics "${HT_PHASE3_METRICS}" \
    --max-steps 256
