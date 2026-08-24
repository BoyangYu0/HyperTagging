#!/usr/bin/env bash
# Immutable replacement entrypoint. The contract argument must be absolute and
# must resolve to the committed snapshot beside this file.
#SBATCH --account=others
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --export=NIL
#SBATCH --output=artifacts/slurm/recon-val-b64-h100nvl-baseline-replacement-20260824-%j.out
#SBATCH --error=artifacts/slurm/recon-val-b64-h100nvl-baseline-replacement-20260824-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_validation_followup.sbatch /absolute/path/to/committed-snapshot/job_contract.json" >&2
  exit 64
fi
: "${SLURM_JOB_ID:?replacement entrypoint requires an allocated Slurm job}"
: "${SLURM_SUBMIT_DIR:?Slurm did not provide a submit directory}"
[[ "${SLURM_SUBMIT_DIR}" = /* ]] || { echo "SLURM_SUBMIT_DIR must be absolute" >&2; exit 64; }

uv_bin="/home/b/Boyang.Yu/.local/bin/uv"
[[ -x "${uv_bin}" ]] || { echo "mandated uv binary unavailable: ${uv_bin}" >&2; exit 64; }
export PATH=/usr/bin:/bin:/opt/slurm/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
unset PYTHONWARNINGS PYTHONBREAKPOINT LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG

repo_root="$(realpath -e "${SLURM_SUBMIT_DIR}")"
snapshot_root="$(realpath -e "$(dirname "${BASH_SOURCE[0]}")")"
cd "${repo_root}"
job_contract="$1"
[[ "${job_contract}" = /* ]] || { echo "contract path must be absolute" >&2; exit 64; }
[[ "$(realpath -e "${job_contract}")" = "${snapshot_root}/job_contract.json" ]] || {
  echo "contract must be the exact immutable snapshot contract" >&2; exit 64;
}
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "${restart_count}" = "0" ]] || { echo "replacement forbids Slurm restarts" >&2; exit 76; }

attempt_root="${repo_root}/artifacts/slurm/reconstruction-replacement-v2/jobs/${SLURM_JOB_ID}/attempt-00"
mkdir -p "${attempt_root}"
stage_log="${attempt_root}/stages.log"
wrapper_status_file="${attempt_root}/wrapper-status.json"
receipt_file="${attempt_root}/receipt.json"
runtime_file="${attempt_root}/verified-runtime.sh"
started_at="$(date --utc --iso-8601=seconds)"
terminal_stage="bootstrap"
run_root=""
telemetry_pid=""

log_stage() {
  terminal_stage="$1"
  printf '%s stage=%s state=%s\n' "$(date --utc --iso-8601=seconds)" "$1" "$2" | tee -a "${stage_log}"
}

stop_telemetry() {
  [[ -n "${telemetry_pid}" ]] || return 0
  if kill -0 "${telemetry_pid}" 2>/dev/null; then kill -TERM "${telemetry_pid}" 2>/dev/null || true; fi
  wait "${telemetry_pid}" || true
  telemetry_pid=""
}

finalize() {
  batch_status=$?
  trap - EXIT
  set +e
  stop_telemetry
  log_stage "${terminal_stage}" "batch_exit_${batch_status}"
  "${uv_bin}" run --frozen python "${snapshot_root}/runtime/scripts/slurm/finalize_reconstruction_fullscale_receipt.py" \
    --receipt "${receipt_file}" --contract "${job_contract}" --attempt-root "${attempt_root}" \
    --stage-log "${stage_log}" --wrapper-status "${wrapper_status_file}" --run-root "${run_root}" \
    --batch-exit-status "${batch_status}" --terminal-stage "${terminal_stage}" \
    --started-at "${started_at}" --completed-at "$(date --utc --iso-8601=seconds)" || true
  exit "${batch_status}"
}
trap finalize EXIT

log_stage bootstrap started
"${uv_bin}" --version > "${attempt_root}/uv-version.txt"
sha256sum "${repo_root}/uv.lock" > "${attempt_root}/uv-lock.sha256"
cp -- "${job_contract}" "${attempt_root}/submitted-job-contract.json"
"${uv_bin}" run --frozen python "${snapshot_root}/runtime/scripts/slurm/verify_snapshot_contract.py" \
  "${job_contract}" --repo-root "${repo_root}" --shell-output "${runtime_file}"
# shellcheck disable=SC1090
source "${runtime_file}"
rm -f -- "${runtime_file}"
log_stage contract_verification passed

scontrol=/opt/slurm/bin/scontrol
"${scontrol}" show job -dd "${SLURM_JOB_ID}" > "${attempt_root}/allocation.txt"
grep -Fq 'gpu:h100nvl:1' "${attempt_root}/allocation.txt" || { echo "exact H100-NVL allocation not present" >&2; exit 76; }
[[ "${SLURM_JOB_GRES:-}" == *"gpu:h100nvl:1"* ]] || { echo "SLURM_JOB_GRES is not exact H100-NVL" >&2; exit 76; }
hostname --fqdn > "${attempt_root}/hostname.txt"
nvidia-smi -q > "${attempt_root}/nvidia-smi.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits > "${attempt_root}/gpu-query.csv"
env | sort > "${attempt_root}/environment.txt"

"${uv_bin}" run --frozen python "${snapshot_root}/runtime/scripts/slurm/preflight_gpu_environment.py" \
  --expected-gres "${expected_gres}" | tee "${attempt_root}/gpu-preflight.json"
"${uv_bin}" run --frozen python - <<'PY' | tee "${attempt_root}/uv-gpu-finite-probe.json"
import hashlib, json, os, sys
from pathlib import Path
import torch

root = Path.cwd()
probe = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
value = probe @ probe
payload = {
    "status": "passed",
    "python_executable": sys.executable,
    "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "torch": torch.__version__, "cuda_build": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "finite_gpu_tensor_op": bool(torch.isfinite(value.float()).all().item()),
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "expected_gres": "gpu:h100nvl:1",
    "python_pin_sha256": hashlib.sha256((root / ".python-version").read_bytes()).hexdigest(),
    "pyproject_sha256": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest(),
    "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
}
if payload["python"] != "3.11.11" or payload["torch"] != "2.7.1+cu126" or payload["cuda_build"] != "12.6":
    raise RuntimeError("Python/torch/CUDA binding mismatch")
if not payload["cuda_available"] or payload["gpu_name"] != "NVIDIA H100 NVL" or not payload["finite_gpu_tensor_op"]:
    raise RuntimeError("fresh H100-NVL finite GPU preflight failed")
if payload["python_pin_sha256"] != "49a506dd32096b010d75205acf3430c9ae6c40351888129499e5a5e487126c93" or payload["pyproject_sha256"] != "607059fb7975f7bf52e7fd21ac848e00af31e08f926bee747470eb211c82a8c2" or payload["uv_lock_sha256"] != "7a18fbd4feed4371fa8e8a740f87720462d58c3a8e283402870f375ab744ad18":
    raise RuntimeError("uv/pin/pyproject/lock binding mismatch")
print(json.dumps(payload, sort_keys=True))
PY
log_stage gpu_preflight passed

run_root="${output_root}/${SLURM_JOB_ID}"
if [[ -e "${run_root}" ]]; then echo "refusing to reuse replacement output ${run_root}" >&2; exit 73; fi
mkdir -p "${run_root}/provenance"
cp -- "${job_contract}" "${run_root}/provenance/job-contract.json"
cp -- "${attempt_root}/allocation.txt" "${run_root}/provenance/allocation.txt"
cp -- "${attempt_root}/nvidia-smi.txt" "${run_root}/provenance/nvidia-smi.txt"
cp -- "${attempt_root}/gpu-preflight.json" "${run_root}/provenance/gpu-preflight.json"
cp -- "${attempt_root}/uv-gpu-finite-probe.json" "${run_root}/provenance/uv-gpu-finite-probe.json"
cp -- "${attempt_root}/environment.txt" "${run_root}/provenance/environment.txt"
cp -- "${attempt_root}/uv-version.txt" "${run_root}/provenance/uv-version.txt"
cp -- "${attempt_root}/uv-lock.sha256" "${run_root}/provenance/uv-lock.sha256"
printf '%s\n' "${attempt_root}" > "${run_root}/provenance/current-attempt.txt"

telemetry_file="${attempt_root}/gpu-telemetry.jsonl"
telemetry_summary="${attempt_root}/gpu-telemetry-summary.json"
"${uv_bin}" run --frozen python "${snapshot_root}/runtime/scripts/slurm/monitor_gpu_telemetry.py" \
  --output "${telemetry_file}" --summary "${telemetry_summary}" --interval-seconds 15 &
telemetry_pid=$!

log_stage trainer started
set +e
"${snapshot_root}/runtime/scripts/slurm/run_with_bounded_requeue.sh" \
  --max-restarts 0 --status-file "${wrapper_status_file}" -- \
  "${uv_bin}" run --frozen python "${snapshot_root}/runtime/scripts/run_reconstruction_fullscale.py" \
  --contract "${job_contract}" --output "${run_root}/result.json" \
  --training-output-dir "${run_root}/training"
trainer_status=$?
set -e
if (( trainer_status != 0 )); then terminal_stage=trainer_failed; exit "${trainer_status}"; fi
stop_telemetry
log_stage gpu_runtime_telemetry completed
terminal_stage=trainer_complete
exit 0
