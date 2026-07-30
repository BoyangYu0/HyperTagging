#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/afs/desy.de/user/b/boyangyu/HyperTagging"
VENV_ROOT="/data/dust/user/boyangyu/uv_env"
BASF2_PYTHON_SITE="${BASF2_PYTHON_SITE:-/data/dust/user/boyangyu/basf2_py38}"
INPUT_ROOT="${INPUT_ROOT:-/pnfs/desy.de/belle/local/belle/MC/release-08-03-00/DB00003335/MC16ri_run2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dust/user/boyangyu/hypertagging/production_10m}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifests/mdst_10m.jsonl}"
TARGET_EVENTS="${TARGET_EVENTS:-10000000}"
EVENTS_PER_TASK="${EVENTS_PER_TASK:-5000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
CONDOR_RUNTIME="${CONDOR_RUNTIME:-7200}"
CONDOR_MEMORY="${CONDOR_MEMORY:-8GB}"
CONDOR_CPUS="${CONDOR_CPUS:-2}"

MODE="dry-run"
REPLAN=0
TASK_ID=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --worker)
      MODE="worker"
      TASK_ID="${2:-}"
      shift 2
      ;;
    --submit)
      MODE="submit"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --replan)
      REPLAN=1
      shift
      ;;
    *)
      echo "Usage: $0 [--dry-run|--submit] [--replan] | --worker TASK_ID" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  echo "Missing virtual environment: ${VENV_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${BASF2_PYTHON_SITE}/awkward" || ! -d "${BASF2_PYTHON_SITE}/pyarrow" ]]; then
  echo "Missing basf2 Python-3.8 dependencies: ${BASF2_PYTHON_SITE}" >&2
  exit 1
fi

if [[ "${MODE}" == "worker" ]]; then
  if [[ -z "${TASK_ID}" ]]; then
    echo "--worker requires a task ID" >&2
    exit 1
  fi
  source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
  source "${VENV_ROOT}/bin/activate"
  export BASF2_PYTHON_SITE
  export OMP_NUM_THREADS="${CONDOR_CPUS}"
  export MKL_NUM_THREADS="${CONDOR_CPUS}"
  cd "${REPO_ROOT}"
  python scripts/mdst_batch_production.py run-task \
    --manifest "${MANIFEST}" \
    --task-id "${TASK_ID}" \
    --repo-root "${REPO_ROOT}"
  exit 0
fi

source "${VENV_ROOT}/bin/activate"
cd "${REPO_ROOT}"
if [[ ! -f "${MANIFEST}" || "${REPLAN}" -eq 1 ]]; then
  PLAN_ARGS=(
    python scripts/mdst_batch_production.py plan
    --input-root "${INPUT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --manifest "${MANIFEST}"
    --target-events "${TARGET_EVENTS}"
    --events-per-task "${EVENTS_PER_TASK}"
  )
  if [[ "${REPLAN}" -eq 1 ]]; then
    PLAN_ARGS+=(--overwrite)
  fi
  "${PLAN_ARGS[@]}"
fi

TASK_COUNT="$(python - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)"
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "Manifest has no tasks: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/manifests"
SUBMIT_FILE="${OUTPUT_ROOT}/manifests/mdst_10m.sub"
SUBMIT_TEXT="$(cat <<EOF
universe = vanilla
executable = ${REPO_ROOT}/scripts/condor/submit_mdst_production_10m.sh
arguments = --worker \$(ProcId)
initialdir = ${REPO_ROOT}
getenv = True
should_transfer_files = NO
request_cpus = ${CONDOR_CPUS}
request_memory = ${CONDOR_MEMORY}
+RequestRuntime = ${CONDOR_RUNTIME}
max_materialize = ${MAX_CONCURRENT}
max_idle = ${MAX_CONCURRENT}
environment = "MANIFEST=${MANIFEST};OUTPUT_ROOT=${OUTPUT_ROOT};VENV_ROOT=${VENV_ROOT};REPO_ROOT=${REPO_ROOT};BASF2_PYTHON_SITE=${BASF2_PYTHON_SITE};CONDOR_CPUS=${CONDOR_CPUS}"
output = ${OUTPUT_ROOT}/logs/mdst-prod-10m-\$(ClusterId).\$(ProcId).out
error = ${OUTPUT_ROOT}/logs/mdst-prod-10m-\$(ClusterId).\$(ProcId).err
log = ${OUTPUT_ROOT}/logs/mdst-prod-10m-\$(ClusterId).log
queue ${TASK_COUNT}
EOF
)"

echo "Manifest: ${MANIFEST}"
echo "Tasks: ${TASK_COUNT}; maximum materialized jobs: ${MAX_CONCURRENT}"
echo "Target input events: ${TARGET_EVENTS}"
if [[ "${MODE}" == "submit" ]]; then
  command -v condor_submit >/dev/null
  printf '%s\n' "${SUBMIT_TEXT}" > "${SUBMIT_FILE}"
  condor_submit "${SUBMIT_FILE}"
else
  printf 'Dry run. Submit description (%s):\n%s\n' "${SUBMIT_FILE}" "${SUBMIT_TEXT}"
  printf 'Submit with:\n%s --submit\n' "$0"
fi
