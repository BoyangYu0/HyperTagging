#!/usr/bin/env bash
set -euo pipefail

readonly REQUEUE_EXIT_CODE=75
max_restarts=""
scontrol=/opt/slurm/bin/scontrol
while (( $# )); do
  case "$1" in
    --max-restarts)
      [[ $# -ge 2 ]] || { echo "--max-restarts requires a value" >&2; exit 64; }
      max_restarts="$2"
      shift 2
      ;;
    --scontrol)
      [[ $# -ge 2 ]] || { echo "--scontrol requires a value" >&2; exit 64; }
      scontrol="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unexpected wrapper argument: $1" >&2
      exit 64
      ;;
  esac
done
if ! [[ "${max_restarts}" =~ ^[0-9]+$ ]] || (( max_restarts > 10 )); then
  echo "max_restarts must be an integer between 0 and 10" >&2
  exit 64
fi
(( $# > 0 )) || { echo "trainer command is required" >&2; exit 64; }

trainer_pid=""
usr1_pending=0
forward_usr1() {
  usr1_pending=1
  if [[ -n "${trainer_pid}" ]] && kill -0 "${trainer_pid}" 2>/dev/null; then
    kill -USR1 "${trainer_pid}" 2>/dev/null || true
  fi
}
forward_termination() {
  if [[ -n "${trainer_pid}" ]] && kill -0 "${trainer_pid}" 2>/dev/null; then
    kill -TERM "${trainer_pid}" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1
trap forward_termination TERM INT

set +e
"$@" &
trainer_pid=$!
if (( usr1_pending )); then
  kill -USR1 "${trainer_pid}" 2>/dev/null || true
fi
while true; do
  wait "${trainer_pid}"
  trainer_status=$?
  if ! kill -0 "${trainer_pid}" 2>/dev/null; then
    break
  fi
done
set -e
trap - USR1 TERM INT

if (( trainer_status != REQUEUE_EXIT_CODE )); then
  exit "${trainer_status}"
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "signal checkpoint requested outside Slurm; refusing requeue" >&2
  exit "${REQUEUE_EXIT_CODE}"
fi
restart_count="${SLURM_RESTART_COUNT:-0}"
if ! [[ "${restart_count}" =~ ^[0-9]+$ ]]; then
  echo "invalid Slurm restart counter" >&2
  exit 76
fi
if (( restart_count >= max_restarts )); then
  echo "bounded requeue limit reached (${restart_count}/${max_restarts})" >&2
  exit 76
fi

"${scontrol}" requeue "${SLURM_JOB_ID}"
exit 0
