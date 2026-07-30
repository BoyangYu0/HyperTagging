#!/usr/bin/env bash
set -euo pipefail
DATA_MANIFEST="${DATA_MANIFEST:-}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/dust/user/boyangyu/hypertagging/level_reconstruction}"
if [[ -z "${DATA_MANIFEST}" || -z "${PRETRAINED_ENCODER}" ]]; then
  echo "Set DATA_MANIFEST and PRETRAINED_ENCODER before rendering the reconstruction job." >&2
  exit 2
fi
printf -v DATA_ARG '%q' "${DATA_MANIFEST}"
printf -v PRETRAINED_ARG '%q' "${PRETRAINED_ENCODER}"
printf -v OUTPUT_ARG '%q' "${OUTPUT_DIR}"
CMD="uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --config configs/level_reconstruction.yaml --data ${DATA_ARG} --pretrained-encoder ${PRETRAINED_ARG} --device cuda --max-steps 1000 --batch-size 64 --output-dir ${OUTPUT_ARG}"
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml --command "$CMD" "$@"
