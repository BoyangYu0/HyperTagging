#!/usr/bin/env bash
set -euo pipefail
DATA_MANIFEST="${DATA_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/dust/user/boyangyu/hypertagging/hyperbolic_pretrain}"
if [[ -z "${DATA_MANIFEST}" ]]; then
  echo "Set DATA_MANIFEST to a v1/v2/v3 parquet, shard directory, or JSONL manifest." >&2
  exit 2
fi
printf -v DATA_ARG '%q' "${DATA_MANIFEST}"
printf -v OUTPUT_ARG '%q' "${OUTPUT_DIR}"
CMD="uv --cache-dir /tmp/uv-cache run python scripts/train_hyperbolic_pretrain.py --config configs/hyperbolic_pretrain.yaml --data ${DATA_ARG} --device cuda --max-steps 1000 --batch-size 64 --output-dir ${OUTPUT_ARG}"
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml --command "$CMD" "$@"
