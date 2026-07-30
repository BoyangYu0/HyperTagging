#!/usr/bin/env bash
set -euo pipefail
CMD='uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --config configs/level_reconstruction.yaml --device cuda --max-steps 1000 --batch-size 64'
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml --command "$CMD" "$@"
