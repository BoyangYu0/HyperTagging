#!/usr/bin/env bash
set -euo pipefail
CMD='uv --cache-dir /tmp/uv-cache run python scripts/train_hyperbolic_pretrain.py --config configs/hyperbolic_pretrain.yaml --device cuda --max-steps 1000 --batch-size 64'
python scripts/slurm/render_slurm_job.py --config configs/slurm/default.yaml --command "$CMD" --dry-run
