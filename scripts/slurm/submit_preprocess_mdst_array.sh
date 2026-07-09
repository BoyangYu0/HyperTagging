#!/usr/bin/env bash
set -euo pipefail
CMD='basf2 scripts/preprocess_mdst.py -- --input /project/agkuhr/users/boyang/data/MC15/mdst*.root --output outputs/processed.parquet --max-events 1000 --overwrite'
python scripts/slurm/render_slurm_job.py --config configs/slurm/default.yaml --command "$CMD" --dry-run
