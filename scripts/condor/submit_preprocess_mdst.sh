#!/usr/bin/env bash
set -euo pipefail
CMD='basf2 scripts/preprocess_mdst.py -- --input /project/agkuhr/users/boyang/data/MC15/mdst*.root --output outputs/processed.parquet --max-events 1000 --overwrite'
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml --command "$CMD" "$@"
