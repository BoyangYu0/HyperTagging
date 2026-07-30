# HTCondor Scripts

The renderers create HTCondor `.sub` files and matching shell executables but
do not submit them automatically.

```bash
python scripts/condor/check_condor_env.py
export DATA_MANIFEST=/data/volume/hypertagging/manifest.jsonl
export PRETRAINED_ENCODER=/data/volume/hypertagging/pretrain/checkpoint.pt
scripts/condor/submit_level_reconstruction.sh \
  --output outputs/condor/level_reconstruction.sub
condor_submit outputs/condor/level_reconstruction.sub
```

The default requests are configured in `configs/condor/default.yaml`.
Submission is always an explicit separate step, except for the production
launcher's explicit `--submit` mode.

`submit_hyperbolic_pretrain.sh` requires `DATA_MANIFEST`.
`submit_level_reconstruction.sh` requires both `DATA_MANIFEST` and
`PRETRAINED_ENCODER`. Optional `OUTPUT_DIR` values should point to the data
volume; wrapper arguments only render files and never submit them.

## Direct-mDST 10M production

`submit_mdst_production_10m.sh` is dry-run by default. It activates
`/data/dust/user/boyangyu/uv_env`, uses the manifest produced by
`scripts/mdst_batch_production.py`, and only calls `condor_submit` with
`--submit`.

```bash
scripts/condor/submit_mdst_production_10m.sh --dry-run
scripts/condor/submit_mdst_production_10m.sh --submit
```

Configuration variables:

- `TARGET_EVENTS` (default `10000000`)
- `EVENTS_PER_TASK` (default `5000`, bounding the in-worker event buffer)
- `MAX_CONCURRENT` (default `50`, applied with `max_materialize`)
- `CONDOR_RUNTIME` in seconds (default `7200`)
- `CONDOR_MEMORY` (default `8GB`)
- `CONDOR_CPUS` (default `2`)
- `INPUT_ROOT`, `OUTPUT_ROOT`, `MANIFEST`, and `BASF2_PYTHON_SITE`

Use `--replan` to overwrite the manifest with current environment settings.
Production is resumable: a valid existing output shard is reported as
`already-complete`.
