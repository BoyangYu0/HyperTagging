# Training and evaluation

## Local CPU verification

```bash
python -m pytest -q
python scripts/train_hyperbolic_pretrain.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
python scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-notebooks
```

These commands use synthetic fixtures. They validate software behavior and do
not measure physics performance.

## Data controls

`stable_split_name` hashes stable `event_uid` values and can group by source
file and/or physics category. Duplicate event IDs are rejected before
splitting. `MaskedFeatureNormalizer` fits each feature only from available
training-split values, stores its state with the checkpoint, and leaves missing
positions at masked zero after transformation.

Production loaders should stratify diagnostics by exact channel ID,
multiplicity, depth, source category, and rare/unseen channels. Exact channel
IDs are not the only channel objective.

## Losses and logging

Hyperbolic runs log LCA/tree-relation loss and accuracy, true Poincare parent
ranking loss/accuracy, radius-depth, pooled-channel, variance, covariance, and
the collapse diagnostics documented in
`docs/hyperbolic_level_autoregressive_reconstruction.md`.

Reconstruction runs log object/no-object and mother-type accuracy, Level-1
pointer precision/recall, component losses, Hungarian matches, and both
teacher-forced/free-rollout results. Full evaluation helpers report exact tree
match, edge precision/recall/F1, validity, p4 closure, node count, and maximum
depth. Production reporting should additionally group reconstruction
efficiency by channel, multiplicity, and depth.

JSONL logging is available through `hypertagging.training.logging`.
Checkpoints contain model, optimizer, scheduler, step, epoch, config, metrics,
an explicit normalization-state payload, and preprocessing schema. Loading
defaults to CPU and `--resume PATH` restores model and optimizer state. Schema
v1 inputs are adapted before normalization/model execution.

## Ablations

The CPU-testable names and matching configs under `configs/ablations/` are:

```text
flat_baseline
heterogeneous_only
hyperbolic_lca_parent
plus_radius_depth
plus_variance_covariance
plus_channel
plus_relation_attention
full_revised
```

Example fixture check:

```bash
python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 1 \
  --ablation heterogeneous_only
```

Do not infer scientific improvement from these fixtures.

## GPU and production boundary

Real data, real model sizes, and long training are HTCondor-only. CUDA outside
Condor is refused unless the command is explicitly tiny and
`--allow-local-tiny-gpu-test` is supplied; that path checks `condor_q`,
`nvidia-smi`, and active GPU processes first. No training script submits a job.

See `docs/condor.md` for render and experiment commands.
