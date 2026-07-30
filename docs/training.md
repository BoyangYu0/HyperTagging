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

The real-parquet CPU pilot path is separate from the fixture-only dry run:

```bash
python scripts/train_hyperbolic_pretrain.py \
  --data /path/to/tiny.parquet --device cpu --max-steps 2 --batch-size 2 \
  --output-dir /tmp/hypertagging-pretrain
python scripts/train_level_reconstruction.py \
  --data /path/to/tiny.parquet \
  --pretrained-encoder /tmp/hypertagging-pretrain/checkpoint.pt \
  --device cpu --max-steps 2 --batch-size 2 \
  --output-dir /tmp/hypertagging-reconstruction
```

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
contextual_euclidean
contextual_hyperbolic_parent_lca
plus_radius_depth
plus_variance_covariance
plus_cross_event_channel
plus_hyperbolic_relation_attention
plus_leaf_pid
plus_scheduled_sampling
full_revised
```

Example fixture check:

```bash
python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 1 \
  --ablation heterogeneous_only
```

Do not infer scientific improvement from these fixtures.

The same `--ablation` value is applied by the real pretrainer and
reconstruction trainer. It controls contextual encoding, physical and
hyperbolic relation stages, the pretraining loss weights, leaf-PID loss,
scheduled sampling, and encoder-transfer eligibility; the selected value is
stored in the checkpoint config.

## GPU and production boundary

Real data, real model sizes, and long training are HTCondor-only. CUDA outside
Condor is refused unless the command is explicitly tiny and
`--allow-local-tiny-gpu-test` is supplied; that path checks `condor_q`,
`nvidia-smi`, and active GPU processes first. No training script submits a job.

See `docs/condor.md` for render and experiment commands.

## Real trainer state and transfer

The parquet data module accepts a file, directory, shards, or JSON/JSONL
manifest; it checks global event UIDs, creates a stable source-aware split,
fits masked normalization on training only, and raises rather than silently
dropping node overflow. Pretraining cycles through the configurable
three-stage curriculum. Reconstruction optimizes every target level in each
batch and validates teacher-forced, seeded scheduled, and free rollout.

Atomic checkpoints include full and encoder-only states, optimizer, scheduler,
AMP scaler, epoch/step/config, git commit, schema/PID/feature specification,
split hash, normalization, metrics, confidence-training state, and RNG states.
`--pretrained-encoder` loads only compatible shared-encoder keys and reports
loaded, missing, unexpected, and shape-mismatched keys. Use
`--freeze-pretrained-encoder-steps` and `--encoder-lr-multiplier` for transfer.

SciPy is a declared production dependency for Hungarian matching. Brute-force
matching is available only to explicitly bounded tiny CPU pilots; there is no
production greedy fallback.
