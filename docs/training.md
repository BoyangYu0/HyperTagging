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

Model architecture is resolved from one of `tiny_cpu`, `gpu_debug`, or
`production_baseline`; every dimension and level-specific query/cardinality
override is serialized into the checkpoint. For example:

```bash
python scripts/train_level_reconstruction.py \
  --config configs/model_presets/production_baseline.yaml \
  --data /data/volume/manifest.jsonl \
  --dataset-index /data/volume/dataset_index.json \
  --pretrained-encoder /data/volume/pretrain/checkpoint.pt \
  --device cuda --output-dir /data/volume/reconstruction
```

Production-baseline encoder transfer rejects major shape mismatches or transfer
coverage below the configured minimum unless the explicit low-coverage
override is supplied. Exact single-worker resume replays the deterministic
epoch iterator through the stored batch index; no unused parquet row-group
cursor is claimed.

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

## Streaming, scheduling, and exact state

Training accepts one Parquet file, a directory, JSON, or production JSONL
manifest (`output_file`, `output`, `path`, and `parquet` are recognized).
Event-row v4 shards are iterated lazily with deterministic shard order,
worker partitioning, bounded shuffle, and configurable prefetch. Masked,
mergeable Welford statistics are fit in a first pass over training only.
`--allow-legacy-conflated` is required for v1-v3 diagnostic runs.

Scheduled sampling participates in optimization. At each level a deterministic
truth/predicted context choice follows the configured constant, linear, cosine,
or inverse-sigmoid schedule. Predicted contexts are aligned to targets by
recursive leaf-source overlap; unrepresentable targets are counted and logged.
The implementation uses per-event micro-rollouts for predicted contexts, which
keeps alignment explicit at the cost of lower throughput than a fully padded
variable-state rollout.

The default target policy is `complete_only`; `reconstructable_partial` and
`diagnostic_all` are explicit alternatives. Allowed mother tokens are derived
per level from eligible training targets. Pointer decoding applies target-level
and node masks, recursive-source conflicts, charge/type compatibility where
configured, and a minimum pointer probability in addition to cardinality.

Resume restores model/encoder/leaf-PID head, optimizer, scheduler, AMP scaler,
Python/NumPy/Torch/CUDA RNG, step/epoch, scheduling state, channel memory bank,
normalization, and split hash. Schema, PID vocabulary, feature specification,
and split mismatches are rejected unless an explicit supported override is
used. YAML precedence is defaults, then YAML, then explicitly supplied CLI
arguments.

## Runtime transform, index, and exact cursor

The streaming data module normalizes static detector-specific track/ECL blocks.
Common and composite values stay in physical units until the model-owned
runtime transform. The same checkpointed buffers normalize the initial context
and every PID/composite-rebuilt context. Categorical compatibility slots are
masked and dedicated embeddings carry their meaning. Pretraining uses this
same transform before contextual encoding.

`scripts/build_dataset_index.py` writes
`hypertagging-dataset-index-v1`: source-safe split inputs, Welford
count/mean/M2, allowed types, bounded capacity/cardinality histograms,
depth/completeness/category counts, and schema/PID/feature contracts.
`--dataset-index` initializes training from those sufficient statistics;
`--rescan-dataset` is explicit.

For `num_workers=0`, checkpoints persist epoch, batch index, events consumed,
and logical shard/row-group/offset cursor fields. Resume deterministically
replays the bounded shuffle to the saved cursor, prioritizing exactness over
restart latency. Exact mid-epoch resume with prefetched workers is rejected.

Scheduled sampling chooses one primary truth or predicted context for each
event and target level. Predicted targets are aligned by recursive leaf sources
and unrepresentable targets are counted. `--auxiliary-teacher-weight` defaults
to zero, so predicted events are not counted a second time.

Validation reports per-level object, pointer, type, and cardinality metrics
with numerator/denominator counts, plus source-aligned rollout, confidence,
p4-closure, and validity metrics over the configured event samples.
