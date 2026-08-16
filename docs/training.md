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

Hyperbolic runs log exact-edge tree-relation loss and accuracy, topology-safe
directed-parent ranking loss/accuracy and its eligible-negative denominator,
radius-depth, pooled-channel, variance, covariance, per-loss gradient norms to
the hyperbolic projection, per-projection gradient norms, and
the collapse diagnostics documented in
`docs/hyperbolic_level_autoregressive_reconstruction.md`.
Validation prefixes the main principal/relation metrics by its named curriculum
view and always emits separate FSP-only and truth-guided-multilevel relation
accuracy/denominator diagnostics, even for a one-batch bounded validation.
Exact structural relation inputs are target-only
by default; `truth_guided_structural_relation_inputs` is a compatibility
ablation and is serialized with the training configuration.

`validate_every` triggers held-out validation over a fixed event cohort.
Scientific mode selects manifest-validation-role UIDs by a stable hash and
serializes them; `validation_batches` is retained for non-scientific CI only.
Validation aggregates total/component losses, relation accuracy,
topology-safe parent ranking, tree distance, radius monotonicity,
variance/covariance/effective rank/boundary fraction, channel retrieval, and
leaf PID accuracy/entropy. Pretraining preserves its configured full-objective
`best.pt`; optional diagnostic best tracks cover principal topology, parent
ranking, tree distance, non-collapse, and channel retrieval without replacing
that principal selection rule. `latest.pt` and `checkpoint.pt` remain separate.
`log_every` controls training-log cadence.

Reconstruction runs log object/no-object and mother-type accuracy, Level-1
pointer precision/recall, component losses, Hungarian matches, and both
teacher-forced/free-rollout results. Full evaluation helpers report exact tree
match, edge precision/recall/F1, validity, p4 closure, node count, and maximum
depth. Production reporting should additionally group reconstruction
efficiency by channel, multiplicity, and depth.

Reconstruction writes independent `best_teacher_forced.pt`,
`best_rollout_edge_f1.pt`, and `best_rollout_tree_validity.pt` tracks, plus
`latest.pt` and `checkpoint.pt`. A rollout track can update only on a validation
step that actually ran rollout. The configurable `best.pt` compatibility alias
records its metric, mode, denominator, validation event UIDs, rollout/constraint
policy, PID mode, thresholds, and selection reason. Resume rejects changed
checkpoint-selection semantics. A rollout checkpoint can become primary only
when its rollout denominator and edge/p4 denominators are nonzero, tree
validity is at least 0.999, p4 closure is complete at the serialized tolerance,
and recursive-source conflicts are zero. Teacher-forced loss remains an
independent diagnostic track.

Source-bound frozen-encoder transfer contracts retain the original
`hypertagging-reconstruction-transfer-probe-v1` 100-step profile. The versioned
`hypertagging-reconstruction-transfer-probe-v2-headwarmup-200` profile is
restricted to the step-3282 calibration and differs only by setting the
optimizer/LR horizon, encoder-freeze horizon, and transferred-leaf-head freeze
horizon to 200. It retains the v1 seed, batches, fixed validation/rollout
cohorts and cadences, architecture, target policy, scientific primary-selection
gates, source immutability, and sealed-test prohibition. Rendering emits an
explicit `sbatch` command but never submits it; the terminal receipt validates
the result's optimizer-step count against the selected version rather than a
hard-coded horizon.

The follow-up
`hypertagging-reconstruction-transfer-probe-v3-query-activation-balance`
profile is restricted to step 3282 and the original 100-step horizon. A
read-only saved-checkpoint diagnostic on restored fixed-validation UIDs found
that every level-1 query was below the unchanged 0.5 object threshold at both
steps 100 and 200; no query also supplied the two pointer probabilities above
0.5 required by its predicted cardinality. The v3 arm therefore changes only
the matched-positive focal weights for the two binary decoder decisions from
2/4 (object/pointer) to 16/16. The value 16 is fixed from the registered
32-query capacity and median two level-1 mothers. It does not change the
encoder, checkpoint, horizon, data roles/cohorts, seed, inference thresholds,
rollout denominators, or primary eligibility gates.

`scripts/diagnose_reconstruction_query_activation.py` is a fail-closed,
read-only validation diagnostic. It restores saved validation UIDs, rejects any
data module with test rows, checks checkpoint hashes before and after use, and
emits schema-versioned finite query/null probabilities, exact inference-stage
counts, Hungarian costs and class margins, derived continue/stop probabilities
(the architecture has no separate depth head), pre-/post-pruning node depth,
representability/loss slices, and gradient reachability.

The first v3 optimization rollout exposed a runtime-normalization support bug:
the data-fitted composite normalizer has zero training observations for the
runtime-only pointer-confidence mean/minimum slots, so its numerical 1e-6
fallback scale mapped an ordinary 0.51 confidence to 510,000 and overflowed
FP16 on the next scheduled-sampling pass. Runtime normalization now uses
fitted statistics only for slots with positive training support and identity
scaling for unsupported runtime-only slots. Available non-finite values still
fail closed. This changes no query gate, threshold, cohort, denominator,
schedule, focal weight, or PID decision.

`configs/hyperbolic_pretrain_pilot.yaml` requires bounded objective-gradient
preflight evidence. It reports raw and weighted magnitudes, active denominators,
projection-specific norms, and pairwise cosines, and checks zero/non-finite or
grossly dominant objectives. The four `pretrain_stage*` configurations are
campaign ablations, not a replacement production default.

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

## uv environments and activation

Use the installed uv binary explicitly. The project environment is synchronized
without relocking, and the CUDA environment is synchronized from its separate
hashed requirements lock:

```bash
/home/b/Boyang.Yu/.local/bin/uv sync --frozen --all-extras
/home/b/Boyang.Yu/.local/bin/uv pip sync --strict --require-hashes \
  --python /project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  environment/gpu/requirements-cu126.lock
```

The committed `uv.lock` records every direct runtime dependency from
`pyproject.toml`, including `scipy` and `PyYAML`, in both the editable-root
dependency list and `requires-dist` metadata. Reproduce and verify the project
environment without manual package restoration:

```bash
/home/b/Boyang.Yu/.local/bin/uv sync --frozen --all-extras
/home/b/Boyang.Yu/.local/bin/uv pip check
/home/b/Boyang.Yu/.local/bin/uv run --frozen --all-extras \
  python scripts/check_uv_lock_direct_dependencies.py
```

Source the tracked helper directly, or use the Bash functions installed below:

```bash
source scripts/activate_env.sh project  # project .venv
source scripts/activate_env.sh gpu      # frozen CUDA 12.6 venv
htenv
htgpu
```

Each activation changes to this checkout and sets `HYPERTAGGING_ENV_MODE` to
`project` or `gpu`.

## GPU and production boundary

Real data, real model sizes, and long training use guarded batch execution.
CUDA outside an authorized batch or bounded diagnostic is refused unless the
command is explicitly tiny and
`--allow-local-tiny-gpu-test` is supplied; that path checks `condor_q`,
`nvidia-smi`, and active GPU processes first. No training script submits a job.

See `docs/condor.md` for render and experiment commands.

## Real trainer state and transfer

The parquet data module accepts a file, directory, shards, or JSON/JSONL
manifest; it checks global event UIDs, creates a stable source-aware split,
fits masked normalization on training only, and raises rather than silently
dropping node overflow. Pretraining progresses once through four checkpointed
phases: 20% FSP-only, 25% truth-guided distance/radius, 30% multilevel channel
memory, and 25% corrupted composites/hard negatives. Explicit step or event
durations and the phase cursor are part of the resume contract; the old cycling
behavior exists only as `legacy_alternating_ablation`. Reconstruction optimizes every target level in each
batch. Teacher-forced validation is batched; seeded scheduled/free checks use
the bounded `evaluation_reference_rollout`. `batched_free_rollout` performs
multi-event, multi-level padded decoding and append with per-event stopping and
CPU reference equivalence. It is not production-ready until the guarded CUDA
smoke and representative memory/throughput profile have run.

Atomic checkpoints include full and encoder-only states, optimizer, scheduler,
AMP scaler, epoch/step/config, git commit, schema/PID/feature specification,
split hash, normalization, metrics, confidence-training state, and RNG states.
`--pretrained-encoder` loads only compatible shared-encoder keys and reports
loaded, missing, unexpected, and shape-mismatched keys. Use
`--freeze-pretrained-encoder-steps` and `--encoder-lr-multiplier` for transfer.

Both real trainers use a versioned optimizer-step schedule with linear warmup
(5% by scientific default, with an explicit cap/override) followed by cosine
decay. The resolved total, warmup, floor, base learning rates, scheduler state,
and current step are serialized. Resume refuses legacy checkpoints without the
new schedule contract instead of silently assigning them a different curve.
Hyperbolic exp/log maps, Poincare distances/radii, and VIC variance/covariance/
effective-rank calculations run in FP32 with AMP disabled around those kernels.

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
The implementation uses per-event reference micro-rollouts for predicted
contexts, which keeps alignment explicit at the cost of lower throughput than
a fully padded variable-state rollout. Weighted set packing and beam search are
bounded evaluation comparators, never training or production defaults.

The default target policy is `complete_only`; `reconstructable_partial` and
`diagnostic_all` are explicit alternatives. A versioned static reduced-PID
mother ontology always rejects unknown and species that occur only as leaves;
K_L and electron tokens remain eligible when the retained truth node is a
complete composite target. Eligible training-target frequencies add hard,
soft, or off empirical level priors.
Pointer decoding applies target-level
and node masks, recursive-source conflicts, charge/type compatibility where
configured, and a minimum pointer probability in addition to cardinality.

`complete_only` is not complete full-event reconstruction: it excludes targets
made incomplete by neutrinos, K_L treatment, acceptance/reconstruction loss,
or other missing daughters. Explicit policy configs live under
`configs/target_policies/`; their metric namespaces and denominators must never
be merged. Run `scripts/report_reconstruction_capacity.py` against the exact
policy-specific dataset index before production training. It reports every
retained level, distributions, configured limits, overflow, and margin and
refuses any overflow. Source/event/neutral multiplicity slices are exact;
channel-frequency slices are exact up to the explicit tracked-signature cap and
publish an overflow/coverage field rather than growing without bound.

Resume restores model/encoder/leaf-PID head, optimizer, scheduler, AMP scaler,
Python/NumPy/Torch/CUDA RNG, step/epoch, scheduling state, channel memory bank,
normalization, and split hash. Schema, PID vocabulary, feature specification,
and split mismatches are rejected unless an explicit supported override is
used. YAML precedence is defaults, then YAML, then explicitly supplied CLI
arguments.

## Runtime transform, index, and exact cursor

The streaming data module normalizes static detector-specific track/ECL blocks;
the optional KLM adapter applies its versioned fixed unit scales and masks.
Common and composite values stay in physical units until the model-owned
runtime transform. The same checkpointed buffers normalize the initial context
and every PID/composite-rebuilt context. Categorical compatibility slots are
masked and dedicated embeddings carry their meaning. Pretraining uses this
same transform before contextual encoding.

`scripts/build_dataset_index.py` writes
`hypertagging-dataset-index-v2`: source-safe split inputs, Welford
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
