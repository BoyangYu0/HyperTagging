# hypertagging-unified

Unified migration workspace for the historical HyperTagging repositories. This
repository now contains migrated, CPU-testable components through Goal B Phase
12. It is still a migration artifact, not a polished reproduction package.

The current revision adds a production-oriented, CPU-verified implementation
of hyperbolic-pretrained level-autoregressive set reconstruction while leaving
the verified v1 preprocessing contract intact:

- `direct-mdst-tree-v2` heterogeneous node blocks and explicit availability;
- v1-to-v2 in-memory adaptation without fabricated detector fields;
- canonical structured two-B channels and unordered Upsilon pair identity;
- track, ECL, and composite adapters in one shared latent/Poincare space;
- LCA, true-Poincare parent, radius-depth, channel, variance, and covariance
  pretraining objectives;
- relation bias applied to actual stair-causal attention logits;
- teacher-forced, scheduled, and free full-tree rollout with daughter-summed
  composite p4;
- fixture-executable dataset, geometry, rollout, and QA notebooks.

These additions are software-validated on tiny CPU fixtures. Full-training
physics performance remains unverified and HTCondor-only.

Scientific behavior is preserved by keeping historical variants separate and by
adding equivalence or smoke tests before each migrated component is treated as
usable. Full training, full-data preprocessing, and performance reproduction
remain GPU/HPC-only and are not verified from repository contents.

## Historical Sources

The unified package consolidates reusable code from:

- `HyperTagging`: original Toy-MC/BASF2 HyperTagging studies.
- `HyperTaggingColab`: cleaner collaboration package structure and embedding
  utilities.
- `graFEI`: early full GraFEI HyperTagging workflow.
- `graFEI_reduced`: reduced/final GraFEI workflow source for many model and
  reconstruction definitions.
- `graFEI_gpt`: GPT-like/autoregressive GraFEI branch.

See `REPOSITORY_MAP.md` for the current compact mapping.

## Data Roots

- Toy-MC inputs after BASF2 generation and before preprocessing:
  `/home/boyang/data/MC`
- Original GraFEI inputs before preprocessing:
  `/home/boyang/data/graFEI`

Derived/preprocessed folders such as `emb/`, `comb/`, `gpt/`, `ConstEmb/`, and
`RegEmb/` are not treated as original inputs.

## What Is Migrated

- Utilities: padding, device, seeds, simple I/O, checkpoint loading with CPU
  `map_location`.
- Data: provisional batch contracts, tiny fixtures, dry-run preprocessing
  adapters, GPT-like collate helpers.
- Models: historical embedding, link, reconstruction, and GPT-like model
  classes where the source class is self-contained.
- Losses: tensor-level embedding, reconstruction, link, and GPT-like losses.
- Training: CPU dry-run loops for embedding, link, reconstruction, and GPT-like
  stages.
- Reconstruction: single-level reconstruction helpers and GraFEI full
  reconstruction evaluation on tiny events.
- Examples: minimal CPU examples for Toy-MC, GraFEI, and GPT-like workflows.

## Known Limitations

- Historical preprocessing scripts still contain hard-coded legacy paths. The
  migrated adapters construct dry-run commands and document intended input
  roots; they do not rewrite scientific preprocessing semantics.
- Full epoch training loops and full-data loader orchestration remain
  production integration work; revised checkpoint, split, normalization, and
  evaluation surfaces are implemented and CPU-tested.
- `graFEI_gpt.models.MultiGPT` was not executable as written. The migrated
  `MultiGPT` preserves the verified autoregressive embedding reconstruction and
  embedding-link branches; the historical PDG/feature branch remains ambiguous.
- Full performance numbers and physics results are not verified from repository
  contents.
- The direct-mDST inspection and four-momentum validation notebooks are
  executable integration artifacts. Historical exploratory notebooks remain
  references in the source repositories.

## CPU Smoke Commands

Create the CPU-only project environment on the data volume (not in the AFS
checkout):

```bash
UV_PROJECT_ENVIRONMENT=/data/dust/user/boyangyu/uv_env \
UV_CACHE_DIR=/data/dust/user/boyangyu/uv_cache \
uv sync --python 3.11 --all-extras
```

Run the full CPU test suite:

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest
```

Run dry-run CLIs:

```bash
uv --cache-dir /tmp/uv-cache run python scripts/train_embedding.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_link.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_reconstruction.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_gpt_like.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/evaluate_reconstruction.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/run_gpt_like.py --dry-run --device cpu
```

Run the new level-autoregressive CPU dry-runs:

```bash
uv --cache-dir /tmp/uv-cache run python scripts/train_hyperbolic_pretrain.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
```

Full CUDA training and full-data preprocessing are HTCondor-only. Local CUDA is
refused unless it is an explicit tiny smoke test guarded by `condor_q` and
`nvidia-smi` checks.  See `docs/hyperbolic_level_autoregressive_reconstruction.md`,
`docs/training.md`, and `docs/condor.md`.

## Direct mDST Preprocessing

The unified repo now includes a direct-mDST preprocessing path under
`src/hypertagging/preprocessing`. Run the producer inside a basf2 environment:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/processed.parquet \
  --schema-version direct-mdst-tree-v4 \
  --max-events 50
```

Validate or inspect output with normal Python:

```bash
/data/dust/user/boyangyu/uv_env/bin/python scripts/verify_preprocessing.py \
  --input /data/dust/user/boyangyu/hypertagging/processed.parquet \
  --all-events --check-tree --check-p4 --check-pid
```

See `docs/preprocessing_design.md` for the schema, legacy compatibility notes,
and the reco-kinematics/truth-topology separation.

New files default to the truth-clean, streamable v4 schema. V1/v2/v3 remain
loadable through the versioned compatibility adapters; trainers require
explicit diagnostic opt-in for their legacy-conflated PID summaries.

Inspect the real parquet schema, variable event/tree structure, GPT attention
contract, and a complete CPU forward/loss/backward/optimizer step in:

```text
notebooks/inspect_preprocessed_parquet_and_gpt_like.ipynb
```

The corresponding direct-tree adapter is
`hypertagging.data.direct_gpt`. It handles variable node counts and tree depths;
the historical fixed-width level collator must not be applied directly to this
schema.

Run all revised v1/v2/v3 inspection notebooks on deterministic CPU fixtures:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-notebook-smoke
```

For real data, set `HYPERTAGGING_PARQUET` and a data-volume
`HYPERTAGGING_FIGURE_DIR`; see `docs/dataset_visualization.md`.

Plan a balanced, exactly sharded 10-million-input-event production and print the
HTCondor submit description without submitting:

```bash
scripts/condor/submit_mdst_production_10m.sh --dry-run
```

After reviewing the manifest, resources, and a small pilot, submit explicitly:

```bash
scripts/condor/submit_mdst_production_10m.sh --submit
```

Run minimal examples:

```bash
uv --cache-dir /tmp/uv-cache run python examples/toy_mc_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/grafei_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/gpt_like_minimal/run_example.py
```

## Equivalence Status

- Exact parity tests exist for selected utilities, losses, model state/forward
  behavior, one-step reconstruction formulas, full-reconstruction evaluation
  formulas, GPT masks/collate behavior, and dry-run CLI surfaces.
- Tiny synthetic fixtures are used for CPU tests. They validate shape,
  conventions, and formula preservation, not final scientific results.
- Components listed in `MIGRATION_PLAN.md` under "must not be refactored" should
  remain unchanged until broader equivalence tests exist.

## Correctness revision: schema-v3 and real CPU pilots

`direct-mdst-tree-v3` is the production schema. Raw generic-mDST tracks use a
data-independent best-p-value fit (pion closest-mass fallback), reconstructed
charge, canonical pion energy, all e/mu/pi/K/p energy hypotheses, and available
PIDLikelihood log-likelihoods. MC PID/charge/p4 occupy separate truth or
diagnostic fields. V1/v2 remain loadable through conservative adapters.

Real (non-dry-run) CPU pilots now accept parquet data:

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

The same CLIs run full CUDA jobs inside HTCondor. Outside Condor, full CUDA is
refused. No job is submitted by either trainer.

## Truth-clean schema-v4 and streaming training

New preprocessing defaults to `direct-mdst-tree-v4`. V4 writes one event per
Parquet row with bounded row-group buffering and a metadata sidecar. It keeps
`daughter_input_pid_histogram` separate from the diagnostic
`daughter_truth_pid_histogram`; only the former is a model input. V1/v2/v3
remain readable, but real trainers reject their legacy-conflated PID contract
unless `--allow-legacy-conflated` is supplied. Such runs are recorded as
diagnostic and are not data-compatible performance measurements.

Raw tracks enter with unknown input PID. A detector-context pass predicts a
charge-compatible leaf PID distribution, rebuilds the differentiable track
energy and physical relations, and a second reconstruction-context pass
predicts Level-1 mothers and pointers. Composite p4 remains an exact recursive
daughter sum. Production JSONL manifests may use `output_file`; the trainers
iterate event rows lazily, fit masked Welford normalization on training only,
and never materialize every batch.

For a schema-v4 pilot:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/input.root --output /data/path/pilot-v4.parquet \
  --schema-version direct-mdst-tree-v4 --max-events 50 \
  --event-buffer-size 32 --row-group-size 16
```
