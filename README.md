# HyperTagging

HyperTagging is a unified, CPU-testable implementation of the historical
HyperTagging, GraFEI, reduced-GraFEI, and GPT-like reconstruction studies. The
current production path uses truth-separated `direct-mdst-tree-v4` events, a
shared hyperbolic encoder, and level-autoregressive set reconstruction.

The software contracts are exercised on small CPU fixtures. A historical
bounded real-mDST pilot verified the Belle II preprocessing API path at its
recorded revision; no current-revision real pilot has run, and full-training
physics performance is not established by this repository. Treat the
[current audit](docs/audits/current_status.md) as the sole mutable readiness
statement; historical reports and run receipts are evidence for their recorded
revisions only.

## Documentation

- [Documentation wiki](https://boyangyu0.github.io/HyperTagging/wiki/)
- [Dashboard](https://boyangyu0.github.io/HyperTagging/wiki/_generated/status/)
- [Local wiki source](docs/index.rst)
- [Training and evaluation guide](docs/training.md)
- [Preprocessing contract](docs/preprocessing_design.md)
- [Full-decay evaluation contract](docs/full_decay_reconstruction_evaluation.md)
- [basf2 and ONNX deployment](docs/basf2_onnx_full_decay.md)

GitHub Pages publishes the validated site from trusted `master` deployments.
Build the same site locally:

```bash
python3.11 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
docs_out="$(mktemp -d)"
.venv-docs/bin/python scripts/build_docs.py --output "$docs_out"
docs_html="$(find "$docs_out" -maxdepth 1 -type d -name html -print -quit)"
.venv-docs/bin/python -m http.server 8000 --directory "$docs_html"
```

The build is offline after dependency installation and does not import the
project, open training data or checkpoints, or require basf2 or CUDA. See the
[wiki maintenance guide](docs/wiki/maintaining.rst) for its publication and
privacy boundary.

## Repository layout

- `src/hypertagging/`: data contracts, preprocessing, models, losses, training,
  reconstruction, evaluation, and deployment integration.
- `configs/`: model presets, ablations, data selections, and guarded production
  contracts.
- `scripts/`: user-facing preprocessing, training, evaluation, documentation,
  and batch-rendering commands.
- `tests/`: CPU contract, regression, and compatibility tests.
- `examples/`: minimal fixture-based CPU runs.
- `notebooks/`: the registry-backed inspection and evidence suite.
- `docs/`: current guides plus explicitly separated audit history.

Compatibility adapters keep v1-v3 inputs readable without making them the
production contract. The [schema history](docs/schema_migration_history.md)
records those boundaries. Exact scientific and implementation invariants for
contributors and coding agents are in [AGENTS.md](AGENTS.md).

## Project environment

Use the existing lock without relocking dependencies:

```bash
uv sync --frozen --all-extras
source scripts/activate_env.sh project
python scripts/check_uv_lock_direct_dependencies.py
```

Keep environments, datasets, checkpoints, notebook outputs, and other large
artifacts outside the checkout. The separately frozen CUDA environment and its
verification procedure are documented in
[environment/gpu/README.md](environment/gpu/README.md).

Run the CPU suite with bounded threads:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q
```

Small end-to-end training checks are available without data or a GPU:

```bash
python scripts/train_hyperbolic_pretrain.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
python scripts/execute_notebook_smoke_tests.py --list
```

Fixture results validate software behavior, not physics performance.

## Core workflows

### Preprocess mDST data

Run the producer inside the documented Belle II release, writing output to a
data volume:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
python3 scripts/preprocess_mdst.py \
  --input /path/to/generic_mdst.root \
  --output /data/volume/hypertagging/processed.parquet \
  --schema-version direct-mdst-tree-v4 \
  --max-events 50
```

Then verify the event rows with ordinary Python:

```bash
python scripts/verify_preprocessing.py \
  --input /data/volume/hypertagging/processed.parquet \
  --all-events --check-tree --check-p4 --check-pid
```

The producer never migrates existing Parquet files in place. Raw-track PID,
detector availability, model-input summaries, truth-only targets, event
identity, and recursive source provenance remain separate contracts.

### Train and evaluate

Real runs consume an immutable source-role selection and authenticated dataset
index. Fit normalization on training rows only, preserve the validation UID
cohort, and never use the sealed test role for tuning. A typical reconstruction
invocation is:

```bash
python scripts/train_level_reconstruction.py \
  --config configs/model_presets/production_baseline.yaml \
  --data /data/volume/manifest.jsonl \
  --dataset-index /data/volume/dataset_index.json \
  --pretrained-encoder /data/volume/pretrain/checkpoint.pt \
  --device cuda \
  --output-dir /data/volume/reconstruction
```

Production jobs use the guarded Slurm or HTCondor workflows documented in
[docs/condor.md](docs/condor.md); rendering a job never submits it. Validate a
trusted pretraining/reconstruction checkpoint pair before running the strict
CPU full-decay evaluator. Greedy inference remains the default. Opt-in bounded
full-depth beam search reports deployable top-1 separately from truth-only
oracle@K metrics.

## Updating the dashboard after training

The dashboard is generated from a small allowlist of tracked evidence declared
in `docs/_ext/wiki_status.py`; it does not poll jobs, inspect arbitrary run
directories, or execute training. Never edit or commit `docs/wiki/_generated/`
or `docs/_build/`.

After a training or evaluation run:

1. Validate the run receipt, then add only the reduced, reviewable metadata
   needed for the dashboard. Keep checkpoints, raw logs, host/user paths,
   scheduler identifiers, and data outside the publication boundary.
2. Append the relevant verification record and update the reviewed current
   status or issue ledger only when their claims actually change. Do not
   overwrite dated receipts; add a new immutable record and update the
   `SOURCE_PATHS` allowlist and its tests in the same branch when a campaign is
   superseded.
3. Regenerate nothing in place. Verify the derived dashboard with:

   ```bash
   python -m pytest -q tests/test_docs_*_cpu.py
   docs_check="$(mktemp -d)"
   python scripts/build_docs.py --output "$docs_check"
   docs_html="$(find "$docs_check" -maxdepth 1 -type d -name html -print -quit)"
   docs_generated="$(find "$docs_check" -type d -name _generated -print -quit)"
   python scripts/validate_docs.py \
     --html "$docs_html" \
     --generated "$docs_generated" \
     --check-generation --workflow
   ```

4. Open a pull request or merge request (PR/MR) containing the evidence and any
   required allowlist/test update. Review the generated status values, source
   hashes, freshness warnings, and scientific-claim boundary before merging.

On GitHub, [.github/workflows/docs.yml](.github/workflows/docs.yml) performs the
PR checks automatically; a successful trusted merge to `master` rebuilds and
deploys Pages. A GitLab mirror may use an MR for the same review flow, but this
repository does not ship a GitLab runner or Pages configuration: its MR
pipeline must run the commands above and the merged revision must reach the
canonical GitHub `master` branch for this Pages deployment to update.

Training itself deliberately never pushes, opens a PR/MR, or publishes the
site. The automatic portion begins when reviewed evidence is committed to the
branch. More detail is in the [maintenance guide](docs/wiki/maintaining.rst).

## Historical scope and limitations

The unified package preserves selected behavior from the original Toy-MC
HyperTagging repository, HyperTaggingColab, GraFEI, `graFEI_reduced`, and
`graFEI_gpt`. Historical paths and variants remain compatibility references,
not current production defaults. The detailed Phase 1-13 decisions and
historical source mapping are preserved in the
[migration provenance](docs/historical_migration_provenance.md). In particular:

- full-data reproduction still depends on external data and checkpoint
  provenance;
- legacy scripts may contain historical site-specific paths;
- synthetic CPU fixtures do not establish convergence, throughput, basf2
  interoperability, or physics quality;
- archived audits, reports, receipts, snapshots, and production data cards are
  retained intentionally and must not be treated as duplicate active guides.

Use the [examples guide](examples/README.md) for minimal runnable paths and the
[notebook guide](notebooks/README.md) for the evidence registry.
