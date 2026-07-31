# Dataset and model visualization

All revised notebooks are generated deterministically and run on CPU without
basf2. If `HYPERTAGGING_PARQUET` is unset they create a tiny schema-v4 fixture
under `/tmp` and label every result as a software fixture. Set the variable to
inspect a real v1 or v2 shard. V1 is adapted in memory; the source parquet is
never modified.

Common parameters:

```bash
export HYPERTAGGING_PARQUET=/data/dust/user/boyangyu/hypertagging/shard.parquet
export HYPERTAGGING_FIGURE_DIR=/data/dust/user/boyangyu/hypertagging/figures
export HYPERTAGGING_NOTEBOOK_SEED=20260730
```

An optional checkpoint is selected with `HYPERTAGGING_CHECKPOINT`.

Execute a real-data copy without adding outputs to the repository:

```bash
export HYPERTAGGING_NOTEBOOK_RUN_DIR=/data/dust/user/boyangyu/hypertagging/notebook-runs
mkdir -p "$HYPERTAGGING_NOTEBOOK_RUN_DIR"
cp notebooks/inspect_preprocessed_dataset.ipynb \
   notebooks/inspect_hyperbolic_pretraining.ipynb \
   notebooks/inspect_level_autoregressive_reconstruction.ipynb \
   notebooks/preprocessing_qa_report.ipynb \
   notebooks/preprocessing_four_momentum_validation.ipynb \
   "$HYPERTAGGING_NOTEBOOK_RUN_DIR/"
for notebook in "$HYPERTAGGING_NOTEBOOK_RUN_DIR"/*.ipynb
do
  /data/dust/user/boyangyu/uv_env/bin/jupyter execute \
    "$notebook" --inplace --timeout=600
done
```

## Notebooks

The authoritative 12-group inventory is in `notebooks/README.md` and
`scripts/execute_notebook_smoke_tests.py`.

- `inspect_preprocessed_dataset.ipynb`: schemas, duplicate IDs, complete PID
  vocabulary, heterogeneous missingness, all level violations, representative
  trees and B branches, p4 closure, and structured channels.
- `inspect_hyperbolic_pretraining.ipynb`: disk/PCA projections, radius-depth
  checks, relation-distance separation, anti-collapse criteria, and pooled
  channel embeddings.
- `inspect_level_autoregressive_reconstruction.ipynb`: stair masks, actual
  relation logits, attention, mother queries, pointers, teacher forcing, free
  rollout, edge errors, and a CPU optimizer step.
- `preprocessing_qa_report.ipynb`: compact closure/level/finite-value checks and
  a machine-readable JSON report.
- `preprocessing_four_momentum_validation.ipynb`: detailed reco-versus-MC
  diagnostics for real samples; it accepts v1 and v2. MC mother p4 remains a
  diagnostic, never a reconstructed target.

Regenerate and execute the fixture suite:

```bash
python scripts/create_dataset_inspection_notebook.py
python scripts/create_hyperbolic_inspection_notebook.py
python scripts/create_reconstruction_inspection_notebook.py
python scripts/create_preprocessing_qa_notebook.py
python scripts/execute_notebook_smoke_tests.py --keep-output /tmp/hypertagging-notebooks
```

Jupyter kernels use local loopback ports. On a restricted sandbox the execution
command may need to run in the normal login-node environment. It is CPU-only
and does not inspect the Condor queue or use a GPU.
# Schema-v3 correctness notebooks

The deterministic suite also generates leaf PID/input-contract, query-capacity
and sparse-loss, real training-pipeline, and production-manifest notebooks.
Their smoke tests require structured CSV/JSON/checkpoint artifacts in addition
to figures. All default to a generated v3 fixture and accept real data through
`HYPERTAGGING_PARQUET`; the manifest notebook uses
`HYPERTAGGING_MANIFEST`.
