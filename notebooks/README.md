# Notebooks

The deterministic notebook suite uses CPU fixtures and schema-v4 by default.
It is software validation, not a physics-performance measurement. The 12
runner groups are:

| Group | Notebook |
|---|---|
| `leaf_composite` | `inspect_leaf_pid_and_composite_inputs.ipynb` |
| `streaming` | `inspect_streaming_dataset.ipynb` |
| `leaf_pid` | `inspect_leaf_input_pid_contract.ipynb` |
| `dataset` | `inspect_preprocessed_dataset.ipynb` |
| `hyperbolic` | `inspect_hyperbolic_pretraining.ipynb` |
| `capacity` | `inspect_query_capacity_and_losses.ipynb` |
| `training` | `inspect_training_pipeline.ipynb` |
| `reconstruction` | `inspect_level_autoregressive_reconstruction.ipynb` |
| `qa` | `preprocessing_qa_report.ipynb` |
| `manifest` | `inspect_production_manifest.ipynb` |
| `four_vector` | `preprocessing_four_momentum_validation.ipynb` |
| `direct_gpt` | `inspect_preprocessed_parquet_and_gpt_like.ipynb` |

Run all groups and retain executed copies outside the repository:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-current-head-notebooks
```

The runner checks semantic JSON/CSV/checkpoint artifacts in addition to
figure creation. Use repeated `--only GROUP` flags for a bounded subset.

`inspect_real_mdst_pilot.ipynb` is separate and deliberately has no fixture
fallback. It requires a published real schema-v4 pilot containing fewer than
100 input events and validates actual provenance, PIDLikelihood availability,
fit/energy selection, reconstructed charge, p4 closure, B-root discovery,
PID/level distributions, and bounded failure examples. See the first cell for
the exact basf2 and execution commands.

Set `HYPERTAGGING_PARQUET` for supported real-data inspection notebooks and
write figures/executed copies to `/tmp` or the configured data volume. Checked-
in notebooks are regenerated from `scripts/create_*_notebook.py`; do not treat
their unexecuted outputs as measurements.
