# Notebooks

The deterministic notebook suite uses CPU fixtures and schema-v4 by default.
It is software validation, not a physics-performance measurement. Ask the
runner for the authoritative count with
`scripts/execute_notebook_smoke_tests.py --list`; the current suite has 15
groups:

| Group | Notebook |
|---|---|
| `leaf_composite` | `inspect_leaf_pid_and_composite_inputs.ipynb` |
| `streaming` | `inspect_streaming_dataset.ipynb` |
| `leaf_pid` | `inspect_leaf_input_pid_contract.ipynb` |
| `dataset` | `inspect_preprocessed_dataset.ipynb` |
| `hyperbolic` | `inspect_hyperbolic_pretraining.ipynb` |
| `exact_geometry` | `inspect_exact_tree_geometry_and_loss_scales.ipynb` |
| `rollout_search` | `inspect_rollout_search_and_calibration.ipynb` |
| `runtime_scaling` | `inspect_runtime_scaling.ipynb` |
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

`inspect_trained_physics_validation.ipynb` is also outside the fixture runner.
It requires `HYPERTAGGING_REAL_PARQUET`, `HYPERTAGGING_DATASET_INDEX`, and
`HYPERTAGGING_TRAINED_CHECKPOINT`, restores checkpoint normalization and
feature/model contracts, and fails with `REAL INPUT REQUIRED` when they are
missing. It never substitutes a fixture for efficiency, purity, calibration,
mass, Mbc/DeltaE, missing-mass, rare-channel, or rollout claims.

`inspect_first_level_ambiguity.ipynb` is a separate diagnostic notebook. Its
bounded fixture output diagnoses Level-0-to-1 factorization behavior; it is not
part of the stable 15-group CI contract and makes no trained-performance claim.

Set `HYPERTAGGING_PARQUET` for supported real-data inspection notebooks and
write figures/executed copies to `/tmp` or the configured data volume. Checked-
in notebooks are regenerated from `scripts/create_*_notebook.py`; do not treat
their unexecuted outputs as measurements.
