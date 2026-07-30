# Notebooks

Deterministically generated, CPU-executable inspection artifacts:

- `inspect_preprocessed_dataset.ipynb`
- `inspect_hyperbolic_pretraining.ipynb`
- `inspect_level_autoregressive_reconstruction.ipynb`
- `preprocessing_qa_report.ipynb`
- `preprocessing_four_momentum_validation.ipynb`
- the retained historical/direct-GPT integration notebook.

Run the revised fixture suite with:

```bash
python scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-notebook-smoke
```

All six notebooks above execute in clearly labelled fixture mode without basf2
or real mDST. Set
`HYPERTAGGING_PARQUET` for a real schema-v1/v2 shard and place figures on the
configured data volume. See `docs/dataset_visualization.md`.
