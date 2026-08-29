# Scripts

Command-line entry points added during migration phases support CPU dry-runs
before full GPU/HPC execution paths.

The `create_*_inspection_notebook.py` generators and
`execute_notebook_smoke_tests.py` provide deterministic v1/v2 fixture
inspection. They are CPU-only and never submit Condor jobs.

Examples:

```bash
uv --cache-dir /tmp/uv-cache run python scripts/train_embedding.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_link.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_reconstruction.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/train_gpt_like.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/evaluate_reconstruction.py --dry-run --device cpu
uv --cache-dir /tmp/uv-cache run python scripts/run_gpt_like.py --dry-run --device cpu
```

`evaluate_reconstruction.py` above is the historical GraFEI-shaped fixture.
For the current level-autoregressive model, use the CPU-only offline evaluator:

```bash
./.venv/bin/python scripts/validate_reconstruction_checkpoint_pair.py \
  --pretraining-checkpoint /path/to/pretraining.pt \
  --reconstruction-checkpoint /path/to/reconstruction.pt

./.venv/bin/python scripts/evaluate_full_decay.py \
  --pretraining-checkpoint /path/to/pretraining.pt \
  --reconstruction-checkpoint /path/to/reconstruction.pt \
  --data /path/to/schema-v4-selection.json \
  --dataset-index /path/to/dataset-index.json \
  --split validation --scope both --max-events 100 \
  --output /path/to/full-decay-evaluation.json
```

Use repeatable `--source-category mixed`/`--source-category ccbar` filters for
separate B-pair and continuum reports. Validation defaults to the checkpoint's
ordered rollout UID cohort and learned-confidence policy. Continuum output is
one row per explicit top-level truth component; it is not an invented two-side
split.

This command reads preprocessed mDST training inputs, not raw mDST/basf2 data
and not GraFEI `pairs`. It hides CUDA, restores the serialized model/data
contracts on CPU, projects to detector FSPs only, and keeps higher-level
particles exclusively in the metric view. The full scientific contract is in
`docs/full_decay_reconstruction_evaluation.md`; training/HPO integration
invariants and open throughput work are tracked in
`docs/full_decay_training_hpo_compatibility_handoff.md` and its JSON companion.
