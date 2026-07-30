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
