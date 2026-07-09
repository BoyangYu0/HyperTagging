# Training

CPU dry-runs:

```bash
uv --cache-dir /tmp/uv-cache run python scripts/train_hyperbolic_pretrain.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2

uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
```

CUDA full training is refused outside SLURM.  Checkpoints should include model,
optimizer, scheduler, step, epoch, config, metrics, and preprocessing schema
version.  JSONL logging is available through `hypertagging.training.logging`.

Tiny local CUDA smoke tests require `--tiny`, small `--max-steps`, small
`--batch-size`, and `--allow-local-tiny-gpu-test`; the guard checks `squeue` and
`nvidia-smi` before running.
