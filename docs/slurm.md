# SLURM Workflow

Full HyperTagging preprocessing and training should run through SLURM.

1. Edit `configs/slurm/default.yaml` to set account and partition.
2. Render a job without submitting:

```bash
python scripts/slurm/render_slurm_job.py --config configs/slurm/default.yaml \
  --command 'uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --device cuda --max-steps 1000'
```

3. Review the rendered script and run `shellcheck` if available.
4. Submit explicitly with `sbatch`.
5. Monitor with `squeue --me`.

The repository never calls `sbatch` automatically.
