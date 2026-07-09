# SLURM Scripts

These templates render jobs but do not call `sbatch` automatically.

```bash
python scripts/slurm/check_slurm_env.py
python scripts/slurm/render_slurm_job.py --config configs/slurm/default.yaml \
  --command 'uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --device cuda --tiny --dry-run'
```

Review the rendered script, set account/partition in `configs/slurm/default.yaml`,
then submit explicitly with `sbatch <job.sh>`.
