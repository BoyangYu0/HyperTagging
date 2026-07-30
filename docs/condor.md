# HTCondor Workflow

Full HyperTagging preprocessing and training should run through HTCondor.

1. Edit `configs/condor/default.yaml` to set the requested CPUs, memory, GPUs,
   and runtime.
2. Render a submit description and executable without submitting:

```bash
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml \
  --command 'uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --device cuda --max-steps 1000' \
  --output outputs/condor/level_reconstruction.sub
```

3. Review the generated `.sub` and `.sh` files and run `shellcheck` on the
   executable if available.
4. Submit explicitly with `condor_submit outputs/condor/level_reconstruction.sub`.
5. Monitor with `condor_q`.

The repository only calls `condor_submit` from a launcher when `--submit` is
given explicitly.
