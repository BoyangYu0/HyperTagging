# HTCondor jobs for HyperTagging

## Purpose

How to create, review, test, and submit HTCondor jobs for this repository.

## Rules

- Search existing submissions before creating new ones.
- Preserve cluster-specific conventions.
- Never run real training locally on GPU.
- Local GPU is only for tiny smoke tests after `condor_q` and `nvidia-smi` checks.
- Real production/GPU training uses the site's guarded HTCondor or Slurm
  workflow. This skill covers HTCondor; bounded CPU diagnostics remain local.
- Do not call `condor_submit` automatically unless explicitly instructed.
- Always provide dry-run render commands first.
- Always log git commit, config, environment, and GPU state.
- Prefer short runtimes for debug jobs.
- Make CPU, memory, GPU, and runtime requests configurable.
- Write logs to `logs/condor`.
- Keep checkpoints and large generated artifacts on the configured data volume.
- Scientific training consumes an immutable source-role selection and its
  authenticated dataset index, with `--scientific-mode`; raw Parquet/JSONL
  prefixes are diagnostic inputs only.

## Workflow

1. Inspect configs and scripts.
2. Render the submit file and executable.
3. Run `shellcheck` on the executable if available.
4. Run a CPU dry-run.
5. Optionally run a local tiny GPU smoke test after the safety guard.
6. Submit with `condor_submit` only when explicitly requested.
7. Monitor with `condor_q`.
8. Inspect logs and checkpoints.

## Template

```text
universe = vanilla
executable = outputs/condor/job.sh
request_cpus = 4
request_memory = 16GB
request_gpus = 1
+RequestRuntime = 1800
output = logs/condor/job-$(ClusterId).$(ProcId).out
error = logs/condor/job-$(ClusterId).$(ProcId).err
log = logs/condor/job-$(ClusterId).log
queue
```

## Commands

```bash
condor_q
condor_submit job.sub
condor_rm <jobid>
nvidia-smi
uv --cache-dir /tmp/uv-cache run pytest
python scripts/condor/render_condor_job.py --dry-run --command '<command>'
```
