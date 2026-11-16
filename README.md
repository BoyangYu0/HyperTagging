# HyperTagging

HyperTagging combines detector-input preprocessing, hyperbolic pretraining and
level-autoregressive full-decay reconstruction. The current data contract is
`direct-mdst-tree-v4`; historical adapters remain compatibility references.

Start with [AGENTS.md](AGENTS.md) for working rules and
[the study loop](docs/study_loop.md) for planning, parallel execution, evaluation,
result tracking and dashboard updates. [Current audit status](docs/audits/current_status.md)
owns readiness claims; CPU fixtures establish software behavior, not physics performance.

## Setup

For a new host or account, follow [remote setup](docs/codex_remote_setup.md).
Reuse the installed environment, or install from the existing lock:

```bash
uv sync --frozen --all-extras --python 3.11
source scripts/activate_env.sh project
python scripts/check_uv_lock_direct_dependencies.py
```

The root lock selects CUDA 12.6 PyTorch wheels, which also run CPU fixtures.
The separately frozen GPU environment and shared-resource rules are in
[GPU setup](environment/gpu/README.md). Production jobs use
[Slurm/HTCondor workflows](docs/condor.md).

## Working and data paths

| Path | Purpose |
| --- | --- |
| `src/hypertagging/` | Data, models, training, reconstruction, evaluation and deployment |
| `scripts/`, `configs/` | Entry points and reusable settings |
| `tests/`, `examples/` | Behavioral regressions and small runnable examples |
| `docs/`, `notebooks/` | Reference guides and notebook source |
| Configured data volume | Datasets, checkpoints, logs, private study records and generated output |

Use distinct study/operation output directories. Keep source and small portable
publication evidence in Git; follow [storage and recovery](docs/repository_storage.md)
for large artifacts. Train on allocated GPUs while evaluating completed checkpoints
on CPU and progressing independent registered studies.

## Training and evaluation

- [Training](docs/training.md): configuration, normalization, checkpoints and resume.
- [Preprocessing](docs/preprocessing_design.md): detector inputs and truth separation.
- [Full-decay evaluation](docs/full_decay_reconstruction_evaluation.md): fixed cohorts,
  native reconstruction, metrics, coverage and uncertainty.
- [Deployment](docs/basf2_onnx_full_decay.md): basf2 and ONNX contracts.
- [Examples](examples/README.md) and [notebooks](notebooks/README.md): inspection tools.

Use matched controls and track actual results, failures, denominators and compute.
Evaluate reconstruction quality as well as training loss; record the next decision
in the study ledger. Keep train, development and primary/test populations distinct.

## Validation

Run tests relevant to the change and reuse unchanged evidence. For the full CPU suite:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

See [test guidance](tests/README.md) for external evidence requirements. CI runs
CPU correctness and documentation checks; notebook execution has its own workflow.
Full training or broad repeated tests are unnecessary for prose-only changes.

## Results and dashboard

The [dashboard](https://boyangyu0.github.io/HyperTagging/wiki/_generated/status/)
shows reviewed public evidence. [Earlier studies](docs/wiki/studies.rst) summarize
historical conclusions and retain complete reports/downloads.

Record publish/defer at each scientific review. Publish meaningful findings,
accumulated diagnoses or material corrections, rather than every test or job update.
Use the existing publication owner and [wiki maintenance](docs/wiki/maintaining.rst)
for source allowlists, local builds, required CI, deployment and live verification.
Do not edit generated dashboard files.
