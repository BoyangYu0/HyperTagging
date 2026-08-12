# Training provenance status — 2026-08-12

The integration source is local branch `training-integration-20260812` at the
clean no-ff merge `83776a2c4f9e7edd776915edf30816730881f9eb`. The two inputs are
preserved without rewriting history:

- `pre-training-master-537d7cc` preserves master tip `537d7cc`.
- `pre-training-focused-8e9d0b8` preserves focused tip `8e9d0b8`.

The 1M-event reduced campaign declares source commit
`f4e54df23b5c60115e475c5d68df4651899d678e` and tree
`b6e3a4118b960e3a4676a61af9601438d56cef96`. The commit object is unavailable
in the inspected repositories, and GitHub returned `not our ref`. Therefore the
tree cannot yet be independently derived from the commit object.

This is a fail-closed scientific provenance blocker, not a CPU-development
blocker. The final non-GPU readiness tranche completed the train+validation
event UID/source/task index gate, evidence-based capacity proof, trainer signal
checkpointing, and exact-GRES Slurm dry-run path without opening the sealed-test
role or submitting a job. The tracked evidence summary is
`configs/training_selection/production_1m_20260812/training_readiness.json`;
large generated evidence remains under the ignored
`artifacts/experiment_readiness/production_1m_20260812/train_035k/` directory.

The corrective CPU-only pass binds all launch values to a v2 hashed job
contract rendered with `--export=NIL` (and rejects `NONE`/`ALL`), proves the
exact typed one-GPU request from exact `ReqTRES` and `AllocTRES` maps plus an
exact typed `TresPerNode`, forwards USR1 through the batch wrapper, and
serializes pending scheduled validation for exact resume. Local V100 evidence
now requires both a fresh host/UUID/model-bound three-sample admission receipt
and a canonically hashed completion receipt proving the bound watchdog-monitored
trainer exited successfully. Scientific rendering requires the explicit
`--local-admission-receipt` and `--local-completion-receipt` pair. The frozen
lock and documented sync command both name PyPI and the PyTorch CUDA 12.6
index. No GPU command, environment installation, Slurm submission, or Slurm
mutation was performed.

Scientific Slurm submission remains forbidden until the production object is
recovered and its tree verified, the frozen GPU environment is installed and
passes its in-allocation preflight, a later explicitly authorized bounded local
V100 microtest produces bound admission and successful-completion receipts, and
this uncommitted tranche is reviewed on a clean immutable tagged commit.

The authoritative machine-readable status is
`configs/training_selection/production_1m_20260812/provenance_status.json`.
Validate it with:

```bash
python scripts/validate_training_provenance.py
python scripts/validate_training_provenance.py --require-scientific-slurm-ready
```

The first command validates the document and reports blockers. The second must
exit nonzero while any scientific blocker remains.
