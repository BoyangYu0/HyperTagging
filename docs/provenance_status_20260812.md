# Training provenance status — 2026-08-14

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
checkpointing, and exact-GRES Slurm execution path without opening the
sealed-test role. The tracked evidence summary is
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
index.

Three serialized V100 diagnostic jobs were submitted on 2026-08-14 after live
queue and typed-GRES checks. Jobs `15744980` and `15745064` failed safely in the
bootstrap, preserving their logs and (for the second) a hashed failure receipt;
the fixes are commits `5f6dfc6` and `63cc269`. Job `15745095` completed in
12:07 on `th-cl-nv01` with exact `gpu:v100:1` request/allocation proof, frozen
CUDA-environment preflight, 10 trainer steps, all curriculum phases, and
256-event validation. Its completion receipt internal SHA-256 is `7da521ee...`,
checkpoint SHA-256 is `6a5fefb6...`, and metrics SHA-256 is `c164cde2...`.
These are diagnostic execution observations, not scientific evidence. The
verified a1e8102 H200 contract was not submitted because live Slurm removed
`gpu:h200nvl` from `kng-cl-nv03` and partition `inter` TRES.

A final fourth serialized V100 diagnostic, job `15745941`, ran from clean
source `0c5e054...` with the actual capacity-approved `small_candidate` and the
frozen 35k train plus validation-only index. It completed four optimizer steps
and phase indices `[0,1,2,3]` in 2:12, then completed one 2-event validation
batch across four named views. Structured stdout reported elapsed time and
event-view throughput after every view. The v2 receipt internal SHA-256 is
`04b8a81c...`, checkpoint SHA-256 is `f203b58e...`, and metrics SHA-256 is
`83587a03...`. Nine periodic telemetry samples were cleaned up normally and
hashed into the receipt; their peaks were 554 MiB, 11% GPU utilization, and
33 C. This is diagnostic runtime evidence only and did not open sealed test.

Both environments are now managed with
`/home/b/Boyang.Yu/.local/bin/uv`. The project frozen all-extras sync preserved
the corrected `uv.lock` SHA-256 `38e6093...`; its editable-root metadata now
contains all 10 direct runtime dependencies, including SciPy and PyYAML. Two
consecutive frozen syncs audited 124 packages without changes, `uv pip check`
passed, and imports reported SciPy 1.18.0 and PyYAML 6.0.3. Strict hashed GPU sync against
`requirements-cu126.lock` made no changes, and both uv package checks plus the
GPU lock-only preflight pass. The tracked `scripts/activate_env.sh` exposes
explicit `project` and `gpu` modes; `.bashrc` functions `htenv` and `htgpu`
were verified in a fresh Bash shell.

Scientific Slurm submission remains forbidden until the production object is
recovered and its tree verified, the frozen GPU environment passes a fresh
in-allocation preflight for the scientific job, and the final reviewed commit
is immutably tagged. A blocked no-submit contract freezes the first full run to
the 35k train selection plus validation and the capacity-approved
`small_candidate`; its prologue refuses execution while these blockers remain.

The authoritative machine-readable status is
`configs/training_selection/production_1m_20260812/provenance_status.json`.
Validate it with:

```bash
python scripts/validate_training_provenance.py
python scripts/validate_training_provenance.py --require-scientific-slurm-ready
```

The first command validates the document and reports blockers. The second must
exit nonzero while any scientific blocker remains.
