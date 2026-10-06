# Frozen one-GPU environment

This environment is separate from the repository's project `uv.lock` (which also selects CUDA wheels). The
resolved lock pins the CUDA 12.6 PyTorch build and every transitive Python
dependency with package hashes for Python 3.11 on Linux x86-64. The initial
readiness tranche generated the lock; later runtime receipts record an installed
environment. Reuse and verify your existing environment when present.

For a new installation, create an immutable environment on a writable volume
owned by the executing account. Replace the placeholder below. Do not recreate
or resynchronize an active frozen environment as part of an unrelated task:

```bash
ht_gpu_env=/path/to/writable/volume/envs/hypertagging-gpu-cu126-v1
uv venv --python 3.11 "$ht_gpu_env"
uv pip sync --strict --require-hashes \
  --python "$ht_gpu_env/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  environment/gpu/requirements-cu126.lock
"$ht_gpu_env/bin/python" scripts/slurm/preflight_gpu_environment.py --lock-only
source "$ht_gpu_env/bin/activate"
```

The `ht_gpu_env` variable above is a shell convenience, not a renderer setting.
Review the selected renderer/config's environment path and update it through its
supported options before rendering a job. `scripts/activate_env.sh gpu` still
selects the historical owner's fixed environment; use direct activation above
for another account or host. The project environment and this hash-pinned GPU
environment have distinct validation contracts. For complete new-account setup,
see [the remote Codex guide](../../docs/codex_remote_setup.md).

The full preflight is intentionally run only inside the exact one-GPU Slurm
allocation because it verifies CUDA availability, visible-device count, GPU
model, GRES, Python, imports, and exact installed distribution versions.

Node-local V100 testing is a separate, explicitly authorized operation. First
create the fresh three-sample admission receipt, then launch only through the
mandatory `run` watchdog subcommand (never invoke `local_microtest` directly):

```bash
python scripts/slurm/v100_local_admission.py admit \
  --output artifacts/local-v100/admission.json \
  --max-steps 10 --batch-size 2 --duration-seconds 300
python scripts/slurm/v100_local_admission.py run \
  --receipt artifacts/local-v100/admission.json \
  --completion-output artifacts/local-v100/completion.json -- \
  /path/to/frozen/bin/python scripts/train_hyperbolic_pretrain.py \
  --config configs/slurm/pretrain_diagnostic.yaml \
  --gpu-execution-mode local_microtest \
  --local-admission-receipt artifacts/local-v100/admission.json \
  --max-steps 10 --batch-size 2
```

The watchdog rechecks telemetry immediately, supplies its short-lived sentinel
and the single visible device, polls at most every 30 seconds, safely signals
and then bounds termination escalation, and writes a hashed completion receipt.
Admission alone is not scientific evidence. Scientific rendering requires both
receipt paths and accepts the completion only when its canonical hash and
admission hash binding are valid, host and GPU identity match, admission was
fresh at monitored start, elapsed time is bounded, `watchdog_reason` is
`trainer_exit`, `trainer_status` is zero, and at least one sample was monitored.
Deadline, trainer-failure, and foreign-process completions fail closed.

The renderer only prints a command; it never submits. Its command uses the
exact requested typed GRES, `--export=NIL` (never `NONE` or `ALL`), and an
absolute positional path to the hashed contract. Once every independent
scientific blocker is cleared, pass the two proof files explicitly as
`--local-admission-receipt ...` and `--local-completion-receipt ...`; neither
flag is sufficient alone.
