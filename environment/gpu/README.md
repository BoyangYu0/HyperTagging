# Frozen one-GPU environment

This environment is separate from the repository's CPU-only `uv.lock`. The
resolved lock pins the CUDA 12.6 PyTorch build and every transitive Python
dependency with package hashes for Python 3.11 on Linux x86-64. This readiness
tranche generated and verified the lock but did not create an environment.

After review, create the immutable environment at the path used by the Slurm
renderer:

```bash
/home/b/Boyang.Yu/.local/bin/uv venv --python 3.11 /project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1
/home/b/Boyang.Yu/.local/bin/uv pip sync --strict --require-hashes \
  --python /project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  environment/gpu/requirements-cu126.lock
/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1/bin/python \
  scripts/slurm/preflight_gpu_environment.py --lock-only
```

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
