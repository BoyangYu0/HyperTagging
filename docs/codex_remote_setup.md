# Start a new Codex project on another device or account

This is the portable handoff for the Git repository
`https://github.com/BoyangYu0/HyperTagging`. Open the checkout containing
`pyproject.toml` and `AGENTS.md`. The surrounding `HyperTagging_uni` directory,
its historical sibling checkouts, and its old `Codex_prompt.md` are not required
and are not part of a fresh clone. Do not use that old implementation prompt as
an instruction to recreate the current pipeline.

This guide prepares a development project and bounded CPU checks. Real data,
trained checkpoints, Belle II software, and scheduler access require separate
provisioning. Repository access alone cannot provide them.

## Access and handoff prerequisites

Treat these identities separately:

| Access | Same account, new device | Different account |
| --- | --- | --- |
| Codex | Sign in on the execution device; confirm the intended workspace. | Sign in with the recipient's own account and permitted workspace. |
| Git repository | Configure Git authentication on that device. | Grant repository access if private; grant write access or use a fork for contributions. |
| Remote Linux host | Set up SSH/VPN access from the new device. | Provision an OS account and authorized SSH key. |
| Data, checkpoints, scheduler | Verify mounts and current permissions. | Grant storage/group and scheduler access separately; use recipient-owned output paths. |

Do not copy someone else's Codex authentication cache, SSH private keys, tokens,
virtual environment, or personal Codex configuration into the repository.
Account-specific tools/plugins are optional for the CPU setup below. Existing
Codex tasks and their conversation history are not required: record decisions,
remaining work, and evidence in repository files instead.

Before moving devices, commit and push the intended changes through the normal
review process, then record the repository URL, branch, and full commit SHA.
Uncommitted files and local-only branches do not arrive with a remote clone.
Check the actual checkout; never assume the default branch contains the desired
handoff. A repository owner must publish these documentation changes before
another device can obtain them from GitHub.

## Remote host workflow

Use a Linux x86-64 host with Git, Bash, Python 3.11 and `uv` available. This is
the recommended target for the existing CUDA-wheel lock, even for CPU fixtures.
A GPU is not required for the checks below. Reserve space for the scientific
stack and package cache. A Windows/macOS client can connect to this host over
SSH; native installation on other platforms is not validated by this guide.

From the client, connect using your provisioned identity:

```bash
ssh YOUR_USER@YOUR_HOST
```

On that host, clone into a new directory you own. Replace the branch below
with the branch recorded by the person handing over the repository:

```bash
git clone https://github.com/BoyangYu0/HyperTagging.git
cd HyperTagging
git fetch origin
git switch --track origin/HANDOFF_BRANCH
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

If the intended branch is already checked out, skip `git switch`. SSH clone
URLs are also supported when your GitHub SSH identity is configured. Compare
HEAD with the recorded SHA before proceeding. For an existing checkout, first
inspect its status and ownership; do not reset another task's work or change
ownership to bypass access errors. Independent users should use separate clones.

Install Codex CLI using the current [official CLI instructions](https://learn.chatgpt.com/docs/codex/cli).
Run it on the remote host so commands execute where the checkout and data live.
For a headless login:

```bash
codex login --device-auth
codex
```

Device login must be enabled in the account/workspace settings. If unavailable,
use the SSH callback forwarding procedure in the
[official authentication guide](https://learn.chatgpt.com/docs/auth).
For a desktop/IDE project, select this Git root on the configured execution
host; merely opening a local clone does not attach it to the remote filesystem.
SSH plus the CLI above is the explicit remote execution path.

## Reproduce the project environment

Run from the Git root, in a fresh checkout with no existing `.venv`. Use the
tracked lock; do not relock as part of onboarding:

```bash
uv --version
uv sync --frozen --all-extras --python 3.11
source scripts/activate_env.sh project
python --version
python scripts/check_uv_lock_direct_dependencies.py
python -c 'import hypertagging, torch; print(hypertagging.__file__); print(torch.__version__)'
```

The root lock selects PyTorch 2.7.1 from the CUDA 12.6 index. CPU execution
still works without a GPU, but this is not a small CPU-only installation.
Allow dependency downloads from the indexes/wheel URLs in `uv.lock`, including
PyPI and `download.pytorch.org`. Do not silently replace this lock with an
unpinned `pip install`. The CI workflow's pip installation is a separate,
non-lock-reproducing path.

The default `.venv` is ignored by Git. Where checkout storage is limited,
create an environment on a writable volume using `UV_PROJECT_ENVIRONMENT`
with `uv sync`, then source that environment's `bin/activate` directly.
`scripts/activate_env.sh project` always selects the checkout's `.venv`;
its `gpu` mode uses a historical fixed site path. Neither mode follows a custom
external environment path. Recreate environments on a new host rather than
copying them; activation scripts and editable installs contain local paths.

Verify a bounded CPU baseline with the selected environment's Python:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pytest -q tests/test_uv_lock_direct_dependencies_cpu.py tests/test_examples_cpu.py
ht_smoke_dir="$(mktemp -d)"
python scripts/train_hyperbolic_pretrain.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2 \
  --output-dir "$ht_smoke_dir/pretrain"
python scripts/train_level_reconstruction.py \
  --dry-run --tiny --device cpu --max-steps 2 --batch-size 2 \
  --output-dir "$ht_smoke_dir/reconstruction"
git diff --check
```

Record the SHA, OS/architecture, Python/uv versions, selected interpreter,
commands, outcomes and output location in the task handoff. Run the broader
CPU suite in `AGENTS.md` when the implementation task warrants it. These checks
establish fixture behavior only. For documentation-only work, use the smaller
pinned environment and build commands in [README.md](../README.md).

## Codex cloud alternative

Connect the repository using an account with access, select the intended
branch/SHA, and configure a cloud environment. Use this manual setup script
from the checkout root after ensuring `uv` and Python 3.11 are available:

```bash
set -euo pipefail
uv sync --frozen --all-extras --python 3.11
.venv/bin/python scripts/check_uv_lock_direct_dependencies.py
```

Use the same commands as a maintenance script if cached environments must
follow dependency changes. During tasks, use `.venv/bin/python` explicitly;
setup-shell activation does not persist. Configure `CUDA_VISIBLE_DEVICES` as
empty and `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` in environment settings.
See [official cloud environment documentation](https://learn.chatgpt.com/docs/environments/cloud-environment)
for setup networking, caching and secret scope. Reconfigure access and settings
for a different account/workspace; do not assume they transfer with Git.

A cloud checkout does not mount the institute's data volumes, CVMFS, or job
scheduler. Use it for code/docs and CPU fixtures. Use the remote host workflow
when work needs those resources. No OpenAI API key is required by HyperTagging's
fixture tests; Codex itself requires its own supported authentication.

## External resources and production work

Before a real run, explicitly supply and validate:

- Readable input data, immutable selection manifests, authenticated dataset
  indexes and shards. See [preprocessing](preprocessing_design.md) and
  [training](training.md). Do not rewrite hashed manifests to bypass lineage
  checks when paths differ; arrange the expected mounts or perform a documented
  relocation with all integrity checks retained.
- Trusted pretraining/reconstruction checkpoints and their lineage. See
  [full-decay evaluation](full_decay_reconstruction_evaluation.md).
- Belle II `release-08-03-00` and CVMFS/site access for raw mDST processing;
  ordinary Python fixtures do not need basf2.
- A writable data/output volume, the recipient's scheduler account/partition
  or Condor requirements, and a separately frozen GPU environment. See
  [GPU setup](../environment/gpu/README.md), [Condor](condor.md), and the
  repository's `skills/condor_jobs/SKILL.md` and `scripts/slurm/` workflows.

Historical absolute user paths, scheduler IDs and authorization receipts are
provenance, not portable defaults or permission for a new campaign. Inspect
selected configs and rendered jobs for old account/path assumptions. Keep real
training in the guarded scheduler workflows; environment setup never submits
jobs. Model export/deployment has its own [basf2/ONNX contract](basf2_onnx_full_decay.md).

## Paste into a new Codex task

```text
Work in this HyperTagging Git root. Read AGENTS.md, README.md,
docs/codex_remote_setup.md and docs/audits/current_status.md. Inspect the
branch, HEAD and git status; preserve existing changes. Verify the intended
handoff revision and the available Python environment. Use bounded CPU
fixtures to check setup. Read the relevant current contract before changing
scientific behavior. Historical sibling checkouts and earlier chat history
are not prerequisites. Report unavailable external data/software/access
explicitly. Do not infer production readiness or job authorization from old
receipts. Then carry out this task: <insert the requested work>.
```

For future handoffs, include the exact revision, unfinished objective, changed
files, checks actually run, known blockers, external artifact identifiers and
access prerequisites. Keep credentials outside the handoff.
