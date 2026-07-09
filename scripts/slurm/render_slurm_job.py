#!/usr/bin/env python
"""Render safe SLURM job scripts for HyperTagging."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def render(config: dict[str, str], command: str) -> str:
    qos = f"#SBATCH --qos={config['qos']}\n" if config.get("qos") else ""
    gres = f"#SBATCH --gres={config['gres']}\n" if config.get("gres") else ""
    setup = config.get("setup", "")
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={config.get('job_name', 'hypertagging')}
#SBATCH --account={config.get('account', '<account>')}
#SBATCH --partition={config.get('partition', '<partition>')}
{qos}#SBATCH --time={config.get('time', '00:30:00')}
#SBATCH --nodes={config.get('nodes', '1')}
#SBATCH --ntasks={config.get('ntasks', '1')}
#SBATCH --cpus-per-task={config.get('cpus_per_task', '4')}
{gres}#SBATCH --mem={config.get('mem', '16G')}
#SBATCH --output={config.get('log_dir', 'logs/slurm')}/%x-%j.out
#SBATCH --error={config.get('log_dir', 'logs/slurm')}/%x-%j.err

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"
mkdir -p {config.get('log_dir', 'logs/slurm')} {config.get('checkpoint_dir', 'outputs/checkpoints')}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
echo "date=$(date)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
git rev-parse HEAD || true
python -c 'import sys; print(sys.executable)'
squeue --me || true
nvidia-smi || true
{setup}
{command}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/slurm/default.yaml")
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    text = render(read_simple_yaml(Path(args.config)), args.command)
    if args.dry_run or not args.output:
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
