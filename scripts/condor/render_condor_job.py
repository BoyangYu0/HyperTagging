#!/usr/bin/env python
"""Render an HTCondor submit description and its job executable."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def render_executable(config: dict[str, str], command: str) -> str:
    log_dir = config.get("log_dir", "logs/condor")
    checkpoint_dir = config.get("checkpoint_dir", "outputs/checkpoints")
    setup = config.get("setup", "")
    cpus = config.get("request_cpus", "1")
    setup_block = f"{setup}\n" if setup else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p {shlex.quote(log_dir)} {shlex.quote(checkpoint_dir)}
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{cpus}}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-{cpus}}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-{cpus}}}"
echo "date=$(date)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
git rev-parse HEAD || true
python -c 'import sys; print(sys.executable)'
condor_q "${{CONDOR_CLUSTER_ID:-}}" || true
nvidia-smi || true
{setup_block}{command}
"""


def render_submit(config: dict[str, str], executable: str, initial_dir: str) -> str:
    job_name = config.get("job_name", "hypertagging")
    log_dir = config.get("log_dir", "logs/condor")
    runtime = config.get("request_runtime", "1800")
    lines = [
        f"universe = {config.get('universe', 'vanilla')}",
        f"executable = {executable}",
        f"initialdir = {initial_dir}",
        "getenv = True",
        "should_transfer_files = NO",
        f"request_cpus = {config.get('request_cpus', '1')}",
        f"request_memory = {config.get('request_memory', '4GB')}",
        f"request_gpus = {config.get('request_gpus', '0')}",
        f"+RequestRuntime = {runtime}",
        f"environment = \"CONDOR_CLUSTER_ID=$(ClusterId);CONDOR_PROCESS_ID=$(ProcId)\"",
        f"output = {log_dir}/{job_name}-$(ClusterId).$(ProcId).out",
        f"error = {log_dir}/{job_name}-$(ClusterId).$(ProcId).err",
        f"log = {log_dir}/{job_name}-$(ClusterId).log",
        "queue",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/condor/default.yaml")
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", default=None, help="Output .sub path; also creates a sibling .sh executable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_simple_yaml(Path(args.config))
    if args.output:
        submit_path = Path(args.output)
        if submit_path.suffix != ".sub":
            parser.error("--output must end in .sub")
        executable_path = submit_path.with_suffix(".sh")
        submit_text = render_submit(config, executable_path.resolve().as_posix(), Path.cwd().resolve().as_posix())
    else:
        executable_path = Path("outputs/condor/job.sh")
        submit_text = render_submit(config, executable_path.resolve().as_posix(), Path.cwd().resolve().as_posix())
    executable_text = render_executable(config, args.command)

    if args.dry_run or not args.output:
        print(f"# {executable_path}\n{executable_text}")
        print(f"# {executable_path.with_suffix('.sub')}\n{submit_text}")
        return 0

    submit_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.get("log_dir", "logs/condor")).mkdir(parents=True, exist_ok=True)
    executable_path.write_text(executable_text, encoding="utf-8")
    executable_path.chmod(0o755)
    submit_path.write_text(submit_text, encoding="utf-8")
    print(submit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
