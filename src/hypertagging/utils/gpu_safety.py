"""GPU and SLURM safety checks for HyperTagging training scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Any


@dataclass(frozen=True)
class CommandSnapshot:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def is_inside_slurm() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"))


def _run(command: tuple[str, ...], timeout: int = 10) -> CommandSnapshot:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return CommandSnapshot(command, result.returncode, result.stdout, result.stderr)
    except Exception as exc:
        return CommandSnapshot(command, 127, "", str(exc))


def get_squeue_snapshot() -> dict[str, CommandSnapshot]:
    return {"squeue_me": _run(("squeue", "--me")), "squeue": _run(("squeue",))}


def get_nvidia_smi_snapshot() -> dict[str, CommandSnapshot]:
    return {
        "nvidia_smi": _run(("nvidia-smi",)),
        "compute_apps": _run(("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")),
        "gpu_state": _run(("nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader")),
    }


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default) if not isinstance(args, dict) else args.get(name, default)


def assert_local_gpu_tiny_test_allowed(args: Any, *, max_steps: int = 10, max_batch_size: int = 2) -> None:
    """Allow local CUDA only for explicit tiny tests after queue/GPU checks."""

    if str(_arg(args, "device", "cpu")).split(":")[0] != "cuda":
        return
    if not _arg(args, "tiny", False):
        raise RuntimeError("Local CUDA requires --tiny.")
    if int(_arg(args, "max_steps", max_steps + 1)) > max_steps:
        raise RuntimeError(f"Local CUDA tiny tests require --max-steps <= {max_steps}.")
    if int(_arg(args, "batch_size", max_batch_size + 1)) > max_batch_size:
        raise RuntimeError(f"Local CUDA tiny tests require --batch-size <= {max_batch_size}.")
    if not _arg(args, "allow_local_tiny_gpu_test", False):
        raise RuntimeError("Local CUDA tiny tests require --allow-local-tiny-gpu-test.")
    snapshots = {**get_squeue_snapshot(), **get_nvidia_smi_snapshot()}
    failed = [name for name, snap in snapshots.items() if snap.returncode != 0]
    if failed:
        raise RuntimeError(f"Cannot verify local GPU safety; failed commands: {failed}")
    compute_apps = snapshots["compute_apps"].stdout.strip()
    if compute_apps:
        raise RuntimeError("Local GPU appears busy: nvidia-smi compute apps are present.")
    squeue = snapshots["squeue"].stdout.strip().splitlines()
    if len(squeue) > 1:
        raise RuntimeError("SLURM queue is non-empty; refusing ambiguous local GPU test.")


def assert_full_training_requires_slurm(args: Any) -> None:
    """Refuse non-tiny CUDA training outside SLURM."""

    if str(_arg(args, "device", "cpu")).split(":")[0] != "cuda":
        return
    if is_inside_slurm():
        return
    if _arg(args, "tiny", False) and _arg(args, "allow_local_tiny_gpu_test", False):
        assert_local_gpu_tiny_test_allowed(args)
        return
    raise RuntimeError("Full CUDA training must run inside SLURM via sbatch.")
