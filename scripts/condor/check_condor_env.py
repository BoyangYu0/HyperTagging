#!/usr/bin/env python
"""Print HTCondor/GPU state without submitting jobs."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.utils.gpu_safety import get_condor_q_snapshot, get_nvidia_smi_snapshot, is_inside_condor


def main() -> int:
    print(f"inside_condor={is_inside_condor()}")
    for name, snap in {**get_condor_q_snapshot(), **get_nvidia_smi_snapshot()}.items():
        print(f"== {name}: returncode={snap.returncode} ==")
        print(snap.stdout or snap.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
