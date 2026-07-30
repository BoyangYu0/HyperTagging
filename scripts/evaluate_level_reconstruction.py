#!/usr/bin/env python
"""Evaluate tiny level-autoregressive reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.training.level_reconstruction_train import run_level_reconstruction_dry_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-events", type=int, default=2)
    args = parser.parse_args(argv)
    if args.device.split(":")[0] != "cpu":
        raise RuntimeError("Evaluation script supports CPU dry-run locally; use HTCondor for CUDA.")
    print(run_level_reconstruction_dry_run(device=args.device, max_steps=1, batch_size=min(args.max_events, 2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
