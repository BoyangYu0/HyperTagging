#!/usr/bin/env python
"""Run hyperbolic pretraining dry-runs or HTCondor-guarded training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.training.hyperbolic_pretrain import run_hyperbolic_pretrain_dry_run
from hypertagging.utils.gpu_safety import assert_full_training_requires_condor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/hyperbolic_pretrain")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--allow-local-tiny-gpu-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assert_full_training_requires_condor(args)
    if not args.dry_run and not args.tiny:
        raise RuntimeError("Full hyperbolic pretraining must be submitted through HTCondor templates.")
    summary = run_hyperbolic_pretrain_dry_run(
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
