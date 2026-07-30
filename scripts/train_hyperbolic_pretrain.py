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
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.utils.gpu_safety import assert_full_training_requires_condor
from hypertagging.training.config import resolve_argparse_namespace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--ablation", choices=sorted(ALL_ABLATIONS), default="full_revised")
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
    parser.add_argument("--channel-memory-size", type=int, default=0)
    parser.add_argument("--allow-local-tiny-gpu-test", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pilot-split-repair", action="store_true")
    parser.add_argument("--allow-legacy-conflated", action="store_true")
    return resolve_argparse_namespace(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assert_full_training_requires_condor(args)
    if args.data:
        result = train_hyperbolic_pretraining(
            PretrainConfig(
                data=args.data,
                output_dir=args.output_dir,
                device=args.device,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                max_events=args.max_events,
                seed=args.seed,
                checkpoint_every=args.checkpoint_every,
                validate_every=args.validate_every,
                resume=args.resume,
                ablation=args.ablation,
                channel_memory_size=args.channel_memory_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                shuffle_buffer_size=args.shuffle_buffer_size,
                persistent_workers=args.persistent_workers,
                pilot_split_repair=args.pilot_split_repair,
                allow_legacy_conflated=args.allow_legacy_conflated,
                log_every=args.log_every,
            )
        )
        print(
            {
                "checkpoint": str(result.checkpoint),
                "log": str(result.log_path),
                "steps": result.steps,
                "loss": result.final_loss,
            }
        )
    else:
        if not args.dry_run and not args.tiny:
            raise RuntimeError("--data is required for real training; use --dry-run for fixtures")
        summary = run_hyperbolic_pretrain_dry_run(
            device=args.device,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            ablation=args.ablation,
            resume=args.resume,
        )
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
