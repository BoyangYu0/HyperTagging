#!/usr/bin/env python
"""Run level-autoregressive reconstruction dry-runs or HTCondor-guarded training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.training.level_reconstruction_train import run_level_reconstruction_dry_run
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    train_level_reconstruction,
)
from hypertagging.models.ablation import ABLATIONS
from hypertagging.utils.gpu_safety import assert_full_training_requires_condor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--freeze-pretrained-encoder-steps", type=int, default=0)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.2)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--max-cardinality", type=int, default=6)
    parser.add_argument("--scheduled-sampling-probability", type=float, default=None)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full_revised")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="outputs/level_reconstruction")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--allow-local-tiny-gpu-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assert_full_training_requires_condor(args)
    if args.data:
        result = train_level_reconstruction(
            ReconstructionConfig(
                data=args.data,
                output_dir=args.output_dir,
                pretrained_encoder=args.pretrained_encoder,
                device=args.device,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                max_events=args.max_events,
                seed=args.seed,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                n_queries=args.n_queries,
                max_cardinality=args.max_cardinality,
                scheduled_sampling_probability=(
                    args.scheduled_sampling_probability
                    if args.scheduled_sampling_probability is not None
                    else (
                        0.25
                        if ABLATIONS[args.ablation].scheduled_sampling
                        else 0.0
                    )
                ),
                freeze_pretrained_encoder_steps=args.freeze_pretrained_encoder_steps,
                encoder_lr_multiplier=args.encoder_lr_multiplier,
                ablation=args.ablation,
                allow_tiny_bruteforce_matching=(
                    args.device == "cpu" and args.max_steps <= 10 and args.batch_size <= 4
                ),
            )
        )
        print(
            {
                "checkpoint": str(result.checkpoint),
                "log": str(result.log_path),
                "steps": result.steps,
                "loss": result.final_loss,
                "transfer": None
                if result.transfer_report is None
                else {
                    "loaded": len(result.transfer_report.loaded_keys),
                    "missing": list(result.transfer_report.missing_keys),
                    "unexpected": list(result.transfer_report.unexpected_keys),
                    "shape_mismatches": list(result.transfer_report.shape_mismatches),
                },
            }
        )
    else:
        if not args.dry_run and not args.tiny:
            raise RuntimeError("--data is required for real training; use --dry-run for fixtures")
        summary = run_level_reconstruction_dry_run(
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
