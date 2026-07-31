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
from hypertagging.training.model_config import MODEL_PRESETS


def _level_int_pairs(value: str):
    if not value:
        return ()
    return tuple(
        (int(level), int(count))
        for level, count in (item.split(":", 1) for item in value.split(","))
    )


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
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument(
        "--channel-pooling",
        choices=("mean_all", "fsp_only", "b_root", "learned_attention", "level_weighted"),
        default="mean_all",
    )
    parser.add_argument("--channel-memory-size", type=int, default=0)
    parser.add_argument("--radius-target-mode", choices=("generation_height_radius", "exact_root_depth_radius", "weak_or_learned_radius"), default="generation_height_radius")
    parser.add_argument("--best-metric", default="validation_loss_total")
    parser.add_argument("--best-mode", choices=("min", "max"), default="min")
    parser.add_argument("--channel-zero-positive-validation-window", type=int, default=3)
    parser.add_argument("--channel-zero-positive-action", choices=("warn", "fail", "ignore"), default="warn")
    parser.add_argument("--hyperbolic-level-encoding", choices=("learned_euclidean", "bounded_tangent_level_embedding", "none"), default="learned_euclidean")
    parser.add_argument("--objective-gradient-diagnostics", action="store_true")
    parser.add_argument("--objective-gradient-diagnostics-every", type=int, default=100)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="tiny_cpu")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--hyper-dim", type=int, default=None)
    parser.add_argument("--tangent-variance-target", type=float, default=None)
    parser.add_argument("--hyper-projection-init-scale", type=float, default=None)
    parser.add_argument("--tangent-scale-mode", choices=("fixed", "learned_bounded"), default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-context-layers", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--n-queries-by-level", type=_level_int_pairs, default=())
    parser.add_argument("--max-cardinality", type=int, default=None)
    parser.add_argument("--max-cardinality-by-level", type=_level_int_pairs, default=())
    parser.add_argument("--allow-local-tiny-gpu-test", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pilot-split-repair", action="store_true")
    parser.add_argument("--allow-legacy-conflated", action="store_true")
    parser.add_argument("--dataset-index", default=None)
    parser.add_argument("--rescan-dataset", action="store_true")
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
                validation_batches=args.validation_batches,
                channel_pooling=args.channel_pooling,
                resume=args.resume,
                ablation=args.ablation,
                channel_memory_size=args.channel_memory_size,
                radius_target_mode=args.radius_target_mode,
                best_metric=args.best_metric,
                best_mode=args.best_mode,
                channel_zero_positive_validation_window=args.channel_zero_positive_validation_window,
                channel_zero_positive_action=args.channel_zero_positive_action,
                hyperbolic_level_encoding=args.hyperbolic_level_encoding,
                objective_gradient_diagnostics=args.objective_gradient_diagnostics,
                objective_gradient_diagnostics_every=(
                    args.objective_gradient_diagnostics_every
                ),
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                shuffle_buffer_size=args.shuffle_buffer_size,
                persistent_workers=args.persistent_workers,
                pilot_split_repair=args.pilot_split_repair,
                allow_legacy_conflated=args.allow_legacy_conflated,
                log_every=args.log_every,
                dataset_index=args.dataset_index,
                rescan_dataset=args.rescan_dataset,
                model_preset=args.model_preset,
                d_model=args.d_model,
                hyper_dim=args.hyper_dim,
                tangent_variance_target=args.tangent_variance_target,
                hyper_projection_init_scale=args.hyper_projection_init_scale,
                tangent_scale_mode=args.tangent_scale_mode,
                n_heads=args.n_heads,
                n_context_layers=args.n_context_layers,
                ffn_dim=args.ffn_dim,
                dropout=args.dropout,
                curvature=args.curvature if args.curvature is not None else 1.0,
                n_queries=args.n_queries,
                n_queries_by_level=tuple(tuple(value) for value in args.n_queries_by_level),
                max_cardinality=args.max_cardinality,
                max_cardinality_by_level=tuple(
                    tuple(value) for value in args.max_cardinality_by_level
                ),
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
