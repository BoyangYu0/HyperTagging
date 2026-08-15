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

from hypertagging.training.hyperbolic_pretrain import run_hyperbolic_pretrain_dry_run  # noqa: E402
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining  # noqa: E402
from hypertagging.models.ablation import ALL_ABLATIONS  # noqa: E402
from hypertagging.utils.gpu_safety import assert_full_training_requires_condor  # noqa: E402
from hypertagging.utils.gpu_safety import ALLOWED_SLURM_GRES  # noqa: E402
from hypertagging.training.config import resolve_argparse_namespace  # noqa: E402
from hypertagging.training.model_config import MODEL_PRESETS  # noqa: E402


def _level_int_pairs(value: str):
    if not value:
        return ()
    return tuple(
        (int(level), int(count))
        for level, count in (item.split(":", 1) for item in value.split(","))
    )


def _int_tuple(value: str):
    return tuple(int(item) for item in value.split(",") if item)


def _string_tuple(value: str):
    return tuple(item for item in value.split(",") if item)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--ablation", choices=sorted(ALL_ABLATIONS), default="full_revised")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gpu-execution-mode",
        choices=("auto", "scientific_slurm", "slurm_diagnostic", "local_microtest"),
        default="auto",
    )
    parser.add_argument("--expected-gres", choices=ALLOWED_SLURM_GRES, default=None)
    parser.add_argument("--local-admission-receipt", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-schedule-total-steps", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-warmup-steps", type=int, default=10_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--amp-init-scale", type=float, default=4096.0)
    parser.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16"), default="float16"
    )
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
    parser.add_argument("--validation-events", type=int, default=None)
    parser.add_argument(
        "--validation-views",
        type=_string_tuple,
        default=(
            "fsp_topology_anticollapse",
            "truth_guided_distance_radius",
            "multilevel_channel_memory",
            "corrupted_composites_hard_negatives",
        ),
    )
    parser.add_argument(
        "--curriculum-mode",
        choices=("progressive", "legacy_alternating_ablation"),
        default="progressive",
    )
    parser.add_argument("--curriculum-phase-steps", type=_int_tuple, default=())
    parser.add_argument("--curriculum-phase-events", type=_int_tuple, default=())
    parser.add_argument("--require-final-curriculum-phase", action="store_true")
    parser.add_argument("--scientific-mode", action="store_true")
    parser.add_argument(
        "--channel-pooling",
        choices=("mean_all", "fsp_only", "b_root", "learned_attention", "level_weighted"),
        default="mean_all",
    )
    parser.add_argument("--channel-memory-size", type=int, default=0)
    parser.add_argument("--radius-target-mode", choices=("generation_height_radius", "exact_root_depth_radius", "weak_or_learned_radius"), default="generation_height_radius")
    parser.add_argument(
        "--best-metric",
        choices=("validation_principal_loss", "validation_full_training_objective"),
        default="validation_full_training_objective",
    )
    parser.add_argument("--best-mode", choices=("min", "max"), default="min")
    parser.add_argument("--channel-zero-positive-validation-window", type=int, default=3)
    parser.add_argument("--channel-zero-positive-action", choices=("warn", "fail", "ignore"), default="warn")
    parser.add_argument("--hyperbolic-level-encoding", choices=("learned_euclidean", "bounded_tangent_level_embedding", "none"), default="learned_euclidean")
    parser.add_argument("--objective-gradient-diagnostics", action="store_true")
    parser.add_argument("--objective-gradient-diagnostics-every", type=int, default=100)
    parser.add_argument("--pilot-objective-preflight", action="store_true")
    parser.add_argument("--objective-dominance-ratio", type=float, default=20.0)
    parser.add_argument(
        "--objective-weighted-loss-tolerance", type=float, default=1e-7
    )
    parser.add_argument(
        "--pilot-objective-violation-action",
        choices=("warn", "fail"),
        default="fail",
    )
    parser.add_argument("--lca-relation-weight", type=float, default=1.0)
    parser.add_argument("--parent-ranking-weight", type=float, default=1.0)
    parser.add_argument("--exact-tree-distance-weight", type=float, default=1.0)
    parser.add_argument("--radius-depth-weight", type=float, default=0.2)
    parser.add_argument("--channel-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--leaf-pid-weight", type=float, default=1.0)
    parser.add_argument("--corruption-class-weight", type=float, default=0.1)
    parser.add_argument("--candidate-correctness-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=0.1)
    parser.add_argument(
        "--truth-guided-structural-relation-inputs",
        action="store_true",
        help=(
            "compatibility ablation: expose exact known truth-guided links to "
            "contextual attention; default keeps them target-only"
        ),
    )
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="tiny_cpu")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--hyper-dim", type=int, default=None)
    parser.add_argument("--tangent-variance-target", type=float, default=None)
    parser.add_argument("--hyper-projection-init-scale", type=float, default=None)
    parser.add_argument("--tangent-scale-mode", choices=("fixed", "learned_bounded"), default=None)
    parser.add_argument("--max-tangent-norm", type=float, default=None)
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
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip=args.gradient_clip,
                lr_schedule_total_steps=args.lr_schedule_total_steps,
                warmup_fraction=args.warmup_fraction,
                warmup_steps=args.warmup_steps,
                max_warmup_steps=args.max_warmup_steps,
                min_lr_ratio=args.min_lr_ratio,
                amp_init_scale=args.amp_init_scale,
                amp_dtype=args.amp_dtype,
                batch_size=args.batch_size,
                max_events=args.max_events,
                seed=args.seed,
                checkpoint_every=args.checkpoint_every,
                validate_every=args.validate_every,
                validation_batches=args.validation_batches,
                validation_events=args.validation_events,
                validation_views=tuple(args.validation_views),
                curriculum_mode=args.curriculum_mode,
                curriculum_phase_steps=tuple(args.curriculum_phase_steps),
                curriculum_phase_events=tuple(args.curriculum_phase_events),
                require_final_curriculum_phase=(
                    args.require_final_curriculum_phase
                ),
                scientific_mode=args.scientific_mode,
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
                pilot_objective_preflight=args.pilot_objective_preflight,
                objective_dominance_ratio=args.objective_dominance_ratio,
                objective_weighted_loss_tolerance=(
                    args.objective_weighted_loss_tolerance
                ),
                pilot_objective_violation_action=(
                    args.pilot_objective_violation_action
                ),
                lca_relation_weight=args.lca_relation_weight,
                parent_ranking_weight=args.parent_ranking_weight,
                exact_tree_distance_weight=args.exact_tree_distance_weight,
                radius_depth_weight=args.radius_depth_weight,
                channel_weight=args.channel_weight,
                variance_weight=args.variance_weight,
                covariance_weight=args.covariance_weight,
                leaf_pid_weight=args.leaf_pid_weight,
                corruption_class_weight=args.corruption_class_weight,
                candidate_correctness_weight=args.candidate_correctness_weight,
                hard_negative_weight=args.hard_negative_weight,
                truth_guided_structural_relation_inputs=(
                    args.truth_guided_structural_relation_inputs
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
                max_tangent_norm=args.max_tangent_norm,
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
