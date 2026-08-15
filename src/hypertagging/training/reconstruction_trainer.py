"""All-level real-parquet reconstruction training and rollout validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from hypertagging.data.capacity import (
    capacity_statistics_from_index,
    dataset_capacity_statistics,
    require_capacity,
)
from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.evaluation.hierarchical_metrics import (
    complete_target_efficiency_counts,
    next_level_metrics,
    p4_closure_rate,
    summarize_rollout,
)
from hypertagging.losses.level_reconstruction import (
    confidence_calibration_metrics,
    level_reconstruction_loss,
)
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    compare_pid_kinematics_modes,
)
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig, cached_context_for_level, level_rollout,
    proposal_ambiguity_metrics,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import (
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from hypertagging.training.checkpoint_selection import (
    RECONSTRUCTION_TRACK_BY_METRIC,
    checkpoint_track_decisions,
    initial_track_values,
    reconstruction_selection_contract,
    rollout_checkpoint_eligibility,
    selection_reason,
)
from hypertagging.training.data_module import RealDataModule, build_real_data_module
from hypertagging.data.streaming import RuntimeFeatureNormalizer, StreamingCursor
from hypertagging.training.logging import JsonlLogger
from hypertagging.training.pretrained_transfer import (
    EncoderTransferReport,
    load_pretrained_encoder,
    optimizer_parameter_groups,
    unfreeze_encoder,
)
from hypertagging.utils.seeds import seed_everything
from hypertagging.utils.tensor_contractions import boolean_matmul
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID, feature_spec_v4
from hypertagging.training.scheduled_sampling import (
    TeacherForcingSchedule,
    aligned_level_targets,
    resolve_unrepresentable_target_policy,
)
from hypertagging.training.model_config import resolve_model_architecture
from hypertagging.training.fixed_validation import select_validation_events
from hypertagging.training.learning_rate import (
    build_warmup_cosine_scheduler,
    resolve_resume_schedule_contract,
)
from hypertagging.training.slurm_signal import (
    PendingValidationInterrupted,
    SafeBoundarySignalController,
    install_safe_boundary_signal_controller,
)


@dataclass(frozen=True)
class ReconstructionConfig:
    data: str
    output_dir: str
    pretrained_encoder: str | None = None
    device: str = "cpu"
    max_steps: int = 2
    batch_size: int = 2
    max_events: int | None = None
    seed: int = 11
    learning_rate: float = 1e-3
    lr_schedule_total_steps: int | None = None
    warmup_fraction: float = 0.05
    warmup_steps: int | None = None
    max_warmup_steps: int = 10_000
    min_lr_ratio: float = 0.0
    encoder_lr_multiplier: float = 0.2
    freeze_pretrained_encoder_steps: int = 0
    gradient_clip: float = 1.0
    checkpoint_every: int = 100
    validate_every: int = 100
    rollout_validate_every: int = 500
    resume: str | None = None
    n_queries: int | None = None
    max_cardinality: int | None = None
    scheduled_sampling_probability: float = 0.25
    allow_tiny_bruteforce_matching: bool = True
    mixed_precision: bool = True
    ablation: str = "full_revised"
    transfer_leaf_pid_head: bool = False
    freeze_leaf_pid_head_steps: int = 0
    leaf_pid_lr_multiplier: float = 1.0
    target_policy: str = "complete_only"
    scheduled_sampling_schedule: str = "linear"
    scheduled_sampling_duration_steps: int = 1000
    num_workers: int = 0
    prefetch_factor: int = 2
    shuffle_buffer_size: int = 1024
    persistent_workers: bool = False
    pilot_split_repair: bool = False
    allow_legacy_conflated: bool = False
    max_validation_events: int = 32
    rollout_validation_events: int = 8
    validation_batch_size: int = 4
    log_every: int = 10
    auxiliary_teacher_weight: float = 0.0
    dataset_index: str | None = None
    rescan_dataset: bool = False
    unrepresentable_target_policy: str = "fallback_teacher"
    level_sampling_mode: str = "all_levels"
    gradient_accumulation: int = 1
    empirical_type_prior_mode: str = "soft"
    diagnostic_forward_interval: int = 10
    model_preset: str = "tiny_cpu"
    d_model: int | None = None
    hyper_dim: int | None = None
    n_heads: int | None = None
    n_context_layers: int | None = None
    ffn_dim: int | None = None
    dropout: float | None = None
    curvature: float | None = None
    n_queries_by_level: tuple[tuple[int, int], ...] = ()
    max_cardinality_by_level: tuple[tuple[int, int], ...] = ()
    minimum_encoder_transfer_coverage: float = 0.9
    allow_low_encoder_transfer_coverage: bool = False
    allow_incomplete_v4_publication: bool = False
    pid_temperature_start: float = 1.0
    pid_temperature_end: float = 0.2
    pid_temperature_duration_steps: int = 1000
    pid_kinematics_mode: str | None = None
    pid_temperature: float = 1.0
    tangent_variance_target: float | None = None
    hyper_projection_init_scale: float | None = None
    tangent_scale_mode: str | None = None
    query_repulsion_weight: float = 0.0
    rollout_pid_kinematics_mode: str = "soft_decision_hard_construction"
    rollout_pid_temperature: float = 0.5
    best_metric: str = "validation_loss_total"
    best_mode: str = "min"
    early_stopping_patience: int | None = None
    pilot_allow_train_validation_fallback: bool = False
    scientific_mode: bool = False
    rollout_min_tree_validity: float = 0.999
    rollout_min_p4_closure: float = 1.0
    rollout_p4_tolerance: float = 1e-6
    rollout_max_recursive_source_conflicts: int = 0
    rollout_required_denominators: tuple[str, ...] = (
        "rollout_validation_events",
        "predicted_edge_denominator",
        "predicted_p4_closure_denominator",
    )
    initial_state_policy: str = "unknown"
    hyperbolic_level_encoding: str = "learned_euclidean"
    type_conditioned_daughter_relation_bias: bool = False


@dataclass(frozen=True)
class ReconstructionTrainingResult:
    checkpoint: Path
    log_path: Path
    steps: int
    final_loss: float
    metrics: dict[str, float]
    transfer_report: EncoderTransferReport | None
    data_module: RealDataModule


def train_level_reconstruction(
    config: ReconstructionConfig,
    *,
    signal_controller: SafeBoundarySignalController | None = None,
) -> ReconstructionTrainingResult:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.validate_every <= 0 or config.rollout_validate_every <= 0:
        raise ValueError("validation cadences must be positive")
    if config.best_mode not in {"min", "max"}:
        raise ValueError("best_mode must be 'min' or 'max'")
    if config.best_metric not in RECONSTRUCTION_TRACK_BY_METRIC:
        raise ValueError(
            "best_metric must be validation_loss_total, predicted_edge_f1, or "
            "predicted_tree_validity_rate"
        )
    if config.scientific_mode and (
        config.best_metric != "predicted_edge_f1" or config.best_mode != "max"
    ):
        raise ValueError(
            "scientific reconstruction requires rollout predicted_edge_f1 as "
            "the maximizing primary metric"
        )
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when supplied")
    if not 0.0 <= config.scheduled_sampling_probability <= 1.0:
        raise ValueError("scheduled_sampling_probability must lie in [0, 1]")
    if config.unrepresentable_target_policy not in {
        "fallback_teacher", "skip_event_level", "masked_representable_only", "recovery_objective"
    }:
        raise ValueError("unknown unrepresentable_target_policy")
    if config.pid_kinematics_mode not in {
        None, "soft_expectation", "temperature_softmax",
        "straight_through_hard", "hard",
    }:
        raise ValueError("unknown pid_kinematics_mode")
    if config.level_sampling_mode not in {
        "all_levels", "one_level_per_event", "stratified_level_sampling"
    }:
        raise ValueError("unknown level_sampling_mode")
    if config.gradient_accumulation != 1:
        raise ValueError(
            "gradient_accumulation other than 1 is not implemented; refusing to "
            "claim an exact-resume order for an unsupported optimizer cadence"
        )
    seed_everything(config.seed)
    if config.resume and config.num_workers > 0:
        raise ValueError(
            "exact streaming resume currently requires num_workers=0; "
            "multiworker resume is intentionally not claimed"
        )
    device = torch.device(config.device)
    resume_payload = (
        load_training_checkpoint(config.resume, map_location="cpu")
        if config.resume
        else None
    )
    data_module = build_real_data_module(
        config.data,
        max_events=config.max_events,
        seed=config.seed,
        pilot_split_repair=config.pilot_split_repair,
        allow_legacy_conflated=config.allow_legacy_conflated,
        shuffle_buffer_size=config.shuffle_buffer_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        normalization_state=(
            resume_payload.get("normalizer_state")
            if resume_payload is not None
            else None
        ),
        dataset_index=config.dataset_index,
        rescan_dataset=config.rescan_dataset,
        target_policy=config.target_policy,
        allow_incomplete_v4_publication=config.allow_incomplete_v4_publication,
        scientific_mode=config.scientific_mode,
    )
    if config.ablation not in ALL_ABLATIONS:
        raise ValueError(f"unknown ablation: {config.ablation}")
    effective_type_relation_bias = (
        config.type_conditioned_daughter_relation_bias
        or ALL_ABLATIONS[config.ablation].type_conditioned_daughter_relation_bias
    )
    architecture = resolve_model_architecture(
        config.model_preset,
        d_model=config.d_model,
        hyper_dim=config.hyper_dim,
        n_heads=config.n_heads,
        n_context_layers=config.n_context_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        curvature=config.curvature,
        n_queries=config.n_queries,
        max_cardinality=config.max_cardinality,
        n_queries_by_level=config.n_queries_by_level,
        max_cardinality_by_level=config.max_cardinality_by_level,
        tangent_variance_target=config.tangent_variance_target,
        hyper_projection_init_scale=config.hyper_projection_init_scale,
        tangent_scale_mode=config.tangent_scale_mode,
        hyperbolic_level_encoding=config.hyperbolic_level_encoding,
        type_conditioned_daughter_relation_bias=effective_type_relation_bias,
    )
    _require_scientific_capacity_report(
        architecture, data_module, scientific_mode=config.scientific_mode
    )
    capacity = (
        capacity_statistics_from_index(
            data_module.dataset_index,
            global_n_queries=architecture.n_queries,
            global_max_cardinality=architecture.max_cardinality,
            n_queries_by_level=dict(architecture.n_queries_by_level),
            max_cardinality_by_level=dict(architecture.max_cardinality_by_level),
        )
        if data_module.dataset_index is not None
        else dataset_capacity_statistics(
            data_module.iter_events("train", shuffle=False),
            global_n_queries=architecture.n_queries,
            global_max_cardinality=architecture.max_cardinality,
            n_queries_by_level=dict(architecture.n_queries_by_level),
            max_cardinality_by_level=dict(architecture.max_cardinality_by_level),
            target_policy=config.target_policy,
        )
    )
    require_capacity(capacity)
    ablation = ALL_ABLATIONS[config.ablation]
    model = build_ablation_model(
        config.ablation,
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=architecture.d_model,
        hyper_dim=architecture.hyper_dim,
        n_queries=architecture.n_queries,
        max_cardinality=architecture.max_cardinality,
        n_heads=architecture.n_heads,
        n_context_layers=architecture.n_context_layers,
        curvature=architecture.curvature,
        ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        n_queries_by_level=architecture.n_queries_by_level,
        max_cardinality_by_level=architecture.max_cardinality_by_level,
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
        hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
        type_conditioned_daughter_relation_bias=(
            architecture.type_conditioned_daughter_relation_bias
        ),
        pid_kinematics_mode=config.pid_kinematics_mode,
        pid_temperature=config.pid_temperature,
    ).to(device)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=data_module.normalizers["common"].mean,
            common_std=data_module.normalizers["common"].std,
            composite_mean=data_module.normalizers["composite"].mean,
            composite_std=data_module.normalizers["composite"].std,
        ).to(device)
    )
    transfer_report = None
    if config.pretrained_encoder and ablation.pretrained_encoder_transfer:
        transfer_report = load_pretrained_encoder(
            model.encoder,
            config.pretrained_encoder,
            map_location=device,
            freeze=config.freeze_pretrained_encoder_steps > 0,
            leaf_pid_head=model.leaf_pid_head,
            transfer_leaf_pid_head=config.transfer_leaf_pid_head,
            freeze_leaf_pid_head=config.freeze_leaf_pid_head_steps > 0,
            minimum_coverage=(
                config.minimum_encoder_transfer_coverage
                if config.model_preset == "production_baseline" else 0.0
            ),
            allow_low_coverage=config.allow_low_encoder_transfer_coverage,
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            base_lr=config.learning_rate,
            encoder_lr_multiplier=config.encoder_lr_multiplier,
            leaf_pid_lr_multiplier=config.leaf_pid_lr_multiplier,
        )
    )
    lr_contract = resolve_resume_schedule_contract(
        resume_payload=resume_payload,
        configured_total_steps=config.lr_schedule_total_steps,
        run_max_steps=config.max_steps,
        warmup_fraction=config.warmup_fraction,
        warmup_steps=config.warmup_steps,
        max_warmup_steps=config.max_warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        base_lrs=[group["lr"] for group in optimizer.param_groups],
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, lr_contract)
    scheduler.hypertagging_contract = lr_contract
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.mixed_precision)
    start_step = 0
    if config.resume:
        payload = restore_training_checkpoint(
            config.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
            expected_schema_version=(
                data_module.source_schema_versions[0]
                if len(data_module.source_schema_versions) == 1
                else None
            ),
            expected_split_manifest_hash=data_module.split_manifest_hash,
            expected_feature_spec_hash=feature_spec_v4()["feature_spec_hash"],
            expected_data_order_contract=_data_order_contract(config, data_module),
            expected_architecture=_architecture_contract(config),
        )
        start_step = int(payload.get("step", 0))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(data_module.split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    cursor = StreamingCursor.from_state_dict(
        (resume_payload or {}).get("streaming_cursor", {})
    )
    epoch = cursor.epoch
    batch_iterator = data_module.batches(
        "train", batch_size=config.batch_size, shuffle=True, epoch=epoch
    )
    for _ in range(cursor.batch_index):
        if next(batch_iterator, None) is None:
            raise ValueError("streaming resume cursor exceeds the saved epoch")
    schedule = TeacherForcingSchedule(
        kind=config.scheduled_sampling_schedule,
        start_probability=1.0,
        end_probability=1.0 - config.scheduled_sampling_probability,
        duration_steps=config.scheduled_sampling_duration_steps,
    )
    constraint_policy = ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=tuple(
            sorted((int(level), tuple(tokens)) for level, tokens in data_module.allowed_types_by_level.items())
        ),
        empirical_type_prior_mode=config.empirical_type_prior_mode,
        initial_state_policy=config.initial_state_policy,
    )
    selection_contract = reconstruction_selection_contract(
        best_metric=config.best_metric,
        best_mode=config.best_mode,
        max_validation_events=config.max_validation_events,
        rollout_validation_events=config.rollout_validation_events,
        rollout_validate_every=config.rollout_validate_every,
        rollout_pid_kinematics_mode=config.rollout_pid_kinematics_mode,
        rollout_pid_temperature=config.rollout_pid_temperature,
        target_policy=config.target_policy,
        constraint_policy=constraint_policy.to_dict(),
        eligibility_gates=_rollout_eligibility_contract(config),
        scientific_mode=config.scientific_mode,
        validation_selection_manifest_hash=(
            data_module.selection_manifest_hash or ""
        ),
    )
    final_metrics: dict[str, float] = dict((resume_payload or {}).get("metrics", {}))
    final_loss = float(final_metrics.get("loss", 0.0))
    restored_state = (resume_payload or {}).get("training_state", {})
    if restored_state and restored_state.get("checkpoint_selection_contract") != selection_contract:
        raise ValueError("resume checkpoint-selection semantics differ from the checkpoint")
    best_metric_value = float(
        restored_state.get(
            "best_metric_value",
            float("inf") if config.best_mode == "min" else float("-inf"),
        )
    )
    checkpoint_track_values = {
        **initial_track_values(),
        **{
            str(key): float(value)
            for key, value in restored_state.get("checkpoint_track_values", {}).items()
        },
    }
    patience_count = int(restored_state.get("early_stopping_patience_count", 0))
    completed_steps = start_step
    last_validation_step = int(restored_state.get("last_validation_step", 0))
    restored_validation_selection = (resume_payload or {}).get(
        "validation_selection", {}
    )
    validation_uids: list[str] = list(
        restored_validation_selection.get("event_uids", [])
    )
    signal_controller = (
        signal_controller or install_safe_boundary_signal_controller()
    )
    if not signal_controller.installed:
        signal_controller.install()

    def run_validation(validation_step: int) -> bool:
        nonlocal best_metric_value, checkpoint_track_values, final_metrics
        nonlocal last_validation_step, patience_count
        validation_metrics = validate_reconstruction(
            model,
            data_module,
            device=device,
            scheduled_sampling_probability=(
                config.scheduled_sampling_probability
                if ablation.scheduled_sampling
                else 0.0
            ),
            seed=config.seed,
            max_validation_events=config.max_validation_events,
            rollout_validation_events=(
                config.rollout_validation_events
                if validation_step % config.rollout_validate_every == 0
                else 0
            ),
            validation_batch_size=config.validation_batch_size,
            target_policy=config.target_policy,
            constraint_policy=constraint_policy,
            rollout_pid_kinematics_mode=config.rollout_pid_kinematics_mode,
            rollout_pid_temperature=config.rollout_pid_temperature,
            pilot_allow_train_validation_fallback=(
                config.pilot_allow_train_validation_fallback
                or config.allow_legacy_conflated
                or config.pilot_split_repair
            ),
            selected_event_uids=validation_uids,
            scientific_mode=config.scientific_mode,
            p4_closure_tolerance=config.rollout_p4_tolerance,
        )
        final_metrics.update(validation_metrics)
        logger.log(step=validation_step, split="validation", **validation_metrics)
        last_validation_step = validation_step
        checkpoint_track_values, selected_tracks = checkpoint_track_decisions(
            validation_metrics, checkpoint_track_values
        )
        for track in selected_tracks:
            _save_reconstruction_checkpoint(
                output_dir / track.filename,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                data_module=data_module,
                step=validation_step,
                metrics=final_metrics,
                streaming_cursor=cursor.state_dict(),
                best_metric_value=best_metric_value,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                validation_uids=validation_uids,
                checkpoint_track_values=checkpoint_track_values,
                checkpoint_selection_contract=selection_contract,
                checkpoint_selection_reason=selection_reason(
                    track, validation_metrics
                ),
            )
        primary_track = RECONSTRUCTION_TRACK_BY_METRIC[config.best_metric]
        eligibility = rollout_checkpoint_eligibility(
            validation_metrics, _rollout_eligibility_contract(config)
        )
        final_metrics["rollout_checkpoint_eligible"] = float(eligibility["eligible"])
        primary_evaluated = (
            config.best_metric in validation_metrics
            and math.isfinite(float(validation_metrics[config.best_metric]))
            and float(validation_metrics.get(primary_track.denominator_metric, 0.0)) > 0
            and (not primary_track.requires_rollout or eligibility["eligible"])
        )
        if primary_evaluated:
            current_metric = float(validation_metrics[config.best_metric])
            improved = (
                current_metric < best_metric_value
                if config.best_mode == "min"
                else current_metric > best_metric_value
            )
            if improved:
                best_metric_value = current_metric
                patience_count = 0
                _save_reconstruction_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    data_module=data_module,
                    step=validation_step,
                    metrics=final_metrics,
                    streaming_cursor=cursor.state_dict(),
                    best_metric_value=best_metric_value,
                    patience_count=patience_count,
                    last_validation_step=last_validation_step,
                    validation_uids=validation_uids,
                    checkpoint_track_values=checkpoint_track_values,
                    checkpoint_selection_contract=selection_contract,
                    checkpoint_selection_reason={
                        **selection_reason(primary_track, validation_metrics),
                        "reason": "new_configured_primary_metric",
                    },
                )
            else:
                patience_count += 1
        _save_reconstruction_checkpoint(
            output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            data_module=data_module,
            step=validation_step,
            metrics=final_metrics,
            streaming_cursor=cursor.state_dict(),
            best_metric_value=best_metric_value,
            patience_count=patience_count,
            last_validation_step=last_validation_step,
            validation_uids=validation_uids,
            checkpoint_track_values=checkpoint_track_values,
            checkpoint_selection_contract=selection_contract,
        )
        return (
            config.early_stopping_patience is not None
            and patience_count >= config.early_stopping_patience
        )

    pending_validation_step = restored_state.get("pending_validation_step")
    resume_validation_requested_stop = False
    if pending_validation_step is not None:
        pending_validation_step = int(pending_validation_step)
        if pending_validation_step != start_step or last_validation_step >= start_step:
            raise ValueError("resume checkpoint contains inconsistent pending validation")
        try:
            with signal_controller.restartable_validation():
                resume_validation_requested_stop = run_validation(
                    pending_validation_step
                )
        except PendingValidationInterrupted:
            signal_controller.exit_after_checkpoint()
    loop_start = config.max_steps if resume_validation_requested_stop else start_step
    for step in range(loop_start, config.max_steps):
        if model.pid_kinematics_mode == "temperature_softmax":
            progress = min(step / max(config.pid_temperature_duration_steps, 1), 1.0)
            model.pid_temperature = (
                config.pid_temperature_start
                + progress * (config.pid_temperature_end - config.pid_temperature_start)
            )
        if (
            config.freeze_pretrained_encoder_steps > 0
            and step == config.freeze_pretrained_encoder_steps
        ):
            unfreeze_encoder(model.encoder)
        if (
            config.freeze_leaf_pid_head_steps > 0
            and step == config.freeze_leaf_pid_head_steps
        ):
            for parameter in model.leaf_pid_head.parameters():
                parameter.requires_grad_(True)
        try:
            next_batch = next(batch_iterator)
        except StopIteration:
            epoch += 1
            cursor.epoch = epoch
            cursor.batch_index = 0
            cursor.events_consumed = 0
            batch_iterator = data_module.batches(
                "train", batch_size=config.batch_size, shuffle=True, epoch=epoch
            )
            try:
                next_batch = next(batch_iterator)
            except StopIteration as error:
                raise ValueError("training split produced no batches") from error
        cursor.batch_index += 1
        cursor.events_consumed += int(next_batch["node_mask"].shape[0])
        valid_levels = sorted(
            {
                int(level)
                for level in next_batch["level_ids"][next_batch["node_mask"]].tolist()
                if int(level) > 0
            }
        )
        if not valid_levels:
            raise ValueError("training batch has no reconstruction target levels")
        # Resolve the small set of target levels before the asynchronous device
        # transfer so the normal CUDA path does not synchronize via tolist().
        batch = {name: value.to(device) for name, value in next_batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config.mixed_precision,
        ):
            optimization_started = time.perf_counter()
            (
                reconstruction_loss,
                leaf_pid_loss,
                component_accumulator,
                level_outputs,
                context_metrics,
            ) = _optimization_loss(
                model,
                batch,
                valid_levels=valid_levels,
                config=config,
                schedule=schedule,
                step=step,
                use_scheduled_sampling=ablation.scheduled_sampling,
                allowed_types_by_level=data_module.allowed_types_by_level,
                constraint_policy=constraint_policy,
            )
            optimization_seconds = max(time.perf_counter() - optimization_started, 1e-9)
            loss = reconstruction_loss + leaf_pid_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final_loss = float(loss.detach().cpu())
        final_metrics = {
            "loss": final_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "leaf_pid_loss": float(leaf_pid_loss.detach().cpu()),
            "levels_trained": float(len(valid_levels)),
            **context_metrics,
            "teacher_forcing_probability": schedule.probability(step),
            "pid_temperature": float(model.pid_temperature),
            "events_per_second": batch["node_mask"].shape[0] / optimization_seconds,
            "target_levels_per_second": context_metrics["optimized_event_level_count"]
            / optimization_seconds,
        }
        if step % max(config.diagnostic_forward_interval, 1) == 0:
            diagnostic_level = valid_levels[0]
            with torch.no_grad():
                final_metrics.update(
                    compare_pid_kinematics_modes(
                        model, batch, target_level=diagnostic_level
                    )
                )
        for name, value in component_accumulator.items():
            final_metrics[f"loss_{name}"] = float(
                (value / len(valid_levels)).detach().cpu()
            )
        if level_outputs:
            last_level, last_output, last_loss = level_outputs[-1]
            final_metrics.update(
                next_level_metrics(
                    last_output.pointer,
                    batch,
                    last_loss.matches,
                    target_level=last_level,
                    target_policy=config.target_policy,
                )
            )
            if last_loss.confidence_targets is not None:
                final_metrics.update(
                    confidence_calibration_metrics(
                        last_output.pointer.confidence_logits,
                        last_loss.confidence_targets,
                    )
                )
        logger.log(step=step + 1, target_levels=valid_levels, **final_metrics)
        completed_steps = step + 1
        should_stop = False
        if (step + 1) % config.validate_every == 0:
            _save_reconstruction_checkpoint(
                output_dir / "signal-checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                data_module=data_module,
                step=step + 1,
                metrics=final_metrics,
                streaming_cursor=cursor.state_dict(),
                best_metric_value=best_metric_value,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                validation_uids=validation_uids,
                checkpoint_track_values=checkpoint_track_values,
                checkpoint_selection_contract=selection_contract,
                pending_validation_step=step + 1,
                termination_reason="scheduled_validation_pending",
            )
            try:
                with signal_controller.restartable_validation():
                    should_stop = run_validation(step + 1)
            except PendingValidationInterrupted:
                signal_controller.exit_after_checkpoint()
        if signal_controller.requested:
            _save_reconstruction_checkpoint(
                output_dir / "signal-checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                data_module=data_module,
                step=step + 1,
                metrics=final_metrics,
                streaming_cursor=cursor.state_dict(),
                best_metric_value=best_metric_value,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                validation_uids=validation_uids,
                checkpoint_track_values=checkpoint_track_values,
                checkpoint_selection_contract=selection_contract,
                termination_reason="sigusr1_safe_optimizer_boundary",
            )
            signal_controller.exit_after_checkpoint()
        if should_stop:
            break
        if (step + 1) % config.checkpoint_every == 0:
            _save_reconstruction_checkpoint(
                output_dir / f"checkpoint-step-{step + 1}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                data_module=data_module,
                step=step + 1,
                metrics=final_metrics,
                streaming_cursor=cursor.state_dict(),
                best_metric_value=best_metric_value,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                validation_uids=validation_uids,
                checkpoint_track_values=checkpoint_track_values,
                checkpoint_selection_contract=selection_contract,
            )
    if last_validation_step != completed_steps:
        validation_metrics = validate_reconstruction(
        model,
        data_module,
        device=device,
        scheduled_sampling_probability=(
            config.scheduled_sampling_probability if ablation.scheduled_sampling else 0.0
        ),
        seed=config.seed,
        max_validation_events=config.max_validation_events,
        rollout_validation_events=config.rollout_validation_events,
        validation_batch_size=config.validation_batch_size,
        target_policy=config.target_policy,
        constraint_policy=constraint_policy,
        rollout_pid_kinematics_mode=config.rollout_pid_kinematics_mode,
        rollout_pid_temperature=config.rollout_pid_temperature,
        pilot_allow_train_validation_fallback=(
            config.pilot_allow_train_validation_fallback
            or config.allow_legacy_conflated
            or config.pilot_split_repair
        ),
        selected_event_uids=validation_uids,
        scientific_mode=config.scientific_mode,
        p4_closure_tolerance=config.rollout_p4_tolerance,
        )
        final_metrics.update(validation_metrics)
        logger.log(step=completed_steps, split="validation", **validation_metrics)
        last_validation_step = completed_steps
        checkpoint_track_values, selected_tracks = checkpoint_track_decisions(
            validation_metrics, checkpoint_track_values
        )
        for track in selected_tracks:
            _save_reconstruction_checkpoint(
                output_dir / track.filename, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, config=config,
                data_module=data_module, step=completed_steps,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                best_metric_value=best_metric_value,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                validation_uids=validation_uids,
                checkpoint_track_values=checkpoint_track_values,
                checkpoint_selection_contract=selection_contract,
                checkpoint_selection_reason=selection_reason(
                    track, validation_metrics
                ),
            )
        primary_track = RECONSTRUCTION_TRACK_BY_METRIC[config.best_metric]
        eligibility = rollout_checkpoint_eligibility(
            validation_metrics, _rollout_eligibility_contract(config)
        )
        final_metrics["rollout_checkpoint_eligible"] = float(
            eligibility["eligible"]
        )
        primary_evaluated = (
            config.best_metric in validation_metrics
            and math.isfinite(float(validation_metrics[config.best_metric]))
            and float(validation_metrics.get(primary_track.denominator_metric, 0.0)) > 0
            and (not primary_track.requires_rollout or eligibility["eligible"])
        )
        if primary_evaluated:
            current_metric = float(validation_metrics[config.best_metric])
            improved = (
                current_metric < best_metric_value
                if config.best_mode == "min" else current_metric > best_metric_value
            )
            if improved:
                best_metric_value = current_metric
                patience_count = 0
                _save_reconstruction_checkpoint(
                    output_dir / "best.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, config=config,
                    data_module=data_module, step=completed_steps,
                    metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                    best_metric_value=best_metric_value,
                    patience_count=patience_count,
                    last_validation_step=last_validation_step,
                    validation_uids=validation_uids,
                    checkpoint_track_values=checkpoint_track_values,
                    checkpoint_selection_contract=selection_contract,
                    checkpoint_selection_reason={
                        **selection_reason(primary_track, validation_metrics),
                        "reason": "new_configured_primary_metric",
                    },
                )
            else:
                patience_count += 1
    _save_reconstruction_checkpoint(
        output_dir / "latest.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, config=config,
        data_module=data_module, step=completed_steps, metrics=final_metrics,
        streaming_cursor=cursor.state_dict(), best_metric_value=best_metric_value,
        patience_count=patience_count, last_validation_step=last_validation_step,
        validation_uids=validation_uids,
        checkpoint_track_values=checkpoint_track_values,
        checkpoint_selection_contract=selection_contract,
    )
    checkpoint = _save_reconstruction_checkpoint(
        output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        data_module=data_module,
        step=completed_steps,
        metrics=final_metrics,
        streaming_cursor=cursor.state_dict(),
        best_metric_value=best_metric_value,
        patience_count=patience_count,
        last_validation_step=last_validation_step,
        validation_uids=validation_uids,
        checkpoint_track_values=checkpoint_track_values,
        checkpoint_selection_contract=selection_contract,
    )
    signal_controller.restore()
    if config.scientific_mode and not math.isfinite(best_metric_value):
        raise RuntimeError(
            "scientific run produced no rollout checkpoint eligible for primary selection"
        )
    return ReconstructionTrainingResult(
        checkpoint=checkpoint,
        log_path=logger.path,
        steps=completed_steps,
        final_loss=final_loss,
        metrics=final_metrics,
        transfer_report=transfer_report,
        data_module=data_module,
    )


def _optimization_loss(
    model: LevelAutoregressiveReconstructor,
    batch: dict[str, torch.Tensor],
    *,
    valid_levels: list[int],
    config: ReconstructionConfig,
    schedule: TeacherForcingSchedule,
    step: int,
    use_scheduled_sampling: bool,
    allowed_types_by_level: dict[int, tuple[int, ...]],
    constraint_policy: ReconstructionConstraintPolicy,
):
    """Choose exactly one primary context per event and target level."""

    level_outputs = []
    component_accumulator: dict[str, torch.Tensor] = {}
    leaf_pid_losses = []
    primary_losses = []
    teacher_primary = []
    predicted_primary = []
    auxiliary_losses = []
    truth_contexts = 0
    predicted_contexts = 0
    sampled_teacher_contexts = 0
    sampled_predicted_contexts = 0
    representable = 0
    truth_targets = 0
    first_divergence: list[int] = []
    fallback_teacher_count = 0
    skipped_event_level_count = 0
    recovery_loss_count = 0
    rollout_call_count = 0
    cached_state_count = 0
    model_forward_count = 0
    selected_by_level = _selected_event_levels(
        batch, valid_levels=valid_levels, mode=config.level_sampling_mode,
        seed=config.seed, step=step, target_policy=config.target_policy,
    )
    first_selected_level = {
        batch_index: next(
            level for level in valid_levels if bool(selected_by_level[level][batch_index])
        )
        for batch_index in range(batch["node_mask"].shape[0])
        if any(bool(selected_by_level[level][batch_index]) for level in valid_levels)
    }
    choices_by_level = {
        target_level: schedule.sample(
            batch["node_mask"].shape[0],
            step=step,
            seed=config.seed + 1009 * target_level,
            device=batch["node_mask"].device,
        )
        for target_level in valid_levels
    }
    if not use_scheduled_sampling:
        for choices in choices_by_level.values():
            choices.fill_(True)
    predicted_rollouts: dict[int, Any] = {}
    if any(
        ((~choices) & selected_by_level[level]).any()
        for level, choices in choices_by_level.items()
    ):
        for batch_index in range(batch["node_mask"].shape[0]):
            if not any(
                bool(selected_by_level[level][batch_index])
                and not bool(choices_by_level[level][batch_index])
                for level in valid_levels
            ):
                continue
            truth_single = _single_event_batch(batch, batch_index)
            with torch.no_grad():
                rollout = level_rollout(
                    model, truth_single, mode="predicted",
                    config=RolloutConfig(
                        max_level=max(max(valid_levels) - 1, 0), root_types=(),
                        exclusive_final=False, use_learned_confidence=False,
                        seed=config.seed + step + batch_index,
                        constraint_policy=constraint_policy,
                        rollout_pid_kinematics_mode=config.rollout_pid_kinematics_mode,
                        rollout_pid_temperature=config.rollout_pid_temperature,
                    ),
                )
            predicted_rollouts[batch_index] = rollout
            rollout_call_count += 1
            cached_state_count += len(rollout.cached_states)
            model_forward_count += len(rollout.steps)
    for target_level in valid_levels:
        choices = choices_by_level[target_level]
        per_level_components: dict[str, list[torch.Tensor]] = {}
        level_entries: list[
            tuple[int, dict[str, torch.Tensor], Any, bool, dict[str, torch.Tensor], int, int]
        ] = []
        for batch_index in range(batch["node_mask"].shape[0]):
            if not bool(selected_by_level[target_level][batch_index]):
                continue
            truth_single = _single_event_batch(batch, batch_index)
            choose_teacher = bool(choices[batch_index])
            recovery_missing = 0
            masked_missing = 0
            sampled_teacher_contexts += int(choose_teacher)
            sampled_predicted_contexts += int(not choose_teacher)
            if choose_teacher:
                truth_contexts += 1
                context = truth_single
                target_override = None
                aligned = aligned_level_targets(
                    truth_single,
                    truth_single,
                    target_level=target_level,
                    target_policy=config.target_policy,
                )
            else:
                predicted_contexts += 1
                with torch.no_grad():
                    context = cached_context_for_level(
                        predicted_rollouts[batch_index], target_level
                    )
                aligned = aligned_level_targets(
                    truth_single,
                    context,
                    target_level=target_level,
                    target_policy=config.target_policy,
                )
                target_override = aligned.target_override
                decision = resolve_unrepresentable_target_policy(
                    config.unrepresentable_target_policy,
                    truth_target_count=aligned.truth_target_count,
                    representable_target_count=aligned.representable_count,
                )
                if aligned.representable_count < aligned.truth_target_count:
                    if decision.use_teacher_context:
                        context = truth_single
                        target_override = None
                        choose_teacher = True
                        predicted_contexts -= 1
                        truth_contexts += 1
                        fallback_teacher_count += 1
                    elif decision.skip_event_level:
                        skipped_event_level_count += 1
                        continue
                    elif decision.add_recovery_objective:
                        recovery_missing = (
                            aligned.truth_target_count - aligned.representable_count
                        )
                        recovery_loss_count += recovery_missing
                    if not decision.use_teacher_context and not decision.skip_event_level:
                        masked_missing = (
                            aligned.truth_target_count - aligned.representable_count
                        )
            truth_targets += aligned.truth_target_count
            representable += aligned.representable_count
            if aligned.representable_count < aligned.truth_target_count:
                first_divergence.append(target_level)
            level_entries.append(
                (
                    batch_index, context, target_override, choose_teacher,
                    truth_single, recovery_missing, masked_missing,
                )
            )
        if level_entries:
            level_batch = _collate_context_batches([entry[1] for entry in level_entries])
            level_batch = _with_allowed_types(
                level_batch, target_level, allowed_types_by_level, constraint_policy
            )
            output = model(level_batch, target_level=target_level)
            model_forward_count += 1
        for entry_index, (
            batch_index, _context, target_override, choose_teacher, truth_single,
            recovery_missing, masked_missing,
        ) in enumerate(level_entries):
            context = _slice_batch_dict(level_batch, entry_index)
            pointer_output = _slice_pointer_output(output.pointer, entry_index)
            loss_batch = dict(context)
            if output.current_p4 is not None:
                loss_batch["p4"] = output.current_p4[entry_index : entry_index + 1]
            loss_output = level_reconstruction_loss(
                pointer_output,
                loss_batch,
                target_level=target_level,
                target_policy=config.target_policy,
                target_override=target_override,
                matching_production=not config.allow_tiny_bruteforce_matching,
                constraint_policy=constraint_policy,
                unrepresentable_target_counts=[masked_missing],
                weights={"query_repulsion": config.query_repulsion_weight},
            )
            recovery_loss = loss_output.total * 0.0
            if recovery_missing:
                count = min(recovery_missing, pointer_output.object_logits.shape[1])
                top_object_logits = pointer_output.object_logits.topk(count, dim=-1).values
                recovery_loss = F.softplus(-top_object_logits).mean()
                per_level_components.setdefault("recovery", []).append(recovery_loss)
            primary_event_loss = loss_output.total + recovery_loss
            primary_losses.append(primary_event_loss)
            (teacher_primary if choose_teacher else predicted_primary).append(primary_event_loss)
            for name, value in loss_output.components.items():
                per_level_components.setdefault(name, []).append(value)
            raw_tracks = (
                context["node_mask"]
                & (context["leaf_kinematics_mode_ids"] == LEAF_MODE_TO_ID["raw_track_predicted_pid"])
                & context["truth_pid_available"]
            )
            if (
                target_level == first_selected_level.get(batch_index)
                and output.leaf_pid_logits is not None
                and raw_tracks.any()
            ):
                leaf_pid_losses.append(
                    F.cross_entropy(
                        output.leaf_pid_logits[entry_index : entry_index + 1][raw_tracks],
                        context["truth_pid_labels"][raw_tracks],
                    )
                )
            if not choose_teacher and config.auxiliary_teacher_weight > 0:
                aux_batch = _with_allowed_types(truth_single, target_level, allowed_types_by_level, constraint_policy)
                aux_output = model(aux_batch, target_level=target_level)
                model_forward_count += 1
                aux_loss_batch = dict(aux_batch)
                if aux_output.current_p4 is not None:
                    aux_loss_batch["p4"] = aux_output.current_p4
                auxiliary_losses.append(
                    level_reconstruction_loss(
                        aux_output.pointer,
                        aux_loss_batch,
                        target_level=target_level,
                        target_policy=config.target_policy,
                        matching_production=not config.allow_tiny_bruteforce_matching,
                        constraint_policy=constraint_policy,
                        weights={"query_repulsion": config.query_repulsion_weight},
                    ).total
                )
        for name, values in per_level_components.items():
            component_accumulator[name] = component_accumulator.get(name, 0) + torch.stack(values).mean()
        # Detached batched teacher view is retained only for metric reporting.
        if step % max(config.diagnostic_forward_interval, 1) == 0:
          with torch.no_grad():
            diagnostic_batch = _with_allowed_types(batch, target_level, allowed_types_by_level, constraint_policy)
            diagnostic_output = model(diagnostic_batch, target_level=target_level)
            model_forward_count += 1
            diagnostic_loss_batch = dict(diagnostic_batch)
            if diagnostic_output.current_p4 is not None:
                diagnostic_loss_batch["p4"] = diagnostic_output.current_p4
            diagnostic_loss = level_reconstruction_loss(
                diagnostic_output.pointer,
                diagnostic_loss_batch,
                target_level=target_level,
                target_policy=config.target_policy,
                matching_production=not config.allow_tiny_bruteforce_matching,
                constraint_policy=constraint_policy,
                weights={"query_repulsion": config.query_repulsion_weight},
            )
          level_outputs.append((target_level, diagnostic_output, diagnostic_loss))
    primary_loss = (
        torch.stack(primary_losses).mean()
        if primary_losses
        else next(model.parameters()).sum() * 0.0
    )
    auxiliary_teacher_loss = (
        torch.stack(auxiliary_losses).mean()
        if auxiliary_losses
        else primary_loss * 0.0
    )
    reconstruction_loss = (
        primary_loss + config.auxiliary_teacher_weight * auxiliary_teacher_loss
    )
    leaf_pid_loss = (
        torch.stack(leaf_pid_losses).mean()
        if leaf_pid_losses
        else primary_loss * 0.0
    )
    total_contexts = truth_contexts + predicted_contexts
    sampled_contexts = sampled_teacher_contexts + sampled_predicted_contexts
    metrics = {
        "context_truth_fraction": truth_contexts / max(total_contexts, 1),
        "context_predicted_fraction": predicted_contexts / max(total_contexts, 1),
        "target_representable_rate": representable / max(truth_targets, 1),
        "unrepresentable_target_count": float(truth_targets - representable),
        "truth_target_count": float(truth_targets),
        "representable_target_count": float(representable),
        "fallback_teacher_count": float(fallback_teacher_count),
        "skipped_event_level_count": float(skipped_event_level_count),
        "recovery_loss_count": float(recovery_loss_count),
        "configured_teacher_probability": schedule.probability(step),
        "sampled_teacher_fraction": sampled_teacher_contexts / max(sampled_contexts, 1),
        "sampled_predicted_fraction": sampled_predicted_contexts / max(sampled_contexts, 1),
        "model_forward_count": float(model_forward_count),
        "rollout_call_count": float(rollout_call_count),
        "cached_state_count": float(cached_state_count),
        "optimized_event_level_count": float(len(primary_losses)),
        "sampled_teacher_count": float(sampled_teacher_contexts),
        "sampled_predicted_count": float(sampled_predicted_contexts),
        "primary_teacher_loss": float(
            torch.stack(teacher_primary).mean().detach().cpu() if teacher_primary else 0.0
        ),
        "primary_predicted_loss": float(
            torch.stack(predicted_primary).mean().detach().cpu() if predicted_primary else 0.0
        ),
        "auxiliary_teacher_loss": float(auxiliary_teacher_loss.detach().cpu()),
        "first_context_divergence_level": float(
            min(first_divergence) if first_divergence else -1
        ),
        "scheduled_sampling_loss": float(
            torch.stack(predicted_primary).mean().detach().cpu() if predicted_primary else 0.0
        ),
    }
    return (
        reconstruction_loss,
        leaf_pid_loss,
        component_accumulator,
        level_outputs,
        metrics,
    )


def _selected_event_levels(
    batch: dict[str, torch.Tensor],
    *,
    valid_levels: list[int],
    mode: str,
    seed: int,
    step: int,
    target_policy: str,
) -> dict[int, torch.Tensor]:
    """Choose reconstruction levels without changing schedule progress."""

    batch_size = batch["node_mask"].shape[0]
    selected = {
        level: torch.zeros(batch_size, dtype=torch.bool, device=batch["node_mask"].device)
        for level in valid_levels
    }
    if mode == "all_levels":
        for values in selected.values():
            values.fill_(True)
        return selected
    generator = torch.Generator().manual_seed(seed + 7919 * step)
    for batch_index in range(batch_size):
        candidates = []
        for level in valid_levels:
            eligible = batch["node_mask"][batch_index] & (
                batch["level_ids"][batch_index] == level
            )
            if target_policy != "diagnostic_all":
                eligible &= batch["valid_reconstruction_target"][batch_index]
            if target_policy == "complete_only":
                eligible &= batch["recursive_reconstructable_complete"][batch_index]
            if eligible.any():
                candidates.append(level)
        if not candidates:
            candidates = [
                level for level in valid_levels
                if bool(
                    (
                        batch["node_mask"][batch_index]
                        & (batch["level_ids"][batch_index] == level)
                    ).any()
                )
            ]
        if not candidates:
            continue
        if mode == "one_level_per_event":
            position = int(torch.randint(len(candidates), (1,), generator=generator))
        elif mode == "stratified_level_sampling":
            position = (step + batch_index) % len(candidates)
        else:
            raise ValueError(f"unknown level sampling mode: {mode}")
        selected[candidates[position]][batch_index] = True
    return selected


def _single_event_batch(
    batch: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, torch.Tensor]:
    batch_size = batch["node_mask"].shape[0]
    single = {
        name: (
            value[batch_index : batch_index + 1]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
            else value
        )
        for name, value in batch.items()
    }
    from hypertagging.reconstruction.level_rollout import _select_nodes

    return _select_nodes(single, single["node_mask"][0])


_NODE_BATCH_FIELDS = {
    "common_features", "common_availability", "track_features",
    "track_availability", "cluster_features", "cluster_availability",
    "klm_features", "klm_availability",
    "composite_features", "composite_availability",
    "daughter_input_pid_histogram", "daughter_truth_pid_histogram",
    "daughter_pid_histogram", "daughter_input_pid_histogram_available",
    "daughter_truth_pid_histogram_available", "daughter_pid_histogram_available",
    "node_kind_ids", "leaf_kinematics_mode_ids", "pid_labels",
    "pid_target_labels", "truth_pid_labels", "truth_pid_available",
    "runtime_composite_type_source_ids", "level_ids", "p4", "charge",
    "parent_ids", "active", "copied", "node_ids", "reco_ids",
    "source_node_ids", "copied_from", "b_side", "node_mask",
    "node_features", "full_truth_daughter_count",
    "retained_truth_daughter_count_expected", "retained_daughter_count",
    "reconstructed_daughter_count", "complete_truth_decay",
    "complete_reconstructable_decay", "recursive_reconstructable_complete",
    "partial_missing_daughters", "contracted_intermediate",
    "valid_reconstruction_target", "truth_root_distance",
    "full_event_max_level", "current_pid_probabilities", "current_pid_tokens",
    "current_pid_available", "daughter_input_pid_source_ids",
    "daughter_truth_pid_source_ids", "model_input_source_ids",
    "truth_supervision_source_ids", "depth_from_retained_root",
    "distance_to_nearest_retained_root",
}
_PAIR_BATCH_FIELDS = {
    "daughter_adjacency", "source_conflict_matrix", "lca_depth",
    "lca_node_id", "edges_to_lca_from_i", "edges_to_lca_from_j",
    "exact_tree_path_distance", "ancestor_descendant_relation",
}


def _collate_context_batches(
    contexts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad dynamic one-event rollout states for one target-level forward."""

    if not contexts:
        raise ValueError("cannot collate an empty context list")
    max_nodes = max(context["node_mask"].shape[1] for context in contexts)
    keys = set.intersection(*(set(context) for context in contexts))
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [context[key] for context in contexts]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        if key == "recursive_leaf_source_mask":
            max_sources = max(value.shape[2] for value in values)
            combined = values[0].new_zeros((len(values), max_nodes, max_sources))
            for index, value in enumerate(values):
                combined[index, : value.shape[1], : value.shape[2]] = value[0]
            output[key] = combined
        elif key in _PAIR_BATCH_FIELDS:
            combined = values[0].new_zeros((len(values), max_nodes, max_nodes))
            for index, value in enumerate(values):
                combined[index, : value.shape[1], : value.shape[2]] = value[0]
            output[key] = combined
        elif key in _NODE_BATCH_FIELDS:
            combined = values[0].new_zeros((len(values), max_nodes, *values[0].shape[2:]))
            if key in {
                "parent_ids", "level_ids", "node_ids", "reco_ids",
                "source_node_ids", "copied_from", "daughter_input_pid_source_ids",
                "daughter_truth_pid_source_ids", "model_input_source_ids",
                "truth_supervision_source_ids",
            }:
                combined.fill_(-1)
            for index, value in enumerate(values):
                combined[index, : value.shape[1]] = value[0]
            output[key] = combined
        elif values[0].ndim > 0 and values[0].shape[0] == 1:
            output[key] = torch.cat(values, dim=0)
        else:
            output[key] = values[0]
    output["node_features"] = output["common_features"]
    output["node_mask"] = output["active"]
    return output


def _slice_batch_dict(
    batch: dict[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    batch_size = batch["node_mask"].shape[0]
    return {
        key: (
            value[index : index + 1]
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size
            else value
        )
        for key, value in batch.items()
    }


def _slice_pointer_output(output: MotherPointerOutput, index: int) -> MotherPointerOutput:
    return MotherPointerOutput(
        object_logits=output.object_logits[index : index + 1],
        type_logits=output.type_logits[index : index + 1],
        pointer_logits=output.pointer_logits[index : index + 1],
        cardinality_logits=output.cardinality_logits[index : index + 1],
        confidence_logits=output.confidence_logits[index : index + 1],
        expected_type_embedding=(
            output.expected_type_embedding[index : index + 1]
            if output.expected_type_embedding is not None else None
        ),
    )


def _with_allowed_types(
    batch: dict[str, torch.Tensor],
    target_level: int,
    allowed_types_by_level: dict[int, tuple[int, ...]],
    constraint_policy: ReconstructionConstraintPolicy | None = None,
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    policy = constraint_policy or ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=((target_level, allowed_types_by_level.get(target_level, ())),)
    )
    allowed, bias = policy.type_constraints(target_level, device=batch["node_mask"].device)
    result["allowed_type_mask"] = allowed
    result["type_logit_bias"] = bias
    result["pointer_validity_mask"] = policy.pointer_validity_mask(batch, target_level)
    return result


@torch.no_grad()
def validate_reconstruction(
    model: LevelAutoregressiveReconstructor,
    data_module: RealDataModule,
    *,
    device: torch.device,
    scheduled_sampling_probability: float = 0.5,
    seed: int = 11,
    max_validation_events: int = 32,
    rollout_validation_events: int = 8,
    validation_batch_size: int = 4,
    target_policy: str = "complete_only",
    constraint_policy: ReconstructionConstraintPolicy | None = None,
    rollout_pid_kinematics_mode: str = "soft_decision_hard_construction",
    rollout_pid_temperature: float = 0.5,
    pilot_allow_train_validation_fallback: bool = False,
    selected_event_uids: list[str] | None = None,
    scientific_mode: bool = False,
    p4_closure_tolerance: float = 1e-6,
) -> dict[str, float]:
    model.eval()
    source = data_module.iter_events("validation", shuffle=False)
    used_train_fallback = False
    if data_module.split_counts.get("validation", 0) == 0:
        if not pilot_allow_train_validation_fallback:
            raise ValueError(
                "validation split is empty; production validation cannot fall back "
                "to training data. Set pilot_allow_train_validation_fallback=True "
                "only for an explicitly diagnostic pilot."
            )
        source = data_module.iter_events("train", shuffle=False)
        used_train_fallback = True
    if scientific_mode and used_train_fallback:
        raise ValueError("scientific validation cannot use the training fallback")
    if validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    restored_uids = tuple(selected_event_uids or ())
    events, fixed_uids, _selection_contract = select_validation_events(
        source,
        limit=max_validation_events,
        scientific_mode=scientific_mode,
        selection_manifest_hash=getattr(data_module, "selection_manifest_hash", None),
        seed=seed,
        restored_event_uids=restored_uids,
    )
    if not events:
        raise ValueError("bounded validation selection contains no events")
    if selected_event_uids is not None and not selected_event_uids:
        selected_event_uids.extend(fixed_uids)
    accumulated: dict[str, list[float]] = {}

    # Dynamic full-tree rollouts remain per-event, while fixed next-level
    # evaluation uses the requested validation batch size.
    for start in range(0, len(events), validation_batch_size):
        batch = data_module.normalize_batch(
            collate_heterogeneous_events(events[start : start + validation_batch_size])
        )
        target_levels = sorted(
            {
                int(level)
                for level in batch["level_ids"][batch["node_mask"]].tolist()
                if int(level) > 0
            }
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        for target_level in target_levels:
            level_batch = _with_allowed_types(
                batch, target_level, data_module.allowed_types_by_level, constraint_policy
            )
            output = model(level_batch, target_level=target_level)
            loss_batch = dict(level_batch)
            if output.current_p4 is not None:
                loss_batch["p4"] = output.current_p4
            loss_output = level_reconstruction_loss(
                output.pointer,
                loss_batch,
                target_level=target_level,
                target_policy=target_policy,
                constraint_policy=constraint_policy,
            )
            accumulated.setdefault("validation_loss_total", []).append(
                float(loss_output.total.detach().cpu())
            )
            for component_name, component_value in loss_output.components.items():
                accumulated.setdefault(
                    f"validation_loss_{component_name}", []
                ).append(float(component_value.detach().cpu()))
            for event_index in range(batch["node_mask"].shape[0]):
                event_pointer = _slice_pointer_output(output.pointer, event_index)
                event_batch = _slice_batch_dict(loss_batch, event_index)
                metrics = next_level_metrics(
                    event_pointer,
                    event_batch,
                    [loss_output.matches[event_index]],
                    target_level=target_level,
                    target_policy=target_policy,
                )
                for name, value in metrics.items():
                    accumulated.setdefault(
                        f"level_{target_level}_{name}", []
                    ).append(float(value))
                    if name.endswith(("_numerator", "_denominator")):
                        accumulated.setdefault(name, []).append(float(value))
                if loss_output.confidence_targets is not None:
                    for name, value in confidence_calibration_metrics(
                        event_pointer.confidence_logits,
                        loss_output.confidence_targets[event_index : event_index + 1],
                    ).items():
                        accumulated.setdefault(name, []).append(float(value))

    rollout_count = min(len(events), rollout_validation_events)
    for event_count, event in enumerate(events[:rollout_count]):
        batch = data_module.normalize_batch(collate_heterogeneous_events([event]))
        batch = {name: value.to(device) for name, value in batch.items()}
        teacher = level_rollout(
            model,
            batch,
            mode="teacher_forced",
            config=RolloutConfig(max_level=8, root_types=(), exclusive_final=False, constraint_policy=constraint_policy, rollout_pid_kinematics_mode=rollout_pid_kinematics_mode, rollout_pid_temperature=rollout_pid_temperature),
        )
        predicted = level_rollout(
            model,
            batch,
            mode="predicted",
            config=RolloutConfig(
                max_level=8,
                root_types=(),
                confidence_trained=True,
                use_learned_confidence=True,
                constraint_policy=constraint_policy,
                rollout_pid_kinematics_mode=rollout_pid_kinematics_mode,
                rollout_pid_temperature=rollout_pid_temperature,
            ),
        )
        bounded = None
        try:
            bounded = level_rollout(
                model,
                batch,
                mode="predicted",
                config=RolloutConfig(
                    max_level=8,
                    root_types=(),
                    confidence_trained=True,
                    use_learned_confidence=True,
                    constraint_policy=constraint_policy,
                    exclusive_resolution="weighted_set_packing",
                    max_resolution_proposals=12,
                    rollout_pid_kinematics_mode=rollout_pid_kinematics_mode,
                    rollout_pid_temperature=rollout_pid_temperature,
                ),
            )
        except ValueError as error:
            if "weighted set packing is bounded" not in str(error):
                raise
        scheduled = level_rollout(
            model,
            batch,
            mode="scheduled",
            config=RolloutConfig(
                max_level=8,
                root_types=(),
                scheduled_sampling_probability=scheduled_sampling_probability,
                seed=seed + event_count,
                constraint_policy=constraint_policy,
                rollout_pid_kinematics_mode=rollout_pid_kinematics_mode,
                rollout_pid_temperature=rollout_pid_temperature,
            ),
        )
        teacher_metrics = summarize_rollout(teacher.batch, batch)
        predicted_metrics = summarize_rollout(predicted.batch, batch)
        predicted_metrics["p4_closure_rate"] = p4_closure_rate(
            predicted.batch, tolerance=p4_closure_tolerance
        )
        leaf_multiplicity = int(
            (
                batch["node_mask"][0]
                & (batch["level_ids"][0] == 0)
            ).sum()
        )
        multiplicity_slice = (
            "low" if leaf_multiplicity <= 4 else "medium" if leaf_multiplicity <= 8 else "high"
        )
        truth_depth = int(batch["level_ids"][batch["node_mask"]].max())
        represented = total_targets = 0
        for target_level in range(1, truth_depth + 1):
            alignment = aligned_level_targets(
                batch,
                predicted.batch,
                target_level=target_level,
                target_policy=target_policy,
            )
            represented += alignment.representable_count
            total_targets += alignment.truth_target_count
        values = {
            "teacher_forced_p4_closure_rate": float(
                teacher_metrics["p4_closure_rate"]
            ),
            "predicted_p4_closure_rate": float(predicted_metrics["p4_closure_rate"]),
            "predicted_tree_validity_rate": float(
                predicted_metrics["tree_validity_rate"]
            ),
            "predicted_edge_denominator": float(
                batch["daughter_adjacency"][batch["node_mask"]].sum()
            ),
            "predicted_p4_closure_denominator": float(
                (
                    predicted.batch["daughter_adjacency"].any(dim=-1)
                    & predicted.batch["node_mask"]
                ).sum()
            ),
            "predicted_recursive_source_conflicts": float(
                _recursive_source_conflicts(predicted.batch)
            ),
            "scheduled_rollout_valid": float(scheduled.valid),
            "representable_target_rate": represented / max(total_targets, 1),
            "representable_target_efficiency_numerator": float(represented),
            "representable_target_efficiency_denominator": float(total_targets),
            f"multiplicity_{multiplicity_slice}_full_tree_exact_match": float(
                predicted_metrics["full_tree_exact_match"]
            ),
            f"depth_{truth_depth}_full_tree_exact_match": float(
                predicted_metrics["full_tree_exact_match"]
            ),
            "channel_pair_full_tree_exact_match": float(
                predicted_metrics["full_tree_exact_match"]
            ),
            **{
                f"predicted_{name}": float(value)
                for name, value in predicted_metrics.items()
                if isinstance(value, (bool, int, float))
            },
        }
        ambiguity: dict[str, list[float]] = {}
        for step_result in predicted.steps:
            context = cached_context_for_level(predicted, step_result.target_level)
            recursive = context.get("recursive_leaf_source_mask")
            if recursive is None:
                continue
            metrics = proposal_ambiguity_metrics(
                list(step_result.proposals),
                list(step_result.accepted),
                total_queries=step_result.model_output.pointer.object_logits.shape[1],
                recursive_leaf_source_mask=recursive[0],
            )
            for name, value in metrics.items():
                ambiguity.setdefault(name, []).append(value)
        values.update(
            {
                name: sum(metric_values) / len(metric_values)
                for name, metric_values in ambiguity.items()
                if metric_values
            }
        )
        values["bounded_resolution_available"] = float(bounded is not None)
        if bounded is not None:
            bounded_metrics = summarize_rollout(bounded.batch, batch)
            values["bounded_full_tree_exact_match"] = float(
                bounded_metrics["full_tree_exact_match"]
            )
            values["greedy_bounded_tree_difference"] = float(
                predicted_metrics["full_tree_exact_match"]
                != bounded_metrics["full_tree_exact_match"]
            )
        complete_correct, complete_eligible = complete_target_efficiency_counts(
            predicted.batch, batch, target_policy=target_policy
        )
        values.update({
            "eligible_complete_target_count": float(complete_eligible),
            "correctly_reconstructed_complete_target_count": float(complete_correct),
            "complete_target_efficiency": complete_correct / max(complete_eligible, 1),
            "complete_target_efficiency_numerator": float(complete_correct),
            "complete_target_efficiency_denominator": float(complete_eligible),
        })
        for name, value in values.items():
            accumulated.setdefault(name, []).append(value)
    model.train()
    output_metrics = _aggregate_metric_lists(accumulated)
    output_metrics.update(
        {
            "validation_events": float(len(events)),
            "validation_teacher_forced_terms": float(
                len(accumulated.get("validation_loss_total", []))
            ),
            "rollout_validation_events": float(rollout_count),
            "validation_batch_size": float(validation_batch_size),
            "validation_used_train_fallback": float(used_train_fallback),
            "p4_closure_tolerance": float(p4_closure_tolerance),
        }
    )
    return output_metrics


def _aggregate_metric_lists(
    accumulated: dict[str, list[float]],
) -> dict[str, float]:
    """Compute macro means and micro ratios from sufficient statistics."""

    output: dict[str, float] = {}
    for name, values in accumulated.items():
        if name.endswith(("_numerator", "_denominator")):
            output[name] = float(sum(values))
        elif values:
            output[name] = output[f"macro_{name}"] = float(sum(values) / len(values))
    for name in accumulated:
        if not name.endswith("_numerator"):
            continue
        stem = name.removesuffix("_numerator")
        denominator_name = f"{stem}_denominator"
        if denominator_name not in accumulated:
            continue
        numerator = sum(accumulated[name])
        denominator = sum(accumulated[denominator_name])
        output[f"micro_{stem}"] = numerator / denominator if denominator else 0.0
    return output


def _recursive_source_conflicts(batch: dict[str, torch.Tensor]) -> int:
    """Count same-level reconstructed nodes that reuse an underlying leaf."""

    sources = batch.get("recursive_leaf_source_mask")
    if sources is None:
        return 0
    conflicts = 0
    for event_index in range(batch["node_mask"].shape[0]):
        active = batch["node_mask"][event_index]
        levels = batch["level_ids"][event_index]
        for level in torch.unique(levels[active & (levels > 0)]).tolist():
            nodes = (active & (levels == int(level))).nonzero(
                as_tuple=False
            ).flatten()
            if nodes.numel() < 2:
                continue
            node_sources = sources[event_index, nodes]
            overlap = boolean_matmul(node_sources, node_sources.T)
            conflicts += int(torch.triu(overlap, diagonal=1).sum())
    return conflicts


def _rollout_eligibility_contract(
    config: ReconstructionConfig,
) -> dict[str, object]:
    return {
        "version": "rollout-checkpoint-eligibility-v1",
        "minimum_tree_validity": float(config.rollout_min_tree_validity),
        "minimum_p4_closure": float(config.rollout_min_p4_closure),
        "p4_closure_tolerance": float(config.rollout_p4_tolerance),
        "maximum_recursive_source_conflicts": int(
            config.rollout_max_recursive_source_conflicts
        ),
        "required_denominators": list(config.rollout_required_denominators),
    }


def _require_scientific_capacity_report(
    architecture: Any,
    data_module: RealDataModule,
    *,
    scientific_mode: bool,
) -> None:
    if (
        scientific_mode
        and architecture.capacity_report_required
        and data_module.dataset_index is None
    ):
        raise ValueError(
            "small_candidate is capacity-report-required: scientific training "
            "requires a checked dataset index before model construction"
        )


def _save_reconstruction_checkpoint(
    path: Path,
    *,
    model: LevelAutoregressiveReconstructor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: ReconstructionConfig,
    data_module: RealDataModule,
    step: int,
    metrics: dict[str, float],
    streaming_cursor: dict[str, int],
    best_metric_value: float = float("inf"),
    patience_count: int = 0,
    last_validation_step: int = 0,
    validation_uids: list[str] | None = None,
    checkpoint_track_values: dict[str, float] | None = None,
    checkpoint_selection_contract: dict[str, object] | None = None,
    checkpoint_selection_reason: dict[str, object] | None = None,
    pending_validation_step: int | None = None,
    termination_reason: str = "running_or_normal_checkpoint",
) -> Path:
    return save_training_checkpoint(
        path,
        model=model,
        encoder=model.encoder,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        config=asdict(config),
        metrics=metrics,
        normalizer_state=data_module.normalization_state(),
        split_manifest_hash=data_module.split_manifest_hash,
        confidence_head_trained=True,
        schedule_state={
            "step": step,
            "teacher_forcing_probability": TeacherForcingSchedule(
                kind=config.scheduled_sampling_schedule,
                start_probability=1.0,
                end_probability=1.0 - config.scheduled_sampling_probability,
                duration_steps=config.scheduled_sampling_duration_steps,
            ).probability(max(step - 1, 0)),
            "kind": config.scheduled_sampling_schedule,
            "learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "lr_schedule_contract": dict(
                getattr(scheduler, "hypertagging_contract", {})
            ),
        },
        feature_contract={
            "feature_spec_revision": feature_spec_v4()["feature_spec_revision"],
            "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
            "model_feature_contract_hash": feature_spec_v4()["model_feature_contract_hash"],
            "track_fit_policies": list(
                data_module.track_fit_policies
            ),
            "pid_reconstruction_mode": model.pid_kinematics_mode,
            "pid_temperature": float(model.pid_temperature),
            "daughter_compatibility": (
                "type_conditioned_relation_aware"
                if config.type_conditioned_daughter_relation_bias
                or ALL_ABLATIONS[config.ablation].type_conditioned_daughter_relation_bias
                else "generic_relation_aware"
            ),
            "exclusive_resolver": {
                "production": "greedy",
                "weighted_set_packing": "bounded_diagnostic_only",
                "beam": "bounded_diagnostic_only",
            },
            "reconstruction_constraint_policy": ReconstructionConstraintPolicy(
                allowed_mother_types_by_level=tuple(
                    sorted((int(level), tuple(tokens)) for level, tokens in data_module.allowed_types_by_level.items())
                ),
                empirical_type_prior_mode=config.empirical_type_prior_mode,
                initial_state_policy=config.initial_state_policy,
            ).to_dict(),
        },
        legacy_conflated_fraction=data_module.legacy_conflated_fraction,
        schema_version=(
            data_module.source_schema_versions[0]
            if len(data_module.source_schema_versions) == 1
            else "mixed"
        ),
        streaming_cursor=streaming_cursor,
        epoch=int(streaming_cursor.get("epoch", 0)),
        data_order_contract={
            **_data_order_contract(config, data_module),
            "epoch": int(streaming_cursor.get("epoch", 0)),
            "batch_index": int(streaming_cursor.get("batch_index", 0)),
        },
        architecture=_architecture_contract(config),
        training_state={
            "best_metric": config.best_metric,
            "best_mode": config.best_mode,
            "best_metric_value": float(best_metric_value),
            "early_stopping_patience_count": int(patience_count),
            "last_validation_step": int(last_validation_step),
            "pending_validation_step": pending_validation_step,
            "checkpoint_track_values": dict(checkpoint_track_values or {}),
            "checkpoint_selection_contract": dict(
                checkpoint_selection_contract or {}
            ),
            "checkpoint_selection_reason": dict(
                checkpoint_selection_reason or {
                    "reason": "latest_or_periodic_state",
                    "metric_name": config.best_metric,
                    "mode": config.best_mode,
                }
            ),
            "lr_schedule_contract": dict(
                getattr(scheduler, "hypertagging_contract", {})
            ),
            "last_rollout_checkpoint_eligibility": rollout_checkpoint_eligibility(
                metrics, _rollout_eligibility_contract(config)
            ),
            "termination_reason": termination_reason,
        },
        validation_selection={
            "split": (
                "train_diagnostic_fallback"
                if metrics.get("validation_used_train_fallback", 0.0)
                else "validation"
            ),
            "event_uids": list(validation_uids or []),
            "rollout_event_uids": list(validation_uids or [])[
                : int(metrics.get("rollout_validation_events", 0.0))
            ],
            "rollout_was_run": bool(
                metrics.get("rollout_validation_events", 0.0) > 0
            ),
            "deterministic": True,
            "strategy": (
                "manifest_validation_role_uid_hash"
                if config.scientific_mode else "non_scientific_ci_prefix"
            ),
            "scientific_mode": bool(config.scientific_mode),
            "selection_manifest_hash": data_module.selection_manifest_hash or "",
            "max_validation_events": int(config.max_validation_events),
            "rollout_validation_events": int(config.rollout_validation_events),
        },
    )


def _data_order_contract(
    config: ReconstructionConfig, data_module: RealDataModule
) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "shuffle_buffer_size": config.shuffle_buffer_size,
        "seed": config.seed,
        "max_events": config.max_events,
        "dataset_index_hash": (
            data_module.dataset_index.get("index_hash", "")
            if data_module.dataset_index else ""
        ),
        "split_hash": data_module.split_manifest_hash,
        "target_policy": config.target_policy,
        "curriculum_order": "level_autoregressive",
        "teacher_forcing_schedule": {
            "kind": config.scheduled_sampling_schedule,
            "duration_steps": config.scheduled_sampling_duration_steps,
            "end_probability": 1.0 - config.scheduled_sampling_probability,
        },
        "level_sampling_mode": config.level_sampling_mode,
        "initial_state_policy": config.initial_state_policy,
        "gradient_accumulation": config.gradient_accumulation,
        "num_workers": config.num_workers,
        "scientific_mode": config.scientific_mode,
    }


def _architecture_contract(config: ReconstructionConfig) -> dict[str, Any]:
    return resolve_model_architecture(
        config.model_preset,
        d_model=config.d_model,
        hyper_dim=config.hyper_dim,
        n_heads=config.n_heads,
        n_context_layers=config.n_context_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        curvature=config.curvature,
        n_queries=config.n_queries,
        max_cardinality=config.max_cardinality,
        n_queries_by_level=config.n_queries_by_level,
        max_cardinality_by_level=config.max_cardinality_by_level,
        tangent_variance_target=config.tangent_variance_target,
        hyper_projection_init_scale=config.hyper_projection_init_scale,
        tangent_scale_mode=config.tangent_scale_mode,
        hyperbolic_level_encoding=config.hyperbolic_level_encoding,
        type_conditioned_daughter_relation_bias=(
            config.type_conditioned_daughter_relation_bias
            or ALL_ABLATIONS[config.ablation].type_conditioned_daughter_relation_bias
        ),
    ).to_dict()


__all__ = [
    "ReconstructionConfig",
    "ReconstructionTrainingResult",
    "train_level_reconstruction",
    "validate_reconstruction",
]
