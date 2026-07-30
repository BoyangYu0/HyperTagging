"""All-level real-parquet reconstruction training and rollout validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hypertagging.data.capacity import dataset_capacity_statistics, require_capacity
from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.evaluation.hierarchical_metrics import next_level_metrics, summarize_rollout
from hypertagging.losses.level_reconstruction import (
    confidence_calibration_metrics,
    level_reconstruction_loss,
)
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import RolloutConfig, level_rollout
from hypertagging.training.checkpointing import (
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from hypertagging.training.data_module import RealDataModule, build_real_data_module
from hypertagging.training.logging import JsonlLogger
from hypertagging.training.pretrained_transfer import (
    EncoderTransferReport,
    load_pretrained_encoder,
    optimizer_parameter_groups,
    unfreeze_encoder,
)
from hypertagging.utils.seeds import seed_everything
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.training.scheduled_sampling import (
    TeacherForcingSchedule,
    aligned_level_targets,
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
    encoder_lr_multiplier: float = 0.2
    freeze_pretrained_encoder_steps: int = 0
    gradient_clip: float = 1.0
    checkpoint_every: int = 100
    resume: str | None = None
    n_queries: int = 8
    max_cardinality: int = 6
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
) -> ReconstructionTrainingResult:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= config.scheduled_sampling_probability <= 1.0:
        raise ValueError("scheduled_sampling_probability must lie in [0, 1]")
    seed_everything(config.seed)
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
    )
    capacity = dataset_capacity_statistics(
        data_module.iter_events("train", shuffle=False),
        global_n_queries=config.n_queries,
        global_max_cardinality=config.max_cardinality,
        target_policy=config.target_policy,
    )
    require_capacity(capacity)
    if config.ablation not in ALL_ABLATIONS:
        raise ValueError(f"unknown ablation: {config.ablation}")
    ablation = ALL_ABLATIONS[config.ablation]
    model = build_ablation_model(
        config.ablation,
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=32,
        hyper_dim=8,
        n_queries=config.n_queries,
    ).to(device)
    # Cardinality capacity is a scientific data contract, not a decode clamp.
    if model.decoder.max_cardinality != config.max_cardinality:
        model.decoder = type(model.decoder)(
            hidden_dim=32,
            n_types=len(PDG_TOKENS),
            n_queries=config.n_queries,
            max_cardinality=config.max_cardinality,
        ).to(device)
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
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            base_lr=config.learning_rate,
            encoder_lr_multiplier=config.encoder_lr_multiplier,
            leaf_pid_lr_multiplier=config.leaf_pid_lr_multiplier,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.max_steps, 1),
    )
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
        )
        start_step = int(payload.get("step", 0))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(data_module.split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    batch_iterator = data_module.batches(
        "train", batch_size=config.batch_size, shuffle=True, epoch=0
    )
    epoch = 0
    schedule = TeacherForcingSchedule(
        kind=config.scheduled_sampling_schedule,
        start_probability=1.0,
        end_probability=1.0 - config.scheduled_sampling_probability,
        duration_steps=config.scheduled_sampling_duration_steps,
    )
    final_loss = 0.0
    final_metrics: dict[str, float] = {}
    for step in range(start_step, config.max_steps):
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
            batch_iterator = data_module.batches(
                "train", batch_size=config.batch_size, shuffle=True, epoch=epoch
            )
            try:
                next_batch = next(batch_iterator)
            except StopIteration as error:
                raise ValueError("training split produced no batches") from error
        batch = {name: value.to(device) for name, value in next_batch.items()}
        valid_levels = sorted(
            {
                int(level)
                for level in batch["level_ids"][batch["node_mask"]].tolist()
                if int(level) > 0
            }
        )
        if not valid_levels:
            raise ValueError("training batch has no reconstruction target levels")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config.mixed_precision,
        ):
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
            )
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
            "leaf_pid_loss": float(leaf_pid_loss.detach().cpu()),
            "levels_trained": float(len(valid_levels)),
            **context_metrics,
            "teacher_forcing_probability": schedule.probability(step),
        }
        for name, value in component_accumulator.items():
            final_metrics[f"loss_{name}"] = float(
                (value / len(valid_levels)).detach().cpu()
            )
        last_level, last_output, last_loss = level_outputs[-1]
        final_metrics.update(
            next_level_metrics(
                last_output.pointer,
                batch,
                last_loss.matches,
                target_level=last_level,
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
            )
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
    )
    final_metrics.update(validation_metrics)
    checkpoint = _save_reconstruction_checkpoint(
        output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        data_module=data_module,
        step=config.max_steps,
        metrics=final_metrics,
    )
    return ReconstructionTrainingResult(
        checkpoint=checkpoint,
        log_path=logger.path,
        steps=config.max_steps,
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
):
    """Batched truth loss plus deterministic per-event predicted micro-rollouts."""

    level_outputs = []
    component_accumulator: dict[str, torch.Tensor] = {}
    leaf_pid_losses = []
    scheduled_losses = []
    truth_contexts = 0
    predicted_contexts = 0
    representable = 0
    truth_targets = 0
    first_divergence: list[int] = []
    teacher_probability = schedule.probability(step) if use_scheduled_sampling else 1.0
    for target_level in valid_levels:
        teacher_batch = _with_allowed_types(
            batch, target_level, allowed_types_by_level
        )
        output = model(teacher_batch, target_level=target_level)
        loss_batch = dict(teacher_batch)
        if output.current_p4 is not None:
            loss_batch["p4"] = output.current_p4
        loss_output = level_reconstruction_loss(
            output.pointer,
            loss_batch,
            target_level=target_level,
            target_policy=config.target_policy,
            matching_production=not config.allow_tiny_bruteforce_matching,
        )
        level_outputs.append((target_level, output, loss_output))
        for name, value in loss_output.components.items():
            component_accumulator[name] = component_accumulator.get(name, 0) + value
        raw_tracks = (
            batch["node_mask"]
            & (
                batch["leaf_kinematics_mode_ids"]
                == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
            )
            & batch["truth_pid_available"]
        )
        if output.leaf_pid_logits is not None and raw_tracks.any():
            leaf_pid_losses.append(
                F.cross_entropy(
                    output.leaf_pid_logits[raw_tracks],
                    batch["truth_pid_labels"][raw_tracks],
                )
            )
        choices = schedule.sample(
            batch["node_mask"].shape[0],
            step=step * 100 + target_level,
            seed=config.seed,
            device=batch["node_mask"].device,
        )
        if not use_scheduled_sampling:
            choices.fill_(True)
        truth_contexts += int(choices.sum())
        for batch_index in (~choices).nonzero(as_tuple=False).flatten().tolist():
            predicted_contexts += 1
            truth_single = _single_event_batch(batch, batch_index)
            with torch.no_grad():
                rollout = level_rollout(
                    model,
                    truth_single,
                    mode="predicted",
                    config=RolloutConfig(
                        max_level=max(target_level - 1, 0),
                        root_types=(),
                        exclusive_final=False,
                        use_learned_confidence=False,
                        seed=config.seed + step + batch_index,
                    ),
                )
                context = rollout.batch
            aligned = aligned_level_targets(
                truth_single,
                context,
                target_level=target_level,
                target_policy=config.target_policy,
            )
            truth_targets += aligned.truth_target_count
            representable += aligned.representable_count
            if aligned.representable_count < aligned.truth_target_count:
                first_divergence.append(target_level)
            context = _with_allowed_types(
                context, target_level, allowed_types_by_level
            )
            predicted_output = model(context, target_level=target_level)
            predicted_loss_batch = dict(context)
            if predicted_output.current_p4 is not None:
                predicted_loss_batch["p4"] = predicted_output.current_p4
            predicted_loss = level_reconstruction_loss(
                predicted_output.pointer,
                predicted_loss_batch,
                target_level=target_level,
                target_policy=config.target_policy,
                target_override=aligned.target_override,
                matching_production=not config.allow_tiny_bruteforce_matching,
            )
            scheduled_losses.append(predicted_loss.total)
    teacher_loss = torch.stack([value[2].total for value in level_outputs]).mean()
    scheduled_loss = (
        torch.stack(scheduled_losses).mean()
        if scheduled_losses
        else teacher_loss * 0.0
    )
    reconstruction_loss = (
        0.5 * (teacher_loss + scheduled_loss)
        if scheduled_losses
        else teacher_loss
    )
    leaf_pid_loss = (
        torch.stack(leaf_pid_losses).mean()
        if leaf_pid_losses
        else reconstruction_loss * 0.0
    )
    total_contexts = truth_contexts + predicted_contexts
    metrics = {
        "context_truth_fraction": truth_contexts / max(total_contexts, 1),
        "context_predicted_fraction": predicted_contexts / max(total_contexts, 1),
        "target_representable_rate": representable / max(truth_targets, 1),
        "first_context_divergence_level": float(
            min(first_divergence) if first_divergence else -1
        ),
        "scheduled_sampling_loss": float(scheduled_loss.detach().cpu()),
    }
    return (
        reconstruction_loss,
        leaf_pid_loss,
        component_accumulator,
        level_outputs,
        metrics,
    )


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


def _with_allowed_types(
    batch: dict[str, torch.Tensor],
    target_level: int,
    allowed_types_by_level: dict[int, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    allowed = torch.zeros(
        len(PDG_TOKENS), dtype=torch.bool, device=batch["node_mask"].device
    )
    tokens = allowed_types_by_level.get(target_level, ())
    if tokens:
        allowed[list(tokens)] = True
    else:
        allowed[:] = True
    result["allowed_type_mask"] = allowed
    result["pointer_validity_mask"] = (
        batch["node_mask"] & (batch["level_ids"] < target_level)
    )
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
) -> dict[str, float]:
    model.eval()
    source = data_module.iter_events("validation", shuffle=False)
    if data_module.split_counts.get("validation", 0) == 0:
        source = data_module.iter_events("train", shuffle=False)
    accumulated: dict[str, list[float]] = {}
    event_count = 0
    for event in source:
        if event_count >= min(max_validation_events, rollout_validation_events):
            break
        batch = data_module.normalize_batch(collate_heterogeneous_events([event]))
        batch = {name: value.to(device) for name, value in batch.items()}
        teacher = level_rollout(
            model,
            batch,
            mode="teacher_forced",
            config=RolloutConfig(max_level=8, root_types=(), exclusive_final=False),
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
            ),
        )
        scheduled = level_rollout(
            model,
            batch,
            mode="scheduled",
            config=RolloutConfig(
                max_level=8,
                root_types=(),
                scheduled_sampling_probability=scheduled_sampling_probability,
                seed=seed + event_count,
            ),
        )
        teacher_metrics = summarize_rollout(teacher.batch, batch)
        predicted_metrics = summarize_rollout(predicted.batch, batch)
        values = {
            "teacher_forced_p4_closure_rate": float(
                teacher_metrics["p4_closure_rate"]
            ),
            "predicted_p4_closure_rate": float(predicted_metrics["p4_closure_rate"]),
            "predicted_tree_validity_rate": float(
                predicted_metrics["tree_validity_rate"]
            ),
            "scheduled_rollout_valid": float(scheduled.valid),
        }
        for name, value in values.items():
            accumulated.setdefault(name, []).append(value)
        event_count += 1
    model.train()
    return {
        name: sum(values) / len(values)
        for name, values in accumulated.items()
    } | {"validation_events": float(event_count)}


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
        },
        legacy_conflated_fraction=data_module.legacy_conflated_fraction,
        schema_version=(
            data_module.source_schema_versions[0]
            if len(data_module.source_schema_versions) == 1
            else "mixed"
        ),
    )


__all__ = [
    "ReconstructionConfig",
    "ReconstructionTrainingResult",
    "train_level_reconstruction",
    "validate_reconstruction",
]
