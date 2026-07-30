"""All-level real-parquet reconstruction training and rollout validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from hypertagging.models.ablation import ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.kinematics import (
    soft_reconstructed_p4_from_leaf_pid,
)
from hypertagging.reconstruction.level_rollout import RolloutConfig, level_rollout
from hypertagging.training.checkpointing import (
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
    data_module = build_real_data_module(
        config.data,
        max_events=config.max_events,
        seed=config.seed,
    )
    capacity = dataset_capacity_statistics(
        data_module.events,
        global_n_queries=config.n_queries,
        global_max_cardinality=config.max_cardinality,
    )
    require_capacity(capacity)
    if config.ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {config.ablation}")
    ablation = ABLATIONS[config.ablation]
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
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            base_lr=config.learning_rate,
            encoder_lr_multiplier=config.encoder_lr_multiplier,
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
            map_location=device,
        )
        start_step = int(payload.get("step", 0))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    batches = list(data_module.batches("train", batch_size=config.batch_size, shuffle=True))
    if not batches:
        raise ValueError("training split produced no batches")
    final_loss = 0.0
    final_metrics: dict[str, float] = {}
    for step in range(start_step, config.max_steps):
        if (
            config.freeze_pretrained_encoder_steps > 0
            and step == config.freeze_pretrained_encoder_steps
        ):
            unfreeze_encoder(model.encoder)
        batch = {
            name: value.to(device)
            for name, value in batches[step % len(batches)].items()
        }
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
        level_outputs = []
        component_accumulator: dict[str, torch.Tensor] = {}
        leaf_pid_losses = []
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config.mixed_precision,
        ):
            for target_level in valid_levels:
                output = model(batch, target_level=target_level)
                loss_batch = dict(batch)
                if ablation.leaf_pid and output.leaf_pid_logits is not None:
                    loss_batch["p4"] = soft_reconstructed_p4_from_leaf_pid(
                        batch,
                        output.leaf_pid_logits,
                    )
                loss_output = level_reconstruction_loss(
                    output.pointer,
                    loss_batch,
                    target_level=target_level,
                    matching_production=not config.allow_tiny_bruteforce_matching,
                )
                level_outputs.append((target_level, output, loss_output))
                for name, value in loss_output.components.items():
                    component_accumulator[name] = component_accumulator.get(name, 0) + value
                raw_tracks = (
                    batch["node_mask"]
                    & (batch["node_kind_ids"] == 1)
                    & batch["truth_pid_available"]
                )
                if (
                    ablation.leaf_pid
                    and output.leaf_pid_logits is not None
                    and raw_tracks.any()
                ):
                    leaf_pid_losses.append(
                        F.cross_entropy(
                            output.leaf_pid_logits[raw_tracks],
                            batch["truth_pid_labels"][raw_tracks],
                        )
                    )
            reconstruction_loss = torch.stack(
                [item[2].total for item in level_outputs]
            ).mean()
            leaf_pid_loss = (
                torch.stack(leaf_pid_losses).mean()
                if leaf_pid_losses
                else reconstruction_loss * 0.0
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


@torch.no_grad()
def validate_reconstruction(
    model: LevelAutoregressiveReconstructor,
    data_module: RealDataModule,
    *,
    device: torch.device,
    scheduled_sampling_probability: float = 0.5,
    seed: int = 11,
) -> dict[str, float]:
    events = data_module.splits["validation"] or data_module.splits["train"][:1]
    if not events:
        return {}
    batch = data_module.normalize_batch(
        collate_heterogeneous_events(events[:1])
    )
    batch = {name: value.to(device) for name, value in batch.items()}
    model.eval()
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
            seed=seed,
        ),
    )
    teacher_metrics = summarize_rollout(teacher.batch, batch)
    predicted_metrics = summarize_rollout(predicted.batch, batch)
    model.train()
    return {
        "teacher_forced_p4_closure_rate": float(teacher_metrics["p4_closure_rate"]),
        "predicted_p4_closure_rate": float(predicted_metrics["p4_closure_rate"]),
        "predicted_tree_validity_rate": float(predicted_metrics["tree_validity_rate"]),
        "scheduled_rollout_valid": float(scheduled.valid),
    }


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
    )


__all__ = [
    "ReconstructionConfig",
    "ReconstructionTrainingResult",
    "train_level_reconstruction",
    "validate_reconstruction",
]
