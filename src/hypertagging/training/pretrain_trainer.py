"""Checkpointed real-parquet contextual hyperbolic pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.data.level_collate import build_lca_depth
from hypertagging.losses.hyperbolic_pretraining import (
    build_tree_relation_targets,
    hyperbolic_pretraining_loss,
    pool_b_branch_embeddings,
)
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.ablation import ABLATIONS
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)
from hypertagging.training.data_module import RealDataModule, build_real_data_module
from hypertagging.training.hyperbolic_pretrain import TreeRelationHead
from hypertagging.training.logging import JsonlLogger
from hypertagging.training.pretraining_curriculum import (
    PretrainingStage,
    build_curriculum_batch,
)
from hypertagging.training.validation import validate_contextual_geometry
from hypertagging.utils.seeds import seed_everything


@dataclass(frozen=True)
class PretrainConfig:
    data: str
    output_dir: str
    device: str = "cpu"
    max_steps: int = 2
    batch_size: int = 2
    max_events: int | None = None
    seed: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    checkpoint_every: int = 100
    validate_every: int = 100
    resume: str | None = None
    curriculum: tuple[str, ...] = (
        PretrainingStage.FSP_ONLY.value,
        PretrainingStage.TRUTH_GUIDED_MULTILEVEL.value,
        PretrainingStage.CORRUPTED_COMPOSITES.value,
    )
    curvature: float = 1.0
    mixed_precision: bool = True
    ablation: str = "full_revised"
    channel_memory_size: int = 0


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    log_path: Path
    steps: int
    final_loss: float
    metrics: dict[str, float]
    data_module: RealDataModule


class ContextualPretrainingModel(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 32,
        hyper_dim: int = 8,
        curvature: float = 1.0,
        use_contextual_encoder: bool = True,
        use_physical_relations: bool = True,
        use_hyperbolic_relations: bool = True,
        channel_memory_size: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = HeterogeneousNodeEncoder(
            d_model=d_model,
            hyper_dim=hyper_dim,
            curvature=curvature,
            use_contextual_encoder=use_contextual_encoder,
            use_physical_context=use_physical_relations,
            use_hyperbolic_refinement=use_hyperbolic_relations,
        )
        self.relation_head = TreeRelationHead(d_model)
        self.leaf_pid_head = torch.nn.Linear(d_model, len(PDG_TOKENS))
        self.channel_memory = ChannelMemoryBank(channel_memory_size, d_model)


class ChannelMemoryBank(torch.nn.Module):
    """Optional detached FIFO of cross-event B-branch channel examples."""

    def __init__(self, capacity: int, embedding_dim: int) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 0)
        self.register_buffer("embeddings", torch.zeros(self.capacity, embedding_dim))
        self.register_buffer("full_ids", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer(
            "reconstructable_ids", torch.zeros(self.capacity, dtype=torch.long)
        )
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def contents(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(self.count)
        return (
            self.embeddings[:count],
            self.full_ids[:count],
            self.reconstructable_ids[:count],
        )

    @torch.no_grad()
    def enqueue(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        full_ids: torch.Tensor,
        reconstructable_ids: torch.Tensor,
    ) -> None:
        if self.capacity == 0:
            return
        selected = mask.reshape(-1)
        additions = embeddings.reshape(-1, embeddings.shape[-1])[selected].detach()
        full = full_ids.reshape(-1)[selected].detach()
        reco = reconstructable_ids.reshape(-1)[selected].detach()
        old_embeddings, old_full, old_reco = self.contents()
        new_embeddings = torch.cat([old_embeddings, additions], dim=0)[-self.capacity :]
        new_full = torch.cat([old_full, full], dim=0)[-self.capacity :]
        new_reco = torch.cat([old_reco, reco], dim=0)[-self.capacity :]
        count = new_embeddings.shape[0]
        self.embeddings.zero_()
        self.full_ids.zero_()
        self.reconstructable_ids.zero_()
        self.embeddings[:count].copy_(new_embeddings)
        self.full_ids[:count].copy_(new_full)
        self.reconstructable_ids[:count].copy_(new_reco)
        self.count.fill_(count)


def train_hyperbolic_pretraining(config: PretrainConfig) -> TrainingResult:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    seed_everything(config.seed)
    device = torch.device(config.device)
    data_module = build_real_data_module(
        config.data,
        max_events=config.max_events,
        seed=config.seed,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    if config.ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {config.ablation}")
    ablation = ABLATIONS[config.ablation]
    model = ContextualPretrainingModel(
        curvature=config.curvature,
        use_contextual_encoder=ablation.contextual_euclidean,
        use_physical_relations=ablation.relation_attention,
        use_hyperbolic_relations=ablation.hyperbolic_relation_attention,
        channel_memory_size=config.channel_memory_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
    batches = list(
        data_module.batches("train", batch_size=config.batch_size, shuffle=True)
    )
    if not batches:
        raise ValueError("training split produced no batches")
    final_loss = 0.0
    final_metrics: dict[str, float] = {}
    for step in range(start_step, config.max_steps):
        batch = _to_device(batches[step % len(batches)], device)
        _add_topology_labels(batch)
        stage = PretrainingStage(config.curriculum[step % len(config.curriculum)])
        curriculum = build_curriculum_batch(batch, stage, seed=config.seed + step)
        train_batch = curriculum.batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config.mixed_precision,
        ):
            encoded = model.encoder(train_batch)
            relation_logits = model.relation_head(encoded.tree_projection)
            targets, relation_mask = build_tree_relation_targets(
                parent_ids=train_batch["parent_ids"],
                lca_depth=train_batch["lca_depth"],
                level_ids=train_batch["level_ids"],
                node_mask=train_batch["node_mask"],
                b_side=train_batch["b_side"],
            )
            branch_embeddings, branch_mask = pool_b_branch_embeddings(
                encoded.channel_projection,
                train_batch["b_side"],
                train_batch["node_mask"],
            )
            memory_embeddings, memory_full_ids, memory_reco_ids = (
                model.channel_memory.contents()
            )
            loss_output = hyperbolic_pretraining_loss(
                z=encoded.hyperbolic_embeddings,
                tree_relation_logits=relation_logits,
                tree_relation_targets=targets,
                tree_relation_mask=relation_mask,
                lca_depth=train_batch["lca_depth"],
                parent_ids=train_batch["parent_ids"],
                level_ids=train_batch["level_ids"],
                node_mask=train_batch["node_mask"],
                b_side=train_batch["b_side"],
                node_kind_ids=train_batch["node_kind_ids"],
                event_ids=train_batch["event_ids"],
                channel_embeddings=branch_embeddings,
                channel_mask=branch_mask,
                full_truth_channel_ids=torch.stack(
                    [
                        train_batch["b1_full_truth_channel_ids"],
                        train_batch["b2_full_truth_channel_ids"],
                    ],
                    dim=-1,
                ),
                reconstructable_channel_ids=torch.stack(
                    [
                        train_batch["b1_reconstructable_channel_ids"],
                        train_batch["b2_reconstructable_channel_ids"],
                    ],
                    dim=-1,
                ),
                channel_branch_count_arrays=train_batch["b_channel_count_arrays"],
                channel_memory_embeddings=memory_embeddings,
                channel_memory_full_truth_ids=memory_full_ids,
                channel_memory_reconstructable_ids=memory_reco_ids,
                weights=_pretraining_weights(config.ablation),
                curvature=config.curvature,
            )
            leaf_pid_loss = (
                _leaf_pid_loss(model, encoded.node_embeddings, train_batch)
                if ablation.leaf_pid
                else encoded.node_embeddings.sum() * 0.0
            )
            loss = loss_output.total + leaf_pid_loss
        model.channel_memory.enqueue(
            branch_embeddings,
            branch_mask,
            torch.stack(
                [
                    train_batch["b1_full_truth_channel_ids"],
                    train_batch["b2_full_truth_channel_ids"],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    train_batch["b1_reconstructable_channel_ids"],
                    train_batch["b2_reconstructable_channel_ids"],
                ],
                dim=-1,
            ),
        )
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
            **{
                f"loss_{name}": float(value.detach().cpu())
                for name, value in loss_output.components.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in loss_output.diagnostics.items()
            },
        }
        logger.log(step=step + 1, stage=stage.value, **final_metrics)
        if (step + 1) % config.checkpoint_every == 0:
            _save_pretrain_checkpoint(
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
    validation_events = data_module.splits["validation"] or data_module.splits["train"][:1]
    if validation_events:
        validation_batch = data_module.normalize_batch(
            collate_heterogeneous_events(validation_events[: config.batch_size])
        )
        validation_metrics = validate_contextual_geometry(
            model.encoder,
            validation_batch,
            device=device,
            curvature=config.curvature,
        )
        final_metrics.update(validation_metrics)
        logger.log(step=config.max_steps, split="validation", **validation_metrics)
    checkpoint = _save_pretrain_checkpoint(
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
    return TrainingResult(
        checkpoint=checkpoint,
        log_path=logger.path,
        steps=config.max_steps,
        final_loss=final_loss,
        metrics=final_metrics,
        data_module=data_module,
    )


def _pretraining_weights(ablation_name: str) -> dict[str, float]:
    config = ABLATIONS[ablation_name]
    return {
        "lca": float(config.lca_parent),
        "parent": float(config.lca_parent),
        "tree_distance": float(config.lca_parent),
        "depth": 0.2 * float(config.radius_depth),
        "channel": 0.2 * float(config.channel_supervision),
        "var": 0.1 * float(config.variance_covariance),
        "cov": 0.01 * float(config.variance_covariance),
    }


def _leaf_pid_loss(
    model: ContextualPretrainingModel,
    embeddings: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    raw_tracks = (
        batch["node_mask"]
        & (batch["node_kind_ids"] == 1)
        & batch["truth_pid_available"]
    )
    if not raw_tracks.any():
        return embeddings.sum() * 0.0
    return F.cross_entropy(
        model.leaf_pid_head(embeddings)[raw_tracks],
        batch["truth_pid_labels"][raw_tracks],
    )


def _add_topology_labels(batch: dict[str, torch.Tensor]) -> None:
    lca = torch.full(
        (*batch["parent_ids"].shape, batch["parent_ids"].shape[1]),
        -1,
        dtype=torch.long,
        device=batch["parent_ids"].device,
    )
    for index in range(batch["parent_ids"].shape[0]):
        count = int(batch["node_mask"][index].sum())
        lca[index, :count, :count] = build_lca_depth(
            batch["parent_ids"][index, :count].cpu(),
            batch["level_ids"][index, :count].cpu(),
        ).to(lca.device)
    batch["lca_depth"] = lca


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _save_pretrain_checkpoint(
    path: Path,
    *,
    model: ContextualPretrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: PretrainConfig,
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
    )


__all__ = [
    "ContextualPretrainingModel",
    "PretrainConfig",
    "TrainingResult",
    "train_hyperbolic_pretraining",
]
