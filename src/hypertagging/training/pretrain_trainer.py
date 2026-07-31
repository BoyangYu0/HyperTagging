"""Checkpointed real-parquet contextual hyperbolic pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.data.level_collate import build_lca_depth
from hypertagging.losses.hyperbolic_pretraining import (
    build_tree_relation_targets,
    hyperbolic_pretraining_loss,
    parent_child_ranking_accuracy,
    pool_b_branch_embeddings,
)
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID, feature_spec_v4
from hypertagging.training.checkpointing import (
    load_training_checkpoint,
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
from hypertagging.data.streaming import RuntimeFeatureNormalizer, StreamingCursor
from hypertagging.utils.seeds import seed_everything
from hypertagging.reconstruction.pid_state import rebuild_runtime_pid_state
from hypertagging.models.level_autoregressive import _runtime_reconstruction_batch
from hypertagging.training.model_config import resolve_model_architecture


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
    num_workers: int = 0
    prefetch_factor: int = 2
    shuffle_buffer_size: int = 1024
    persistent_workers: bool = False
    pilot_split_repair: bool = False
    allow_legacy_conflated: bool = False
    log_every: int = 10
    dataset_index: str | None = None
    rescan_dataset: bool = False
    corruption_objective: str = "invalid_candidate"
    model_preset: str = "tiny_cpu"
    d_model: int | None = None
    hyper_dim: int | None = None
    n_heads: int | None = None
    n_context_layers: int | None = None
    ffn_dim: int | None = None
    dropout: float | None = None
    n_queries: int | None = None
    n_queries_by_level: tuple[tuple[int, int], ...] = ()
    max_cardinality: int | None = None
    max_cardinality_by_level: tuple[tuple[int, int], ...] = ()
    validation_batches: int = 4
    channel_pooling: str = "mean_all"


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
        n_heads: int = 4,
        n_context_layers: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        channel_pooling: str = "mean_all",
    ) -> None:
        super().__init__()
        self.encoder = HeterogeneousNodeEncoder(
            d_model=d_model,
            hyper_dim=hyper_dim,
            curvature=curvature,
            n_heads=n_heads,
            n_context_layers=n_context_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_contextual_encoder=use_contextual_encoder,
            use_physical_context=use_physical_relations,
            use_hyperbolic_refinement=use_hyperbolic_relations,
        )
        self.relation_head = TreeRelationHead(d_model)
        self.leaf_pid_head = torch.nn.Linear(d_model, len(PDG_TOKENS))
        self.candidate_correctness_head = torch.nn.Linear(d_model, 1)
        self.corruption_type_head = torch.nn.Linear(d_model, 5)
        self.channel_memory = ChannelMemoryBank(channel_memory_size, d_model)
        if channel_pooling not in {
            "mean_all", "b_root", "learned_attention", "level_weighted"
        }:
            raise ValueError(f"unknown channel pooling mode: {channel_pooling}")
        self.channel_pooling = channel_pooling
        self.channel_pool_score = (
            torch.nn.Linear(d_model, 1)
            if channel_pooling == "learned_attention" else None
        )
        self.runtime_feature_normalizer = RuntimeFeatureNormalizer.identity(12, 13)

    def set_runtime_feature_normalizer(
        self, normalizer: RuntimeFeatureNormalizer
    ) -> None:
        self.runtime_feature_normalizer = normalizer

    def normalize_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        common, common_mask, composite, composite_mask = (
            self.runtime_feature_normalizer.normalize_runtime(
                batch["common_features"],
                batch["common_availability"],
                batch["composite_features"],
                batch["composite_availability"],
            )
        )
        result = dict(batch)
        result["common_features"] = common
        result["common_availability"] = common_mask
        result["composite_features"] = composite
        result["composite_availability"] = composite_mask
        result["node_features"] = common
        return result

    def encode_runtime(
        self,
        batch: dict[str, torch.Tensor],
        *,
        attention_mask: torch.Tensor,
    ):
        """Run the same detector-PID then PID-conditioned context as reconstruction."""

        first_batch = self.normalize_batch(batch)
        first = self.encoder(first_batch, attention_mask=attention_mask)
        leaf_pid_logits = self.leaf_pid_head(first.reconstruction_projection)
        runtime = rebuild_runtime_pid_state(batch, leaf_pid_logits, hard=False)
        second_batch = _runtime_reconstruction_batch(
            batch,
            runtime,
            normalizer=self.runtime_feature_normalizer,
            canonical_batch=first_batch,
            use_canonical=False,
        )
        second_batch["curriculum_attention_mask"] = attention_mask
        second = self.encoder(second_batch, attention_mask=attention_mask)
        return second, leaf_pid_logits, second_batch


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
    if config.validate_every <= 0 or config.validation_batches <= 0:
        raise ValueError("validate_every and validation_batches must be positive")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    seed_everything(config.seed)
    if config.resume and config.num_workers > 0:
        raise ValueError("exact streaming resume currently requires num_workers=0")
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
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    if config.ablation not in ALL_ABLATIONS:
        raise ValueError(f"unknown ablation: {config.ablation}")
    ablation = ALL_ABLATIONS[config.ablation]
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
        n_queries_by_level=config.n_queries_by_level,
        max_cardinality=config.max_cardinality,
        max_cardinality_by_level=config.max_cardinality_by_level,
    )
    model = ContextualPretrainingModel(
        d_model=architecture.d_model,
        hyper_dim=architecture.hyper_dim,
        curvature=architecture.curvature,
        n_heads=architecture.n_heads,
        n_context_layers=architecture.n_context_layers,
        ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        use_contextual_encoder=ablation.contextual_euclidean,
        use_physical_relations=ablation.relation_attention,
        use_hyperbolic_relations=ablation.hyperbolic_relation_attention,
        channel_memory_size=config.channel_memory_size,
        channel_pooling=config.channel_pooling,
    ).to(device)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=data_module.normalizers["common"].mean,
            common_std=data_module.normalizers["common"].std,
            composite_mean=data_module.normalizers["composite"].mean,
            composite_std=data_module.normalizers["composite"].std,
        ).to(device)
    )
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
            scaler=scaler,
            map_location=device,
            expected_schema_version=(
                data_module.source_schema_versions[0]
                if len(data_module.source_schema_versions) == 1
                else None
            ),
            expected_split_manifest_hash=data_module.split_manifest_hash,
            expected_feature_spec_hash=feature_spec_v4()["feature_spec_hash"],
            expected_data_order_contract=_pretrain_data_order_contract(
                config, data_module
            ),
            expected_architecture=architecture.to_dict(),
        )
        start_step = int(payload.get("step", 0))
    (output_dir / "split_manifest.json").write_text(
        json.dumps(data_module.split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    final_loss = 0.0
    final_metrics: dict[str, float] = {}
    best_validation_loss = float("inf")
    last_validation_step = 0
    for step in range(start_step, config.max_steps):
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
        batch = _to_device(next_batch, device)
        _add_topology_labels(batch)
        stage = PretrainingStage(config.curriculum[step % len(config.curriculum)])
        curriculum = build_curriculum_batch(
            batch, stage, seed=config.seed + step,
            corruption_objective=config.corruption_objective,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config.mixed_precision,
        ):
            encoded, leaf_pid_logits, train_batch = model.encode_runtime(
                curriculum.batch,
                attention_mask=curriculum.batch["curriculum_attention_mask"],
            )
            structural_mask = curriculum.structural_positive_mask
            relation_logits = model.relation_head(encoded.tree_projection)
            targets, relation_mask = build_tree_relation_targets(
                parent_ids=train_batch["parent_ids"],
                lca_depth=train_batch["lca_depth"],
                level_ids=train_batch["level_ids"],
                node_mask=structural_mask,
                b_side=train_batch["b_side"],
            )
            branch_embeddings, branch_mask = pool_b_branch_embeddings(
                encoded.channel_projection,
                train_batch["b_side"],
                train_batch["node_mask"],
                mode=model.channel_pooling,
                level_ids=train_batch["level_ids"],
                attention_logits=(
                    model.channel_pool_score(encoded.channel_projection).squeeze(-1)
                    if model.channel_pool_score is not None else None
                ),
            )
            valid_channel_event = valid_b_root_channel_mask(
                train_batch,
                corrupted_node_mask=curriculum.corrupted_node_mask,
                corruption_objective=config.corruption_objective,
            )
            branch_mask = branch_mask & valid_channel_event[:, None]
            memory_embeddings, memory_full_ids, memory_reco_ids = (
                model.channel_memory.contents()
            )
            channel_structural_features = torch.cat(
                [
                    train_batch["b_channel_count_arrays"],
                    train_batch["b_depth_pid_count_arrays"].flatten(start_dim=2),
                    train_batch["b_branch_multiplicity_summaries"],
                    train_batch["b_intermediate_count_arrays"],
                ],
                dim=-1,
            )
            loss_output = hyperbolic_pretraining_loss(
                z=encoded.hyperbolic_embeddings,
                tree_relation_logits=relation_logits,
                tree_relation_targets=targets,
                tree_relation_mask=relation_mask,
                lca_depth=train_batch["lca_depth"],
                parent_ids=train_batch["parent_ids"],
                level_ids=train_batch["level_ids"],
                node_mask=structural_mask,
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
                channel_branch_count_arrays=channel_structural_features,
                channel_memory_embeddings=memory_embeddings,
                channel_memory_full_truth_ids=memory_full_ids,
                channel_memory_reconstructable_ids=memory_reco_ids,
                weights=_pretraining_weights(config.ablation),
                curvature=config.curvature,
                full_event_max_level=train_batch.get("full_event_max_level"),
            )
            leaf_pid_loss = (
                _leaf_pid_loss(leaf_pid_logits, train_batch)
                if ablation.leaf_pid
                else encoded.node_embeddings.sum() * 0.0
            )
            corruption_logits = model.corruption_type_head(encoded.node_embeddings)
            correctness_logits = model.candidate_correctness_head(
                encoded.node_embeddings
            ).squeeze(-1)
            corruption_nodes = (
                train_batch["node_mask"] & (train_batch["level_ids"] > 0)
            )
            corruption_loss = (
                F.cross_entropy(
                    corruption_logits[corruption_nodes],
                    curriculum.corruption_code[corruption_nodes],
                )
                if corruption_nodes.any()
                else encoded.node_embeddings.sum() * 0.0
            )
            correctness_target = (~curriculum.corrupted_node_mask).float()
            correctness_loss = (
                F.binary_cross_entropy_with_logits(
                    correctness_logits[corruption_nodes],
                    correctness_target[corruption_nodes],
                )
                if corruption_nodes.any()
                else encoded.node_embeddings.sum() * 0.0
            )
            hard_negative_loss = _hard_negative_tree_loss(
                encoded.hyperbolic_embeddings,
                curriculum.hard_negative_pairs,
                curvature=config.curvature,
            )
            loss = (
                loss_output.total
                + leaf_pid_loss
                + 0.1 * corruption_loss
                + 0.1 * correctness_loss
                + 0.1 * hard_negative_loss
            )
        enqueue_mask = (
            branch_mask
            & train_batch.get(
                "b_root_discovery_valid", torch.ones_like(branch_mask[:, 0])
            )[:, None]
        )
        if stage is PretrainingStage.CORRUPTED_COMPOSITES:
            enqueue_mask = torch.zeros_like(enqueue_mask)
        model.channel_memory.enqueue(
            branch_embeddings,
            enqueue_mask,
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
            "corruption_loss": float(corruption_loss.detach().cpu()),
            "candidate_correctness_loss": float(correctness_loss.detach().cpu()),
            "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
            "hard_negative_count": float(curriculum.hard_negative_pairs.shape[0]),
            **{
                f"loss_{name}": float(value.detach().cpu())
                for name, value in loss_output.components.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in loss_output.diagnostics.items()
            },
        }
        if (step + 1) % config.log_every == 0 or step == start_step or step + 1 == config.max_steps:
            logger.log(step=step + 1, stage=stage.value, **final_metrics)
        if (step + 1) % config.validate_every == 0:
            validation_metrics = _validate_pretraining(
                model,
                data_module,
                device=device,
                config=config,
            )
            final_metrics.update(validation_metrics)
            logger.log(step=step + 1, split="validation", **validation_metrics)
            last_validation_step = step + 1
            _save_pretrain_checkpoint(
                output_dir / "latest.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, data_module=data_module, step=step + 1,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
            )
            validation_loss = validation_metrics.get("validation_loss_total", float("inf"))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                _save_pretrain_checkpoint(
                    output_dir / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, data_module=data_module, step=step + 1,
                    metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                )
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
                streaming_cursor=cursor.state_dict(),
            )
    if last_validation_step != config.max_steps:
        validation_metrics = _validate_pretraining(
            model, data_module, device=device, config=config
        )
        final_metrics.update(validation_metrics)
        logger.log(step=config.max_steps, split="validation", **validation_metrics)
        validation_loss = validation_metrics.get("validation_loss_total", float("inf"))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            _save_pretrain_checkpoint(
                output_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, data_module=data_module, step=config.max_steps,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
            )
    _save_pretrain_checkpoint(
        output_dir / "latest.pt",
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        config=config, data_module=data_module, step=config.max_steps,
        metrics=final_metrics, streaming_cursor=cursor.state_dict(),
    )
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
        streaming_cursor=cursor.state_dict(),
    )
    return TrainingResult(
        checkpoint=checkpoint,
        log_path=logger.path,
        steps=config.max_steps,
        final_loss=final_loss,
        metrics=final_metrics,
        data_module=data_module,
    )


@torch.no_grad()
def _validate_pretraining(
    model: ContextualPretrainingModel,
    data_module: RealDataModule,
    *,
    device: torch.device,
    config: PretrainConfig,
) -> dict[str, float]:
    """Aggregate bounded held-out objective and representation diagnostics."""

    model.eval()
    split = "validation" if data_module.split_counts.get("validation", 0) else "train"
    totals: dict[str, list[float]] = {}
    retrieval_embeddings: list[torch.Tensor] = []
    retrieval_ids: list[torch.Tensor] = []
    batch_count = event_count = 0
    for raw_batch in data_module.batches(
        split, batch_size=config.batch_size, shuffle=False
    ):
        if batch_count >= config.validation_batches:
            break
        batch_count += 1
        event_count += int(raw_batch["node_mask"].shape[0])
        batch = _to_device(raw_batch, device)
        _add_topology_labels(batch)
        curriculum = build_curriculum_batch(
            batch, PretrainingStage.TRUTH_GUIDED_MULTILEVEL,
            seed=config.seed + batch_count,
            corruption_objective=config.corruption_objective,
        )
        encoded, leaf_pid_logits, validation_batch = model.encode_runtime(
            curriculum.batch,
            attention_mask=curriculum.batch["curriculum_attention_mask"],
        )
        relation_logits = model.relation_head(encoded.tree_projection)
        targets, relation_mask = build_tree_relation_targets(
            parent_ids=validation_batch["parent_ids"],
            lca_depth=validation_batch["lca_depth"],
            level_ids=validation_batch["level_ids"],
            node_mask=validation_batch["node_mask"],
            b_side=validation_batch["b_side"],
        )
        attention_logits = (
            model.channel_pool_score(encoded.channel_projection).squeeze(-1)
            if model.channel_pool_score is not None else None
        )
        branch_embeddings, branch_mask = pool_b_branch_embeddings(
            encoded.channel_projection,
            validation_batch["b_side"],
            validation_batch["node_mask"],
            mode=model.channel_pooling,
            level_ids=validation_batch["level_ids"],
            attention_logits=attention_logits,
        )
        valid_channel = valid_b_root_channel_mask(validation_batch)
        branch_mask &= valid_channel[:, None]
        full_channel_ids = torch.stack(
            [
                validation_batch["b1_full_truth_channel_ids"],
                validation_batch["b2_full_truth_channel_ids"],
            ],
            dim=-1,
        )
        structural_features = torch.cat(
            [
                validation_batch["b_channel_count_arrays"],
                validation_batch["b_depth_pid_count_arrays"].flatten(start_dim=2),
                validation_batch["b_branch_multiplicity_summaries"],
                validation_batch["b_intermediate_count_arrays"],
            ],
            dim=-1,
        )
        loss_output = hyperbolic_pretraining_loss(
            z=encoded.hyperbolic_embeddings,
            tree_relation_logits=relation_logits,
            tree_relation_targets=targets,
            tree_relation_mask=relation_mask,
            lca_depth=validation_batch["lca_depth"],
            parent_ids=validation_batch["parent_ids"],
            level_ids=validation_batch["level_ids"],
            node_mask=validation_batch["node_mask"],
            b_side=validation_batch["b_side"],
            node_kind_ids=validation_batch["node_kind_ids"],
            event_ids=validation_batch["event_ids"],
            channel_embeddings=branch_embeddings,
            channel_mask=branch_mask,
            full_truth_channel_ids=full_channel_ids,
            reconstructable_channel_ids=torch.stack(
                [
                    validation_batch["b1_reconstructable_channel_ids"],
                    validation_batch["b2_reconstructable_channel_ids"],
                ], dim=-1,
            ),
            channel_branch_count_arrays=structural_features,
            weights=_pretraining_weights(config.ablation),
            curvature=config.curvature,
            full_event_max_level=validation_batch.get("full_event_max_level"),
        )
        leaf_loss = _leaf_pid_loss(leaf_pid_logits, validation_batch)
        total_loss = loss_output.total + leaf_loss
        totals.setdefault("validation_loss_total", []).append(float(total_loss))
        totals.setdefault("validation_loss_leaf_pid", []).append(float(leaf_loss))
        for name, value in loss_output.components.items():
            totals.setdefault(f"validation_loss_{name}", []).append(float(value))
        for name, value in loss_output.diagnostics.items():
            totals.setdefault(f"validation_{name}", []).append(float(value))
        if relation_mask.any():
            accuracy = (relation_logits.argmax(dim=-1)[relation_mask] == targets[relation_mask]).float().mean()
            totals.setdefault("validation_relation_accuracy", []).append(float(accuracy))
        ranking = parent_child_ranking_accuracy(
            encoded.hyperbolic_embeddings,
            validation_batch["parent_ids"],
            validation_batch["node_mask"],
            curvature=config.curvature,
            lca_depth=validation_batch["lca_depth"],
            tree_relation_targets=targets,
            b_side=validation_batch["b_side"],
        )
        totals.setdefault("validation_parent_ranking_accuracy", []).append(float(ranking))
        correlation = loss_output.diagnostics.get("radius_level_correlation")
        if correlation is not None:
            totals.setdefault("validation_radius_level_monotonicity", []).append(float(-correlation))
        raw_tracks = (
            validation_batch["node_mask"]
            & (validation_batch["leaf_kinematics_mode_ids"] == LEAF_MODE_TO_ID["raw_track_predicted_pid"])
            & validation_batch["truth_pid_available"]
        )
        if raw_tracks.any():
            probabilities = torch.softmax(leaf_pid_logits[raw_tracks], dim=-1)
            totals.setdefault("validation_leaf_pid_accuracy", []).append(
                float((probabilities.argmax(dim=-1) == validation_batch["truth_pid_labels"][raw_tracks]).float().mean())
            )
            totals.setdefault("validation_leaf_pid_entropy", []).append(
                float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)).mean())
            )
        selected = branch_mask.reshape(-1) & (full_channel_ids.reshape(-1) > 0)
        if selected.any():
            retrieval_embeddings.append(branch_embeddings.reshape(-1, branch_embeddings.shape[-1])[selected].cpu())
            retrieval_ids.append(full_channel_ids.reshape(-1)[selected].cpu())
    if retrieval_embeddings:
        embeddings = F.normalize(torch.cat(retrieval_embeddings), dim=-1)
        ids = torch.cat(retrieval_ids)
        similarity = embeddings @ embeddings.T
        similarity.fill_diagonal_(float("-inf"))
        has_peer = (ids[:, None] == ids[None, :]).fill_diagonal_(False).any(dim=-1)
        if has_peer.any():
            nearest = similarity.argmax(dim=-1)
            totals["validation_channel_retrieval_accuracy"] = [
                float((ids[nearest[has_peer]] == ids[has_peer]).float().mean())
            ]
    model.train()
    metrics = {
        name: sum(values) / len(values)
        for name, values in totals.items()
        if values
    }
    metrics["validation_batches"] = float(batch_count)
    metrics["validation_events"] = float(event_count)
    return metrics


def _pretraining_weights(ablation_name: str) -> dict[str, float]:
    config = ALL_ABLATIONS[ablation_name]
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
    leaf_pid_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    raw_tracks = (
        batch["node_mask"]
        & (
            batch["leaf_kinematics_mode_ids"]
            == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
        )
        & batch["truth_pid_available"]
    )
    if not raw_tracks.any():
        return leaf_pid_logits.sum() * 0.0
    return F.cross_entropy(
        leaf_pid_logits[raw_tracks],
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


def valid_b_root_channel_mask(
    batch: dict[str, torch.Tensor],
    *,
    corrupted_node_mask: torch.Tensor | None = None,
    corruption_objective: str = "denoising",
) -> torch.Tensor:
    """Exclude invalid/fallback B assignments from current-batch channel loss."""

    reference = batch["node_mask"][:, 0]
    valid = batch.get(
        "b_root_discovery_valid", torch.ones_like(reference)
    ).bool().clone()
    valid &= ~batch.get(
        "b_root_discovery_fallback", torch.zeros_like(reference)
    ).bool()
    if corruption_objective == "invalid_candidate" and corrupted_node_mask is not None:
        valid &= ~corrupted_node_mask.any(dim=-1)
    return valid


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
    streaming_cursor: dict[str, int],
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
        legacy_conflated_fraction=data_module.legacy_conflated_fraction,
        schema_version=(
            data_module.source_schema_versions[0]
            if len(data_module.source_schema_versions) == 1
            else "mixed"
        ),
        streaming_cursor=streaming_cursor,
        epoch=int(streaming_cursor.get("epoch", 0)),
        data_order_contract={
            **_pretrain_data_order_contract(config, data_module),
            "epoch": int(streaming_cursor.get("epoch", 0)),
            "batch_index": int(streaming_cursor.get("batch_index", 0)),
        },
        architecture=resolve_model_architecture(
            config.model_preset,
            d_model=config.d_model,
            hyper_dim=config.hyper_dim,
            n_heads=config.n_heads,
            n_context_layers=config.n_context_layers,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            curvature=config.curvature,
            n_queries=config.n_queries,
            n_queries_by_level=config.n_queries_by_level,
            max_cardinality=config.max_cardinality,
            max_cardinality_by_level=config.max_cardinality_by_level,
        ).to_dict(),
    )


def _pretrain_data_order_contract(
    config: PretrainConfig, data_module: RealDataModule
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
        "target_policy": "complete_only",
        "curriculum_order": list(config.curriculum),
        "teacher_forcing_schedule": None,
        "level_sampling_mode": "curriculum_stage",
        "gradient_accumulation": 1,
        "num_workers": config.num_workers,
    }


def _hard_negative_tree_loss(
    z: torch.Tensor,
    pairs: torch.Tensor,
    *,
    margin: float = 1.0,
    curvature: float = 1.0,
) -> torch.Tensor:
    if not pairs.numel():
        return z.sum() * 0.0
    from hypertagging.models.hyperbolic import distance

    values = []
    for batch_index, left, right in pairs.tolist():
        values.append(
            distance(
                z[batch_index, left],
                z[batch_index, right],
                curvature=curvature,
            )
        )
    distances = torch.stack(values)
    return torch.relu(margin - distances).mean()


__all__ = [
    "ContextualPretrainingModel",
    "PretrainConfig",
    "TrainingResult",
    "train_hyperbolic_pretraining",
    "valid_b_root_channel_mask",
]
