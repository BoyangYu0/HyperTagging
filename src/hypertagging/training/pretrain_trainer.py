"""Checkpointed real-parquet contextual hyperbolic pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn.functional as F

from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.data.tree_geometry import build_exact_tree_geometry
from hypertagging.losses.hyperbolic_pretraining import (
    build_topology_safe_parent_negative_mask,
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
from hypertagging.training.checkpoint_selection import (
    PRETRAIN_CHECKPOINT_TRACKS,
    checkpoint_track_decisions,
    initial_track_values,
    selection_reason,
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
    tangent_variance_target: float | None = None
    hyper_projection_init_scale: float | None = None
    tangent_scale_mode: str | None = None
    radius_target_mode: str = "generation_height_radius"
    best_metric: str = "validation_full_training_objective"
    best_mode: str = "min"
    channel_zero_positive_validation_window: int = 3
    channel_zero_positive_action: str = "warn"
    hyperbolic_level_encoding: str = "learned_euclidean"
    objective_gradient_diagnostics: bool = False
    objective_gradient_diagnostics_every: int = 100
    pilot_objective_preflight: bool = False
    objective_dominance_ratio: float = 100.0
    pilot_objective_violation_action: str = "warn"
    lca_relation_weight: float = 1.0
    parent_ranking_weight: float = 1.0
    exact_tree_distance_weight: float = 1.0
    radius_depth_weight: float = 0.2
    channel_weight: float = 0.2
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    leaf_pid_weight: float = 1.0
    corruption_class_weight: float = 0.1
    candidate_correctness_weight: float = 0.1
    hard_negative_weight: float = 0.1
    truth_guided_structural_relation_inputs: bool = False


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
        hyper_projection_init_scale: float = 0.05,
        tangent_scale_mode: str = "fixed",
        hyperbolic_level_encoding: str = "learned_euclidean",
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
            hyper_projection_init_scale=hyper_projection_init_scale,
            tangent_scale_mode=tangent_scale_mode,
            hyperbolic_level_encoding=hyperbolic_level_encoding,
        )
        self.relation_head = TreeRelationHead(d_model)
        self.leaf_pid_head = torch.nn.Linear(d_model, len(PDG_TOKENS))
        self.candidate_correctness_head = torch.nn.Linear(d_model, 1)
        self.corruption_type_head = torch.nn.Linear(d_model, 5)
        self.channel_memory = ChannelMemoryBank(channel_memory_size, d_model)
        if channel_pooling not in {
            "mean_all", "fsp_only", "b_root", "learned_attention", "level_weighted"
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
    """Optional detached ring buffer of cross-event channel examples."""

    def __init__(self, capacity: int, embedding_dim: int) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 0)
        self.register_buffer("embeddings", torch.zeros(self.capacity, embedding_dim))
        self.register_buffer("full_ids", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer(
            "reconstructable_ids", torch.zeros(self.capacity, dtype=torch.long)
        )
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool))

    def contents(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.capacity == 0:
            return self.embeddings, self.full_ids, self.reconstructable_ids
        order = (
            torch.arange(self.capacity, device=self.embeddings.device) + self.cursor
        ) % self.capacity
        order = order[self.valid[order]]
        return (
            self.embeddings[order],
            self.full_ids[order],
            self.reconstructable_ids[order],
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
        if additions.shape[0] == 0:
            return
        additions = additions[-self.capacity :]
        full = full[-self.capacity :]
        reco = reco[-self.capacity :]
        positions = (
            self.cursor
            + torch.arange(additions.shape[0], device=additions.device)
        ) % self.capacity
        self.embeddings[positions] = additions
        self.full_ids[positions] = full
        self.reconstructable_ids[positions] = reco
        self.valid[positions] = True
        self.cursor.copy_((self.cursor + additions.shape[0]) % self.capacity)
        self.count.copy_(
            torch.clamp(self.count + additions.shape[0], max=self.capacity)
        )


def train_hyperbolic_pretraining(config: PretrainConfig) -> TrainingResult:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.validate_every <= 0 or config.validation_batches <= 0:
        raise ValueError("validate_every and validation_batches must be positive")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if config.best_mode not in {"min", "max"}:
        raise ValueError("best_mode must be 'min' or 'max'")
    if config.best_metric not in {
        "validation_principal_loss",
        "validation_full_training_objective",
    }:
        raise ValueError(
            "best_metric must explicitly select validation_principal_loss or "
            "validation_full_training_objective"
        )
    if config.radius_target_mode not in {
        "generation_height_radius", "exact_root_depth_radius",
        "weak_or_learned_radius",
    }:
        raise ValueError("unknown radius_target_mode")
    if config.channel_zero_positive_action not in {"warn", "fail", "ignore"}:
        raise ValueError("channel_zero_positive_action must be warn, fail, or ignore")
    if config.channel_zero_positive_validation_window <= 0:
        raise ValueError("channel zero-positive validation window must be positive")
    if config.objective_gradient_diagnostics_every <= 0:
        raise ValueError("objective gradient diagnostic cadence must be positive")
    if config.pilot_objective_preflight and not config.objective_gradient_diagnostics:
        raise ValueError("pilot objective preflight requires gradient diagnostics")
    if config.objective_dominance_ratio <= 1:
        raise ValueError("objective dominance ratio must exceed one")
    if config.pilot_objective_violation_action not in {"warn", "fail"}:
        raise ValueError("pilot objective violation action must be warn or fail")
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
        tangent_variance_target=config.tangent_variance_target,
        hyper_projection_init_scale=config.hyper_projection_init_scale,
        tangent_scale_mode=config.tangent_scale_mode,
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
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
        hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
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
    restored_training_state = (resume_payload or {}).get("training_state", {})
    if restored_training_state and (
        restored_training_state.get("best_metric") != config.best_metric
        or restored_training_state.get("best_mode") != config.best_mode
    ):
        raise ValueError("resume best-metric selection differs from the checkpoint")
    best_validation_loss = float(restored_training_state.get(
        "best_metric_value",
        float("inf") if config.best_mode == "min" else float("-inf"),
    ))
    diagnostic_track_values = {
        **initial_track_values(PRETRAIN_CHECKPOINT_TRACKS),
        **{
            str(key): float(value)
            for key, value in restored_training_state.get(
                "diagnostic_checkpoint_track_values", {}
            ).items()
        },
    }
    last_validation_step = int(restored_training_state.get("last_validation_step", 0))
    zero_positive_windows = int(restored_training_state.get(
        "channel_zero_positive_validation_windows", 0
    ))
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
            truth_guided_structural_relation_inputs=(
                config.truth_guided_structural_relation_inputs
            ),
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
                lca_node_id=train_batch["lca_node_id"],
                edges_to_lca_from_i=train_batch["edges_to_lca_from_i"],
                edges_to_lca_from_j=train_batch["edges_to_lca_from_j"],
            )
            parent_negative_mask = build_topology_safe_parent_negative_mask(
                targets, structural_mask,
                train_batch["ancestor_descendant_relation"],
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
                exact_tree_path_distance=train_batch["exact_tree_path_distance"],
                parent_negative_mask=parent_negative_mask,
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
                weights=_pretraining_weights(config),
                curvature=config.curvature,
                full_event_max_level=train_batch.get("full_event_max_level"),
                tangent_variance_target=architecture.tangent_variance_target,
                radius_target_mode=config.radius_target_mode,
                depth_from_retained_root=train_batch["depth_from_retained_root"],
                distance_to_nearest_retained_root=train_batch[
                    "distance_to_nearest_retained_root"
                ],
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
                + config.leaf_pid_weight * leaf_pid_loss
                + config.corruption_class_weight * corruption_loss
                + config.candidate_correctness_weight * correctness_loss
                + config.hard_negative_weight * hard_negative_loss
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
        should_log_gradients = (
            (step + 1) % config.log_every == 0
            or step == start_step
            or step + 1 == config.max_steps
        )
        gradient_metrics: dict[str, float] = {}
        if should_log_gradients:
            hyper_parameters = tuple(model.encoder.hyper_projection.parameters())
            per_loss = {
                **loss_output.components,
                "leaf_pid": leaf_pid_loss,
                "corruption": corruption_loss,
                "candidate_correctness": correctness_loss,
                "hard_negative": hard_negative_loss,
            }
            loss_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
            for name, value in per_loss.items():
                gradients = torch.autograd.grad(
                    value,
                    hyper_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                loss_gradients[name] = gradients
                gradient_metrics[f"gradient_loss_{name}_to_hyper_projection"] = (
                    _tensor_gradient_norm(gradients)
                )
            if "depth" in loss_gradients and "tree_distance" in loss_gradients:
                gradient_metrics[
                    "gradient_cosine_radius_tree_distance_hyper_projection"
                ] = _gradient_cosine(
                    loss_gradients["depth"], loss_gradients["tree_distance"]
                )
        if (
            config.objective_gradient_diagnostics
            and (
                (step + 1) % config.objective_gradient_diagnostics_every == 0
                or step == start_step
                or step + 1 == config.max_steps
            )
        ):
            objective_values = {
                "lca": loss_output.components["lca"],
                "parent": loss_output.components["parent"],
                "tree_distance": loss_output.components["tree_distance"],
                "radius": loss_output.components["depth"],
                "channel": loss_output.components["channel"],
                "variance": loss_output.components["var"],
                "covariance": loss_output.components["cov"],
                "leaf_pid": leaf_pid_loss,
                "corruption_class": corruption_loss,
                "candidate_correctness": correctness_loss,
                "hard_negative": hard_negative_loss,
            }
            objective_report = objective_gradient_diagnostics(
                objective_values,
                pretraining_projection_parameter_groups(model),
            )
            for group_name, values in objective_report["gradient_norms"].items():
                for objective_name, value in values.items():
                    gradient_metrics[
                        f"objective_gradient_norm_{group_name}_{objective_name}"
                    ] = value
            for group_name, matrix in objective_report["gradient_cosines"].items():
                for row_name, row in matrix.items():
                    for column_name, value in row.items():
                        gradient_metrics[
                            f"objective_gradient_cosine_{group_name}_{row_name}_{column_name}"
                        ] = value
            gradient_metrics["objective_zero_gradient_count"] = float(
                len(objective_report["zero_gradient_objectives"])
            )
            if config.pilot_objective_preflight:
                component_weights = _pretraining_weights(config)
                objective_weights = {
                    "lca": component_weights["lca"],
                    "parent": component_weights["parent"],
                    "tree_distance": component_weights["tree_distance"],
                    "radius": component_weights["depth"],
                    "channel": component_weights["channel"],
                    "variance": component_weights["var"],
                    "covariance": component_weights["cov"],
                    "leaf_pid": config.leaf_pid_weight,
                    "corruption_class": config.corruption_class_weight,
                    "candidate_correctness": config.candidate_correctness_weight,
                    "hard_negative": config.hard_negative_weight,
                }
                diagnostic = loss_output.diagnostics
                objective_denominators = {
                    "lca": float(diagnostic["active_denominator_lca"].detach().cpu()),
                    "parent": float(
                        diagnostic["parent_ranking_accuracy_denominator"].detach().cpu()
                    ),
                    "tree_distance": float(
                        diagnostic["active_denominator_tree_distance"].detach().cpu()
                    ),
                    "radius": float(diagnostic["active_denominator_radius"].detach().cpu()),
                    "channel": float(diagnostic["channel_active_anchors"].detach().cpu()),
                    "variance": float(
                        diagnostic["active_denominator_variance"].detach().cpu()
                    ),
                    "covariance": float(
                        diagnostic["active_denominator_covariance"].detach().cpu()
                    ),
                    "leaf_pid": float(
                        (
                            train_batch["node_mask"]
                            & (train_batch["level_ids"] == 0)
                            & train_batch["truth_pid_available"]
                        ).sum().detach().cpu()
                    ),
                    "corruption_class": float(corruption_nodes.sum().detach().cpu()),
                    "candidate_correctness": float(corruption_nodes.sum().detach().cpu()),
                    "hard_negative": float(curriculum.hard_negative_pairs.shape[0]),
                }
                preflight = objective_preflight_report(
                    objective_values,
                    objective_weights,
                    objective_denominators,
                    objective_report,
                    dominance_ratio=config.objective_dominance_ratio,
                    action=config.pilot_objective_violation_action,
                )
                gradient_metrics["objective_preflight_pass"] = float(preflight["pass"])
                gradient_metrics["objective_weighted_dominance_ratio"] = float(
                    preflight["weighted_dominance_ratio"]
                )
                gradient_metrics["objective_preflight_violation_count"] = float(
                    len(preflight["violations"])
                )
                for objective_name, row in preflight["objectives"].items():
                    for field, value in row.items():
                        gradient_metrics[
                            f"objective_preflight_{objective_name}_{field}"
                        ] = float(value)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if should_log_gradients:
            for name in (
                "tree_head",
                "reconstruction_head",
                "channel_head",
                "hyper_projection",
                "tangent_scale",
            ):
                module = getattr(model.encoder, name)
                gradient_metrics[f"gradient_projection_{name}"] = (
                    _parameter_gradient_norm(module.parameters())
                )
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
            "active_denominator_leaf_pid": float(
                (
                    train_batch["node_mask"]
                    & (train_batch["level_ids"] == 0)
                    & train_batch.get(
                        "truth_pid_available",
                        torch.ones_like(train_batch["node_mask"]),
                    )
                ).sum().detach().cpu()
            ),
            **{
                f"loss_{name}": float(value.detach().cpu())
                for name, value in loss_output.components.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in loss_output.diagnostics.items()
            },
            **gradient_metrics,
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
            zero_positive_windows = _check_channel_positive_window(
                validation_metrics, zero_positive_windows, config
            )
            diagnostic_track_values, selected_diagnostic_tracks = (
                checkpoint_track_decisions(
                    validation_metrics,
                    diagnostic_track_values,
                    PRETRAIN_CHECKPOINT_TRACKS,
                )
            )
            for track in selected_diagnostic_tracks:
                _save_pretrain_checkpoint(
                    output_dir / track.filename,
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, data_module=data_module, step=step + 1,
                    metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                    best_metric_value=best_validation_loss,
                    last_validation_step=last_validation_step,
                    zero_positive_windows=zero_positive_windows,
                    diagnostic_track_values=diagnostic_track_values,
                    checkpoint_selection_reason=selection_reason(
                        track, validation_metrics
                    ),
                )
            if config.best_metric not in validation_metrics:
                raise ValueError(
                    f"best_metric {config.best_metric!r} is absent from validation metrics"
                )
            validation_loss = float(validation_metrics[config.best_metric])
            improved = (
                validation_loss < best_validation_loss
                if config.best_mode == "min"
                else validation_loss > best_validation_loss
            )
            if improved:
                best_validation_loss = validation_loss
                _save_pretrain_checkpoint(
                    output_dir / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, data_module=data_module, step=step + 1,
                    metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                    best_metric_value=best_validation_loss,
                    last_validation_step=last_validation_step,
                    zero_positive_windows=zero_positive_windows,
                    diagnostic_track_values=diagnostic_track_values,
                    checkpoint_selection_reason={
                        "metric_name": config.best_metric,
                        "mode": config.best_mode,
                        "value": validation_loss,
                        "denominator_name": "validation_batches",
                        "denominator": validation_metrics["validation_batches"],
                        "reason": "new_principal_configured_checkpoint",
                    },
                )
            _save_pretrain_checkpoint(
                output_dir / "latest.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, data_module=data_module, step=step + 1,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                best_metric_value=best_validation_loss,
                last_validation_step=last_validation_step,
                zero_positive_windows=zero_positive_windows,
                diagnostic_track_values=diagnostic_track_values,
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
                best_metric_value=best_validation_loss,
                last_validation_step=last_validation_step,
                zero_positive_windows=zero_positive_windows,
                diagnostic_track_values=diagnostic_track_values,
            )
    if last_validation_step != config.max_steps:
        validation_metrics = _validate_pretraining(
            model, data_module, device=device, config=config
        )
        final_metrics.update(validation_metrics)
        logger.log(step=config.max_steps, split="validation", **validation_metrics)
        last_validation_step = config.max_steps
        zero_positive_windows = _check_channel_positive_window(
            validation_metrics, zero_positive_windows, config
        )
        diagnostic_track_values, selected_diagnostic_tracks = checkpoint_track_decisions(
            validation_metrics,
            diagnostic_track_values,
            PRETRAIN_CHECKPOINT_TRACKS,
        )
        for track in selected_diagnostic_tracks:
            _save_pretrain_checkpoint(
                output_dir / track.filename,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, data_module=data_module, step=config.max_steps,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                best_metric_value=best_validation_loss,
                last_validation_step=last_validation_step,
                zero_positive_windows=zero_positive_windows,
                diagnostic_track_values=diagnostic_track_values,
                checkpoint_selection_reason=selection_reason(
                    track, validation_metrics
                ),
            )
        validation_loss = float(validation_metrics[config.best_metric])
        improved = (
            validation_loss < best_validation_loss
            if config.best_mode == "min" else validation_loss > best_validation_loss
        )
        if improved:
            best_validation_loss = validation_loss
            _save_pretrain_checkpoint(
                output_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, data_module=data_module, step=config.max_steps,
                metrics=final_metrics, streaming_cursor=cursor.state_dict(),
                best_metric_value=best_validation_loss,
                last_validation_step=last_validation_step,
                zero_positive_windows=zero_positive_windows,
                diagnostic_track_values=diagnostic_track_values,
                checkpoint_selection_reason={
                    "metric_name": config.best_metric,
                    "mode": config.best_mode,
                    "value": validation_loss,
                    "denominator_name": "validation_batches",
                    "denominator": validation_metrics["validation_batches"],
                    "reason": "new_principal_configured_checkpoint",
                },
            )
    _save_pretrain_checkpoint(
        output_dir / "latest.pt",
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        config=config, data_module=data_module, step=config.max_steps,
        metrics=final_metrics, streaming_cursor=cursor.state_dict(),
        best_metric_value=best_validation_loss,
        last_validation_step=last_validation_step,
        zero_positive_windows=zero_positive_windows,
        diagnostic_track_values=diagnostic_track_values,
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
        best_metric_value=best_validation_loss,
        last_validation_step=last_validation_step,
        zero_positive_windows=zero_positive_windows,
        diagnostic_track_values=diagnostic_track_values,
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
        tangent_variance_target=config.tangent_variance_target,
        hyper_projection_init_scale=config.hyper_projection_init_scale,
        tangent_scale_mode=config.tangent_scale_mode,
        hyperbolic_level_encoding=config.hyperbolic_level_encoding,
    )
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
        validation_stage = PretrainingStage(
            config.curriculum[(batch_count - 1) % len(config.curriculum)]
        )
        curriculum = build_curriculum_batch(
            batch, validation_stage,
            seed=config.seed + batch_count,
            corruption_objective=config.corruption_objective,
            truth_guided_structural_relation_inputs=(
                config.truth_guided_structural_relation_inputs
            ),
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
            node_mask=curriculum.structural_positive_mask,
            b_side=validation_batch["b_side"],
            lca_node_id=validation_batch["lca_node_id"],
            edges_to_lca_from_i=validation_batch["edges_to_lca_from_i"],
            edges_to_lca_from_j=validation_batch["edges_to_lca_from_j"],
        )
        parent_negative_mask = build_topology_safe_parent_negative_mask(
            targets, curriculum.structural_positive_mask,
            validation_batch["ancestor_descendant_relation"],
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
        valid_channel = valid_b_root_channel_mask(
            validation_batch,
            corrupted_node_mask=curriculum.corrupted_node_mask,
            corruption_objective=config.corruption_objective,
        )
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
            exact_tree_path_distance=validation_batch["exact_tree_path_distance"],
            parent_negative_mask=parent_negative_mask,
            parent_ids=validation_batch["parent_ids"],
            level_ids=validation_batch["level_ids"],
            node_mask=curriculum.structural_positive_mask,
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
            weights=_pretraining_weights(config),
            curvature=config.curvature,
            full_event_max_level=validation_batch.get("full_event_max_level"),
            tangent_variance_target=architecture.tangent_variance_target,
            radius_target_mode=config.radius_target_mode,
            depth_from_retained_root=validation_batch["depth_from_retained_root"],
            distance_to_nearest_retained_root=validation_batch[
                "distance_to_nearest_retained_root"
            ],
        )
        ablation = ALL_ABLATIONS[config.ablation]
        leaf_loss = (
            _leaf_pid_loss(leaf_pid_logits, validation_batch)
            if ablation.leaf_pid
            else encoded.node_embeddings.sum() * 0.0
        )
        corruption_nodes = validation_batch["node_mask"] & (
            validation_batch["level_ids"] > 0
        )
        corruption_logits = model.corruption_type_head(encoded.node_embeddings)
        correctness_logits = model.candidate_correctness_head(
            encoded.node_embeddings
        ).squeeze(-1)
        corruption_loss = (
            F.cross_entropy(
                corruption_logits[corruption_nodes],
                curriculum.corruption_code[corruption_nodes],
            )
            if corruption_nodes.any()
            else encoded.node_embeddings.sum() * 0.0
        )
        correctness_loss = (
            F.binary_cross_entropy_with_logits(
                correctness_logits[corruption_nodes],
                (~curriculum.corrupted_node_mask)[corruption_nodes].float(),
            )
            if corruption_nodes.any()
            else encoded.node_embeddings.sum() * 0.0
        )
        hard_negative_loss = _hard_negative_tree_loss(
            encoded.hyperbolic_embeddings,
            curriculum.hard_negative_pairs,
            curvature=config.curvature,
        )
        principal_loss = loss_output.total + config.leaf_pid_weight * leaf_loss
        full_objective = (
            principal_loss
            + config.corruption_class_weight * corruption_loss
            + config.candidate_correctness_weight * correctness_loss
            + config.hard_negative_weight * hard_negative_loss
        )
        totals.setdefault("validation_principal_loss", []).append(float(principal_loss))
        stage_prefix = f"validation_{validation_stage.value}"
        totals.setdefault(f"{stage_prefix}_principal_loss", []).append(
            float(principal_loss)
        )
        totals.setdefault(f"{stage_prefix}_relation_accuracy", []).append(
            float(
                (relation_logits.argmax(dim=-1)[relation_mask] == targets[relation_mask])
                .float()
                .mean()
            )
            if relation_mask.any()
            else 0.0
        )
        # Always evaluate the two scientifically distinct representation
        # views, even when a bounded validation loader contains fewer batches
        # than the training curriculum has stages.
        for diagnostic_stage in (
            PretrainingStage.FSP_ONLY,
            PretrainingStage.TRUTH_GUIDED_MULTILEVEL,
        ):
            accuracy, denominator = _stage_relation_validation(
                model,
                batch,
                stage=diagnostic_stage,
                config=config,
                seed=config.seed + 10_000 + batch_count,
            )
            diagnostic_prefix = f"validation_{diagnostic_stage.value}"
            totals.setdefault(
                f"{diagnostic_prefix}_relation_accuracy_separate", []
            ).append(accuracy)
            totals.setdefault(
                f"{diagnostic_prefix}_relation_denominator_separate", []
            ).append(denominator)
        totals.setdefault("validation_full_training_objective", []).append(
            float(full_objective)
        )
        # Compatibility alias; checkpoint selection must use one of the two
        # explicitly named metrics above.
        totals.setdefault("validation_loss_total", []).append(float(full_objective))
        totals.setdefault("validation_loss_leaf_pid", []).append(float(leaf_loss))
        totals.setdefault("validation_loss_corruption_class", []).append(
            float(corruption_loss)
        )
        totals.setdefault("validation_loss_candidate_correctness", []).append(
            float(correctness_loss)
        )
        totals.setdefault("validation_loss_hard_negative", []).append(
            float(hard_negative_loss)
        )
        totals.setdefault("validation_hard_negative_count", []).append(
            float(curriculum.hard_negative_pairs.shape[0])
        )
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
            parent_negative_mask=parent_negative_mask,
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
            totals["validation_channel_retrieval_queries"] = [
                float(has_peer.sum())
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


def _pretraining_weights(config: PretrainConfig) -> dict[str, float]:
    ablation = ALL_ABLATIONS[config.ablation]
    return {
        "lca": config.lca_relation_weight * float(ablation.lca_relation),
        "parent": config.parent_ranking_weight * float(ablation.parent_ranking),
        "tree_distance": config.exact_tree_distance_weight * float(
            ablation.exact_tree_distance
        ),
        "depth": config.radius_depth_weight * float(ablation.radius_depth),
        "channel": config.channel_weight * float(ablation.channel_supervision),
        "var": config.variance_weight * float(ablation.variance_covariance),
        "cov": config.covariance_weight * float(ablation.variance_covariance),
    }


@torch.no_grad()
def _stage_relation_validation(
    model: ContextualPretrainingModel,
    batch: dict[str, torch.Tensor],
    *,
    stage: PretrainingStage,
    config: PretrainConfig,
    seed: int,
) -> tuple[float, float]:
    """Evaluate relation classification under one explicit information view."""

    curriculum = build_curriculum_batch(
        batch,
        stage,
        seed=seed,
        corruption_objective=config.corruption_objective,
        truth_guided_structural_relation_inputs=(
            config.truth_guided_structural_relation_inputs
        ),
    )
    encoded, _leaf_pid_logits, diagnostic_batch = model.encode_runtime(
        curriculum.batch,
        attention_mask=curriculum.batch["curriculum_attention_mask"],
    )
    logits = model.relation_head(encoded.tree_projection)
    targets, relation_mask = build_tree_relation_targets(
        parent_ids=diagnostic_batch["parent_ids"],
        lca_depth=diagnostic_batch["lca_depth"],
        level_ids=diagnostic_batch["level_ids"],
        node_mask=curriculum.structural_positive_mask,
        b_side=diagnostic_batch["b_side"],
        lca_node_id=diagnostic_batch["lca_node_id"],
        edges_to_lca_from_i=diagnostic_batch["edges_to_lca_from_i"],
        edges_to_lca_from_j=diagnostic_batch["edges_to_lca_from_j"],
    )
    denominator = float(relation_mask.sum())
    accuracy = (
        float(
            (logits.argmax(dim=-1)[relation_mask] == targets[relation_mask])
            .float()
            .mean()
        )
        if relation_mask.any()
        else 0.0
    )
    return accuracy, denominator


def _tensor_gradient_norm(
    gradients: tuple[torch.Tensor | None, ...],
) -> float:
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0


def _gradient_cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> float:
    pairs = [
        (a.detach().flatten(), b.detach().flatten())
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not pairs:
        return 0.0
    a = torch.cat([pair[0] for pair in pairs])
    b = torch.cat([pair[1] for pair in pairs])
    return float(F.cosine_similarity(a, b, dim=0, eps=1e-12).cpu())


PRINCIPAL_PRETRAINING_OBJECTIVES = (
    "lca",
    "parent",
    "tree_distance",
    "radius",
    "channel",
    "variance",
    "covariance",
    "leaf_pid",
)

OPTIONAL_PRETRAINING_OBJECTIVES = (
    "corruption_class",
    "candidate_correctness",
    "hard_negative",
)


def pretraining_projection_parameter_groups(
    model: ContextualPretrainingModel,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return the five documented gradient-interference parameter groups."""

    excluded = (
        "tree_head.",
        "reconstruction_head.",
        "channel_head.",
        "hyper_projection.",
        "tangent_scale.",
    )
    shared = tuple(
        parameter
        for name, parameter in model.encoder.named_parameters()
        if not name.startswith(excluded)
    )
    return {
        "shared_encoder": shared,
        "tree_projection": tuple(model.encoder.tree_head.parameters()),
        "hyperbolic_projection": tuple(
            model.encoder.hyper_projection.parameters()
        )
        + tuple(model.encoder.tangent_scale.parameters()),
        "reconstruction_projection": tuple(
            model.encoder.reconstruction_head.parameters()
        ),
        "channel_projection": tuple(model.encoder.channel_head.parameters()),
    }


def objective_gradient_diagnostics(
    objectives: dict[str, torch.Tensor],
    parameter_groups: dict[str, tuple[torch.nn.Parameter, ...]],
) -> dict[str, Any]:
    """Pairwise objective-gradient cosines without changing optimization."""

    missing = sorted(set(PRINCIPAL_PRETRAINING_OBJECTIVES) - set(objectives))
    if missing:
        raise ValueError("missing principal objective(s): " + ", ".join(missing))
    objective_order = PRINCIPAL_PRETRAINING_OBJECTIVES + tuple(
        name for name in OPTIONAL_PRETRAINING_OBJECTIVES if name in objectives
    )
    report: dict[str, Any] = {
        "objective_order": list(objective_order),
        "gradient_norms": {},
        "gradient_cosines": {},
        "zero_gradient_objectives": [],
    }
    zero_pairs: set[str] = set()
    for group_name, parameters in parameter_groups.items():
        vectors: dict[str, torch.Tensor] = {}
        norms: dict[str, float] = {}
        for objective_name in objective_order:
            value = objectives[objective_name]
            gradients = (
                torch.autograd.grad(
                    value,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                if value.requires_grad and parameters
                else tuple(None for _ in parameters)
            )
            vector = torch.cat(
                [
                    (
                        gradient.detach().reshape(-1)
                        if gradient is not None
                        else parameter.detach().new_zeros(parameter.numel())
                    )
                    for parameter, gradient in zip(parameters, gradients, strict=True)
                ]
            ) if parameters else value.detach().new_zeros(1)
            vectors[objective_name] = vector
            norm = float(torch.linalg.vector_norm(vector.float()).cpu())
            norms[objective_name] = norm
            if norm == 0.0:
                zero_pairs.add(f"{group_name}:{objective_name}")
        report["gradient_norms"][group_name] = norms
        report["gradient_cosines"][group_name] = {
            left: {
                right: float(
                    F.cosine_similarity(
                        vectors[left].float(),
                        vectors[right].float(),
                        dim=0,
                        eps=1e-12,
                    ).cpu()
                )
                for right in objective_order
            }
            for left in objective_order
        }
    report["zero_gradient_objectives"] = sorted(zero_pairs)
    return report


def objective_preflight_report(
    objectives: dict[str, torch.Tensor],
    weights: dict[str, float],
    denominators: dict[str, float],
    gradient_report: dict[str, Any],
    *,
    dominance_ratio: float = 100.0,
    action: str = "warn",
) -> dict[str, Any]:
    """Validate intended pilot objectives without changing optimization."""

    if action not in {"warn", "fail"}:
        raise ValueError("objective preflight action must be warn or fail")
    rows: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    all_projection_norms = gradient_report.get("gradient_norms", {})
    monitored_groups = (
        "shared_encoder",
        "tree_projection",
        "hyperbolic_projection",
    )
    weighted_nonzero: list[tuple[str, float]] = []
    for name, value in objectives.items():
        raw = float(value.detach().cpu())
        weight = float(weights.get(name, 0.0))
        weighted = raw * weight
        denominator = float(denominators.get(name, 0.0))
        group_norms = {
            group: float(all_projection_norms.get(group, {}).get(name, 0.0))
            for group in monitored_groups
        }
        rows[name] = {
            "raw_loss": raw,
            "configured_weight": weight,
            "weighted_magnitude": weighted,
            "active_denominator": denominator,
            **{
                f"{group}_gradient_norm": norm
                for group, norm in group_norms.items()
            },
        }
        if weight == 0:
            continue
        if not all(
            math.isfinite(number)
            for number in (raw, weighted, *group_norms.values())
        ):
            violations.append(f"{name}:non_finite")
        if denominator <= 0:
            violations.append(f"{name}:zero_denominator")
        if not any(norm > 0 for norm in group_norms.values()):
            violations.append(f"{name}:zero_gradient")
        if abs(weighted) > 0:
            weighted_nonzero.append((name, abs(weighted)))
    if len(weighted_nonzero) >= 2:
        largest_name, largest = max(weighted_nonzero, key=lambda item: item[1])
        smallest_name, smallest = min(weighted_nonzero, key=lambda item: item[1])
        ratio = largest / max(smallest, 1e-30)
        if ratio > dominance_ratio:
            violations.append(
                f"weighted_dominance:{largest_name}/{smallest_name}={ratio:.6g}"
            )
    else:
        ratio = 1.0
    for name in ("variance", "covariance"):
        if weights.get(name, 0.0) and denominators.get(name, 0.0) < 2:
            violations.append(f"{name}:insufficient_samples")
    if weights.get("channel", 0.0) and denominators.get("channel", 0.0) <= 0:
        violations.append("channel:no_positive_pairs")
    report = {
        "objectives": rows,
        "projection_gradient_norms": {
            projection: {
                name: float(values.get(name, 0.0)) for name in objectives
            }
            for projection, values in all_projection_norms.items()
        },
        "pairwise_gradient_cosines": gradient_report.get(
            "pairwise_cosines", gradient_report.get("gradient_cosines", {})
        ),
        "weighted_dominance_ratio": ratio,
        "dominance_threshold": float(dominance_ratio),
        "violations": sorted(set(violations)),
        "pass": not violations,
    }
    if violations:
        message = "objective pilot preflight: " + ", ".join(report["violations"])
        if action == "fail":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return report


def _parameter_gradient_norm(parameters) -> float:
    return _tensor_gradient_norm(
        tuple(parameter.grad for parameter in parameters)
    )


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
    required = {
        "lca_depth", "lca_node_id", "edges_to_lca_from_i",
        "edges_to_lca_from_j", "exact_tree_path_distance",
        "depth_from_retained_root", "distance_to_nearest_retained_root",
        "ancestor_descendant_relation",
    }
    if required.issubset(batch):
        return
    if batch["parent_ids"].is_cuda:
        missing = sorted(required - set(batch))
        raise RuntimeError(
            "topology geometry must be collated on CPU before CUDA training; "
            f"missing fields: {missing}"
        )
    # Compatibility path for tiny standalone CPU callers. Geometry is built
    # exactly once per event and every derived tensor reuses that result.
    lca = torch.full(
        (*batch["parent_ids"].shape, batch["parent_ids"].shape[1]),
        -1,
        dtype=torch.long,
        device=batch["parent_ids"].device,
    )
    geometries = []
    counts = []
    for index in range(batch["parent_ids"].shape[0]):
        count = int(batch["node_mask"][index].sum())
        geometry = build_exact_tree_geometry(batch["parent_ids"][index, :count])
        geometries.append(geometry)
        counts.append(count)
        valid_lca = geometry.lca_node_id >= 0
        event_lca = torch.full_like(geometry.lca_node_id, -1)
        event_lca[valid_lca] = batch["level_ids"][index, :count][
            geometry.lca_node_id[valid_lca]
        ]
        lca[index, :count, :count] = event_lca
    batch["lca_depth"] = lca
    pair_fields = (
        "lca_node_id",
        "edges_to_lca_from_i",
        "edges_to_lca_from_j",
        "exact_tree_path_distance",
    )
    for field in pair_fields:
        batch[field] = torch.full_like(lca, -1)
    batch["depth_from_retained_root"] = torch.full_like(batch["parent_ids"], -1)
    batch["distance_to_nearest_retained_root"] = torch.full_like(
        batch["parent_ids"], -1
    )
    batch["ancestor_descendant_relation"] = torch.zeros_like(lca, dtype=torch.bool)
    for index, (count, geometry) in enumerate(zip(counts, geometries, strict=True)):
        for field in pair_fields:
            batch[field][index, :count, :count] = getattr(geometry, field).to(lca.device)
        batch["depth_from_retained_root"][index, :count] = (
            geometry.depth_from_retained_root.to(lca.device)
        )
        batch["distance_to_nearest_retained_root"][index, :count] = (
            geometry.distance_to_nearest_retained_root.to(lca.device)
        )
        positions = torch.arange(count, device=lca.device)
        batch["ancestor_descendant_relation"][index, :count, :count] = (
            ((geometry.lca_node_id == positions[:, None])
             | (geometry.lca_node_id == positions[None, :]))
            & ~torch.eye(count, dtype=torch.bool, device=lca.device)
        )


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
    best_metric_value: float = float("inf"),
    last_validation_step: int = 0,
    zero_positive_windows: int = 0,
    diagnostic_track_values: dict[str, float] | None = None,
    checkpoint_selection_reason: dict[str, object] | None = None,
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
        feature_contract={
            "feature_spec_revision": feature_spec_v4()["feature_spec_revision"],
            "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
            "model_feature_contract_hash": feature_spec_v4()["model_feature_contract_hash"],
            "track_fit_policies": list(
                data_module.track_fit_policies
            ),
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
            **_pretrain_data_order_contract(config, data_module),
            "epoch": int(streaming_cursor.get("epoch", 0)),
            "batch_index": int(streaming_cursor.get("batch_index", 0)),
        },
        architecture=resolve_model_architecture(
            config.model_preset,
            d_model=config.d_model,
            hyper_dim=config.hyper_dim,
            tangent_variance_target=config.tangent_variance_target,
            hyper_projection_init_scale=config.hyper_projection_init_scale,
            tangent_scale_mode=config.tangent_scale_mode,
            hyperbolic_level_encoding=config.hyperbolic_level_encoding,
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
        training_state={
            "best_metric": config.best_metric,
            "best_mode": config.best_mode,
            "best_metric_value": float(best_metric_value),
            "last_validation_step": int(last_validation_step),
            "channel_zero_positive_validation_windows": int(
                zero_positive_windows
            ),
            "diagnostic_checkpoint_track_values": dict(
                diagnostic_track_values or {}
            ),
            "checkpoint_selection_reason": dict(
                checkpoint_selection_reason
                or {
                    "reason": "latest_or_periodic_state",
                    "metric_name": config.best_metric,
                    "mode": config.best_mode,
                }
            ),
        },
    )


def _check_channel_positive_window(
    metrics: dict[str, float],
    previous: int,
    config: PretrainConfig,
) -> int:
    positives = float(metrics.get("validation_channel_positive_pairs", 0.0))
    current = previous + 1 if positives <= 0 else 0
    if current >= config.channel_zero_positive_validation_window:
        message = (
            "channel objective had zero positive pairs for "
            f"{current} consecutive validation windows; channel_memory_size="
            f"{config.channel_memory_size}"
        )
        if config.channel_zero_positive_action == "fail":
            raise RuntimeError(message)
        if config.channel_zero_positive_action == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return current


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

    indices = pairs.to(device=z.device, dtype=torch.long)
    distances = distance(
        z[indices[:, 0], indices[:, 1]],
        z[indices[:, 0], indices[:, 2]],
        curvature=curvature,
    )
    return torch.relu(margin - distances).mean()


__all__ = [
    "ChannelMemoryBank",
    "ContextualPretrainingModel",
    "PRINCIPAL_PRETRAINING_OBJECTIVES",
    "PretrainConfig",
    "TrainingResult",
    "objective_gradient_diagnostics",
    "objective_preflight_report",
    "pretraining_projection_parameter_groups",
    "train_hyperbolic_pretraining",
    "valid_b_root_channel_mask",
]
