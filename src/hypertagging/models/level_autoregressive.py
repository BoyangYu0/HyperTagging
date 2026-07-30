"""Level-autoregressive set reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.hyperbolic import HyperbolicNodeEncoder
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.mother_pointer import MotherPointerDecoder, MotherPointerOutput
from hypertagging.models.relation_attention import RelationAwareSetTransformer
from hypertagging.models.relations import RelationBias
from hypertagging.models.stair_masks import context_mask_for_level, stair_attention_mask
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES as CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES as COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES as COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES as TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, validate_pid_tokens


@dataclass(frozen=True)
class LevelReconstructionOutput:
    target_level: int
    pointer: MotherPointerOutput
    node_embeddings: torch.Tensor
    hyperbolic_embeddings: torch.Tensor
    context_mask: torch.Tensor
    relation_bias: torch.Tensor
    attention_weights: torch.Tensor
    tree_projection: torch.Tensor
    reconstruction_projection: torch.Tensor
    channel_projection: torch.Tensor
    leaf_pid_logits: torch.Tensor | None = None


class LevelAutoregressiveReconstructor(nn.Module):
    """Tree-biased encoder plus pointer-based mother set decoder."""

    def __init__(
        self,
        *,
        n_features: int,
        n_types: int,
        hidden_dim: int = 32,
        hyper_dim: int = 16,
        n_queries: int = 8,
        n_heads: int = 4,
        n_context_layers: int = 2,
        encoder_mode: str = "heterogeneous",
        use_contextual_encoder: bool = True,
        use_relation_bias: bool = True,
        use_hyperbolic_relation_refinement: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_mode = encoder_mode
        self.use_contextual_encoder = use_contextual_encoder
        # The public argument is retained for source compatibility, but the
        # scientific contract has exactly one model vocabulary.
        n_types = len(PDG_TOKENS)
        if encoder_mode == "flat":
            self.encoder: nn.Module = HyperbolicNodeEncoder(
                n_features=n_features,
                n_pid=len(PDG_TOKENS),
                hidden_dim=hidden_dim,
                hyper_dim=hyper_dim,
            )
        elif encoder_mode == "heterogeneous":
            self.encoder = HeterogeneousNodeEncoder(
                d_model=hidden_dim,
                hyper_dim=hyper_dim,
                n_pid=len(PDG_TOKENS),
                n_heads=n_heads,
                n_context_layers=n_context_layers,
                use_contextual_encoder=use_contextual_encoder,
                use_physical_context=use_relation_bias,
                use_hyperbolic_refinement=use_hyperbolic_relation_refinement,
            )
        else:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}")
        self.flat_relation_bias = RelationBias(hidden_dim=hidden_dim, enabled=use_relation_bias)
        self.flat_contextualizer = RelationAwareSetTransformer(
            hidden_dim,
            n_heads=n_heads,
            n_layers=n_context_layers,
        )
        self.decoder = MotherPointerDecoder(hidden_dim=hidden_dim, n_types=n_types, n_queries=n_queries)
        self.leaf_pid_head = nn.Linear(hidden_dim, len(PDG_TOKENS))

    @property
    def relation_bias(self) -> nn.Module:
        """Compatibility handle for gradient/ablation inspection."""

        if self.encoder_mode == "heterogeneous":
            return self.encoder.physical_relation_bias  # type: ignore[attr-defined]
        return self.flat_relation_bias

    def forward(self, batch: dict[str, torch.Tensor], *, target_level: int = 1) -> LevelReconstructionOutput:
        if self.encoder_mode == "heterogeneous":
            batch = _upgrade_flat_batch(batch)
            encoded = self.encoder(
                batch,
                attention_mask=stair_attention_mask(batch["level_ids"], batch["node_mask"]),
            )
            h = encoded.node_embeddings
            z = encoded.hyperbolic_embeddings
            tree_projection = encoded.tree_projection
            reconstruction_projection = encoded.reconstruction_projection
            channel_projection = encoded.channel_projection
            relation_bias = encoded.physical_relation_bias + (
                encoded.hyperbolic_relation_bias
                if self.encoder.use_hyperbolic_refinement  # type: ignore[attr-defined]
                else torch.zeros_like(encoded.physical_relation_bias)
            )
            attention_weights = encoded.attention_weights
        else:
            h, z = self.encoder(
                batch["node_features"],
                batch["pid_labels"],
                batch["level_ids"],
                batch["charge"],
            )
            tree_projection = h
            reconstruction_projection = h
            channel_projection = h
            relation_bias = self.flat_relation_bias(
                p4=batch["p4"],
                charge=batch["charge"],
                level_ids=batch["level_ids"],
                z_hyperbolic=z,
                node_mask=batch["node_mask"],
                node_kind_ids=batch.get("node_kind_ids"),
                copied=batch.get("copied"),
                source_node_ids=batch.get("source_node_ids"),
            )
            if self.use_contextual_encoder:
                h, attention_weights = self.flat_contextualizer(
                    reconstruction_projection,
                    relation_bias=relation_bias,
                    attention_mask=stair_attention_mask(batch["level_ids"], batch["node_mask"]),
                    node_mask=batch["node_mask"],
                )
            else:
                h = reconstruction_projection
                attention_weights = relation_bias.new_zeros(
                    (*relation_bias.shape[:1], 1, *relation_bias.shape[-2:])
                )
        context_mask = context_mask_for_level(batch["level_ids"], batch["node_mask"], target_level)
        pointer = self.decoder(h, context_mask, target_level=target_level)
        return LevelReconstructionOutput(
            target_level,
            pointer,
            h,
            z,
            context_mask,
            relation_bias,
            attention_weights,
            tree_projection,
            reconstruction_projection,
            channel_projection,
            self.leaf_pid_head(h),
        )


def construct_mother_p4(pointer_logits: torch.Tensor, p4: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
    """Construct mother p4 from daughter pointers, never from MC mother p4."""

    weights = (pointer_logits > 0).float() if hard else torch.sigmoid(pointer_logits)
    return torch.einsum("bqn,bnf->bqf", weights, p4)


def _upgrade_flat_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upgrade legacy/tiny batches without claiming detector-specific values."""

    if "common_features" in batch:
        return batch
    p4 = batch["p4"]
    charge = batch["charge"]
    levels = batch["level_ids"]
    active = batch["node_mask"].bool()
    copied = batch.get("copied", torch.zeros_like(active))
    adjacency = batch["daughter_adjacency"]
    mass2 = p4[..., 3].square() - p4[..., :3].square().sum(dim=-1)
    common = torch.stack(
        [
            p4[..., 0],
            p4[..., 1],
            p4[..., 2],
            p4[..., 3],
            mass2.clamp_min(0).sqrt(),
            charge,
            batch["pid_labels"].float(),
            levels.float().clamp_min(0),
            active.float(),
            copied.float(),
            adjacency.sum(dim=-1).float(),
            torch.zeros_like(charge),
        ],
        dim=-1,
    )
    common_availability = active.unsqueeze(-1).expand_as(common).clone()
    common_availability[..., -1] = False
    batch = dict(batch)
    batch["common_features"] = common
    batch["common_availability"] = common_availability
    shape = (*active.shape, len(TRACK_FEATURE_NAMES))
    batch["track_features"] = p4.new_zeros(shape)
    batch["track_availability"] = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    shape = (*active.shape, len(CLUSTER_FEATURE_NAMES))
    batch["cluster_features"] = p4.new_zeros(shape)
    batch["cluster_availability"] = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    shape = (*active.shape, len(COMPOSITE_FEATURE_NAMES))
    composite = p4.new_zeros(shape)
    composite[..., :4] = torch.einsum("bmn,bnf->bmf", adjacency.float(), p4)
    composite[..., 4] = torch.einsum("bmn,bn->bm", adjacency.float(), charge)
    composite[..., 5] = adjacency.sum(dim=-1)
    composite[..., 8] = (
        torch.einsum("bmn,bn->bm", adjacency.float(), copied.float())
        / adjacency.sum(dim=-1).clamp_min(1)
    )
    composite_availability = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    has_daughters = adjacency.any(dim=-1)
    for index in (0, 1, 2, 3, 4, 5, 8):
        composite_availability[..., index] = has_daughters
    batch["composite_features"] = composite
    batch["composite_availability"] = composite_availability
    histogram = p4.new_zeros((*active.shape, len(PDG_TOKENS)))
    daughter_tokens = (
        batch["pid_labels"][:, None, :]
        .expand(-1, active.shape[1], -1)
    )
    validate_pid_tokens(batch["pid_labels"][active], name="upgraded batch PID labels")
    histogram.scatter_add_(-1, daughter_tokens, adjacency.float())
    batch["daughter_pid_histogram"] = histogram
    batch["daughter_pid_histogram_available"] = has_daughters
    if "node_kind_ids" in batch:
        kinds = batch["node_kind_ids"]
    else:
        kinds = torch.full_like(levels, NODE_KIND_TO_ID["unknown"])
        kinds[levels > 0] = NODE_KIND_TO_ID["composite"]
        kinds[(levels == 0) & (charge != 0)] = NODE_KIND_TO_ID["track"]
    batch["node_kind_ids"] = kinds
    default_ids = torch.arange(
        active.shape[1],
        device=p4.device,
        dtype=torch.long,
    )[None].expand_as(levels)
    batch.setdefault("node_ids", default_ids)
    batch.setdefault("reco_ids", torch.full_like(levels, -1))
    batch.setdefault("source_node_ids", default_ids)
    batch.setdefault("copied_from", torch.full_like(levels, -1))
    batch.setdefault("active", active)
    batch.setdefault("b_side", torch.full_like(levels, -1))
    return batch
