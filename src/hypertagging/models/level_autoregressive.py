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
from hypertagging.preprocessing.schema_v2 import (
    CLUSTER_FEATURE_NAMES,
    COMMON_FEATURE_NAMES,
    COMPOSITE_FEATURE_NAMES,
    NODE_KIND_TO_ID,
    TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS


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
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_mode = encoder_mode
        if encoder_mode == "flat":
            self.encoder: nn.Module = HyperbolicNodeEncoder(
                n_features=n_features,
                n_pid=max(n_types, 4096),
                hidden_dim=hidden_dim,
                hyper_dim=hyper_dim,
            )
        elif encoder_mode == "heterogeneous":
            self.encoder = HeterogeneousNodeEncoder(
                d_model=hidden_dim,
                hyper_dim=hyper_dim,
                n_pid=max(n_types, 4096),
            )
        else:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}")
        self.relation_bias = RelationBias(hidden_dim=hidden_dim, enabled=use_relation_bias)
        self.contextualizer = RelationAwareSetTransformer(
            hidden_dim,
            n_heads=n_heads,
            n_layers=n_context_layers,
        )
        self.decoder = MotherPointerDecoder(hidden_dim=hidden_dim, n_types=n_types, n_queries=n_queries)

    def forward(self, batch: dict[str, torch.Tensor], *, target_level: int = 1) -> LevelReconstructionOutput:
        if self.encoder_mode == "heterogeneous":
            batch = _upgrade_flat_batch(batch)
            encoded = self.encoder(batch)
            h = encoded.node_embeddings
            z = encoded.hyperbolic_embeddings
            tree_projection = encoded.tree_projection
            reconstruction_projection = encoded.reconstruction_projection
            channel_projection = encoded.channel_projection
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
        context_mask = context_mask_for_level(batch["level_ids"], batch["node_mask"], target_level)
        relation_bias = self.relation_bias(
            p4=batch["p4"],
            charge=batch["charge"],
            level_ids=batch["level_ids"],
            z_hyperbolic=z,
            node_mask=batch["node_mask"],
            node_kind_ids=batch.get("node_kind_ids"),
            copied=batch.get("copied"),
            source_node_ids=batch.get("source_node_ids"),
        )
        h, attention_weights = self.contextualizer(
            reconstruction_projection,
            relation_bias=relation_bias,
            attention_mask=stair_attention_mask(batch["level_ids"], batch["node_mask"]),
            node_mask=batch["node_mask"],
        )
        pointer = self.decoder(h, context_mask)
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
        .clamp(0, len(PDG_TOKENS) - 1)
    )
    histogram.scatter_add_(-1, daughter_tokens, adjacency.float())
    batch["daughter_pid_histogram"] = histogram
    batch["daughter_pid_histogram_available"] = has_daughters
    kinds = torch.full_like(levels, NODE_KIND_TO_ID["unknown"])
    kinds[levels > 0] = NODE_KIND_TO_ID["composite"]
    kinds[(levels == 0) & (charge != 0)] = NODE_KIND_TO_ID["track"]
    kinds[(levels == 0) & (batch["pid_labels"] == 22)] = NODE_KIND_TO_ID["ecl_cluster"]
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
