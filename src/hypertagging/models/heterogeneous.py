"""Type-specific detector/composite frontends with one shared latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.hyperbolic import expmap0
from hypertagging.preprocessing.schema_v2 import (
    CLUSTER_FEATURE_NAMES,
    COMMON_FEATURE_NAMES,
    COMPOSITE_FEATURE_NAMES,
    NODE_KINDS,
    TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS


def masked_mean_pool(
    embeddings: torch.Tensor,
    daughter_adjacency: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant daughter pooling for every possible mother."""

    weights = daughter_adjacency.to(embeddings.dtype)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("bmn,bnd->bmd", weights, embeddings) / denominator


class _MaskedBlockEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2 * n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        clean = torch.nan_to_num(values)
        masked = torch.where(availability, clean, torch.zeros_like(clean))
        return self.projection(torch.cat([masked, availability.to(clean.dtype)], dim=-1))


class CommonNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(COMMON_FEATURE_NAMES), d_model)


class TrackNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(TRACK_FEATURE_NAMES), d_model)


class ClusterNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(CLUSTER_FEATURE_NAMES), d_model)


class CompositeNodeEncoder(nn.Module):
    """Encode reco-derived structure together with a pooled daughter summary."""

    def __init__(self, d_model: int, n_pid: int = len(PDG_TOKENS)) -> None:
        super().__init__()
        self.structural = _MaskedBlockEncoder(len(COMPOSITE_FEATURE_NAMES), d_model)
        self.pid_histogram = nn.Sequential(
            nn.Linear(n_pid + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.combine = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        daughter_summary: torch.Tensor,
        daughter_pid_histogram: torch.Tensor,
        histogram_available: torch.Tensor,
    ) -> torch.Tensor:
        structural = self.structural(features, availability)
        histogram = torch.nan_to_num(daughter_pid_histogram)
        histogram = torch.where(
            histogram_available.unsqueeze(-1),
            histogram,
            torch.zeros_like(histogram),
        )
        histogram = histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1.0)
        histogram_h = self.pid_histogram(
            torch.cat([histogram, histogram_available.unsqueeze(-1).to(histogram.dtype)], dim=-1)
        )
        return self.combine(torch.cat([structural, daughter_summary, histogram_h], dim=-1))


@dataclass(frozen=True)
class HeterogeneousEncoderOutput:
    node_embeddings: torch.Tensor
    hyperbolic_embeddings: torch.Tensor
    tree_projection: torch.Tensor
    reconstruction_projection: torch.Tensor
    channel_projection: torch.Tensor
    daughter_summary: torch.Tensor


class HeterogeneousNodeEncoder(nn.Module):
    """Different frontends, shared normalization/context space and Poincare ball."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        hyper_dim: int = 16,
        n_pid: int = 4096,
        max_level: int = 32,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.curvature = curvature
        self.common_encoder = CommonNodeEncoder(d_model)
        self.track_encoder = TrackNodeEncoder(d_model)
        self.cluster_encoder = ClusterNodeEncoder(d_model)
        self.composite_encoder = CompositeNodeEncoder(d_model)
        self.other_encoder = nn.Parameter(torch.zeros(d_model))
        self.pid_embedding = nn.Embedding(n_pid, d_model)
        self.node_kind_embedding = nn.Embedding(len(NODE_KINDS), d_model)
        self.level_embedding = nn.Embedding(max_level + 2, d_model)
        availability_width = (
            len(COMMON_FEATURE_NAMES)
            + len(TRACK_FEATURE_NAMES)
            + len(CLUSTER_FEATURE_NAMES)
            + len(COMPOSITE_FEATURE_NAMES)
            + 1
        )
        self.availability_encoder = nn.Linear(availability_width, d_model)
        self.shared_norm = nn.LayerNorm(d_model)
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.tree_head = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, d_model)
        self.channel_head = nn.Linear(d_model, d_model)
        self.hyper_projection = nn.Linear(d_model, hyper_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> HeterogeneousEncoderOutput:
        common = self.common_encoder(batch["common_features"], batch["common_availability"])
        track = self.track_encoder(batch["track_features"], batch["track_availability"])
        cluster = self.cluster_encoder(batch["cluster_features"], batch["cluster_availability"])
        kinds = batch["node_kind_ids"].clamp(0, len(NODE_KINDS) - 1)
        pid = batch["pid_labels"].abs().clamp(0, self.pid_embedding.num_embeddings - 1)
        levels = batch["level_ids"].clamp(0, self.level_embedding.num_embeddings - 1)
        availability = torch.cat(
            [
                batch["common_availability"],
                batch["track_availability"],
                batch["cluster_availability"],
                batch["composite_availability"],
                batch["daughter_pid_histogram_available"].unsqueeze(-1),
            ],
            dim=-1,
        ).to(common.dtype)

        specific = self.other_encoder.view(1, 1, -1).expand_as(common).clone()
        specific = torch.where((kinds == 1).unsqueeze(-1), track, specific)
        specific = torch.where((kinds == 2).unsqueeze(-1), cluster, specific)
        pre_composite = self.shared_norm(
            common
            + specific
            + self.pid_embedding(pid)
            + self.node_kind_embedding(kinds)
            + self.level_embedding(levels)
            + self.availability_encoder(availability)
        )
        daughter_summary = masked_mean_pool(pre_composite, batch["daughter_adjacency"])
        composite = self.composite_encoder(
            batch["composite_features"],
            batch["composite_availability"],
            daughter_summary,
            batch["daughter_pid_histogram"],
            batch["daughter_pid_histogram_available"],
        )
        specific = torch.where((kinds == 3).unsqueeze(-1), composite, specific)
        h0 = self.shared_norm(
            common
            + specific
            + self.pid_embedding(pid)
            + self.node_kind_embedding(kinds)
            + self.level_embedding(levels)
            + self.availability_encoder(availability)
        )
        h = self.shared_norm(h0 + self.shared_mlp(h0))
        h = h * batch["node_mask"].unsqueeze(-1)
        tree = self.tree_head(h)
        reconstruction = self.reconstruction_head(h)
        channel = self.channel_head(h)
        z = expmap0(self.hyper_projection(tree), curvature=self.curvature)
        z = z * batch["node_mask"].unsqueeze(-1)
        return HeterogeneousEncoderOutput(h, z, tree, reconstruction, channel, daughter_summary)


def composite_token_from_daughters(
    *,
    daughter_mask: torch.Tensor,
    p4: torch.Tensor,
    charge: torch.Tensor,
    pid_labels: torch.Tensor,
    daughter_embeddings: torch.Tensor,
    pointer_confidence: torch.Tensor | None = None,
    copied: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Shared truth-guided/predicted composite construction for identical links."""

    weights = daughter_mask.to(p4.dtype)
    summed_p4 = torch.einsum("bn,bnf->bf", weights, p4)
    summed_charge = torch.einsum("bn,bn->b", weights, charge)
    count = weights.sum(dim=-1)
    pooled = torch.einsum("bn,bnd->bd", weights, daughter_embeddings) / count.unsqueeze(-1).clamp_min(1)
    confidence = (
        torch.where(daughter_mask, pointer_confidence, torch.ones_like(pointer_confidence))
        if pointer_confidence is not None
        else None
    )
    confidence_mean = (
        torch.einsum("bn,bn->b", weights, pointer_confidence) / count.clamp_min(1)
        if pointer_confidence is not None
        else torch.zeros_like(count)
    )
    confidence_min = (
        confidence.min(dim=-1).values if confidence is not None else torch.zeros_like(count)
    )
    copied_fraction = (
        torch.einsum("bn,bn->b", weights, copied.to(weights.dtype)) / count.clamp_min(1)
        if copied is not None
        else torch.zeros_like(count)
    )
    features = torch.stack(
        [
            summed_p4[:, 0],
            summed_p4[:, 1],
            summed_p4[:, 2],
            summed_p4[:, 3],
            summed_charge,
            count,
            confidence_mean,
            confidence_min,
            copied_fraction,
        ],
        dim=-1,
    )
    availability = torch.ones_like(features, dtype=torch.bool)
    if pointer_confidence is None:
        availability[:, 6:8] = False
    histogram = torch.zeros(
        (*pid_labels.shape[:-1], len(PDG_TOKENS)),
        dtype=p4.dtype,
        device=p4.device,
    )
    histogram.scatter_add_(
        -1,
        pid_labels.clamp(0, len(PDG_TOKENS) - 1),
        weights,
    )
    return {
        "p4": summed_p4,
        "charge": summed_charge,
        "features": features,
        "availability": availability,
        "daughter_summary": pooled,
        "daughter_pid_histogram": histogram,
        "daughter_pid_histogram_available": count > 0,
    }


__all__ = [
    "ClusterNodeEncoder",
    "CommonNodeEncoder",
    "CompositeNodeEncoder",
    "HeterogeneousEncoderOutput",
    "HeterogeneousNodeEncoder",
    "TrackNodeEncoder",
    "composite_token_from_daughters",
    "masked_mean_pool",
]
