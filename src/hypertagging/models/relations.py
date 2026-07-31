"""Relation-aware attention bias features."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.models.hyperbolic import (
    hyperbolic_pairwise_distance,
    logmap0,
    radius,
)
from hypertagging.preprocessing.schema_v2 import NODE_KINDS


PHYSICAL_RELATION_SCALING_VERSION = "physical-relations-logscale-v1"


def _signed_log_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Compress dimensional physical values under an explicit fixed contract."""

    return torch.sign(value) * torch.log1p(value.abs() / float(scale))


def _physical_features(
    *,
    p4: torch.Tensor,
    charge: torch.Tensor,
    level_ids: torch.Tensor,
    node_kind_ids: torch.Tensor | None,
    copied: torch.Tensor | None,
    source_node_ids: torch.Tensor | None,
) -> torch.Tensor:
    level_diff = (level_ids[:, :, None] - level_ids[:, None, :]).float()
    same_level = (level_ids[:, :, None] == level_ids[:, None, :]).float()
    charge_sum = charge[:, :, None] + charge[:, None, :]
    pair_p4 = p4[:, :, None, :] + p4[:, None, :, :]
    mass2 = pair_p4[..., 3] ** 2 - (pair_p4[..., :3] ** 2).sum(dim=-1)
    mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    momentum_dot = torch.einsum("bif,bjf->bij", p4[..., :3], p4[..., :3])
    same_kind = (
        torch.zeros_like(level_diff)
        if node_kind_ids is None
        else (node_kind_ids[:, :, None] == node_kind_ids[:, None, :]).float()
    )
    copied_conflict = torch.zeros_like(level_diff)
    if copied is not None and source_node_ids is not None:
        same_source = source_node_ids[:, :, None] == source_node_ids[:, None, :]
        copied_conflict = (same_source & (copied[:, :, None] | copied[:, None, :])).float()
    return torch.stack(
        [
            level_diff / 8.0,
            same_level,
            charge_sum / 4.0,
            _signed_log_scale(mass, 1.0),
            _signed_log_scale(pair_p4[..., 3], 1.0),
            _signed_log_scale(momentum_dot, 1.0),
            same_kind,
            copied_conflict,
        ],
        dim=-1,
    )


class PhysicalRelationBias(nn.Module):
    """Stage-A relation bias using only data-compatible physical inputs.

    Continuous GeV-valued features follow
    ``physical-relations-logscale-v1``: level and charge use fixed divisors;
    mass, energy and momentum-dot use signed ``log1p(abs(x)/1 GeV^k)``.
    Binary indicators stay in {0,1}. Node kinds use a collision-free symmetric
    pair embedding rather than a summed scalar code.
    """

    def __init__(self, hidden_dim: int = 32, *, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.scaling_version = PHYSICAL_RELATION_SCALING_VERSION
        pair_dim = max(4, hidden_dim // 4)
        self.pair_kind_embedding = nn.Embedding(len(NODE_KINDS) ** 2, pair_dim)
        self.net = nn.Sequential(
            nn.Linear(8 + pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        *,
        p4: torch.Tensor,
        charge: torch.Tensor,
        level_ids: torch.Tensor,
        node_mask: torch.Tensor,
        node_kind_ids: torch.Tensor | None = None,
        copied: torch.Tensor | None = None,
        source_node_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = _physical_features(
            p4=p4,
            charge=charge,
            level_ids=level_ids,
            node_kind_ids=node_kind_ids,
            copied=copied,
            source_node_ids=source_node_ids,
        )
        if node_kind_ids is None:
            pair_embedding = features.new_zeros((*features.shape[:-1], self.pair_kind_embedding.embedding_dim))
        else:
            left = node_kind_ids[:, :, None].clamp(0, len(NODE_KINDS) - 1)
            right = node_kind_ids[:, None, :].clamp(0, len(NODE_KINDS) - 1)
            lower = torch.minimum(left, right)
            upper = torch.maximum(left, right)
            pair_embedding = self.pair_kind_embedding(lower * len(NODE_KINDS) + upper)
        combined = torch.cat([features, pair_embedding], dim=-1)
        bias = self.net(combined).squeeze(-1) if self.enabled else features[..., 0] * 0.0
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        return bias.masked_fill(~valid, -1e4)


class HyperbolicRelationBias(nn.Module):
    """Optional Stage-B bias after contextual hyperbolic projection."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        enabled: bool = True,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.curvature = curvature
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, *, z_hyperbolic: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        d_h = hyperbolic_pairwise_distance(
            z_hyperbolic,
            node_mask,
            curvature=self.curvature,
        )
        radii = radius(z_hyperbolic, curvature=self.curvature)
        tangent = logmap0(z_hyperbolic, curvature=self.curvature)
        tangent_dot = torch.einsum("bif,bjf->bij", tangent, tangent)
        features = torch.stack(
            [
                torch.log1p(d_h),
                torch.log1p(radii[:, :, None].expand_as(d_h)),
                torch.log1p(radii[:, None, :].expand_as(d_h)),
                _signed_log_scale(tangent_dot, 1.0),
            ],
            dim=-1,
        )
        bias = self.net(features).squeeze(-1) if self.enabled else d_h * 0.0
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        return bias.masked_fill(~valid, -1e4)


class RelationBias(nn.Module):
    """Learn a pairwise attention bias from level, charge, p4, and hyperbolic distance."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        enabled: bool = True,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.curvature = curvature
        self.physical = PhysicalRelationBias(hidden_dim, enabled=enabled)
        self.hyperbolic = HyperbolicRelationBias(
            hidden_dim,
            enabled=enabled,
            curvature=curvature,
        )

    def forward(
        self,
        *,
        p4: torch.Tensor,
        charge: torch.Tensor,
        level_ids: torch.Tensor,
        z_hyperbolic: torch.Tensor,
        node_mask: torch.Tensor,
        node_kind_ids: torch.Tensor | None = None,
        copied: torch.Tensor | None = None,
        source_node_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.physical(
            p4=p4,
            charge=charge,
            level_ids=level_ids,
            node_mask=node_mask,
            node_kind_ids=node_kind_ids,
            copied=copied,
            source_node_ids=source_node_ids,
        ) + self.hyperbolic(z_hyperbolic=z_hyperbolic, node_mask=node_mask)


__all__ = [
    "HyperbolicRelationBias",
    "PHYSICAL_RELATION_SCALING_VERSION",
    "PhysicalRelationBias",
    "RelationBias",
]
