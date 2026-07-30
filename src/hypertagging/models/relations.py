"""Relation-aware attention bias features."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.models.hyperbolic import hyperbolic_pairwise_distance, radius


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
    if node_kind_ids is None:
        same_kind = torch.zeros_like(level_diff)
        kind_pair = torch.zeros_like(level_diff)
    else:
        same_kind = (node_kind_ids[:, :, None] == node_kind_ids[:, None, :]).float()
        kind_pair = (
            node_kind_ids[:, :, None].float() + node_kind_ids[:, None, :].float()
        ) / 8.0
    copied_conflict = torch.zeros_like(level_diff)
    if copied is not None and source_node_ids is not None:
        same_source = source_node_ids[:, :, None] == source_node_ids[:, None, :]
        copied_conflict = (same_source & (copied[:, :, None] | copied[:, None, :])).float()
    return torch.stack(
        [
            level_diff,
            same_level,
            charge_sum,
            mass,
            pair_p4[..., 3],
            momentum_dot,
            same_kind,
            kind_pair,
            copied_conflict,
        ],
        dim=-1,
    )


class PhysicalRelationBias(nn.Module):
    """Stage-A relation bias using only data-compatible physical inputs."""

    def __init__(self, hidden_dim: int = 32, *, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.net = nn.Sequential(nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

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
        bias = self.net(features).squeeze(-1) if self.enabled else features[..., 0] * 0.0
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
        tangent_dot = torch.einsum("bif,bjf->bij", z_hyperbolic, z_hyperbolic)
        features = torch.stack(
            [
                d_h,
                radii[:, :, None].expand_as(d_h),
                radii[:, None, :].expand_as(d_h),
                tangent_dot,
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


__all__ = ["HyperbolicRelationBias", "PhysicalRelationBias", "RelationBias"]
