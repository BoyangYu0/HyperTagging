"""Relation-aware attention bias features."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.models.hyperbolic import hyperbolic_pairwise_distance, radius


class RelationBias(nn.Module):
    """Learn a pairwise attention bias from level, charge, p4, and hyperbolic distance."""

    def __init__(self, hidden_dim: int = 32, *, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.net = nn.Sequential(nn.Linear(11, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

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
        d_h = hyperbolic_pairwise_distance(z_hyperbolic, node_mask)
        level_diff = (level_ids[:, :, None] - level_ids[:, None, :]).float()
        same_level = (level_ids[:, :, None] == level_ids[:, None, :]).float()
        charge_sum = charge[:, :, None] + charge[:, None, :]
        pair_p4 = p4[:, :, None, :] + p4[:, None, :, :]
        mass2 = pair_p4[..., 3] ** 2 - (pair_p4[..., :3] ** 2).sum(dim=-1)
        mass = torch.sqrt(torch.clamp(mass2, min=0.0))
        radii = radius(z_hyperbolic)
        radius_i = radii[:, :, None].expand_as(d_h)
        radius_j = radii[:, None, :].expand_as(d_h)
        if node_kind_ids is None:
            same_kind = torch.zeros_like(d_h)
            kind_pair = torch.zeros_like(d_h)
        else:
            same_kind = (node_kind_ids[:, :, None] == node_kind_ids[:, None, :]).float()
            kind_pair = (
                node_kind_ids[:, :, None].float() + node_kind_ids[:, None, :].float()
            ) / 8.0
        copied_conflict = torch.zeros_like(d_h)
        if copied is not None and source_node_ids is not None:
            same_source = source_node_ids[:, :, None] == source_node_ids[:, None, :]
            copied_conflict = (same_source & (copied[:, :, None] | copied[:, None, :])).float()
        features = torch.stack(
            [
                d_h,
                radius_i,
                radius_j,
                level_diff,
                same_level,
                charge_sum,
                mass,
                pair_p4[..., 3],
                same_kind,
                kind_pair,
                copied_conflict,
            ],
            dim=-1,
        )
        bias = self.net(features).squeeze(-1) if self.enabled else torch.zeros_like(d_h)
        return bias.masked_fill(~(node_mask[:, :, None] & node_mask[:, None, :]), -1e4)
