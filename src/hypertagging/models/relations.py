"""Relation-aware attention bias features."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.models.hyperbolic import hyperbolic_pairwise_distance


class RelationBias(nn.Module):
    """Learn a pairwise attention bias from level, charge, p4, and hyperbolic distance."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        *,
        p4: torch.Tensor,
        charge: torch.Tensor,
        level_ids: torch.Tensor,
        z_hyperbolic: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        d_h = hyperbolic_pairwise_distance(z_hyperbolic, node_mask)
        level_diff = (level_ids[:, :, None] - level_ids[:, None, :]).float()
        same_level = (level_ids[:, :, None] == level_ids[:, None, :]).float()
        charge_sum = charge[:, :, None] + charge[:, None, :]
        pair_p4 = p4[:, :, None, :] + p4[:, None, :, :]
        mass2 = pair_p4[..., 3] ** 2 - (pair_p4[..., :3] ** 2).sum(dim=-1)
        mass = torch.sqrt(torch.clamp(mass2, min=0.0))
        features = torch.stack([d_h, level_diff, same_level, charge_sum, mass, pair_p4[..., 3]], dim=-1)
        bias = self.net(features).squeeze(-1)
        return bias.masked_fill(~(node_mask[:, :, None] & node_mask[:, None, :]), -1e4)
