"""GPT-like/autoregressive losses migrated from historical scripts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hypertagging.losses.embedding_losses import EPSILON
from hypertagging.losses.link_losses import link_metrics


def distance(pred: torch.Tensor, goal: torch.Tensor, level_mask: torch.Tensor) -> torch.Tensor:
    """Historical graFEI_gpt reconstruction distance loss."""

    robust_loss = F.mse_loss(input=pred[level_mask], target=goal[level_mask])
    particle_mask = goal[level_mask].sum(dim=-1) != 0
    accurate_loss = F.l1_loss(input=pred[level_mask][particle_mask], target=goal[level_mask][particle_mask])
    return 10 * robust_loss + accurate_loss


def radius_loss(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Historical graFEI_gpt autoregressive radius loss."""

    r_euclidean = torch.norm(pred[mask], dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - epsilon))
    r_goal = 0.9 * torch.sqrt(1 - goal[mask] / 100) + 0.1
    return F.l1_loss(input=r_poincare, target=r_goal)


__all__ = ["distance", "link_metrics", "radius_loss"]
