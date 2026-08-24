"""Reconstruction losses migrated from historical scripts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_class_weight(goal: torch.Tensor, mask: torch.Tensor, pad_lenth: int | None = None) -> torch.Tensor:
    """Historical class-weight helper, preserving the misspelled parameter name."""

    class_weight = F.normalize((1 / torch.bincount(goal[mask])).nan_to_num(posinf=1), dim=-1)
    if pad_lenth:
        class_weight = F.pad(class_weight, (0, pad_lenth - len(class_weight)), "constant", 1)
    return class_weight


def recover_pdg(pred: torch.Tensor) -> torch.Tensor:
    """Recover PDG class ids by argmax."""

    return torch.argmax(pred, dim=-1)


def pdg_metrics(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Historical PDG cross-entropy and accuracy."""

    pdg_loss = F.cross_entropy(input=pred[mask], target=goal[mask])
    pdg_acc = (torch.argmax(pred, dim=-1)[mask] == goal[mask]).float().mean()
    return pdg_loss, pdg_acc


def momentum_metrics(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
    *,
    spatial_weight: float = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Momentum MSE/MAE used in reconstruction scripts.

    `spatial_weight=3` matches `graFEI/Reconstruction.py`; use `1` for the
    reduced variant and `None` through `plain_momentum_metrics` for noR.
    """

    mse = F.mse_loss(input=pred[mask][..., :3], target=goal[mask][..., :3]) * spatial_weight + F.mse_loss(
        input=pred[mask][..., 3],
        target=goal[mask][..., 3],
    )
    mae = F.l1_loss(input=pred[mask], target=goal[mask])
    return mse, mae


def plain_momentum_metrics(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HyperTagging/noR reconstruction momentum MSE/MAE variant."""

    mse = F.mse_loss(input=pred[mask], target=goal[mask])
    mae = F.l1_loss(input=pred[mask], target=goal[mask])
    return mse, mae


def embedding_mse_distance(pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Historical noR embedding target distance after model embedding."""

    return F.mse_loss(input=pred, target=goal)


def embedding_cosine_distance(pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Historical GraFEI cosine embedding distance after model embedding."""

    return (1 - F.cosine_similarity(pred.nan_to_num(nan=1.0), goal)).mean()
