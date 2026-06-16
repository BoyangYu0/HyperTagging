"""Link-prediction losses migrated from historical scripts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def link_metrics(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy link loss and argmax accuracy under a boolean mask."""

    link_loss = F.cross_entropy(input=pred[mask], target=goal[mask])
    link_acc = (torch.argmax(pred, dim=-1)[mask] == goal[mask]).float().mean()
    return link_loss, link_acc


def link_cross_entropy(
    logits: torch.Tensor,
    links: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Historical link-prediction cross-entropy term."""

    return F.cross_entropy(input=logits[padding_mask], target=links[padding_mask])


def transfer_link_metrics(
    pred: torch.Tensor,
    goal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Historical reconstruction transfer-link MSE and argmax agreement."""

    link_loss = F.mse_loss(input=pred[mask], target=goal[mask])
    link_acc = (torch.argmax(pred, dim=-1)[mask] == torch.argmax(goal, dim=-1)[mask]).float().mean()
    return link_loss, link_acc
