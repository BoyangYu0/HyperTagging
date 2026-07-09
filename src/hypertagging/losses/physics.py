"""Physics consistency losses for daughter-sum reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hypertagging.models.level_autoregressive import construct_mother_p4


def invariant_mass(p4: torch.Tensor) -> torch.Tensor:
    mass2 = p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(dim=-1)
    return torch.sqrt(torch.clamp(mass2, min=0.0))


def soft_daughter_sum_p4(pointer_logits: torch.Tensor, daughter_p4: torch.Tensor) -> torch.Tensor:
    return construct_mother_p4(pointer_logits, daughter_p4, hard=False)


def p4_sum_consistency_loss(pointer_logits: torch.Tensor, daughter_p4: torch.Tensor, target_p4: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(soft_daughter_sum_p4(pointer_logits, daughter_p4), target_p4)


def charge_consistency_loss(pointer_logits: torch.Tensor, daughter_charge: torch.Tensor, target_charge: torch.Tensor) -> torch.Tensor:
    weights = torch.sigmoid(pointer_logits)
    pred = torch.einsum("bqn,bn->bq", weights, daughter_charge)
    return F.mse_loss(pred, target_charge)
