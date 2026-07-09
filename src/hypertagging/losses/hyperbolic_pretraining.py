"""Dense hyperbolic pretraining losses for topology supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hypertagging.models.hyperbolic import logmap0, radius


@dataclass(frozen=True)
class HyperbolicLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def same_pair_bce(pair_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask[:, :, None] & mask[:, None, :]
    return F.binary_cross_entropy_with_logits(pair_logits[pair_mask], labels.float()[pair_mask])


def parent_child_margin_loss(
    z: torch.Tensor,
    parent_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    tangent = logmap0(z)
    d = torch.cdist(tangent, tangent)
    d = d.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.0)
    losses = []
    for batch_index in range(z.shape[0]):
        valid = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        for child in valid.tolist():
            parent = int(parent_ids[batch_index, child])
            if parent < 0:
                continue
            negatives = valid[(valid != parent) & (valid != child)]
            if negatives.numel() == 0:
                continue
            neg_dist = d[batch_index, child, negatives].min()
            losses.append(F.relu(d[batch_index, child, parent] + margin - neg_dist))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def radius_depth_loss(z: torch.Tensor, level_ids: torch.Tensor, mask: torch.Tensor, *, scale: float = 0.15) -> torch.Tensor:
    target = (level_ids.float().clamp_min(0) + 1.0) * scale
    pred = radius(z)
    return F.mse_loss(pred[mask], target[mask])


def hyperbolic_pretraining_loss(
    *,
    z: torch.Tensor,
    same_mother_logits: torch.Tensor,
    same_branch_logits: torch.Tensor,
    same_mother: torch.Tensor,
    same_branch: torch.Tensor,
    parent_ids: torch.Tensor,
    level_ids: torch.Tensor,
    node_mask: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> HyperbolicLossOutput:
    weights = {"same_mother": 1.0, "same_branch": 1.0, "parent": 1.0, "radius": 0.2, **(weights or {})}
    components = {
        "same_mother": same_pair_bce(same_mother_logits, same_mother, node_mask),
        "same_branch": same_pair_bce(same_branch_logits, same_branch, node_mask),
        "parent": parent_child_margin_loss(z, parent_ids, node_mask),
        "radius": radius_depth_loss(z, level_ids, node_mask),
    }
    total = sum(components[name] * weights[name] for name in components)
    return HyperbolicLossOutput(total=total, components=components)
