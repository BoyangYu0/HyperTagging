"""Experimental unordered B-membership head; separate from production hierarchy.

Two optional groups compete with an unassigned source class. Truth memberships
enter only the loss. The frozen-encoder pilot does not imply full-model quality.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirectMembershipHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.nodes = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.queries = nn.Parameter(torch.randn(2, hidden_dim) * .02)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.refine = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.background = nn.Linear(hidden_dim, 1)
        self.object_head = nn.Linear(hidden_dim, 1)

    def forward(self, features, node_mask):
        if not bool(node_mask.any(dim=1).all()):
            raise ValueError("direct membership requires at least one source per event")
        h = self.nodes(features)
        pooled = (h * node_mask[..., None]).sum(1) / node_mask.sum(1, keepdim=True)
        q = self.queries[None] + pooled[:, None]
        attended, _ = self.attention(q, h, h, key_padding_mask=~node_mask, need_weights=False)
        q = self.norm(q + attended)
        q = self.norm(q + self.refine(q))
        membership = torch.einsum("bnd,bkd->bnk", h, q) / h.shape[-1] ** .5
        return torch.cat((self.background(h), membership), -1), self.object_head(q).squeeze(-1)


def direct_membership_loss(logits, objects, targets, node_mask):
    """Exact two-slot permutation matching, including empty continuum targets."""
    alternatives = []
    valid = node_mask & (targets >= 0)
    probability = logits.softmax(-1)[..., 1:].transpose(1, 2)
    for swap in (False, True):
        labels = torch.where(targets > 0, 3 - targets, targets) if swap else targets
        safe = labels.clamp_min(0)
        target_masks = torch.stack((labels == 1, labels == 2), 1) & valid[:, None]
        present = target_masks.any(-1)
        ce = F.cross_entropy(logits.transpose(1, 2), safe, reduction="none")
        ce = (ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        intersection = (probability * target_masks).sum(-1)
        union = ((probability + target_masks - probability * target_masks) * valid[:, None]).sum(-1)
        overlap = ((1 - (intersection + 1e-6) / (union + 1e-6)) * present).sum(-1) / present.sum(-1).clamp_min(1)
        presence = F.binary_cross_entropy_with_logits(objects, present.float(), reduction="none").mean(-1)
        alternatives.append(ce + overlap + presence)
    loss = torch.stack(alternatives).min(0).values
    available = valid.any(-1)
    return (loss * available).sum() / available.sum().clamp_min(1)
