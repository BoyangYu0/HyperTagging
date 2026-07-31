"""Bounded, disabled-by-default first-level set-decoder ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FirstLevelAmbiguityAblation:
    whole_set_compatibility_scorer: bool = False
    iterative_within_mother_pointer: bool = False
    type_conditioned_daughter_relation_bias: bool = False
    max_iterative_daughters: int = 12


class WholeSetCompatibilityScorer(nn.Module):
    """Score a proposed set after independent pointer probabilities."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        query: torch.Tensor,
        node_embeddings: torch.Tensor,
        proposal_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = proposal_mask.to(node_embeddings.dtype)
        pooled = torch.einsum("bqn,bnh->bqh", weights, node_embeddings)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1)
        maximum = node_embeddings[:, None].masked_fill(
            ~proposal_mask[..., None], float("-inf")
        ).amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.net(torch.cat([query, pooled, maximum], dim=-1)).squeeze(-1)


def iterative_pointer_mask(
    pointer_logits: torch.Tensor,
    *,
    cardinality: int,
    relation_bias: torch.Tensor | None = None,
    max_daughters: int = 12,
) -> torch.Tensor:
    """Greedy bounded within-mother decoding; never enumerates combinations."""

    if cardinality < 0 or cardinality > max_daughters:
        raise ValueError("iterative daughter cardinality exceeds the bounded ablation")
    selected = torch.zeros_like(pointer_logits, dtype=torch.bool)
    score = pointer_logits.clone()
    for _ in range(cardinality):
        position = score.masked_fill(selected, float("-inf")).argmax(dim=-1)
        selected.scatter_(1, position[:, None], True)
        if relation_bias is not None:
            score = score + relation_bias.gather(
                1, position[:, None, None].expand(-1, 1, score.shape[-1])
            ).squeeze(1)
    return selected


def type_conditioned_relation_bias(
    type_probabilities: torch.Tensor,
    type_relation_table: torch.Tensor,
) -> torch.Tensor:
    """Expected per-daughter bias under the predicted mother-type distribution."""

    return torch.einsum("bqt,tn->bqn", type_probabilities, type_relation_table)


__all__ = [
    "FirstLevelAmbiguityAblation",
    "WholeSetCompatibilityScorer",
    "iterative_pointer_mask",
    "type_conditioned_relation_bias",
]
