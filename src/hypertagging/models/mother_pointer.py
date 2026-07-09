"""Pointer-network style mother-node prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MotherPointerOutput:
    object_logits: torch.Tensor
    type_logits: torch.Tensor
    pointer_logits: torch.Tensor
    cardinality_logits: torch.Tensor
    confidence_logits: torch.Tensor


class MotherPointerDecoder(nn.Module):
    """Decode unordered mother query slots into symbolic reconstruction decisions."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_types: int,
        max_cardinality: int = 6,
        n_queries: int = 8,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, n_types)
        self.cardinality_head = nn.Linear(hidden_dim, max_cardinality + 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor) -> MotherPointerOutput:
        batch_size = context.shape[0]
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _weights = self.cross_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        q = self.pointer_query(attended)
        k = self.pointer_key(context)
        pointer_logits = torch.einsum("bqh,bnh->bqn", q, k) / (context.shape[-1] ** 0.5)
        pointer_logits = pointer_logits.masked_fill(~context_mask[:, None, :], -1e4)
        return MotherPointerOutput(
            object_logits=self.object_head(attended).squeeze(-1),
            type_logits=self.type_head(attended),
            pointer_logits=pointer_logits,
            cardinality_logits=self.cardinality_head(attended),
            confidence_logits=self.confidence_head(attended).squeeze(-1),
        )
