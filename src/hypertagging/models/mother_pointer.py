"""Pointer-network style mother-node prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.preprocessing.pid_filter import PDG_TOKENS


@dataclass(frozen=True)
class MotherPointerOutput:
    object_logits: torch.Tensor
    type_logits: torch.Tensor
    pointer_logits: torch.Tensor
    cardinality_logits: torch.Tensor
    confidence_logits: torch.Tensor
    expected_type_embedding: torch.Tensor | None = None


class MotherPointerDecoder(nn.Module):
    """Decode unordered mother query slots into symbolic reconstruction decisions."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_types: int,
        max_cardinality: int = 6,
        n_queries: int = 8,
        max_level: int = 32,
    ) -> None:
        super().__init__()
        if n_types != len(PDG_TOKENS):
            n_types = len(PDG_TOKENS)
        self.n_queries = n_queries
        self.max_cardinality = max_cardinality
        self.query = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.target_level_embedding = nn.Embedding(max_level + 1, hidden_dim)
        self.type_embedding = nn.Embedding(n_types, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, n_types)
        self.cardinality_head = nn.Linear(hidden_dim, max_cardinality + 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.pointer_query = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        target_level: int,
        allowed_type_mask: torch.Tensor | None = None,
        pointer_validity_mask: torch.Tensor | None = None,
    ) -> MotherPointerOutput:
        if target_level < 0 or target_level >= self.target_level_embedding.num_embeddings:
            raise ValueError(f"target_level={target_level} is outside decoder level capacity")
        batch_size = context.shape[0]
        level = self.target_level_embedding.weight[target_level]
        queries = (self.query + level).unsqueeze(0).expand(batch_size, -1, -1)
        attended, _weights = self.cross_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        type_logits = self.type_head(attended)
        if allowed_type_mask is not None:
            if allowed_type_mask.shape != (type_logits.shape[-1],):
                raise ValueError("allowed_type_mask must have one entry per reduced PID token")
            if not bool(allowed_type_mask.any()):
                raise ValueError("allowed_type_mask rejects every mother type")
            type_logits = type_logits.masked_fill(~allowed_type_mask[None, None, :], -1e4)
        expected_type = torch.softmax(type_logits, dim=-1) @ self.type_embedding.weight
        q = self.pointer_query(torch.cat([attended, expected_type], dim=-1))
        k = self.pointer_key(context)
        pointer_logits = torch.einsum("bqh,bnh->bqn", q, k) / (context.shape[-1] ** 0.5)
        pointer_logits = pointer_logits.masked_fill(~context_mask[:, None, :], -1e4)
        if pointer_validity_mask is not None:
            if pointer_validity_mask.shape != pointer_logits.shape:
                raise ValueError("pointer_validity_mask must match pointer logits")
            pointer_logits = pointer_logits.masked_fill(~pointer_validity_mask, -1e4)
        return MotherPointerOutput(
            object_logits=self.object_head(attended).squeeze(-1),
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=self.cardinality_head(attended),
            confidence_logits=self.confidence_head(attended).squeeze(-1),
            expected_type_embedding=expected_type,
        )
