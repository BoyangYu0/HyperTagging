"""Pointer-network style mother-node prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.first_level_ablations import type_conditioned_relation_bias
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, STATIC_MOTHER_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KINDS


@dataclass(frozen=True)
class MotherPointerOutput:
    object_logits: torch.Tensor
    type_logits: torch.Tensor
    pointer_logits: torch.Tensor
    cardinality_logits: torch.Tensor
    confidence_logits: torch.Tensor
    expected_type_embedding: torch.Tensor | None = None
    query_node_compatibility_bias: torch.Tensor | None = None


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
        type_conditioned_daughter_relation_bias: bool = False,
    ) -> None:
        super().__init__()
        if n_types != len(PDG_TOKENS):
            n_types = len(PDG_TOKENS)
        self.n_queries = n_queries
        self.max_cardinality = max_cardinality
        self.type_conditioned_daughter_relation_bias = bool(
            type_conditioned_daughter_relation_bias
        )
        self.query = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.target_level_embedding = nn.Embedding(max_level + 1, hidden_dim)
        self.type_embedding = nn.Embedding(n_types, hidden_dim)
        self.query_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=1, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, n_types)
        self.cardinality_head = nn.Linear(hidden_dim, max_cardinality + 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.pointer_query = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        if self.type_conditioned_daughter_relation_bias:
            self.type_relation_table = nn.Parameter(torch.zeros(n_types, n_types))
            self.compatibility_kind_embedding = nn.Embedding(
                len(NODE_KINDS), hidden_dim
            )
            self.compatibility_query = nn.Linear(3 * hidden_dim, hidden_dim)
            self.compatibility_node = nn.Linear(3 * hidden_dim + 3, hidden_dim)
        else:
            self.register_parameter("type_relation_table", None)
            self.compatibility_kind_embedding = None
            self.compatibility_query = None
            self.compatibility_node = None
        static_mask = torch.zeros(n_types, dtype=torch.bool)
        static_mask[list(STATIC_MOTHER_TOKENS)] = True
        self.register_buffer("static_mother_type_mask", static_mask)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        target_level: int,
        allowed_type_mask: torch.Tensor | None = None,
        type_logit_bias: torch.Tensor | None = None,
        pointer_validity_mask: torch.Tensor | None = None,
        node_pid_probabilities: torch.Tensor | None = None,
        node_charge: torch.Tensor | None = None,
        node_kind_ids: torch.Tensor | None = None,
        node_level_ids: torch.Tensor | None = None,
        node_relation_summary: torch.Tensor | None = None,
    ) -> MotherPointerOutput:
        if target_level < 0 or target_level >= self.target_level_embedding.num_embeddings:
            raise ValueError(f"target_level={target_level} is outside decoder level capacity")
        batch_size = context.shape[0]
        level = self.target_level_embedding.weight[target_level]
        queries = (self.query + level).unsqueeze(0).expand(batch_size, -1, -1)
        interacted, _weights = self.query_self_attention(
            queries, queries, queries, need_weights=False
        )
        queries = self.query_norm(queries + interacted)
        attended, _weights = self.cross_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        attended = self.cross_norm(queries + attended)
        attended = self.output_norm(attended + self.query_ffn(attended))
        type_logits = self.type_head(attended)
        if type_logit_bias is not None:
            if type_logit_bias.shape != (type_logits.shape[-1],):
                raise ValueError("type_logit_bias must have one entry per reduced PID token")
            type_logits = type_logits + type_logit_bias[None, None, :]
        effective_allowed = self.static_mother_type_mask
        if allowed_type_mask is not None:
            if allowed_type_mask.shape != (type_logits.shape[-1],):
                raise ValueError("allowed_type_mask must have one entry per reduced PID token")
            effective_allowed = effective_allowed & allowed_type_mask
            if not bool(effective_allowed.any()):
                raise ValueError("allowed_type_mask rejects every mother type")
        type_logits = type_logits.masked_fill(~effective_allowed[None, None, :], -1e4)
        expected_type = torch.softmax(type_logits, dim=-1) @ self.type_embedding.weight
        q = self.pointer_query(torch.cat([attended, expected_type], dim=-1))
        k = self.pointer_key(context)
        pointer_logits = torch.einsum("bqh,bnh->bqn", q, k) / (context.shape[-1] ** 0.5)
        compatibility_bias = None
        if self.type_conditioned_daughter_relation_bias:
            required = {
                "node_pid_probabilities": node_pid_probabilities,
                "node_charge": node_charge,
                "node_kind_ids": node_kind_ids,
                "node_level_ids": node_level_ids,
                "node_relation_summary": node_relation_summary,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "type-conditioned relation bias requires " + ", ".join(missing)
                )
            assert node_pid_probabilities is not None
            assert node_charge is not None
            assert node_kind_ids is not None
            assert node_level_ids is not None
            assert node_relation_summary is not None
            if node_pid_probabilities.shape != (
                batch_size, context.shape[1], type_logits.shape[-1]
            ):
                raise ValueError("node_pid_probabilities has an invalid shape")
            assert self.type_relation_table is not None
            assert self.compatibility_kind_embedding is not None
            assert self.compatibility_query is not None
            assert self.compatibility_node is not None
            type_probabilities = torch.softmax(type_logits, dim=-1)
            node_type = node_pid_probabilities.to(self.type_embedding.weight.dtype)
            node_type_embedding = node_type @ self.type_embedding.weight
            kind_embedding = self.compatibility_kind_embedding(
                node_kind_ids.clamp(0, len(NODE_KINDS) - 1)
            )
            scalar_features = torch.stack(
                [
                    torch.tanh(node_charge.to(context.dtype)),
                    node_level_ids.to(context.dtype)
                    / float(self.target_level_embedding.num_embeddings - 1),
                    torch.tanh(node_relation_summary.to(context.dtype)),
                ],
                dim=-1,
            )
            compatibility_q = self.compatibility_query(
                torch.cat(
                    [
                        attended,
                        expected_type,
                        level.view(1, 1, -1).expand(batch_size, attended.shape[1], -1),
                    ],
                    dim=-1,
                )
            )
            compatibility_k = self.compatibility_node(
                torch.cat(
                    [context, node_type_embedding, kind_embedding, scalar_features],
                    dim=-1,
                )
            )
            compatibility_bias = (
                torch.einsum("bqh,bnh->bqn", compatibility_q, compatibility_k)
                / (context.shape[-1] ** 0.5)
                + type_conditioned_relation_bias(
                    type_probabilities,
                    self.type_relation_table,
                    node_type,
                )
            )
            pointer_logits = pointer_logits + compatibility_bias
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
            query_node_compatibility_bias=compatibility_bias,
        )


def source_conflict_penalty(
    pointer_logits: torch.Tensor,
    source_conflict: torch.Tensor,
    *,
    object_logits: torch.Tensor | None = None,
    active_query_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean conflict probability over valid active-query opportunities.

    Object probabilities prevent no-object slots from contributing arbitrary
    pointer conflicts. A supplied ``active_query_mask`` is authoritative and
    may be boolean (matched/active) or a continuous query weight. ``query_mask``
    excludes padded slots. Both numerator and denominator use the same weights,
    so decoder capacity alone cannot rescale the objective.
    """

    if source_conflict.shape != (
        pointer_logits.shape[0],
        pointer_logits.shape[-1],
        pointer_logits.shape[-1],
    ):
        raise ValueError("source_conflict must have shape [B, N, N]")
    if object_logits is not None and object_logits.shape != pointer_logits.shape[:2]:
        raise ValueError("object_logits must have shape [B, Q]")
    if active_query_mask is not None and active_query_mask.shape != pointer_logits.shape[:2]:
        raise ValueError("active_query_mask must have shape [B, Q]")
    if query_mask is not None and query_mask.shape != pointer_logits.shape[:2]:
        raise ValueError("query_mask must have shape [B, Q]")
    probability = torch.sigmoid(pointer_logits)
    pair = probability.unsqueeze(-1) * probability.unsqueeze(-2)
    upper = torch.triu(source_conflict, diagonal=1).to(pair.dtype)
    if active_query_mask is not None:
        query_weight = active_query_mask.to(pair.dtype)
    elif object_logits is not None:
        query_weight = torch.sigmoid(object_logits).to(pair.dtype)
    else:
        # Backward-compatible standalone behavior: every query is active.
        query_weight = pair.new_ones(pointer_logits.shape[:2])
    if query_mask is not None:
        query_weight = query_weight * query_mask.to(pair.dtype)
    opportunities = query_weight[:, :, None, None] * upper[:, None]
    denominator = opportunities.sum()
    return (pair * opportunities).sum() / denominator.clamp_min(
        torch.finfo(pair.dtype).eps
    )


def constrained_daughter_decode(
    probabilities: torch.Tensor,
    *,
    cardinality: int,
    pointer_mask: torch.Tensor,
    source_conflict: torch.Tensor,
    min_probability: float = 0.5,
    insufficient_policy: str = "invalid",
) -> tuple[torch.Tensor, bool]:
    """Greedy constrained top-k with explicit low-score behavior."""

    if probabilities.ndim != 1:
        raise ValueError("constrained_daughter_decode expects one proposal")
    if pointer_mask.shape != probabilities.shape:
        raise ValueError("pointer_mask shape differs from pointer probabilities")
    if source_conflict.shape != (probabilities.numel(), probabilities.numel()):
        raise ValueError("source_conflict has an invalid shape")
    if insufficient_policy not in {"invalid", "reduce"}:
        raise ValueError("insufficient_policy must be 'invalid' or 'reduce'")
    selected = torch.zeros_like(pointer_mask)
    candidates = (
        pointer_mask
        & torch.isfinite(probabilities)
        & (probabilities >= float(min_probability))
    ).nonzero(as_tuple=False).flatten()
    order = candidates[
        torch.argsort(probabilities[candidates], descending=True, stable=True)
    ]
    for index in order.tolist():
        chosen = selected.nonzero(as_tuple=False).flatten()
        if chosen.numel() and source_conflict[index, chosen].any():
            continue
        selected[index] = True
        if int(selected.sum()) >= cardinality:
            break
    enough = int(selected.sum()) == cardinality
    return selected, bool(enough or insufficient_policy == "reduce")


__all__ = [
    "MotherPointerDecoder",
    "MotherPointerOutput",
    "constrained_daughter_decode",
    "source_conflict_penalty",
]
