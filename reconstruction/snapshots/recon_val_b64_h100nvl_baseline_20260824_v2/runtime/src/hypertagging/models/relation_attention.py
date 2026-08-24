"""Relation-biased self-attention for unordered levelized node sets."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RelationAwareSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, n_nodes, _ = x.shape
        qkv = self.qkv(x).view(batch_size, n_nodes, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + relation_bias[:, None, :, :]
        allowed = attention_mask[:, None, :, :] & node_mask[:, None, :, None]
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(allowed, weights, torch.zeros_like(weights))
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, n_nodes, self.d_model)
        output = self.output(attended) * node_mask.unsqueeze(-1)
        return output, weights if return_attention else None


class RelationAwareSetLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, feedforward_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = RelationAwareSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ff_dim = feedforward_dim or 2 * d_model
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        update, weights = self.attention(
            self.norm1(x),
            relation_bias=relation_bias,
            attention_mask=attention_mask,
            node_mask=node_mask,
            return_attention=return_attention,
        )
        x = x + self.dropout(update)
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x * node_mask.unsqueeze(-1), weights


class RelationAwareSetTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, feedforward_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RelationAwareSetLayer(d_model, n_heads, feedforward_dim, dropout)
            for _ in range(n_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        weights: torch.Tensor | None = None
        for layer in self.layers:
            x, weights = layer(
                x,
                relation_bias=relation_bias,
                attention_mask=attention_mask,
                node_mask=node_mask,
                return_attention=return_attention,
            )
        return x, weights


__all__ = [
    "RelationAwareSelfAttention",
    "RelationAwareSetLayer",
    "RelationAwareSetTransformer",
]
