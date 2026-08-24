"""Stair-causal masks for level-autoregressive set reconstruction."""

from __future__ import annotations

import torch


def context_mask_for_level(level_ids: torch.Tensor, node_mask: torch.Tensor, target_level: int) -> torch.Tensor:
    """Return nodes in S_{<=target_level-1}."""

    return node_mask & (level_ids >= 0) & (level_ids < target_level)


def stair_attention_mask(level_ids: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Return [B, N, N] mask where each node attends only to same/lower levels."""

    valid = node_mask[:, :, None] & node_mask[:, None, :]
    return valid & (level_ids[:, None, :] <= level_ids[:, :, None])


def query_to_context_mask(level_ids: torch.Tensor, node_mask: torch.Tensor, target_level: int, n_queries: int) -> torch.Tensor:
    """Return [B, Q, N] mask for target-level query slots."""

    context = context_mask_for_level(level_ids, node_mask, target_level)
    return context[:, None, :].expand(-1, n_queries, -1)
