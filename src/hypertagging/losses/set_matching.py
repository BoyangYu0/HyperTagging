"""Permutation-invariant matching for next-level mother sets."""

from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as F


def jaccard_cost(pointer_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pointer_logits)
    target = target_masks.float()
    inter = torch.minimum(pred[:, None, :], target[None, :, :]).sum(dim=-1)
    union = torch.maximum(pred[:, None, :], target[None, :, :]).sum(dim=-1).clamp_min(1e-6)
    return 1.0 - inter / union


def matching_cost(
    *,
    type_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    target_types: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """Return [Q, M] cost for one event."""

    type_cost = -F.log_softmax(type_logits, dim=-1)[:, target_types]
    return type_cost + jaccard_cost(pointer_logits, target_masks)


def hungarian_or_greedy(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Deterministic assignment with scipy when available and brute-force fallback."""

    q, m = cost.shape
    if m == 0 or q == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]

        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [(int(row), int(col)) for row, col in zip(rows, cols, strict=True) if row < q and col < m]
    except Exception:
        pass
    if max(q, m) <= 8:
        cost_for_assignment = cost.detach()
        best_score = None
        best: list[tuple[int, int]] = []
        for rows in permutations(range(q), min(q, m)):
            pairs = list(zip(rows, range(min(q, m)), strict=True))
            score = sum(float(cost_for_assignment[row, col]) for row, col in pairs)
            if best_score is None or score < best_score:
                best_score = score
                best = pairs
        return best
    used: set[int] = set()
    pairs = []
    for col in range(m):
        available = [(float(cost[row, col]), row) for row in range(q) if row not in used]
        if not available:
            break
        _score, row = min(available)
        used.add(row)
        pairs.append((row, col))
    return pairs
