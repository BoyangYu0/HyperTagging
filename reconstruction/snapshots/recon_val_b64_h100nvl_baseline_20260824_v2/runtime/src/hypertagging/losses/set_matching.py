"""Permutation-invariant matching for next-level mother sets."""

from __future__ import annotations

from itertools import combinations, permutations

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
    object_logits: torch.Tensor | None = None,
    cardinality_logits: torch.Tensor | None = None,
    charge_cost: torch.Tensor | None = None,
    p4_cost: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return [Q, M] cost for one event."""

    weights = {
        "type": 1.0,
        "pointer": 1.0,
        "object": 0.2,
        "cardinality": 0.2,
        "charge": 0.1,
        "p4": 0.1,
        **(weights or {}),
    }
    type_cost = -F.log_softmax(type_logits, dim=-1)[:, target_types]
    total = weights["type"] * type_cost + weights["pointer"] * jaccard_cost(
        pointer_logits,
        target_masks,
    )
    if object_logits is not None:
        total = total + weights["object"] * F.softplus(-object_logits)[:, None]
    if cardinality_logits is not None:
        cardinalities = target_masks.sum(dim=-1).long()
        if cardinalities.numel() and int(cardinalities.max()) >= cardinality_logits.shape[-1]:
            raise OverflowError(
                f"truth daughter cardinality {int(cardinalities.max())} exceeds "
                f"decoder capacity {cardinality_logits.shape[-1] - 1}"
            )
        total = total + weights["cardinality"] * (
            -F.log_softmax(cardinality_logits, dim=-1)[:, cardinalities]
        )
    if charge_cost is not None:
        total = total + weights["charge"] * charge_cost
    if p4_cost is not None:
        total = total + weights["p4"] * p4_cost
    return total


def hungarian_assignment(
    cost: torch.Tensor,
    *,
    production: bool = True,
    allow_bruteforce: bool = False,
    brute_force_limit: int = 8,
) -> list[tuple[int, int]]:
    """Deterministic Hungarian assignment with an explicit tiny-test fallback."""

    q, m = cost.shape
    if m == 0 or q == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]
    except ImportError as exc:
        if production or not allow_bruteforce:
            raise RuntimeError(
                "SciPy is required for production Hungarian matching. Install the "
                "declared training dependency; greedy matching is not supported."
            ) from exc
    else:
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [
            (int(row), int(col))
            for row, col in zip(rows, cols, strict=True)
            if row < q and col < m
        ]
    if max(q, m) > brute_force_limit:
        raise RuntimeError(
            "brute-force assignment is restricted to tiny tests with "
            f"max dimension <= {brute_force_limit}"
        )
    size = min(q, m)
    detached = cost.detach()
    best_score = float("inf")
    best: list[tuple[int, int]] = []
    if q >= m:
        for rows in combinations(range(q), size):
            for ordered_rows in permutations(rows):
                pairs = list(zip(ordered_rows, range(m), strict=True))
                score = sum(float(detached[row, col]) for row, col in pairs)
                if score < best_score:
                    best_score, best = score, pairs
    else:
        for cols in combinations(range(m), size):
            for ordered_cols in permutations(cols):
                pairs = list(zip(range(q), ordered_cols, strict=True))
                score = sum(float(detached[row, col]) for row, col in pairs)
                if score < best_score:
                    best_score, best = score, pairs
    return best


def hungarian_or_greedy(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Compatibility alias: Hungarian normally, brute force only for tiny tests.

    Despite the historical name, there is no greedy fallback.
    """

    return hungarian_assignment(cost, production=False, allow_bruteforce=True)


__all__ = [
    "hungarian_assignment",
    "hungarian_or_greedy",
    "jaccard_cost",
    "matching_cost",
]
