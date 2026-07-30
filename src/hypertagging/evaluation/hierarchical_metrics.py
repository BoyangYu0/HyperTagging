"""Level, edge, rollout, channel, and embedding metrics for revised models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hypertagging.models.hyperbolic import distance, radius
from hypertagging.losses.level_reconstruction import targets_for_level
from hypertagging.models.mother_pointer import MotherPointerOutput


@dataclass(frozen=True)
class EdgeMetrics:
    precision: float
    recall: float
    f1: float
    exact_match: bool


def edge_set(batch: dict[str, torch.Tensor], batch_index: int = 0) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    node_ids = batch["node_ids"][batch_index]
    active = batch["node_mask"][batch_index]
    for parent, child in batch["daughter_adjacency"][batch_index].nonzero(as_tuple=False).tolist():
        if bool(active[parent]) and bool(active[child]):
            edges.add((int(node_ids[parent]), int(node_ids[child])))
    return edges


def edge_metrics(
    predicted: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
) -> EdgeMetrics:
    predicted_edges = edge_set(predicted)
    truth_edges = edge_set(truth)
    true_positive = len(predicted_edges & truth_edges)
    precision = true_positive / len(predicted_edges) if predicted_edges else float(not truth_edges)
    recall = true_positive / len(truth_edges) if truth_edges else float(not predicted_edges)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EdgeMetrics(precision, recall, f1, predicted_edges == truth_edges)


def p4_closure_rate(batch: dict[str, torch.Tensor], tolerance: float = 1e-6) -> float:
    closed = 0
    composites = 0
    for batch_index in range(batch["p4"].shape[0]):
        for mother in range(batch["p4"].shape[1]):
            daughters = batch["daughter_adjacency"][batch_index, mother]
            if not daughters.any():
                continue
            composites += 1
            residual = batch["p4"][batch_index, mother] - batch["p4"][batch_index, daughters].sum(dim=0)
            closed += int(bool(torch.all(residual.abs() <= tolerance)))
    return closed / composites if composites else 1.0


def tree_validity_rate(batch: dict[str, torch.Tensor]) -> float:
    valid_events = 0
    for batch_index in range(batch["p4"].shape[0]):
        active_count = int(batch["node_mask"][batch_index].sum())
        adjacency = batch["daughter_adjacency"][batch_index, :active_count, :active_count]
        levels = batch["level_ids"][batch_index, :active_count]
        valid = not bool(torch.diagonal(adjacency).any())
        if valid:
            for mother, child in adjacency.nonzero(as_tuple=False).tolist():
                if int(levels[mother]) <= int(levels[child]):
                    valid = False
                    break
        valid_events += int(valid)
    return valid_events / batch["p4"].shape[0]


def radius_level_correlation(
    z: torch.Tensor,
    levels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    radii = radius(z)[mask]
    level_values = levels[mask].float()
    if radii.numel() < 2 or radii.std(unbiased=False) == 0 or level_values.std(unbiased=False) == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([radii, level_values]))[0, 1])


def summarize_rollout(
    predicted: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
) -> dict[str, Any]:
    edges = edge_metrics(predicted, truth)
    return {
        "full_tree_exact_match": edges.exact_match,
        "edge_precision": edges.precision,
        "edge_recall": edges.recall,
        "edge_f1": edges.f1,
        "tree_validity_rate": tree_validity_rate(predicted),
        "p4_closure_rate": p4_closure_rate(predicted),
        "predicted_nodes": int(predicted["node_mask"].sum()),
        "truth_nodes": int(truth["node_mask"].sum()),
        "predicted_max_level": int(predicted["level_ids"][predicted["node_mask"]].max()),
        "truth_max_level": int(truth["level_ids"][truth["node_mask"]].max()),
    }


def tree_relation_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> float:
    return float((logits.argmax(dim=-1)[pair_mask] == targets[pair_mask]).float().mean()) if pair_mask.any() else 0.0


def parent_ranking_accuracy(
    z: torch.Tensor,
    parent_ids: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    correct = []
    for batch_index in range(z.shape[0]):
        valid = node_mask[batch_index].nonzero(as_tuple=False).flatten()
        for child in valid.tolist():
            parent = int(parent_ids[batch_index, child])
            negatives = valid[(valid != child) & (valid != parent)]
            if parent < 0 or not negatives.numel():
                continue
            positive = distance(z[batch_index, child], z[batch_index, parent])
            negative = distance(
                z[batch_index, child].expand_as(z[batch_index, negatives]),
                z[batch_index, negatives],
            ).min()
            correct.append(positive < negative)
    return float(torch.stack(correct).float().mean()) if correct else 0.0


def next_level_metrics(
    output: MotherPointerOutput,
    batch: dict[str, torch.Tensor],
    matches: list[list[tuple[int, int]]],
    *,
    target_level: int,
    pointer_threshold: float = 0.5,
) -> dict[str, float]:
    target_types, target_masks, _, _ = targets_for_level(batch, target_level)
    object_target = torch.zeros_like(output.object_logits, dtype=torch.bool)
    type_correct, pointer_tp, pointer_fp, pointer_fn = 0, 0, 0, 0
    matched_count = 0
    for batch_index, event_matches in enumerate(matches):
        context = batch["node_mask"][batch_index] & (
            batch["level_ids"][batch_index] < target_level
        )
        for query, target in event_matches:
            object_target[batch_index, query] = True
            matched_count += 1
            type_correct += int(
                int(output.type_logits[batch_index, query].argmax())
                == int(target_types[batch_index][target])
            )
            predicted = (
                torch.sigmoid(output.pointer_logits[batch_index, query, context])
                >= pointer_threshold
            )
            truth = target_masks[batch_index][target]
            pointer_tp += int((predicted & truth).sum())
            pointer_fp += int((predicted & ~truth).sum())
            pointer_fn += int((~predicted & truth).sum())
    object_prediction = torch.sigmoid(output.object_logits) >= 0.5
    object_accuracy = float((object_prediction == object_target).float().mean())
    pointer_precision = pointer_tp / (pointer_tp + pointer_fp) if pointer_tp + pointer_fp else 1.0
    pointer_recall = pointer_tp / (pointer_tp + pointer_fn) if pointer_tp + pointer_fn else 1.0
    return {
        "object_no_object_accuracy": object_accuracy,
        "mother_type_accuracy": type_correct / matched_count if matched_count else 0.0,
        "pointer_precision": pointer_precision,
        "pointer_recall": pointer_recall,
    }


def channel_generalization_slices(
    training_channel_ids: torch.Tensor,
    evaluation_channel_ids: torch.Tensor,
    *,
    rare_max_count: int = 2,
) -> dict[str, torch.Tensor]:
    unique, counts = torch.unique(training_channel_ids[training_channel_ids > 0], return_counts=True)
    count_by_id = {int(channel): int(count) for channel, count in zip(unique, counts)}
    unseen = torch.tensor(
        [int(channel) not in count_by_id for channel in evaluation_channel_ids],
        dtype=torch.bool,
        device=evaluation_channel_ids.device,
    )
    rare = torch.tensor(
        [
            int(channel) in count_by_id and count_by_id[int(channel)] <= rare_max_count
            for channel in evaluation_channel_ids
        ],
        dtype=torch.bool,
        device=evaluation_channel_ids.device,
    )
    return {"rare": rare, "unseen": unseen, "seen_non_rare": ~(rare | unseen)}


__all__ = [
    "EdgeMetrics",
    "edge_metrics",
    "edge_set",
    "p4_closure_rate",
    "radius_level_correlation",
    "tree_relation_accuracy",
    "parent_ranking_accuracy",
    "next_level_metrics",
    "channel_generalization_slices",
    "summarize_rollout",
    "tree_validity_rate",
]
