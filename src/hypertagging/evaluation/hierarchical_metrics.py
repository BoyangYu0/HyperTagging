"""Level, edge, rollout, channel, and embedding metrics for revised models."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
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


@dataclass(frozen=True)
class CanonicalTreeMetrics:
    subtree_exact_match: float
    full_tree_exact_match: bool
    edge_precision: float
    edge_recall: float
    edge_f1: float
    mother_type_accuracy: float
    leaf_assignment_accuracy: float
    recursive_source_overlap: float
    tree_edit_like_distance: float
    first_divergence_level: int
    root_reconstruction_success: bool


@dataclass(frozen=True)
class SourceAlignmentResult:
    matches: tuple[tuple[int, int], ...]
    mother_type_accuracy: float
    mean_source_jaccard: float


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
    predicted_signatures = canonical_tree_signatures(predicted)
    truth_signatures = canonical_tree_signatures(truth)
    predicted_edges = _canonical_edges(predicted, predicted_signatures)
    truth_edges = _canonical_edges(truth, truth_signatures)
    intersection = predicted_edges & truth_edges
    true_positive = sum(intersection.values())
    predicted_count = sum(predicted_edges.values())
    truth_count = sum(truth_edges.values())
    precision = true_positive / predicted_count if predicted_count else float(not truth_count)
    recall = true_positive / truth_count if truth_count else float(not predicted_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EdgeMetrics(precision, recall, f1, predicted_edges == truth_edges)


def canonical_tree_signatures(
    batch: dict[str, torch.Tensor],
    batch_index: int = 0,
) -> dict[int, tuple[Any, ...]]:
    """Canonical signatures independent of generated composite node IDs."""

    active = batch["node_mask"][batch_index]
    adjacency = batch["daughter_adjacency"][batch_index]
    pid = batch.get("pid_target_labels", batch["pid_labels"])[batch_index]
    memo: dict[int, tuple[Any, ...]] = {}
    visiting: set[int] = set()

    def signature(node: int) -> tuple[Any, ...]:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError(f"cycle detected while canonicalizing node {node}")
        visiting.add(node)
        daughters = [
            int(child)
            for child in adjacency[node].nonzero(as_tuple=False).flatten().tolist()
            if bool(active[child])
        ]
        if daughters:
            result: tuple[Any, ...] = (
                "mother",
                int(pid[node]),
                tuple(sorted((signature(child) for child in daughters), key=repr)),
            )
        else:
            if "recursive_leaf_source_mask" in batch:
                sources = tuple(
                    int(value)
                    for value in batch["recursive_leaf_source_mask"][batch_index, node]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
            elif "reco_ids" in batch and int(batch["reco_ids"][batch_index, node]) >= 0:
                sources = (int(batch["reco_ids"][batch_index, node]),)
            else:
                sources = (int(batch["source_node_ids"][batch_index, node]),)
            result = ("leaf", sources)
        visiting.remove(node)
        memo[node] = result
        return result

    for node in active.nonzero(as_tuple=False).flatten().tolist():
        signature(int(node))
    return memo


def canonical_tree_metrics(
    predicted: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
) -> CanonicalTreeMetrics:
    predicted_signatures = canonical_tree_signatures(predicted)
    truth_signatures = canonical_tree_signatures(truth)
    predicted_counter = Counter(predicted_signatures.values())
    truth_counter = Counter(truth_signatures.values())
    common = predicted_counter & truth_counter
    subtree_denominator = max(sum(truth_counter.values()), 1)
    subtree_exact = sum(common.values()) / subtree_denominator
    predicted_roots = _root_signature_counter(predicted, predicted_signatures)
    truth_roots = _root_signature_counter(truth, truth_signatures)
    edges = edge_metrics(predicted, truth)
    aligned_result = _align_batch_nodes_by_source(predicted, truth)
    aligned = list(aligned_result.matches)
    type_correct = 0
    type_total = 0
    source_scores = []
    leaf_correct = 0
    leaf_total = 0
    for predicted_node, truth_node in aligned:
        pred_sig = predicted_signatures[predicted_node]
        truth_sig = truth_signatures[truth_node]
        if pred_sig[0] == "mother" and truth_sig[0] == "mother":
            type_total += 1
            type_correct += int(pred_sig[1] == truth_sig[1])
        if pred_sig[0] == "leaf" and truth_sig[0] == "leaf":
            leaf_total += 1
            leaf_correct += int(pred_sig == truth_sig)
        pred_sources = _signature_leaf_sources(pred_sig)
        truth_sources = _signature_leaf_sources(truth_sig)
        union = pred_sources | truth_sources
        source_scores.append(
            len(pred_sources & truth_sources) / len(union) if union else 1.0
        )
    edit_numerator = (
        sum((predicted_counter - truth_counter).values())
        + sum((truth_counter - predicted_counter).values())
    )
    edit_denominator = max(sum(predicted_counter.values()) + sum(truth_counter.values()), 1)
    return CanonicalTreeMetrics(
        subtree_exact_match=subtree_exact,
        full_tree_exact_match=predicted_roots == truth_roots,
        edge_precision=edges.precision,
        edge_recall=edges.recall,
        edge_f1=edges.f1,
        mother_type_accuracy=type_correct / type_total if type_total else 1.0,
        leaf_assignment_accuracy=leaf_correct / leaf_total if leaf_total else 1.0,
        recursive_source_overlap=sum(source_scores) / len(source_scores) if source_scores else 0.0,
        tree_edit_like_distance=edit_numerator / edit_denominator,
        first_divergence_level=_first_divergence_level(
            predicted,
            truth,
            predicted_signatures,
            truth_signatures,
        ),
        root_reconstruction_success=bool(truth_roots) and bool(predicted_roots & truth_roots),
    )


def _canonical_edges(
    batch: dict[str, torch.Tensor],
    signatures: dict[int, tuple[Any, ...]],
) -> Counter[tuple[tuple[Any, ...], tuple[Any, ...]]]:
    output: Counter[tuple[tuple[Any, ...], tuple[Any, ...]]] = Counter()
    for parent, child in batch["daughter_adjacency"][0].nonzero(as_tuple=False).tolist():
        if parent in signatures and child in signatures:
            output[(signatures[parent], signatures[child])] += 1
    return output


def _root_signature_counter(
    batch: dict[str, torch.Tensor],
    signatures: dict[int, tuple[Any, ...]],
) -> Counter[tuple[Any, ...]]:
    adjacency = batch["daughter_adjacency"][0]
    has_parent = adjacency.any(dim=0)
    return Counter(
        signatures[node]
        for node in signatures
        if not bool(has_parent[node])
    )


def _align_by_signature(
    predicted: dict[int, tuple[Any, ...]],
    truth: dict[int, tuple[Any, ...]],
) -> list[tuple[int, int]]:
    truth_by_signature: dict[tuple[Any, ...], list[int]] = {}
    for node, signature in truth.items():
        truth_by_signature.setdefault(signature, []).append(node)
    output = []
    for node, signature in sorted(predicted.items()):
        candidates = truth_by_signature.get(signature, [])
        if candidates:
            output.append((node, candidates.pop(0)))
    return output


def align_subtrees_by_source(
    predicted: list[dict[str, Any]],
    truth: list[dict[str, Any]],
) -> SourceAlignmentResult:
    """Hungarian source/topology alignment that deliberately excludes type."""

    if not predicted or not truth:
        return SourceAlignmentResult((), float(not predicted and not truth), 0.0)
    cost = torch.zeros((len(predicted), len(truth)), dtype=torch.float64)
    jaccard = torch.zeros_like(cost)
    for left, pred in enumerate(predicted):
        pred_sources = set(pred["sources"])
        for right, target in enumerate(truth):
            truth_sources = set(target["sources"])
            union = pred_sources | truth_sources
            score = (
                len(pred_sources & truth_sources) / len(union) if union else 1.0
            )
            jaccard[left, right] = score
            depth_delta = abs(int(pred.get("depth", 0)) - int(target.get("depth", 0)))
            count_delta = abs(
                int(pred.get("daughter_count", 0))
                - int(target.get("daughter_count", 0))
            )
            cost[left, right] = 1 - score + 0.05 * depth_delta + 0.05 * count_delta
    from hypertagging.losses.set_matching import hungarian_assignment

    matches = tuple(
        hungarian_assignment(cost, production=False, allow_bruteforce=True)
    )
    type_accuracy = sum(
        int(predicted[left].get("type") == truth[right].get("type"))
        for left, right in matches
    ) / max(len(matches), 1)
    mean_jaccard = sum(float(jaccard[left, right]) for left, right in matches) / max(
        len(matches), 1
    )
    return SourceAlignmentResult(matches, type_accuracy, mean_jaccard)


def _align_batch_nodes_by_source(
    predicted: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
) -> SourceAlignmentResult:
    def records(batch: dict[str, torch.Tensor]) -> tuple[list[int], list[dict[str, Any]]]:
        active = batch["node_mask"][0].nonzero(as_tuple=False).flatten().tolist()
        output = []
        for node in active:
            sources = set(
                batch["recursive_leaf_source_mask"][0, node]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            output.append(
                {
                    "type": int(
                        batch.get("pid_target_labels", batch["pid_labels"])[0, node]
                    ),
                    "sources": sources,
                    "depth": int(batch["level_ids"][0, node]),
                    "daughter_count": int(
                        batch["daughter_adjacency"][0, node].sum()
                    ),
                }
            )
        return [int(value) for value in active], output

    predicted_nodes, predicted_records = records(predicted)
    truth_nodes, truth_records = records(truth)
    result = align_subtrees_by_source(predicted_records, truth_records)
    return SourceAlignmentResult(
        tuple(
            (predicted_nodes[left], truth_nodes[right])
            for left, right in result.matches
        ),
        result.mother_type_accuracy,
        result.mean_source_jaccard,
    )


def _signature_leaf_sources(signature: tuple[Any, ...]) -> set[int]:
    if signature[0] == "leaf":
        return set(signature[1])
    output: set[int] = set()
    for child in signature[2]:
        output.update(_signature_leaf_sources(child))
    return output


def _first_divergence_level(
    predicted: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    predicted_signatures: dict[int, tuple[Any, ...]],
    truth_signatures: dict[int, tuple[Any, ...]],
) -> int:
    levels = sorted(
        set(int(value) for value in truth["level_ids"][truth["node_mask"]].tolist())
        | set(int(value) for value in predicted["level_ids"][predicted["node_mask"]].tolist())
    )
    for level in levels:
        predicted_at_level = Counter(
            predicted_signatures[node]
            for node in predicted_signatures
            if int(predicted["level_ids"][0, node]) == level
        )
        truth_at_level = Counter(
            truth_signatures[node]
            for node in truth_signatures
            if int(truth["level_ids"][0, node]) == level
        )
        if predicted_at_level != truth_at_level:
            return level
    return -1


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
            visiting: set[int] = set()
            visited: set[int] = set()

            def visit(node: int) -> bool:
                if node in visiting:
                    return False
                if node in visited:
                    return True
                visiting.add(node)
                for child in adjacency[node].nonzero(as_tuple=False).flatten().tolist():
                    if not visit(int(child)):
                        return False
                visiting.remove(node)
                visited.add(node)
                return True

            valid = all(visit(node) for node in range(active_count))
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
    canonical = canonical_tree_metrics(predicted, truth)
    return {
        "full_tree_exact_match": canonical.full_tree_exact_match,
        "canonical_subtree_exact_match": canonical.subtree_exact_match,
        "edge_precision": canonical.edge_precision,
        "edge_recall": canonical.edge_recall,
        "edge_f1": canonical.edge_f1,
        "tree_edit_like_distance": canonical.tree_edit_like_distance,
        "first_divergence_level": canonical.first_divergence_level,
        "root_reconstruction_success": canonical.root_reconstruction_success,
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
    "CanonicalTreeMetrics",
    "SourceAlignmentResult",
    "align_subtrees_by_source",
    "canonical_tree_metrics",
    "canonical_tree_signatures",
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
