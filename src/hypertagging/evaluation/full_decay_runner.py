"""Serialization and structural diagnostics for offline decay-tree inference."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import torch

from hypertagging.evaluation.full_decay_metrics import canonical_fsp_membership
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.hierarchical_inference import (
    HierarchicalInferenceResult,
)


ROLLOUT_STOP_REASONS = {
    0: "not_stopped",
    1: "no_valid_new_mother",
    2: "configured_root_reconstructed",
    3: "maximum_level",
}


def inference_diagnostics(
    result: HierarchicalInferenceResult,
    event_index: int = 0,
    *,
    p4_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Return JSON-safe validity and coverage counters for one event."""

    batch = result.batch
    active = batch["node_mask"][event_index].bool().detach().cpu()
    adjacency = batch["daughter_adjacency"][event_index].bool().detach().cpu()
    adjacency &= active[:, None] & active[None, :]
    parents = batch["parent_ids"][event_index].long().detach().cpu()
    levels = batch["level_ids"][event_index].long().detach().cpu()
    p4 = batch["p4"][event_index].detach().cpu()
    sources, canonical_keys = canonical_fsp_membership(batch, event_index)
    detector_sources = (
        batch["recursive_leaf_source_mask"][event_index]
        .bool()
        .detach()
        .cpu()
    )

    indegree = adjacency.sum(dim=0)
    roots = active & (indegree == 0)
    active_positions = active.nonzero(as_tuple=False).flatten()
    expected_parent = torch.full_like(parents, -1)
    has_parent = indegree == 1
    if has_parent.any():
        expected_parent[has_parent] = adjacency[:, has_parent].long().argmax(dim=0)
    parent_consistent = bool(
        torch.equal(parents[active_positions], expected_parent[active_positions])
    )
    edge_positions = adjacency.nonzero(as_tuple=False)
    levels_ordered = all(
        int(levels[parent]) > int(levels[child])
        for parent, child in edge_positions.tolist()
    )

    closure_numerator = 0
    closure_denominator = 0
    maximum_closure_residual = 0.0
    for mother in active_positions.tolist():
        daughters = adjacency[mother]
        if not daughters.any():
            continue
        residual = p4[mother] - p4[daughters].sum(dim=0)
        maximum = float(residual.abs().max())
        maximum_closure_residual = max(maximum_closure_residual, maximum)
        closure_numerator += int(maximum <= p4_tolerance)
        closure_denominator += 1

    root_sources = sources[roots]
    source_usage = (
        root_sources.sum(dim=0)
        if root_sources.numel()
        else torch.zeros(sources.shape[-1], dtype=torch.long)
    )
    source_width = len(canonical_keys)
    covered_sources = int((source_usage > 0).sum())
    duplicate_sources = int((source_usage > 1).sum())
    detector_usage = (
        detector_sources[roots].sum(dim=0)
        if bool(roots.any())
        else torch.zeros(detector_sources.shape[-1], dtype=torch.long)
    )
    detector_source_width = int(
        result.input_audit.detector_source_counts[event_index]
    )
    duplicate_detector_sources = int(
        (detector_usage[:detector_source_width] > 1).sum()
    )
    conflicting_mother_count = 0
    conflicting_detector_resource_count = 0
    for mother in active_positions.tolist():
        daughters = (adjacency[mother] & active).nonzero(
            as_tuple=False
        ).flatten()
        if daughters.numel() < 2:
            continue
        daughter_usage = detector_sources[daughters].sum(dim=0)
        conflicts = int((daughter_usage[:detector_source_width] > 1).sum())
        conflicting_mother_count += int(conflicts > 0)
        conflicting_detector_resource_count += conflicts
    # Associated ECL/KLM inputs may remain as disconnected forest roots and
    # therefore share detector provenance without any reconstructed mother
    # double-using them.  Keep that global overlap visible, but structural
    # validity depends on conflicts among daughters actually combined.
    recursive_sources_disjoint = conflicting_mother_count == 0
    node_kinds = batch["node_kind_ids"][event_index].detach().cpu()
    composite = active & (node_kinds == NODE_KIND_TO_ID["composite"])
    input_fsp_roots = roots & ~composite
    stop_code = int(result.rollout.stop_code[event_index])
    empty_level_count = (
        int(result.rollout.empty_level_counts[event_index])
        if result.rollout.empty_level_counts is not None
        else 0
    )
    structurally_valid = bool(
        levels_ordered
        and (indegree <= 1).all()
        and parent_consistent
        and duplicate_sources == 0
        and recursive_sources_disjoint
    )

    return {
        "stop_code": stop_code,
        "stop_reason": ROLLOUT_STOP_REASONS.get(stop_code, f"unknown_{stop_code}"),
        "levels_completed": int(result.rollout.levels_completed[event_index]),
        "empty_level_count": empty_level_count,
        "rollout_event_valid": bool(result.rollout.event_valid_mask[event_index]),
        "configured_root_completed": bool(
            result.rollout.root_completed_mask[event_index]
        ),
        "active_node_count": int(active.sum()),
        "predicted_composite_count": int(composite.sum()),
        "forest_root_count": int(roots.sum()),
        "leftover_input_fsp_count": int(input_fsp_roots.sum()),
        "b_root_count": int(result.b_root_mask[event_index].sum()),
        "continuum_root_count": int(result.continuum_root_mask[event_index].sum()),
        "evaluation_slice_multiplicity": int(
            result.evaluation_slice_multiplicity[event_index]
        ),
        "accepted_mother_count": sum(
            int(mask[event_index].sum())
            for mask in result.rollout.accepted_query_masks
        ),
        "acyclic_by_strict_level_order": bool(levels_ordered),
        "single_parent": bool((indegree <= 1).all()),
        "parent_adjacency_consistent": parent_consistent,
        "inference_structurally_valid": structurally_valid,
        "forest_root_sources_disjoint": duplicate_sources == 0,
        "duplicate_root_source_count": duplicate_sources,
        "forest_root_detector_resources_disjoint": (
            duplicate_detector_sources == 0
        ),
        "duplicate_root_detector_resource_count": duplicate_detector_sources,
        "detector_resource_count": detector_source_width,
        "recursive_detector_sources_disjoint": recursive_sources_disjoint,
        "source_conflicting_mother_count": conflicting_mother_count,
        "source_conflicting_detector_resource_count": (
            conflicting_detector_resource_count
        ),
        "input_source_coverage_numerator": covered_sources,
        "input_source_coverage_denominator": source_width,
        "input_source_coverage": (
            covered_sources / source_width if source_width else None
        ),
        "p4_closure_numerator": closure_numerator,
        "p4_closure_denominator": closure_denominator,
        "p4_closure_eligible": closure_denominator > 0,
        "p4_closure_rate": (
            closure_numerator / closure_denominator
            if closure_denominator
            else None
        ),
        "maximum_p4_closure_residual": maximum_closure_residual,
        "p4_closure_tolerance": float(p4_tolerance),
        "forest_root_only_pointer_policy_enforced": True,
    }


def serialize_reconstructed_tree(
    result: HierarchicalInferenceResult,
    event_index: int = 0,
) -> dict[str, Any]:
    """Serialize active predicted nodes without exposing padded query slots."""

    batch = result.batch
    active = batch["node_mask"][event_index].bool().detach().cpu()
    adjacency = batch["daughter_adjacency"][event_index].bool().detach().cpu()
    sources, canonical_keys = canonical_fsp_membership(batch, event_index)
    detector_sources = (
        batch["recursive_leaf_source_mask"][event_index]
        .bool()
        .detach()
        .cpu()
    )
    forest = result.forest_root_mask[event_index].detach().cpu()
    b_roots = result.b_root_mask[event_index].detach().cpu()
    continuum_roots = result.continuum_root_mask[event_index].detach().cpu()
    slices = result.evaluation_slice_root_mask[event_index].detach().cpu()
    pid_values = batch.get("current_pid_tokens", batch["pid_labels"])
    nodes: list[dict[str, Any]] = []
    for position in active.nonzero(as_tuple=False).flatten().tolist():
        pid_token = int(pid_values[event_index, position])
        raw_sources = sources[position].nonzero(as_tuple=False).flatten().tolist()
        nodes.append(
            {
                "position": position,
                "node_id": int(batch["node_ids"][event_index, position]),
                "node_kind_id": int(batch["node_kind_ids"][event_index, position]),
                "level": int(batch["level_ids"][event_index, position]),
                "pid_token": pid_token,
                "pdg": int(PDG_TOKENS[pid_token]),
                "charge": float(batch["charge"][event_index, position]),
                "p4": [
                    float(value)
                    for value in batch["p4"][event_index, position].detach().cpu()
                ],
                "parent_position": int(batch["parent_ids"][event_index, position]),
                "daughter_positions": [
                    int(value)
                    for value in (adjacency[position] & active)
                    .nonzero(as_tuple=False)
                    .flatten()
                ],
                "fsp_source_keys": [canonical_keys[index] for index in raw_sources],
                "detector_resource_indices": detector_sources[position]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist(),
                "is_forest_root": bool(forest[position]),
                "is_b_root": bool(b_roots[position]),
                "is_continuum_root": bool(continuum_roots[position]),
                "is_evaluation_slice_root": bool(slices[position]),
            }
        )
    return {"nodes": nodes}


def summarize_inference_diagnostics(
    diagnostics: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate event-level structural diagnostics without hiding failures."""

    rows = list(diagnostics)
    stop_counts = Counter(str(row["stop_reason"]) for row in rows)
    closure_numerator = sum(int(row["p4_closure_numerator"]) for row in rows)
    closure_denominator = sum(int(row["p4_closure_denominator"]) for row in rows)
    source_numerator = sum(int(row["input_source_coverage_numerator"]) for row in rows)
    source_denominator = sum(int(row["input_source_coverage_denominator"]) for row in rows)

    def rate(name: str) -> dict[str, float | int | None]:
        numerator = sum(int(bool(row[name])) for row in rows)
        denominator = len(rows)
        return {
            "value": numerator / denominator if denominator else None,
            "numerator": numerator,
            "denominator": denominator,
        }

    return {
        "event_count": len(rows),
        "total_empty_level_count": sum(
            int(row.get("empty_level_count", 0)) for row in rows
        ),
        "stop_reason_counts": dict(sorted(stop_counts.items())),
        "configured_root_completion": rate("configured_root_completed"),
        "rollout_event_valid": rate("rollout_event_valid"),
        "acyclic_by_strict_level_order": rate("acyclic_by_strict_level_order"),
        "single_parent": rate("single_parent"),
        "parent_adjacency_consistent": rate("parent_adjacency_consistent"),
        "inference_structurally_valid": rate("inference_structurally_valid"),
        "forest_root_sources_disjoint": rate("forest_root_sources_disjoint"),
        "forest_root_detector_resources_disjoint": rate(
            "forest_root_detector_resources_disjoint"
        ),
        "recursive_detector_sources_disjoint": rate(
            "recursive_detector_sources_disjoint"
        ),
        "p4_closure": {
            "value": (
                closure_numerator / closure_denominator
                if closure_denominator
                else None
            ),
            "numerator": closure_numerator,
            "denominator": closure_denominator,
            "eligible": closure_denominator > 0,
        },
        "input_source_coverage": {
            "value": (
                source_numerator / source_denominator
                if source_denominator
                else None
            ),
            "numerator": source_numerator,
            "denominator": source_denominator,
        },
        "maximum_p4_closure_residual": max(
            (float(row["maximum_p4_closure_residual"]) for row in rows),
            default=0.0,
        ),
    }


__all__ = [
    "ROLLOUT_STOP_REASONS",
    "inference_diagnostics",
    "serialize_reconstructed_tree",
    "summarize_inference_diagnostics",
]
