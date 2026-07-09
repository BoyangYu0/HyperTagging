"""Collation and topology-label construction for levelized events."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from hypertagging.data.level_batch import LevelBatch, LevelEvent


def collate_level_events(events: Sequence[LevelEvent], *, max_query_slots: int | None = None) -> LevelBatch:
    """Pad variable-size levelized events into a dense CPU-testable batch."""

    if not events:
        raise ValueError("collate_level_events requires at least one event")
    batch_size = len(events)
    max_nodes = max(event.node_features.shape[0] for event in events)
    feature_dim = events[0].node_features.shape[-1]
    max_level = max(int(event.level_ids.max().item()) for event in events)
    query_slots = max_query_slots or max(1, max_nodes)

    node_features = torch.zeros((batch_size, max_nodes, feature_dim), dtype=torch.float32)
    p4 = torch.zeros((batch_size, max_nodes, 4), dtype=torch.float32)
    charge = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    level_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    pid_labels = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    parent_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    active = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    copied = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    copied_from = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    same_mother = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    same_branch = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    lca_depth = torch.full((batch_size, max_nodes, max_nodes), -1, dtype=torch.long)
    query_mask = torch.zeros((batch_size, max_level + 1, query_slots), dtype=torch.bool)
    event_ids = torch.tensor([event.event_id for event in events], dtype=torch.long)

    for batch_index, event in enumerate(events):
        n_nodes = event.node_features.shape[0]
        node_features[batch_index, :n_nodes] = event.node_features
        p4[batch_index, :n_nodes] = event.p4
        charge[batch_index, :n_nodes] = event.charge
        level_ids[batch_index, :n_nodes] = event.level_ids
        pid_labels[batch_index, :n_nodes] = event.pid_labels
        parent_ids[batch_index, :n_nodes] = event.parent_ids
        adjacency[batch_index, :n_nodes, :n_nodes] = event.daughter_adjacency
        active[batch_index, :n_nodes] = event.active
        copied[batch_index, :n_nodes] = event.copied
        copied_from[batch_index, :n_nodes] = event.copied_from
        same_mother[batch_index, :n_nodes, :n_nodes] = build_same_mother(event.parent_ids)
        same_branch[batch_index, :n_nodes, :n_nodes] = build_same_branch(event.parent_ids)
        lca_depth[batch_index, :n_nodes, :n_nodes] = build_lca_depth(event.parent_ids, event.level_ids)
        for level in range(max_level + 1):
            count = int((event.level_ids == level).sum().item())
            query_mask[batch_index, level, : min(query_slots, max(1, count))] = True

    return LevelBatch(
        node_features=node_features,
        node_mask=active.clone(),
        level_ids=level_ids,
        pid_labels=pid_labels,
        p4=p4,
        charge=charge,
        daughter_adjacency=adjacency,
        parent_ids=parent_ids,
        active=active,
        copied=copied,
        copied_from=copied_from,
        same_mother=same_mother,
        same_branch=same_branch,
        lca_depth=lca_depth,
        query_mask=query_mask,
        event_ids=event_ids,
    )


def build_same_mother(parent_ids: torch.Tensor) -> torch.Tensor:
    parents = parent_ids[:, None]
    valid = parent_ids >= 0
    return (parents == parents.T) & valid[:, None] & valid[None, :]


def build_same_branch(parent_ids: torch.Tensor) -> torch.Tensor:
    n_nodes = parent_ids.numel()
    ancestors = [_ancestor_set(parent_ids, index) for index in range(n_nodes)]
    out = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
    for i in range(n_nodes):
        for j in range(n_nodes):
            out[i, j] = bool(ancestors[i] & ancestors[j]) or i == j
    return out


def build_lca_depth(parent_ids: torch.Tensor, level_ids: torch.Tensor) -> torch.Tensor:
    n_nodes = parent_ids.numel()
    ancestor_paths = [_ancestor_path(parent_ids, index) for index in range(n_nodes)]
    out = torch.full((n_nodes, n_nodes), -1, dtype=torch.long)
    for i in range(n_nodes):
        for j in range(n_nodes):
            common = set(ancestor_paths[i]) & set(ancestor_paths[j])
            if common:
                out[i, j] = int(max(level_ids[list(common)]).item())
            elif i == j:
                out[i, j] = int(level_ids[i].item())
    return out


def _ancestor_set(parent_ids: torch.Tensor, index: int) -> set[int]:
    return set(_ancestor_path(parent_ids, index))


def _ancestor_path(parent_ids: torch.Tensor, index: int) -> list[int]:
    path = [index]
    seen = {index}
    parent = int(parent_ids[index].item())
    while parent >= 0:
        if parent in seen:
            raise ValueError(f"cycle detected while building ancestor path for {index}")
        path.append(parent)
        seen.add(parent)
        parent = int(parent_ids[parent].item())
    return path
