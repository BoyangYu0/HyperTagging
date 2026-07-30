"""V1/v2 parquet loading and dense heterogeneous model batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from hypertagging.data.level_batch import LevelEvent
from hypertagging.preprocessing.schema_v2 import (
    NODE_KIND_TO_ID,
)
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES as CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES as COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES as COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES as TRACK_FEATURE_NAMES,
    load_payload_v3,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, validate_pid_tokens


@dataclass(frozen=True)
class HeterogeneousEvent:
    event_id: int
    event_uid: str
    common_features: torch.Tensor
    common_availability: torch.Tensor
    track_features: torch.Tensor
    track_availability: torch.Tensor
    cluster_features: torch.Tensor
    cluster_availability: torch.Tensor
    composite_features: torch.Tensor
    composite_availability: torch.Tensor
    daughter_pid_histogram: torch.Tensor
    daughter_pid_histogram_available: torch.Tensor
    node_kind_ids: torch.Tensor
    pid_labels: torch.Tensor
    pid_target_labels: torch.Tensor
    truth_pid_labels: torch.Tensor
    truth_pid_available: torch.Tensor
    level_ids: torch.Tensor
    p4: torch.Tensor
    charge: torch.Tensor
    parent_ids: torch.Tensor
    daughter_adjacency: torch.Tensor
    active: torch.Tensor
    copied: torch.Tensor
    node_ids: torch.Tensor
    reco_ids: torch.Tensor
    source_node_ids: torch.Tensor
    recursive_leaf_source_mask: torch.Tensor
    copied_from: torch.Tensor
    b_side: torch.Tensor
    b_channel_count_arrays: torch.Tensor
    b1_channel_id: int = 0
    b2_channel_id: int = 0
    b1_full_truth_channel_id: int = 0
    b2_full_truth_channel_id: int = 0
    b1_reconstructable_channel_id: int = 0
    b2_reconstructable_channel_id: int = 0
    channel_similarity: float = 0.0
    source_file: str = ""
    source_category: str = ""
    schema_version: str = ""


def load_heterogeneous_events(
    path: str | Path,
    *,
    limit: int | None = None,
    max_nodes: int | None = None,
    overflow_strategy: str = "raise",
) -> list[HeterogeneousEvent]:
    """Load any direct-mDST schema into the corrected v3 model contract."""

    payload = load_payload_v3(path)
    records = payload["events"] if limit is None else payload["events"][:limit]
    output: list[HeterogeneousEvent] = []
    for event in records:
        if max_nodes is not None and len(event["nodes"]) > max_nodes:
            if overflow_strategy == "drop":
                continue
            raise OverflowError(
                f"event {event.get('event_uid', event.get('event_id'))} has "
                f"{len(event['nodes'])} nodes, exceeding max_nodes={max_nodes}; "
                "use overflow_strategy='drop' only with explicit accounting"
            )
        output.append(_event_from_record(event))
    return output


def _event_from_record(event: dict[str, Any]) -> HeterogeneousEvent:
    nodes = sorted(event["nodes"], key=lambda node: (int(node["level"]), int(node["node_id"])))
    id_to_position = {int(node["node_id"]): index for index, node in enumerate(nodes)}
    count = len(nodes)
    adjacency = torch.zeros((count, count), dtype=torch.bool)
    parent_ids = torch.full((count,), -1, dtype=torch.long)
    for position, node in enumerate(nodes):
        parent_ids[position] = id_to_position.get(int(node["parent_id"]), -1)
        for daughter_id in node["daughter_ids"]:
            if int(daughter_id) in id_to_position:
                adjacency[position, id_to_position[int(daughter_id)]] = True
    b_side = _b_side_labels(nodes, id_to_position, event)
    input_tokens = torch.tensor(
        [int(node.get("input_pid_token", node.get("token", 0))) for node in nodes],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [int(node.get("pid_target_token", node.get("token", 0))) for node in nodes],
        dtype=torch.long,
    )
    validate_pid_tokens(input_tokens, name="parquet input PID tokens")
    validate_pid_tokens(target_tokens, name="parquet target PID tokens")
    truth_available = torch.tensor(
        [node.get("truth_pid_token") is not None for node in nodes],
        dtype=torch.bool,
    )
    truth_tokens = torch.tensor(
        [int(node.get("truth_pid_token") or 0) for node in nodes],
        dtype=torch.long,
    )
    validate_pid_tokens(truth_tokens, name="parquet truth PID tokens")
    recursive_source_mask = _recursive_source_mask(nodes)
    return HeterogeneousEvent(
        event_id=int(event["event_id"]),
        event_uid=str(event.get("event_uid", event["event_id"])),
        common_features=_records_tensor(nodes, "common_features", COMMON_FEATURE_NAMES),
        common_availability=_records_tensor(
            nodes,
            "common_availability",
            COMMON_FEATURE_NAMES,
            dtype=torch.bool,
        ),
        track_features=_records_tensor(nodes, "track_features", TRACK_FEATURE_NAMES),
        track_availability=_records_tensor(
            nodes,
            "track_availability",
            TRACK_FEATURE_NAMES,
            dtype=torch.bool,
        ),
        cluster_features=_records_tensor(nodes, "cluster_features", CLUSTER_FEATURE_NAMES),
        cluster_availability=_records_tensor(
            nodes,
            "cluster_availability",
            CLUSTER_FEATURE_NAMES,
            dtype=torch.bool,
        ),
        composite_features=_records_tensor(nodes, "composite_features", COMPOSITE_FEATURE_NAMES),
        composite_availability=_records_tensor(
            nodes,
            "composite_availability",
            COMPOSITE_FEATURE_NAMES,
            dtype=torch.bool,
        ),
        daughter_pid_histogram=torch.tensor(
            [node["daughter_pid_histogram"] for node in nodes],
            dtype=torch.float32,
        ),
        daughter_pid_histogram_available=torch.tensor(
            [bool(node["daughter_pid_histogram_available"]) for node in nodes],
            dtype=torch.bool,
        ),
        node_kind_ids=torch.tensor([int(node["node_kind_id"]) for node in nodes], dtype=torch.long),
        pid_labels=input_tokens,
        pid_target_labels=target_tokens,
        truth_pid_labels=truth_tokens,
        truth_pid_available=truth_available,
        level_ids=torch.tensor([int(node["level"]) for node in nodes], dtype=torch.long),
        p4=torch.tensor(
            [[float(node[name]) for name in ("px", "py", "pz", "energy")] for node in nodes],
            dtype=torch.float32,
        ),
        charge=torch.tensor(
            [float(node.get("reco_charge", node.get("charge", 0.0)) or 0.0) for node in nodes],
            dtype=torch.float32,
        ),
        parent_ids=parent_ids,
        daughter_adjacency=adjacency,
        active=torch.tensor([bool(node["active"]) for node in nodes], dtype=torch.bool),
        copied=torch.tensor([bool(node["copied"]) for node in nodes], dtype=torch.bool),
        node_ids=torch.tensor([int(node["node_id"]) for node in nodes], dtype=torch.long),
        reco_ids=torch.tensor(
            [_stable_reco_id(str(node.get("reco_id", ""))) for node in nodes],
            dtype=torch.long,
        ),
        source_node_ids=torch.tensor(
            [
                int(node.get("source_node_id", -1))
                if int(node.get("source_node_id", -1)) >= 0
                else (
                    int(node.get("copied_from", -1))
                    if int(node.get("copied_from", -1)) >= 0
                    else int(node["node_id"])
                )
                for node in nodes
            ],
            dtype=torch.long,
        ),
        recursive_leaf_source_mask=recursive_source_mask,
        copied_from=torch.tensor(
            [int(node.get("copied_from", -1)) for node in nodes],
            dtype=torch.long,
        ),
        b_side=b_side,
        b_channel_count_arrays=torch.tensor(
            [
                event.get("b1_channel_count_array", [0] * len(PDG_TOKENS)),
                event.get("b2_channel_count_array", [0] * len(PDG_TOKENS)),
            ],
            dtype=torch.float32,
        ),
        b1_channel_id=int(event.get("b1_channel_id", 0)),
        b2_channel_id=int(event.get("b2_channel_id", 0)),
        b1_full_truth_channel_id=int(event.get("b1_full_truth_channel_id", 0)),
        b2_full_truth_channel_id=int(event.get("b2_full_truth_channel_id", 0)),
        b1_reconstructable_channel_id=int(
            event.get("b1_reconstructable_channel_id", event.get("b1_channel_id", 0))
        ),
        b2_reconstructable_channel_id=int(
            event.get("b2_reconstructable_channel_id", event.get("b2_channel_id", 0))
        ),
        channel_similarity=float(event.get("structured_channel_similarity", 0.0)),
        source_file=str(event.get("source_file", "")),
        source_category=str(event.get("source_category", "")),
        schema_version=str(event.get("schema_version", "")),
    )


def _recursive_source_mask(nodes: list[dict[str, Any]]) -> torch.Tensor:
    per_node: list[list[str]] = []
    all_sources: set[str] = set()
    for node in nodes:
        sources = [str(value) for value in node.get("recursive_leaf_source_ids", []) if str(value)]
        if not sources and not node.get("daughter_ids"):
            fallback = str(node.get("reco_object_id") or node.get("reco_id") or "")
            sources = [fallback] if fallback else []
        unique = sorted(set(sources))
        per_node.append(unique)
        all_sources.update(unique)
    ordered = sorted(all_sources)
    position = {source: index for index, source in enumerate(ordered)}
    output = torch.zeros((len(nodes), len(ordered)), dtype=torch.bool)
    for node_index, sources in enumerate(per_node):
        for source in sources:
            output[node_index, position[source]] = True
    return output


def _records_tensor(
    nodes: list[dict[str, Any]],
    block: str,
    names: tuple[str, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [[node[block][name] for name in names] for node in nodes],
        dtype=dtype,
    )


def _b_side_labels(
    nodes: list[dict[str, Any]],
    id_to_position: dict[int, int],
    event: dict[str, Any],
) -> torch.Tensor:
    labels = torch.full((len(nodes),), -1, dtype=torch.long)
    for side, field in enumerate(("b1_root_id", "b2_root_id")):
        root = int(event.get(field, -1))
        if root not in id_to_position:
            continue
        stack = [root]
        while stack:
            node_id = stack.pop()
            position = id_to_position[node_id]
            if labels[position] >= 0:
                continue
            labels[position] = side
            stack.extend(int(child) for child in nodes[position]["daughter_ids"] if int(child) in id_to_position)
    return labels


def heterogeneous_from_level_event(event: LevelEvent) -> HeterogeneousEvent:
    """Upgrade a tiny/legacy in-memory event using only genuinely known values."""

    count = event.node_features.shape[0]
    mass2 = event.p4[:, 3].square() - event.p4[:, :3].square().sum(dim=-1)
    mass = mass2.clamp_min(0).sqrt()
    copied = event.copied.bool()
    common = torch.stack(
        [
            event.p4[:, 0],
            event.p4[:, 1],
            event.p4[:, 2],
            event.p4[:, 3],
            mass,
            event.charge,
            event.pid_labels.float(),
            event.level_ids.float(),
            event.active.float(),
            copied.float(),
            event.daughter_adjacency.sum(dim=-1).float(),
            torch.zeros(count),
        ],
        dim=-1,
    )
    common_mask = torch.ones_like(common, dtype=torch.bool)
    common_mask[:, -1] = False
    validate_pid_tokens(event.pid_labels, name="LevelEvent.pid_labels")
    if event.node_kind_ids is not None:
        kinds = event.node_kind_ids.clone()
    else:
        kinds = torch.full((count,), NODE_KIND_TO_ID["unknown"], dtype=torch.long)
        kinds[event.level_ids > 0] = NODE_KIND_TO_ID["composite"]
        kinds[(event.level_ids == 0) & (event.charge != 0)] = NODE_KIND_TO_ID["track"]

    track = torch.zeros((count, len(TRACK_FEATURE_NAMES)))
    track_mask = torch.zeros_like(track, dtype=torch.bool)
    cluster = torch.zeros((count, len(CLUSTER_FEATURE_NAMES)))
    cluster_mask = torch.zeros_like(cluster, dtype=torch.bool)
    composite = torch.zeros((count, len(COMPOSITE_FEATURE_NAMES)))
    composite_mask = torch.zeros_like(composite, dtype=torch.bool)
    pid_hist = torch.zeros((count, len(PDG_TOKENS)))
    pid_hist_available = torch.zeros(count, dtype=torch.bool)
    for mother in range(count):
        daughters = event.daughter_adjacency[mother].nonzero(as_tuple=False).flatten()
        if not daughters.numel():
            continue
        composite[mother, :4] = event.p4[daughters].sum(dim=0)
        composite[mother, 4] = event.charge[daughters].sum()
        composite[mother, 5] = daughters.numel()
        composite[mother, 8] = copied[daughters].float().mean()
        composite_mask[mother, [0, 1, 2, 3, 4, 5, 8]] = True
        pid_hist[mother].scatter_add_(
            0,
            event.pid_labels[daughters],
            torch.ones_like(daughters, dtype=torch.float32),
        )
        pid_hist_available[mother] = True
    node_ids = torch.arange(count, dtype=torch.long)
    return HeterogeneousEvent(
        event_id=event.event_id,
        event_uid=str(event.event_id),
        common_features=common,
        common_availability=common_mask,
        track_features=track,
        track_availability=track_mask,
        cluster_features=cluster,
        cluster_availability=cluster_mask,
        composite_features=composite,
        composite_availability=composite_mask,
        daughter_pid_histogram=pid_hist,
        daughter_pid_histogram_available=pid_hist_available,
        node_kind_ids=kinds,
        pid_labels=event.pid_labels,
        pid_target_labels=event.pid_labels,
        truth_pid_labels=event.pid_labels,
        truth_pid_available=torch.ones(count, dtype=torch.bool),
        level_ids=event.level_ids,
        p4=event.p4,
        charge=event.charge,
        parent_ids=event.parent_ids,
        daughter_adjacency=event.daughter_adjacency,
        active=event.active,
        copied=copied,
        node_ids=node_ids,
        reco_ids=torch.full((count,), -1, dtype=torch.long),
        source_node_ids=node_ids,
        recursive_leaf_source_mask=_fixture_recursive_source_mask(event),
        copied_from=event.copied_from,
        b_side=torch.full((count,), -1, dtype=torch.long),
        b_channel_count_arrays=torch.zeros((2, len(PDG_TOKENS))),
    )


def _fixture_recursive_source_mask(event: LevelEvent) -> torch.Tensor:
    count = event.p4.shape[0]
    representatives = [
        int(event.copied_from[index])
        if int(event.copied_from[index]) >= 0
        else index
        for index in range(count)
    ]
    leaves = sorted({representatives[index] for index in range(count) if not event.daughter_adjacency[index].any()})
    leaf_position = {leaf: index for index, leaf in enumerate(leaves)}
    output = torch.zeros((count, len(leaves)), dtype=torch.bool)
    for node in range(count):
        stack = [node]
        while stack:
            current = stack.pop()
            daughters = event.daughter_adjacency[current].nonzero(as_tuple=False).flatten().tolist()
            if daughters:
                stack.extend(daughters)
            else:
                representative = representatives[current]
                if representative in leaf_position:
                    output[node, leaf_position[representative]] = True
    return output


def collate_heterogeneous_events(events: Sequence[HeterogeneousEvent]) -> dict[str, torch.Tensor]:
    """Pad heterogeneous events without assigning semantics to padded zeroes."""

    if not events:
        raise ValueError("collate_heterogeneous_events requires at least one event")
    batch_size = len(events)
    max_nodes = max(event.common_features.shape[0] for event in events)
    output: dict[str, torch.Tensor] = {}
    feature_fields = {
        "common_features": events[0].common_features.shape[-1],
        "common_availability": events[0].common_availability.shape[-1],
        "track_features": events[0].track_features.shape[-1],
        "track_availability": events[0].track_availability.shape[-1],
        "cluster_features": events[0].cluster_features.shape[-1],
        "cluster_availability": events[0].cluster_availability.shape[-1],
        "composite_features": events[0].composite_features.shape[-1],
        "composite_availability": events[0].composite_availability.shape[-1],
        "daughter_pid_histogram": events[0].daughter_pid_histogram.shape[-1],
    }
    bool_fields = {
        "common_availability",
        "track_availability",
        "cluster_availability",
        "composite_availability",
    }
    for field, width in feature_fields.items():
        dtype = torch.bool if field in bool_fields else torch.float32
        output[field] = torch.zeros((batch_size, max_nodes, width), dtype=dtype)
    vector_defaults = {
        "node_kind_ids": 0,
        "pid_labels": 0,
        "pid_target_labels": 0,
        "truth_pid_labels": 0,
        "truth_pid_available": 0,
        "level_ids": -1,
        "charge": 0,
        "parent_ids": -1,
        "active": 0,
        "copied": 0,
        "node_ids": -1,
        "reco_ids": -1,
        "source_node_ids": -1,
        "copied_from": -1,
        "b_side": -1,
        "daughter_pid_histogram_available": 0,
    }
    bool_vectors = {
        "active",
        "copied",
        "daughter_pid_histogram_available",
        "truth_pid_available",
    }
    for field, default in vector_defaults.items():
        dtype = torch.bool if field in bool_vectors else (
            torch.float32 if field == "charge" else torch.long
        )
        output[field] = torch.full((batch_size, max_nodes), default, dtype=dtype)
    output["p4"] = torch.zeros((batch_size, max_nodes, 4), dtype=torch.float32)
    output["daughter_adjacency"] = torch.zeros(
        (batch_size, max_nodes, max_nodes),
        dtype=torch.bool,
    )
    max_sources = max(event.recursive_leaf_source_mask.shape[1] for event in events)
    output["recursive_leaf_source_mask"] = torch.zeros(
        (batch_size, max_nodes, max_sources),
        dtype=torch.bool,
    )
    for batch_index, event in enumerate(events):
        n_nodes = event.common_features.shape[0]
        for field in feature_fields:
            output[field][batch_index, :n_nodes] = getattr(event, field)
        for field in vector_defaults:
            output[field][batch_index, :n_nodes] = getattr(event, field)
        output["p4"][batch_index, :n_nodes] = event.p4
        output["daughter_adjacency"][batch_index, :n_nodes, :n_nodes] = event.daughter_adjacency
        n_sources = event.recursive_leaf_source_mask.shape[1]
        output["recursive_leaf_source_mask"][batch_index, :n_nodes, :n_sources] = (
            event.recursive_leaf_source_mask
        )
    output["node_mask"] = output["active"].clone()
    output["node_features"] = output["common_features"]
    output["event_ids"] = torch.tensor([event.event_id for event in events], dtype=torch.long)
    output["b1_channel_ids"] = torch.tensor([event.b1_channel_id for event in events], dtype=torch.long)
    output["b2_channel_ids"] = torch.tensor([event.b2_channel_id for event in events], dtype=torch.long)
    output["b1_full_truth_channel_ids"] = torch.tensor(
        [event.b1_full_truth_channel_id for event in events], dtype=torch.long
    )
    output["b2_full_truth_channel_ids"] = torch.tensor(
        [event.b2_full_truth_channel_id for event in events], dtype=torch.long
    )
    output["b1_reconstructable_channel_ids"] = torch.tensor(
        [event.b1_reconstructable_channel_id for event in events], dtype=torch.long
    )
    output["b2_reconstructable_channel_ids"] = torch.tensor(
        [event.b2_reconstructable_channel_id for event in events], dtype=torch.long
    )
    output["channel_similarity"] = torch.tensor(
        [event.channel_similarity for event in events],
        dtype=torch.float32,
    )
    output["b_channel_count_arrays"] = torch.stack(
        [event.b_channel_count_arrays for event in events]
    )
    return output


def _stable_reco_id(value: str) -> int:
    if not value:
        return -1
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:15], 16)


__all__ = [
    "HeterogeneousEvent",
    "collate_heterogeneous_events",
    "heterogeneous_from_level_event",
    "load_heterogeneous_events",
]
