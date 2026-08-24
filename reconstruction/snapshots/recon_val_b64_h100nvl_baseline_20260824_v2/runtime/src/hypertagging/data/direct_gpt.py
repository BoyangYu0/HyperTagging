"""Adapters from direct-mDST parquet trees to GPT-like model batches.

The historical GPT collator assumes a fixed number of particles at every
level. Direct mDST trees have variable level widths, so they require a
different teacher-forcing layout:

- reconstructed leaves are visible embedding inputs;
- higher-level nodes are zero-valued query slots;
- a level query can attend only to strictly lower levels;
- every active slot targets the embedding of its truth-guided node; and
- child slots link to the sequence position of their retained parent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import torch


DIRECT_FEATURE_NAMES: tuple[str, ...] = (
    "mass",
    "charge",
    "energy",
    "prodTime",
    "x",
    "y",
    "z",
    "px",
    "py",
    "pz",
    "nDaughters",
)


@dataclass(frozen=True)
class DirectGptEvent:
    """One direct-tree event ordered from leaves to successively higher levels."""

    event_id: int
    pdg_tokens: torch.Tensor
    features: torch.Tensor
    level_ids: torch.Tensor
    parent_indices: torch.Tensor
    masses: torch.Tensor
    node_ids: torch.Tensor


def load_direct_gpt_events(
    path: str | Path,
    *,
    limit: int | None = None,
    max_nodes: int | None = None,
) -> list[DirectGptEvent]:
    """Load direct-tree events from one preprocessing parquet shard."""

    payload = ak.to_list(ak.from_parquet(path))[0]
    if payload.get("schema_version") != "direct-mdst-tree-v1":
        raise ValueError(f"Unsupported preprocessing schema: {payload.get('schema_version')!r}")

    records = payload["events"] if limit is None else payload["events"][:limit]
    events: list[DirectGptEvent] = []
    for event in records:
        nodes = sorted(event["nodes"], key=lambda node: (int(node["level"]), int(node["node_id"])))
        if max_nodes is not None and len(nodes) > max_nodes:
            continue
        node_id_to_position = {int(node["node_id"]): index for index, node in enumerate(nodes)}
        parent_indices = [
            node_id_to_position.get(int(node["parent_id"]), -1)
            if int(node["parent_id"]) >= 0
            else -1
            for node in nodes
        ]
        events.append(
            DirectGptEvent(
                event_id=int(event["event_id"]),
                pdg_tokens=torch.tensor([int(node["token"]) for node in nodes], dtype=torch.long),
                features=torch.tensor(
                    [[float(node[name]) for name in DIRECT_FEATURE_NAMES] for node in nodes],
                    dtype=torch.float32,
                ),
                level_ids=torch.tensor([int(node["level"]) for node in nodes], dtype=torch.long),
                parent_indices=torch.tensor(parent_indices, dtype=torch.long),
                masses=torch.tensor([float(node["mass"]) for node in nodes], dtype=torch.float32),
                node_ids=torch.tensor([int(node["node_id"]) for node in nodes], dtype=torch.long),
            )
        )
    return events


def collate_direct_gpt_events(events: Sequence[DirectGptEvent]) -> dict[str, torch.Tensor]:
    """Pad variable direct-tree events for the particle embedder and GPT adapter."""

    if not events:
        raise ValueError("collate_direct_gpt_events requires at least one event")
    feature_dim = events[0].features.shape[-1]
    if any(event.features.shape[-1] != feature_dim for event in events):
        raise ValueError("All direct GPT events must have the same feature dimension")

    batch_size = len(events)
    max_nodes = max(len(event.pdg_tokens) for event in events)
    pdg = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    feature = torch.zeros((batch_size, max_nodes, feature_dim), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    level_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    parent_indices = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    mass = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    node_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long)

    for batch_index, event in enumerate(events):
        count = len(event.pdg_tokens)
        pdg[batch_index, :count] = event.pdg_tokens
        feature[batch_index, :count] = event.features
        padding_mask[batch_index, :count] = True
        level_ids[batch_index, :count] = event.level_ids
        parent_indices[batch_index, :count] = event.parent_indices
        mass[batch_index, :count] = event.masses
        node_ids[batch_index, :count] = event.node_ids

    return {
        "pdg": pdg,
        "feature": feature,
        "padding_mask": padding_mask,
        "level_ids": level_ids,
        "parent_indices": parent_indices,
        "mass": mass,
        "node_ids": node_ids,
        "event_ids": torch.tensor([event.event_id for event in events], dtype=torch.long),
    }


def build_direct_multi_gpt_batch(
    particle_embeddings: torch.Tensor,
    structure: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build a variable-level teacher-forcing batch for :class:`MultiGPT`."""

    padding_mask = structure["padding_mask"].bool()
    level_ids = structure["level_ids"]
    if particle_embeddings.shape[:2] != padding_mask.shape:
        raise ValueError(
            "particle_embeddings leading dimensions must match padding_mask: "
            f"{tuple(particle_embeddings.shape[:2])} != {tuple(padding_mask.shape)}"
        )

    visible_leaves = padding_mask & (level_ids == 0)
    model_input = torch.zeros_like(particle_embeddings)
    model_input[visible_leaves] = particle_embeddings[visible_leaves]
    level_code = torch.zeros_like(level_ids, dtype=torch.float32)
    level_code[padding_mask] = torch.exp(-level_ids[padding_mask].float())

    batch_size, max_nodes = padding_mask.shape
    source_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    for batch_index in range(batch_size):
        active = padding_mask[batch_index]
        levels = level_ids[batch_index]
        for query_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            query_level = int(levels[query_index])
            if query_level == 0:
                allowed = active & (levels == 0)
            else:
                allowed = active & (levels < query_level)
            source_mask[batch_index, query_index] = ~allowed
        source_mask[batch_index, ~active] = False

    mass_categories = torch.round(structure["mass"]).long().clamp_(0, 100)
    batch = {
        "emb": model_input,
        "target": particle_embeddings,
        "src_mask": source_mask,
        "links": structure["parent_indices"].long(),
        "mass": mass_categories,
        "lvl_code": level_code,
    }
    validate_direct_multi_gpt_batch(batch)
    return batch


def validate_direct_multi_gpt_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate the direct-tree MultiGPT tensor contract."""

    required = {"emb", "target", "src_mask", "links", "mass", "lvl_code"}
    missing = required - batch.keys()
    if missing:
        raise KeyError(f"Missing direct MultiGPT fields: {sorted(missing)}")
    emb = batch["emb"]
    if emb.ndim != 3 or batch["target"].shape != emb.shape:
        raise ValueError("emb and target must have matching [batch, nodes, embedding] shapes")
    batch_size, nodes, _ = emb.shape
    if batch["src_mask"].shape != (batch_size, nodes, nodes):
        raise ValueError("src_mask must have shape [batch, nodes, nodes]")
    for name in ("links", "mass", "lvl_code"):
        if batch[name].shape != (batch_size, nodes):
            raise ValueError(f"{name} must have shape [batch, nodes]")
    if batch["src_mask"].dtype is not torch.bool:
        raise TypeError("direct-tree src_mask must be boolean")
    if torch.any(batch["links"] >= nodes):
        raise ValueError("links contains an out-of-range parent position")
    if not torch.isfinite(emb).all() or not torch.isfinite(batch["target"]).all():
        raise ValueError("direct MultiGPT embeddings must be finite")
    return batch


__all__ = [
    "DIRECT_FEATURE_NAMES",
    "DirectGptEvent",
    "build_direct_multi_gpt_batch",
    "collate_direct_gpt_events",
    "load_direct_gpt_events",
    "validate_direct_multi_gpt_batch",
]
