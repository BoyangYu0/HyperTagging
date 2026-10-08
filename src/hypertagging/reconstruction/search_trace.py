"""Detached, truth-free search records for post-inference survival diagnostics.

Only reconstructed topology, dense inference leaf IDs and model outputs are
recorded. Evaluation identities and truth must be joined after search returns.
Tracing never supplies a score, proposal, mask or stopping decision to search.
"""
from __future__ import annotations

import torch


def state_record(batch: dict[str, torch.Tensor]) -> dict:
    active = batch["node_mask"][0].nonzero().flatten().tolist()
    adjacency = batch["daughter_adjacency"][0].bool()
    ids = batch["node_ids"][0].tolist()
    cache: dict[int, list[int]] = {}

    def leaves(position: int) -> list[int]:
        if position not in cache:
            daughters = adjacency[position].nonzero().flatten().tolist()
            cache[position] = (sorted({leaf for child in daughters for leaf in leaves(child)})
                               if daughters else [int(ids[position])])
        return cache[position]

    return {
        "memberships": [leaves(p) for p in range(batch["node_mask"].shape[1])],
        "active": active,
        "roots": [p for p in active if int(batch["parent_ids"][0, p]) < 0],
        "composites": [p for p in active if bool(adjacency[p].any())],
        "types": batch["pid_labels"][0].tolist(),
        "levels": batch["level_ids"][0].tolist(),
        "children": [adjacency[p].nonzero().flatten().tolist() for p in range(adjacency.shape[0])],
    }


def proposal_record(candidate) -> dict:
    p = candidate.proposal
    return {"query": int(p.query_id), "type": int(p.mother_type),
            "daughters": list(p.daughter_positions), "log_score": float(candidate.log_score)}


def decode_record(output, state, eligible) -> dict:
    """Record context eligibility before alias and joint proposal filtering."""
    pointer = output.pointer
    return {
        "stage": "decode", "level": int(output.target_level),
        "state": state_record(state), "eligible": eligible.nonzero().flatten().tolist(),
        "object_probability": torch.sigmoid(pointer.object_logits[0]).tolist(),
        "pointer_probability": torch.sigmoid(pointer.pointer_logits[0]).tolist(),
        "generated": [], "query_retained": [], "level_retained": [],
    }
