"""Deterministic structured decay-channel representations.

Signatures use the already-pruned and contracted :class:`EventTree`. Reco
kinematics, node copies, and reconstructed object IDs are deliberately absent.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable

from hypertagging.preprocessing.mdst_tree_builder import EventTree
from hypertagging.preprocessing.pid_filter import tokenize_pdg


B_PDGS = frozenset({511, 521, 531})


def conjugate_pdg(pdg: int) -> int:
    """Return the charge-conjugate PDG for the retained vocabulary."""

    # Self-conjugate states in the reduced vocabulary.
    if abs(int(pdg)) in {0, 22, 111, 130, 310, 443, 300553}:
        return int(pdg)
    return -int(pdg)


def canonical_decay_signature(
    tree: EventTree,
    node_id: int,
    *,
    charge_conjugate_normalize: bool = False,
) -> str:
    """Return a recursively sorted reduced-PID signature as canonical JSON."""

    raw = _signature_tuple(tree, node_id, conjugate=False)
    if charge_conjugate_normalize:
        conjugated = _signature_tuple(tree, node_id, conjugate=True)
        raw = min(raw, conjugated, key=_signature_json)
    return _signature_json(raw)


def _signature_tuple(tree: EventTree, node_id: int, *, conjugate: bool) -> tuple[int, tuple[Any, ...]]:
    node = tree.nodes[node_id]
    pdg = conjugate_pdg(node.pdg) if conjugate else node.pdg
    daughters = tuple(
        sorted(
            (_signature_tuple(tree, child_id, conjugate=conjugate) for child_id in node.daughter_ids),
            key=_signature_json,
        )
    )
    return tokenize_pdg(pdg), daughters


def _signature_json(signature: tuple[int, tuple[Any, ...]]) -> str:
    return json.dumps(signature, separators=(",", ":"), ensure_ascii=True)


def deterministic_channel_id(signature: str | None) -> int:
    """Return a stable non-negative 60-bit ID; zero denotes unavailable."""

    if not signature:
        return 0
    return int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:15], 16)


def find_b_branches(tree: EventTree) -> list[int]:
    """Find the two retained B-side roots deterministically."""

    y4s = [node_id for node_id, node in tree.nodes.items() if abs(node.pdg) == 300553]
    candidates: list[int] = []
    for root_id in sorted(y4s):
        candidates.extend(
            child_id
            for child_id in tree.nodes[root_id].daughter_ids
            if abs(tree.nodes[child_id].pdg) in B_PDGS
        )
    if len(candidates) < 2:
        candidates = [
            node_id
            for node_id, node in tree.nodes.items()
            if abs(node.pdg) in B_PDGS
            and (
                node.parent_id is None
                or abs(tree.nodes[node.parent_id].pdg) not in B_PDGS
            )
        ]
    return sorted(dict.fromkeys(candidates))[:2]


def branch_node_ids(tree: EventTree, root_id: int) -> list[int]:
    """Return unique branch nodes; copied subtrees are represented once per node."""

    output: list[int] = []
    seen: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        output.append(node_id)
        for child_id in tree.nodes[node_id].daughter_ids:
            visit(child_id)

    visit(root_id)
    return output


def channel_count_array(
    tree: EventTree,
    root_id: int,
    *,
    conjugate: bool = False,
) -> dict[str, Any]:
    """Return a depth-aware sparse count representation for channel similarity."""

    token_counts: Counter[int] = Counter()
    depth_token_counts: Counter[tuple[int, int]] = Counter()
    depth_counts: Counter[int] = Counter()
    branch_multiplicities: list[int] = []
    selected_intermediate_counts: Counter[int] = Counter()
    stack = [(root_id, 0)]
    seen_sources: set[int] = set()
    while stack:
        node_id, depth = stack.pop()
        node = tree.nodes[node_id]
        # Copies carry the same topological meaning as their source. Avoid
        # double counting a source inside one branch representation.
        source_id = node.source_node_id if node.source_node_id is not None else (
            node.copied_from if node.copied_from is not None else node.node_id
        )
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        token = int(tokenize_pdg(conjugate_pdg(node.pdg) if conjugate else node.pdg))
        token_counts[token] += 1
        depth_token_counts[(depth, token)] += 1
        depth_counts[depth] += 1
        if node.daughter_ids:
            branch_multiplicities.append(len(node.daughter_ids))
            if abs(node.pdg) in {111, 310, 3122, 411, 421, 431, 4122, 413, 423, 433}:
                selected_intermediate_counts[token] += 1
        stack.extend((child_id, depth + 1) for child_id in reversed(node.daughter_ids))
    return {
        "pid_counts": _counter_records(token_counts, ("token",)),
        "depth_pid_counts": [
            {"depth": depth, "token": token, "count": count}
            for (depth, token), count in sorted(depth_token_counts.items())
        ],
        "depth_counts": _counter_records(depth_counts, ("depth",)),
        "selected_intermediate_counts": _counter_records(
            selected_intermediate_counts,
            ("token",),
        ),
        "branch_multiplicities": sorted(branch_multiplicities),
        "n_nodes": sum(token_counts.values()),
        "max_relative_depth": max(depth_counts, default=0),
    }


def _counter_records(counter: Counter[Any], names: tuple[str, ...]) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    for key, count in sorted(counter.items()):
        values = key if isinstance(key, tuple) else (key,)
        records.append({**dict(zip(names, map(int, values))), "count": int(count)})
    return records


def structured_channel_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Weighted-Jaccard similarity of depth/PID counts and multiplicities."""

    left_counts = _count_vector(left)
    right_counts = _count_vector(right)
    keys = left_counts.keys() | right_counts.keys()
    if not keys:
        return 1.0
    intersection = sum(min(left_counts.get(key, 0.0), right_counts.get(key, 0.0)) for key in keys)
    union = sum(max(left_counts.get(key, 0.0), right_counts.get(key, 0.0)) for key in keys)
    return float(intersection / union) if union else 1.0


def _count_vector(channel: dict[str, Any]) -> dict[tuple[Any, ...], float]:
    output: dict[tuple[Any, ...], float] = {}
    for record in channel.get("pid_counts", []):
        output[("pid", int(record["token"]))] = float(record["count"])
    for record in channel.get("depth_pid_counts", []):
        output[("depth_pid", int(record["depth"]), int(record["token"]))] = float(record["count"])
    for index, value in enumerate(channel.get("branch_multiplicities", [])):
        output[("multiplicity", index, int(value))] = 0.5
    return output


def unordered_b_pair_signature(signatures: Iterable[str | None]) -> str | None:
    valid = sorted(signature for signature in signatures if signature)
    if len(valid) != 2:
        return None
    return json.dumps(valid, separators=(",", ":"), ensure_ascii=True)


def event_channel_record(
    tree: EventTree,
    *,
    charge_conjugate_normalize: bool = False,
) -> dict[str, Any]:
    """Build separate B-side and unordered event-level channel fields."""

    b_ids = find_b_branches(tree)
    signatures = [
        canonical_decay_signature(
            tree,
            node_id,
            charge_conjugate_normalize=charge_conjugate_normalize,
        )
        for node_id in b_ids
    ]
    conjugate_flags = []
    for node_id in b_ids:
        raw = _signature_tuple(tree, node_id, conjugate=False)
        conjugated = _signature_tuple(tree, node_id, conjugate=True)
        conjugate_flags.append(
            charge_conjugate_normalize
            and _signature_json(conjugated) < _signature_json(raw)
        )
    arrays = [
        channel_count_array(tree, node_id, conjugate=conjugate)
        for node_id, conjugate in zip(b_ids, conjugate_flags)
    ]
    while len(signatures) < 2:
        signatures.append(None)
        arrays.append(_empty_count_array())
        b_ids.append(-1)
    pair_signature = unordered_b_pair_signature(signatures)
    return {
        "charge_conjugate_normalized": bool(charge_conjugate_normalize),
        "b1_root_id": int(b_ids[0]),
        "b2_root_id": int(b_ids[1]),
        "b1_channel_signature": signatures[0],
        "b2_channel_signature": signatures[1],
        "b1_channel_id": deterministic_channel_id(signatures[0]),
        "b2_channel_id": deterministic_channel_id(signatures[1]),
        "b1_channel_counts": arrays[0],
        "b2_channel_counts": arrays[1],
        "exact_channel_equal": bool(signatures[0] and signatures[0] == signatures[1]),
        "structured_channel_similarity": structured_channel_similarity(arrays[0], arrays[1]),
        "same_event": True,
        "y4s_channel_signature": pair_signature,
        "y4s_channel_id": deterministic_channel_id(pair_signature),
    }


def _empty_count_array() -> dict[str, Any]:
    return {
        "pid_counts": [],
        "depth_pid_counts": [],
        "depth_counts": [],
        "selected_intermediate_counts": [],
        "branch_multiplicities": [],
        "n_nodes": 0,
        "max_relative_depth": 0,
    }


__all__ = [
    "branch_node_ids",
    "canonical_decay_signature",
    "channel_count_array",
    "conjugate_pdg",
    "deterministic_channel_id",
    "event_channel_record",
    "find_b_branches",
    "structured_channel_similarity",
    "unordered_b_pair_signature",
]
