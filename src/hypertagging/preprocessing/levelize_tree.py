"""Levelisation and level-wise training sample construction."""

from __future__ import annotations

from dataclasses import dataclass

from hypertagging.preprocessing.mdst_tree_builder import EventTree


@dataclass(frozen=True)
class LevelSample:
    """One adjacent-level training sample for an event."""

    event_id: int
    level: int
    input_node_ids: list[int]
    target_node_ids: list[int]
    links: list[int]


def assign_levels(tree: EventTree) -> None:
    """Assign reconstruction levels with leaves at level 0."""

    memo: dict[int, int] = {}
    visiting: set[int] = set()

    def level(node_id: int) -> int:
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id}")
        if node_id in memo:
            return memo[node_id]
        visiting.add(node_id)
        node = tree.nodes[node_id]
        if not node.daughter_ids:
            value = 0
        else:
            value = 1 + max(level(child_id) for child_id in node.daughter_ids)
        visiting.remove(node_id)
        memo[node_id] = value
        node.level = value
        return value

    for node_id in tree.nodes:
        level(node_id)


def nodes_by_level(tree: EventTree) -> dict[int, list[int]]:
    """Return node ids grouped by assigned level."""

    levels: dict[int, list[int]] = {}
    for node in tree.nodes.values():
        if node.level < 0:
            raise ValueError("assign_levels must be called before nodes_by_level")
        levels.setdefault(node.level, []).append(node.node_id)
    for node_ids in levels.values():
        node_ids.sort()
    return dict(sorted(levels.items()))


def adjacent_level_samples(tree: EventTree) -> list[LevelSample]:
    """Build level-wise input/target/link samples.

    For each target level ``t + 1``, all nodes with level ``<= t`` form the
    input set.  The link label for each input node is the target mother index
    when its retained parent is at ``t + 1`` and ``-1`` otherwise.
    """

    grouped = nodes_by_level(tree)
    if not grouped:
        return []
    samples: list[LevelSample] = []
    max_level = max(grouped)
    for source_level in range(max_level):
        input_node_ids = sorted(node_id for node_id, node in tree.nodes.items() if node.level <= source_level)
        target_node_ids = grouped.get(source_level + 1, [])
        target_index = {node_id: index for index, node_id in enumerate(target_node_ids)}
        links: list[int] = []
        for node_id in input_node_ids:
            parent_id = tree.nodes[node_id].parent_id
            if parent_id is not None and parent_id in target_index:
                links.append(target_index[parent_id])
            else:
                links.append(-1)
        samples.append(
            LevelSample(
                event_id=tree.event_id,
                level=source_level,
                input_node_ids=input_node_ids,
                target_node_ids=target_node_ids,
                links=links,
            )
        )
    return samples


def legacy_depth_samples(tree: EventTree, *, channel: int = 0) -> list[dict[str, object]]:
    """Return records compatible with the historical awkward parquet fields."""

    records: list[dict[str, object]] = []
    for sample in adjacent_level_samples(tree):
        input_nodes = [tree.nodes[node_id] for node_id in sample.input_node_ids]
        records.append(
            {
                "feature": [node.feature() for node in input_nodes],
                "channel": channel,
                "evtNum": tree.event_id,
                "depth": sample.level,
                "seq_len": len(input_nodes),
                "arrayIndex": sample.input_node_ids,
                "motherPDG": [
                    tree.nodes[node.parent_id].token if node.parent_id is not None else 0
                    for node in input_nodes
                ],
                "motherIndex": sample.links,
                "E_Rec": _energy_reconstruction_fraction(tree, sample.input_node_ids),
                "targetNodeIds": sample.target_node_ids,
            }
        )
    return records


def _energy_reconstruction_fraction(tree: EventTree, node_ids: list[int]) -> float:
    roots = [tree.nodes[node_id] for node_id in tree.root_ids if node_id in tree.nodes]
    root_energy = sum(node.p4.energy for node in roots)
    leaf_energy = sum(tree.nodes[node_id].p4.energy for node_id in node_ids if not tree.nodes[node_id].daughter_ids)
    if root_energy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, leaf_energy / root_energy))
