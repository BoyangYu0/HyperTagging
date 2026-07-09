"""Export and load preprocessed HyperTagging event trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import awkward as ak

from hypertagging.preprocessing.levelize_tree import assign_levels, legacy_depth_samples, nodes_by_level
from hypertagging.preprocessing.mdst_tree_builder import EventTree, TreeNode


SCHEMA_VERSION = "direct-mdst-tree-v1"


def export_trees(
    trees: Iterable[EventTree],
    output: str | Path,
    *,
    summary: dict[str, object] | None = None,
    legacy_levels: bool = True,
) -> Path:
    """Write event trees to an awkward parquet file."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_records: list[dict[str, object]] = []
    level_records: list[dict[str, object]] = []
    for tree in trees:
        assign_levels(tree)
        event_records.append(_event_record(tree))
        if legacy_levels:
            level_records.extend(legacy_depth_samples(tree, channel=int(tree.metadata.get("channel", 0))))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "events": event_records,
        "legacy_levels": level_records,
        "summary_json": json.dumps(summary or {}, sort_keys=True),
    }
    ak.to_parquet(ak.Array([payload]), output_path)
    return output_path


def load_processed(path: str | Path) -> ak.Array:
    """Load a processed direct-mDST parquet file."""

    return ak.from_parquet(path)


def _event_record(tree: EventTree) -> dict[str, object]:
    grouped = nodes_by_level(tree)
    node_ids = sorted(tree.nodes)
    return {
        "event_id": tree.event_id,
        "root_ids": tree.root_ids,
        "levels": [{"level": level, "node_ids": ids} for level, ids in grouped.items()],
        "nodes": [_node_record(tree.nodes[node_id]) for node_id in node_ids],
        "metadata_json": json.dumps(dict(tree.metadata), sort_keys=True),
    }


def _node_record(node: TreeNode) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "pdg": node.pdg,
        "token": node.token,
        "charge": node.charge,
        "mass": node.p4.mass,
        "energy": node.p4.energy,
        "px": node.p4.px,
        "py": node.p4.py,
        "pz": node.p4.pz,
        "level": node.level,
        "parent_id": -1 if node.parent_id is None else node.parent_id,
        "daughter_ids": node.daughter_ids,
        "reco_id": "" if node.reco_id is None else node.reco_id,
        "mc_id": -1 if node.mc_id is None else node.mc_id,
        "copied_from": -1 if node.copied_from is None else node.copied_from,
        "source_node_id": -1 if node.source_node_id is None else node.source_node_id,
        "flags": sorted(node.flags),
        "prodTime": node.prod_time,
        "x": node.vertex[0],
        "y": node.vertex[1],
        "z": node.vertex[2],
        "nDaughters": node.n_daughters,
        "mc_px": None if node.mc_p4 is None else node.mc_p4.px,
        "mc_py": None if node.mc_p4 is None else node.mc_p4.py,
        "mc_pz": None if node.mc_p4 is None else node.mc_p4.pz,
        "mc_energy": None if node.mc_p4 is None else node.mc_p4.energy,
    }
