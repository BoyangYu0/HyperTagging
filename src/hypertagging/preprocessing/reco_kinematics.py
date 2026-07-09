"""Reco-level four-vector construction utilities."""

from __future__ import annotations

from hypertagging.preprocessing.mdst_tree_builder import EventTree, FourVector, recompute_mother_p4_from_daughters


def daughter_sum_p4(tree: EventTree, node_id: int) -> FourVector:
    """Return p4 from retained daughters for one node."""

    node = tree.nodes[node_id]
    return FourVector.sum(tree.nodes[child_id].p4 for child_id in node.daughter_ids)


def enforce_reco_mother_p4(tree: EventTree) -> None:
    """Recompute all mother p4 from reco daughters."""

    recompute_mother_p4_from_daughters(tree)
