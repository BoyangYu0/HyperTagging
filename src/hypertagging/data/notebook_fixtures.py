"""Deterministic schema-v2 fixture used by notebook smoke tests."""

from __future__ import annotations

from pathlib import Path

from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import (
    EventTree,
    FourVector,
    TreeNode,
    recompute_mother_p4_from_daughters,
    _annotate_reconstruction_contract,
)
from hypertagging.preprocessing.schema_v2 import export_trees_v2
from hypertagging.preprocessing.schema_v3 import export_trees_v3
from hypertagging.preprocessing.schema_v4 import export_trees_v4
from hypertagging.reconstruction.kinematics import (
    CHARGED_STABLE_NAMES,
    track_energy_hypotheses,
)
import torch
from hypertagging.preprocessing.channels import event_channel_record
from hypertagging.preprocessing.export_dataset import export_trees


def notebook_fixture_trees() -> list[EventTree]:
    trees = [_fixture_event(9001, conjugate=True), _fixture_event(9002, conjugate=False)]
    trees[1].nodes[5].flags.add("unmatched_reco")
    source = trees[1].nodes[5]
    clone = trees[1].nodes[6]
    clone.pdg = source.pdg
    clone.charge = source.charge
    clone.p4 = source.p4
    clone.reco_id = source.reco_id
    clone.node_kind = source.node_kind
    clone.track_features = dict(source.track_features)
    clone.cluster_features = dict(source.cluster_features)
    clone.mc_p4 = source.mc_p4
    clone.copied_from = 5
    clone.source_node_id = 5
    clone.flags.add("copied")
    for tree in trees:
        recompute_mother_p4_from_daughters(tree)
        _annotate_reconstruction_contract(tree, [])
        tree.full_truth_channel_record = event_channel_record(tree)
        tree.full_truth_channel_record_cc = event_channel_record(
            tree,
            charge_conjugate_normalize=True,
        )
        assign_levels(tree)
    return trees


def write_notebook_fixture(path: str | Path) -> Path:
    return export_trees_v2(
        notebook_fixture_trees(),
        path,
        summary={
            "fixture": True,
            "events": 2,
            "pid_summary": {
                "pdg_before": {"321": 2, "-321": 1, "22": 1, "211": 2, "-211": 2},
                "pdg_after": {"321": 2, "-321": 1, "22": 1, "211": 2, "-211": 2},
            },
        },
        charge_conjugate_normalize=True,
    )


def write_notebook_fixture_v1(path: str | Path) -> Path:
    """Write the same trees through the unchanged schema-v1 exporter."""

    trees = notebook_fixture_trees()
    return export_trees(
        trees,
        path,
        summary={
            "fixture": True,
            "events": len(trees),
            "nodes_before_pruning": sum(len(tree.nodes) for tree in trees),
            "nodes_after_pruning": sum(len(tree.nodes) for tree in trees),
            "unmatched_reco": 1,
            "collection": {"track_records": 7, "ecl_records": 1},
            "entry_sequences": None,
        },
    )


def write_notebook_fixture_v3(path: str | Path) -> Path:
    """Write the corrected tiny CPU training/notebook fixture."""

    trees = notebook_fixture_trees()
    return export_trees_v3(
        trees,
        path,
        summary={
            "fixture": True,
            "events": len(trees),
            "leaf_contract": "fixed_hypothesis_candidate",
        },
        charge_conjugate_normalize=True,
    )


def write_notebook_fixture_v4(
    path: str | Path,
    *,
    event_buffer_size: int = 1,
    row_group_size: int = 1,
) -> Path:
    """Write the truth-clean event-row fixture used by real trainers."""

    trees = notebook_fixture_trees()
    for tree in trees:
        for node in tree.nodes.values():
            if not node.daughter_ids and node.node_kind == "track":
                node.leaf_kinematics_mode = "raw_track_predicted_pid"
                node.input_pid_token = 0
                p3 = torch.tensor([node.p4.px, node.p4.py, node.p4.pz])
                energies = track_energy_hypotheses(p3).tolist()
                node.p4 = FourVector(
                    node.p4.px, node.p4.py, node.p4.pz, float(energies[2])
                )
                node.energy_source = "canonical_pion_hypothesis"
                node.track_energy_hypotheses = dict(
                    zip(CHARGED_STABLE_NAMES, energies)
                )
                node.track_energy_availability = {
                    name: True for name in CHARGED_STABLE_NAMES
                }
        recompute_mother_p4_from_daughters(tree)
    return export_trees_v4(
        trees,
        path,
        summary={"fixture": True, "events": len(trees)},
        charge_conjugate_normalize=True,
        event_buffer_size=event_buffer_size,
        row_group_size=row_group_size,
    )


def _fixture_event(event_id: int, *, conjugate: bool) -> EventTree:
    tree = EventTree(
        event_id=event_id,
        metadata={
            "event_uid": f"fixture:0:{event_id}:0",
            "source_file": "fixture.root",
            "source_category": "fixture",
            "fixture": 1,
        },
    )
    tree.add_node(TreeNode(0, 300553, 0, FourVector(0, 0, 0, 0), daughter_ids=[1, 2]))
    tree.add_node(TreeNode(1, 521, 1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[3, 4]))
    tree.add_node(TreeNode(2, -521, -1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[5, 6]))
    tree.add_node(
        TreeNode(
            3,
            321,
            1,
            FourVector(0.30, 0.02, 0.01, 0.62),
            parent_id=1,
            reco_id=f"Track:{event_id}:3",
            node_kind="track",
            mc_p4=FourVector(0.31, 0.02, 0.01, 0.63),
            track_features={"fit_p_value": 0.91, "d0": 0.01, "z0": -0.03},
        )
    )
    tree.add_node(
        TreeNode(
            4,
            -211,
            -1,
            FourVector(-0.12, 0.16, 0.02, 0.32),
            parent_id=1,
            reco_id=f"Track:{event_id}:4",
            node_kind="track",
            mc_p4=FourVector(-0.11, 0.15, 0.02, 0.33),
            track_features={"fit_p_value": 0.72, "d0": -0.02, "z0": 0.05},
        )
    )
    if conjugate:
        pdg5, charge5, kind5, reco5 = -321, -1.0, "track", f"Track:{event_id}:5"
        cluster_features = {}
        track_features = {"fit_p_value": 0.84, "d0": 0.03, "z0": 0.01}
    else:
        pdg5, charge5, kind5, reco5 = 22, 0.0, "ecl_cluster", f"ECLCluster:{event_id}:5"
        cluster_features = {
            "cluster_energy": 0.58,
            "theta": 1.1,
            "phi": -2.7,
            "time": 0.8,
            "e9_over_e21": 0.92,
            "n_crystals": 7.0,
            "photon_hypothesis": 1.0,
            "track_matched": 0.0,
        }
        track_features = {}
    tree.add_node(
        TreeNode(
            5,
            pdg5,
            charge5,
            FourVector(-0.28, -0.03, -0.01, 0.58),
            parent_id=2,
            reco_id=reco5,
            node_kind=kind5,
            mc_p4=FourVector(-0.27, -0.03, -0.01, 0.59),
            track_features=track_features,
            cluster_features=cluster_features,
        )
    )
    tree.add_node(
        TreeNode(
            6,
            211,
            1,
            FourVector(0.10, -0.14, -0.02, 0.30),
            parent_id=2,
            reco_id=f"Track:{event_id}:6",
            node_kind="track",
            mc_p4=FourVector(0.11, -0.13, -0.02, 0.31),
            track_features={"fit_p_value": 0.67, "d0": 0.0, "z0": 0.02},
        )
    )
    tree.root_ids = [0]
    return tree


__all__ = [
    "notebook_fixture_trees",
    "write_notebook_fixture",
    "write_notebook_fixture_v1",
    "write_notebook_fixture_v3",
    "write_notebook_fixture_v4",
]
