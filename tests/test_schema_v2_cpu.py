import json

import torch

from hypertagging.data.heterogeneous import load_heterogeneous_events
from hypertagging.preprocessing.channels import (
    canonical_decay_signature,
    deterministic_channel_id,
    event_channel_record,
    structured_channel_similarity,
    unordered_b_pair_signature,
)
from hypertagging.preprocessing.export_dataset import export_trees
from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import (
    EventTree,
    FourVector,
    TreeNode,
    recompute_mother_p4_from_daughters,
)
from hypertagging.preprocessing.schema_v2 import (
    SCHEMA_VERSION_V2,
    export_trees_v2,
    load_payload_v2,
)


def _two_b_tree(*, conjugate=True):
    tree = EventTree(event_id=77, metadata={"event_uid": "1:2:77:3", "source_file": "tiny.root"})
    tree.add_node(TreeNode(0, 300553, 0, FourVector(0, 0, 0, 0), daughter_ids=[1, 2]))
    tree.add_node(TreeNode(1, 521, 1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[3, 4]))
    tree.add_node(TreeNode(2, -521, -1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[5, 6]))
    tree.add_node(
        TreeNode(
            3,
            321,
            1,
            FourVector(0.2, 0.0, 0.0, 0.6),
            parent_id=1,
            reco_id="Track:3",
            node_kind="track",
            track_features={"d0": 0.01, "fit_p_value": 0.8},
        )
    )
    tree.add_node(
        TreeNode(
            4,
            -211,
            -1,
            FourVector(-0.1, 0.1, 0.0, 0.3),
            parent_id=1,
            reco_id="Track:4",
            node_kind="track",
        )
    )
    tree.add_node(
        TreeNode(
            5,
            -321 if conjugate else 22,
            -1 if conjugate else 0,
            FourVector(-0.2, 0.0, 0.0, 0.6),
            parent_id=2,
            reco_id="Track:5" if conjugate else "ECLCluster:5",
            node_kind="track" if conjugate else "ecl_cluster",
            cluster_features={} if conjugate else {"cluster_energy": 0.6, "time": 1.2},
        )
    )
    tree.add_node(
        TreeNode(
            6,
            211,
            1,
            FourVector(0.1, -0.1, 0.0, 0.3),
            parent_id=2,
            reco_id="Track:6",
            node_kind="track",
        )
    )
    tree.root_ids = [0]
    recompute_mother_p4_from_daughters(tree)
    assign_levels(tree)
    return tree


def test_v2_round_trip_and_heterogeneous_blocks(tmp_path):
    tree = _two_b_tree(conjugate=False)
    path = export_trees_v2([tree], tmp_path / "v2.parquet")
    payload = load_payload_v2(path)

    assert payload["schema_version"] == SCHEMA_VERSION_V2
    event = payload["events"][0]
    nodes = {node["node_id"]: node for node in event["nodes"]}
    assert nodes[3]["node_kind"] == "track"
    assert nodes[3]["track_availability"]["d0"]
    assert not any(nodes[3]["cluster_availability"].values())
    assert nodes[5]["node_kind"] == "ecl_cluster"
    assert nodes[5]["cluster_availability"]["time"]
    assert not any(nodes[5]["track_availability"].values())
    assert nodes[1]["node_kind"] == "composite"
    assert nodes[1]["composite_availability"]["daughter_sum_energy"]
    assert not any(nodes[1]["track_availability"].values())
    assert len(event["b1_channel_count_array"]) == 41
    assert event["same_event"]
    assert not event["exact_channel_equal"]

    loaded = load_heterogeneous_events(path)
    assert loaded[0].common_features.shape[-1] == 12
    assert loaded[0].node_kind_ids.unique().numel() >= 3
    assert torch.isfinite(loaded[0].common_features).all()


def test_v1_adapter_preserves_original_fields_and_does_not_fabricate_detector_values(tmp_path):
    tree = _two_b_tree()
    path = export_trees([tree], tmp_path / "v1.parquet")
    original = load_payload_v2(path)
    event = original["events"][0]
    node = next(node for node in event["nodes"] if node["node_id"] == 3)

    assert original["source_schema_version"] == "direct-mdst-tree-v1"
    assert node["px"] == tree.nodes[3].p4.px
    assert node["reco_id"] == "Track:3"
    assert node["node_kind"] == "track"
    assert not any(node["track_availability"].values())
    assert not any(node["cluster_availability"].values())
    composite = next(node for node in event["nodes"] if node["node_id"] == 1)
    assert composite["composite_availability"]["daughter_sum_energy"]
    assert composite["composite_features"]["daughter_sum_energy"] == composite["energy"]


def test_channel_signatures_charge_conjugation_and_ids_are_stable():
    tree = _two_b_tree()
    raw_left = canonical_decay_signature(tree, 1)
    raw_right = canonical_decay_signature(tree, 2)
    normalized_left = canonical_decay_signature(tree, 1, charge_conjugate_normalize=True)
    normalized_right = canonical_decay_signature(tree, 2, charge_conjugate_normalize=True)

    assert raw_left != raw_right
    assert normalized_left == normalized_right
    assert deterministic_channel_id(normalized_left) == deterministic_channel_id(normalized_right)
    record = event_channel_record(tree, charge_conjugate_normalize=True)
    assert record["exact_channel_equal"]
    assert record["structured_channel_similarity"] == 1.0
    before_copy_metadata = canonical_decay_signature(tree, 1)
    tree.nodes[3].copied_from = 4
    tree.nodes[3].source_node_id = 4
    assert canonical_decay_signature(tree, 1) == before_copy_metadata


def test_structured_similarity_and_unordered_pair_are_symmetric():
    left = {"pid_counts": [{"token": 1, "count": 2}], "depth_pid_counts": [], "branch_multiplicities": []}
    right = {"pid_counts": [{"token": 1, "count": 1}], "depth_pid_counts": [], "branch_multiplicities": []}
    assert structured_channel_similarity(left, right) == structured_channel_similarity(right, left)
    first = unordered_b_pair_signature(["b", "a"])
    second = unordered_b_pair_signature(["a", "b"])
    assert first == second == json.dumps(["a", "b"], separators=(",", ":"))


def test_channel_multiplicity_similarity_uses_canonical_counts_not_list_positions():
    common = {
        "pid_counts": [],
        "depth_pid_counts": [],
        "selected_intermediate_counts": [],
    }
    left = {
        **common,
        "branch_multiplicities": [2, 3],
        "branch_multiplicity_counts": [
            {"multiplicity": 2, "count": 1},
            {"multiplicity": 3, "count": 1},
        ],
    }
    reordered = {
        **common,
        "branch_multiplicities": [3, 2],
        "branch_multiplicity_counts": list(
            reversed(left["branch_multiplicity_counts"])
        ),
    }
    extra_two_body = {
        **common,
        "branch_multiplicities": [2, 2, 3],
        "branch_multiplicity_counts": [
            {"multiplicity": 2, "count": 2},
            {"multiplicity": 3, "count": 1},
        ],
    }
    assert structured_channel_similarity(left, reordered) == 1.0
    assert 0.0 < structured_channel_similarity(left, extra_two_body) < 1.0


def test_same_event_is_not_the_same_as_same_channel():
    record = event_channel_record(_two_b_tree(conjugate=False))
    assert record["same_event"]
    assert not record["exact_channel_equal"]
