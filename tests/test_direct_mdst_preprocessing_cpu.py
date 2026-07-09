import math

from hypertagging.preprocessing.export_dataset import export_trees, load_processed
from hypertagging.preprocessing.levelize_tree import adjacent_level_samples, assign_levels, nodes_by_level
from hypertagging.preprocessing.mdst_tree_builder import (
    EventTree,
    FourVector,
    MCRecord,
    RecoRecord,
    TreeNode,
    build_truth_guided_tree,
    copy_shared_daughters,
    recompute_mother_p4_from_daughters,
    validate_tree,
)
from hypertagging.preprocessing.pid_filter import PidFilter, tokenize_pdg


def _toy_records():
    mc = [
        MCRecord(0, 300553, 0.0, None, FourVector(0, 0, 0, 10.58), "Upsilon(4S)", True),
        MCRecord(1, 521, 1.0, 0, FourVector(0.1, 0, 0, 5.29), "B+", True),
        MCRecord(2, 421, 0.0, 1, FourVector(0.4, 0, 0, 2.0), "D0", True),
        MCRecord(3, 321, 1.0, 2, FourVector(0.2, 0, 0, 0.6), "K+", True),
        MCRecord(4, -211, -1.0, 2, FourVector(0.1, 0.1, 0, 0.4), "pi-", True),
        MCRecord(5, 113, 0.0, 1, FourVector(0, 0, 0, 0.7), "rho0", True),
        MCRecord(6, 22, 0.0, 5, FourVector(0, 0.2, 0, 0.2), "gamma", True),
    ]
    reco = [
        RecoRecord("trk-k", 321, 1.0, FourVector(0.25, 0.0, 0.0, 0.65), 3),
        RecoRecord("trk-pi", -211, -1.0, FourVector(0.10, 0.15, 0.0, 0.35), 4),
        RecoRecord("ecl-gamma", 22, 0.0, FourVector(0.0, 0.20, 0.0, 0.20), 6),
    ]
    return mc, reco


def test_truth_guided_tree_prunes_pid_and_recomputes_mother_p4():
    pid_filter = PidFilter()
    mc, reco = _toy_records()

    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco, pid_filter=pid_filter)
    assign_levels(tree)
    stats = validate_tree(tree)

    pdgs = {node.pdg for node in tree.nodes.values()}
    assert 113 not in pdgs
    assert 22 in pdgs
    assert stats["max_abs_p4_diff"] == 0.0
    d0 = next(node for node in tree.nodes.values() if node.pdg == 421)
    daughters = [tree.nodes[child_id] for child_id in d0.daughter_ids]
    assert math.isclose(d0.p4.energy, sum(child.p4.energy for child in daughters))
    assert not math.isclose(d0.p4.energy, 2.0)
    assert pid_filter.summary.dropped["name_not_allowed"] == 1


def test_levelisation_and_adjacent_level_links():
    mc, reco = _toy_records()
    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco)
    assign_levels(tree)

    grouped = nodes_by_level(tree)
    assert 0 in grouped
    assert max(grouped) >= 2
    samples = adjacent_level_samples(tree)
    assert samples
    assert all(len(sample.links) == len(sample.input_node_ids) for sample in samples)


def test_copy_shared_daughters_clones_subtree():
    tree = EventTree(event_id=1)
    tree.add_node(TreeNode(0, 521, 1, FourVector(0, 0, 0, 0), daughter_ids=[2]))
    tree.add_node(TreeNode(1, -521, -1, FourVector(0, 0, 0, 0), daughter_ids=[2]))
    tree.add_node(TreeNode(2, 211, 1, FourVector(1, 0, 0, 1.2), parent_id=0))
    tree.root_ids = [0, 1]

    copied = copy_shared_daughters(tree)
    recompute_mother_p4_from_daughters(tree)
    assign_levels(tree)
    validate_tree(tree)

    assert copied == 1
    copied_nodes = [node for node in tree.nodes.values() if node.copied_from == 2]
    assert len(copied_nodes) == 1
    assert copied_nodes[0].p4.energy == tree.nodes[2].p4.energy


def test_export_contains_canonical_and_legacy_views(tmp_path):
    mc, reco = _toy_records()
    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco)
    assign_levels(tree)
    output = export_trees([tree], tmp_path / "processed.parquet", summary={"events": 1})

    payload = load_processed(output).to_list()[0]
    assert payload["schema_version"] == "direct-mdst-tree-v1"
    assert payload["events"][0]["event_id"] == 12
    assert payload["legacy_levels"]
    first_level = payload["legacy_levels"][0]
    assert "feature" in first_level
    assert "motherIndex" in first_level
    assert first_level["feature"][0][0] in {tokenize_pdg(node["pdg"]) for node in payload["events"][0]["nodes"]}
