import torch

from hypertagging.evaluation.hierarchical_metrics import canonical_tree_metrics, tree_validity_rate


def _tree(mother_id, mother_type=4, missing=False):
    node_ids = torch.tensor([[10, 11, mother_id]])
    adjacency = torch.zeros((1, 3, 3), dtype=torch.bool)
    adjacency[0, 2, 0] = True
    if not missing:
        adjacency[0, 2, 1] = True
    sources = torch.tensor([[[1, 0], [0, 1], [1, 1]]], dtype=torch.bool)
    return {
        "node_ids": node_ids,
        "node_mask": torch.ones((1, 3), dtype=torch.bool),
        "daughter_adjacency": adjacency,
        "pid_labels": torch.tensor([[2, 2, mother_type]]),
        "pid_target_labels": torch.tensor([[2, 2, mother_type]]),
        "recursive_leaf_source_mask": sources,
        "reco_ids": torch.tensor([[100, 101, -1]]),
        "source_node_ids": node_ids.clone(),
        "level_ids": torch.tensor([[0, 0, 1]]),
        "p4": torch.zeros((1, 3, 4)),
    }


def test_different_generated_ids_are_canonical_exact_matches():
    metrics = canonical_tree_metrics(_tree(20), _tree(999))
    assert metrics.full_tree_exact_match
    assert metrics.edge_f1 == 1
    assert metrics.first_divergence_level == -1
    assert tree_validity_rate(_tree(20)) == 1


def test_wrong_type_and_missing_edge_are_detected():
    assert not canonical_tree_metrics(_tree(20, mother_type=5), _tree(99)).full_tree_exact_match
    missing = canonical_tree_metrics(_tree(20, missing=True), _tree(99))
    assert not missing.full_tree_exact_match
    assert missing.edge_recall < 1


def test_actual_cycle_detection():
    cyclic = _tree(20)
    cyclic["daughter_adjacency"][0, 0, 2] = True
    assert tree_validity_rate(cyclic) == 0
