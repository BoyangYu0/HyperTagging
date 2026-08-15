import torch

from hypertagging.training.reconstruction_trainer import _collate_context_batches


def _context(nodes: int) -> dict[str, torch.Tensor]:
    active = torch.ones((1, nodes), dtype=torch.bool)
    return {
        "active": active,
        "node_mask": active,
        "common_features": torch.ones((1, nodes, 12)),
        "klm_features": torch.ones((1, nodes, 9)),
        "klm_availability": torch.ones((1, nodes, 9), dtype=torch.bool),
        "daughter_input_pid_source_ids": torch.arange(nodes).reshape(1, nodes),
        "daughter_truth_pid_source_ids": torch.arange(nodes).reshape(1, nodes),
        "model_input_source_ids": torch.arange(nodes).reshape(1, nodes),
        "truth_supervision_source_ids": torch.arange(nodes).reshape(1, nodes),
        "depth_from_retained_root": torch.arange(nodes).reshape(1, nodes),
        "distance_to_nearest_retained_root": torch.arange(nodes).reshape(1, nodes),
        "daughter_adjacency": torch.eye(nodes, dtype=torch.bool).reshape(1, nodes, nodes),
        "recursive_leaf_source_mask": torch.ones((1, nodes, nodes + 2), dtype=torch.bool),
    }


def test_dynamic_context_collation_pads_schema_v4_node_fields():
    combined = _collate_context_batches([_context(5), _context(3)])

    assert combined["node_mask"].shape == (2, 5)
    assert combined["common_features"].shape == (2, 5, 12)
    assert combined["klm_features"].shape == (2, 5, 9)
    assert combined["daughter_adjacency"].shape == (2, 5, 5)
    assert combined["recursive_leaf_source_mask"].shape == (2, 5, 7)
    assert combined["model_input_source_ids"].shape == (2, 5)
    assert combined["model_input_source_ids"][1, 3:].tolist() == [-1, -1]
    assert combined["klm_features"][1, 3:].count_nonzero() == 0


if __name__ == "__main__":
    test_dynamic_context_collation_pads_schema_v4_node_fields()
