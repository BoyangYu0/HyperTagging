import torch

from hypertagging.losses.level_reconstruction import targets_for_level


def _batch():
    return {
        "node_features": torch.zeros(1, 4, 2),
        "node_mask": torch.ones(1, 4, dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 1, 1]]),
        "daughter_adjacency": torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]],
            dtype=torch.bool,
        ),
        "pid_labels": torch.tensor([[1, 2, 3, 4]]),
        "pid_target_labels": torch.tensor([[1, 2, 3, 4]]),
        "p4": torch.zeros(1, 4, 4),
        "charge": torch.zeros(1, 4),
        "valid_reconstruction_target": torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
        "recursive_reconstructable_complete": torch.tensor(
            [[1, 1, 1, 0]], dtype=torch.bool
        ),
        "complete_reconstructable_decay": torch.tensor(
            [[1, 1, 1, 0]], dtype=torch.bool
        ),
        "partial_missing_daughters": torch.tensor([[0, 0, 0, 1]], dtype=torch.bool),
    }


def test_complete_only_is_default_and_partial_is_explicit():
    complete = targets_for_level(_batch(), 1)[0][0]
    partial = targets_for_level(
        _batch(), 1, target_policy="reconstructable_partial"
    )[0][0]
    assert complete.tolist() == [3]
    assert partial.tolist() == [3, 4]
