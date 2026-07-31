import torch

from hypertagging.evaluation.hierarchical_metrics import complete_target_efficiency_counts


def test_complete_efficiency_uses_eligible_mother_denominator():
    truth = {
        "node_mask": torch.ones((1, 4), dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 1, 1]]),
        "valid_reconstruction_target": torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
        "recursive_reconstructable_complete": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        "recursive_leaf_source_mask": torch.tensor(
            [[[1, 0], [0, 1], [1, 1], [1, 0]]], dtype=torch.bool
        ),
        "pid_labels": torch.tensor([[1, 2, 3, 4]]),
        "pid_target_labels": torch.tensor([[1, 2, 3, 4]]),
    }
    predicted = {key: value.clone() for key, value in truth.items()}
    correct, eligible = complete_target_efficiency_counts(predicted, truth)
    assert (correct, eligible) == (1, 1)
    correct_partial, eligible_partial = complete_target_efficiency_counts(
        predicted, truth, target_policy="reconstructable_partial"
    )
    assert (correct_partial, eligible_partial) == (2, 2)

