import torch

from hypertagging.evaluation.hierarchical_metrics import next_level_metrics
from hypertagging.models.mother_pointer import MotherPointerOutput


def _batch():
    return {
        "node_features": torch.zeros((1, 4, 2)),
        "node_mask": torch.ones((1, 4), dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 1, 1]]),
        "daughter_adjacency": torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]],
            dtype=torch.bool,
        ),
        "pid_labels": torch.tensor([[1, 2, 3, 4]]),
        "pid_target_labels": torch.tensor([[1, 2, 3, 4]]),
        "p4": torch.zeros((1, 4, 4)),
        "charge": torch.zeros((1, 4)),
        "valid_reconstruction_target": torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
        "recursive_reconstructable_complete": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
    }


def test_next_level_validation_uses_requested_target_policy():
    pointer = MotherPointerOutput(
        object_logits=torch.full((1, 2), 10.0),
        type_logits=torch.zeros((1, 2, 5)),
        pointer_logits=torch.zeros((1, 2, 4)),
        cardinality_logits=torch.zeros((1, 2, 5)),
        confidence_logits=torch.zeros((1, 2)),
    )
    complete = next_level_metrics(
        pointer, _batch(), [[(0, 0)]], target_level=1, target_policy="complete_only"
    )
    partial = next_level_metrics(
        pointer, _batch(), [[(0, 0), (1, 1)]], target_level=1,
        target_policy="reconstructable_partial",
    )
    assert complete["object_precision_denominator"] == 2
    assert complete["object_precision_numerator"] == 1
    assert partial["object_precision_numerator"] == 2
