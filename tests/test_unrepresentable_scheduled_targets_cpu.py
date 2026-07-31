import torch
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.mother_pointer import MotherPointerOutput

from hypertagging.training.scheduled_sampling import (
    aligned_level_targets,
    resolve_unrepresentable_target_policy,
)


def test_absent_earlier_composite_falls_back_instead_of_all_no_object():
    # Truth level 2 depends on a level-1 composite with source set {a,b}; the
    # predicted context contains only the two leaves because that composite was
    # missed in the prior rollout step.
    truth = {
        "node_mask": torch.tensor([[True, True, True, True]]),
        "level_ids": torch.tensor([[0, 0, 1, 2]]),
        "valid_reconstruction_target": torch.tensor([[False, False, True, True]]),
        "recursive_reconstructable_complete": torch.ones((1, 4), dtype=torch.bool),
        "daughter_adjacency": torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]]],
            dtype=torch.bool,
        ),
        "recursive_leaf_source_mask": torch.tensor(
            [[[1, 0], [0, 1], [1, 1], [1, 1]]], dtype=torch.bool
        ),
        "pid_labels": torch.tensor([[2, 3, 8, 9]]),
        "pid_target_labels": torch.tensor([[2, 3, 8, 9]]),
        "p4": torch.ones((1, 4, 4)),
        "charge": torch.zeros((1, 4)),
    }
    predicted = {
        key: value[:, :2] if value.ndim == 2 else value[:, :2, :]
        for key, value in truth.items()
        if key not in {"daughter_adjacency"}
    }
    predicted["daughter_adjacency"] = torch.zeros((1, 2, 2), dtype=torch.bool)

    aligned = aligned_level_targets(truth, predicted, target_level=2, min_daughters=1)
    assert aligned.truth_target_count == 1
    assert aligned.representable_count == 0
    decision = resolve_unrepresentable_target_policy(
        "fallback_teacher",
        truth_target_count=aligned.truth_target_count,
        representable_target_count=aligned.representable_count,
    )
    assert decision.use_teacher_context
    assert not decision.use_representable_subset
    assert not decision.skip_event_level

    pointer = MotherPointerOutput(
        object_logits=torch.tensor([[-10.0]]),
        type_logits=torch.zeros((1, 1, 41)),
        pointer_logits=torch.zeros((1, 1, 2)),
        cardinality_logits=torch.zeros((1, 1, 3)),
        confidence_logits=torch.zeros((1, 1)),
    )
    empty_override = ([torch.empty(0, dtype=torch.long)],
                      [torch.zeros((0, 2), dtype=torch.bool)],
                      [torch.zeros((0, 4))], [torch.zeros(0)])
    loss = level_reconstruction_loss(
        pointer, predicted, target_level=2, target_override=empty_override,
        unrepresentable_target_counts=[1],
    )
    assert float(loss.components["object"]) == 0.0
