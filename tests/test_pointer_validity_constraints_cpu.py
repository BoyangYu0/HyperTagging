import torch

from hypertagging.models.mother_pointer import (
    constrained_daughter_decode,
    source_conflict_penalty,
)


def test_source_overlap_and_low_probability_are_rejected():
    probabilities = torch.tensor([0.95, 0.9, 0.2, 0.8])
    conflict = torch.zeros(4, 4, dtype=torch.bool)
    conflict[0, 1] = conflict[1, 0] = True
    selected, valid = constrained_daughter_decode(
        probabilities,
        cardinality=3,
        pointer_mask=torch.ones(4, dtype=torch.bool),
        source_conflict=conflict,
        min_probability=0.5,
        insufficient_policy="reduce",
    )
    assert valid
    assert selected.tolist() == [True, False, False, True]


def test_source_conflict_penalty_has_gradients():
    logits = torch.randn(1, 2, 4, requires_grad=True)
    conflict = torch.zeros(1, 4, 4, dtype=torch.bool)
    conflict[:, 0, 1] = conflict[:, 1, 0] = True
    loss = source_conflict_penalty(logits, conflict)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
