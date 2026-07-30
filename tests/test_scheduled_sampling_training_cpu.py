import torch

from hypertagging.training.scheduled_sampling import (
    TeacherForcingSchedule,
    align_context_by_recursive_sources,
)


def test_schedule_is_reproducible_and_stateful():
    schedule = TeacherForcingSchedule(
        kind="linear", start_probability=1.0, end_probability=0.2, duration_steps=10
    )
    assert schedule.probability(0) == 1.0
    assert schedule.probability(10) == 0.2
    choices_a = schedule.sample(8, step=4, seed=17)
    choices_b = schedule.sample(8, step=4, seed=17)
    assert torch.equal(choices_a, choices_b)


def test_recursive_source_alignment_counts_unrepresentable_targets():
    predicted = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    truth = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    alignment = align_context_by_recursive_sources(predicted, truth)
    assert alignment.predicted_to_truth.tolist() == [0, -1]
    assert alignment.representable.tolist() == [True, False]
