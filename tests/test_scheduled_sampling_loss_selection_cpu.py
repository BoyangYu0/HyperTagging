import torch

from hypertagging.training.scheduled_sampling import combine_sampled_context_losses


def test_sampled_context_is_primary_and_not_double_counted():
    teacher = torch.tensor([1.0, 2.0], requires_grad=True)
    predicted = torch.tensor([10.0, 20.0], requires_grad=True)
    choose_teacher = torch.tensor([True, False])
    total, metrics = combine_sampled_context_losses(
        teacher, predicted, choose_teacher, auxiliary_teacher_weight=0.0
    )
    torch.testing.assert_close(total, torch.tensor(10.5))
    total.backward()
    torch.testing.assert_close(teacher.grad, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(predicted.grad, torch.tensor([0.0, 0.5]))
    assert metrics["sampled_teacher_count"] == 1
    assert metrics["sampled_predicted_count"] == 1

