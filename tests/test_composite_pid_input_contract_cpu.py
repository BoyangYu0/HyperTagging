import torch

from hypertagging.reconstruction.pid_state import (
    hard_daughter_pid_histograms,
    soft_daughter_pid_histograms,
)


def test_truth_changes_do_not_change_composite_input_histogram():
    probabilities = torch.zeros(1, 3, 41)
    probabilities[0, 0, 0] = 1
    probabilities[0, 1, 4] = 1
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    adjacency[0, 2, :2] = True
    truth_a = torch.tensor([[3, 4, 8]])
    truth_b = torch.tensor([[6, 7, 8]])

    input_a = soft_daughter_pid_histograms(probabilities, adjacency)
    input_b = soft_daughter_pid_histograms(probabilities, adjacency)
    truth_hist_a = hard_daughter_pid_histograms(truth_a, adjacency)
    truth_hist_b = hard_daughter_pid_histograms(truth_b, adjacency)

    torch.testing.assert_close(input_a, input_b)
    assert not torch.equal(truth_hist_a, truth_hist_b)
    assert input_a[0, 2, 0] == 1  # unknown raw-track input stays unknown


def test_soft_runtime_histogram_is_differentiable():
    logits = torch.randn(1, 3, 41, requires_grad=True)
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    adjacency[0, 2, :2] = True
    histogram = soft_daughter_pid_histograms(logits.softmax(-1), adjacency)
    histogram[0, 2].square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
