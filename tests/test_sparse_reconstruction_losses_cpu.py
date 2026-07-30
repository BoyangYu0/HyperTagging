import torch

from hypertagging.losses.level_reconstruction import focal_binary_cross_entropy_with_logits
from hypertagging.losses.physics import p4_sum_consistency_loss


def test_positive_sparse_examples_receive_weighted_gradient():
    logits = torch.full((100,), -2.0, requires_grad=True)
    targets = torch.zeros(100)
    targets[0] = 1
    loss = focal_binary_cross_entropy_with_logits(
        logits,
        targets,
        positive_weight=20,
        gamma=2,
    )
    loss.backward()
    assert abs(float(logits.grad[0])) > abs(float(logits.grad[1]))
    assert torch.isfinite(logits.grad).all()


def test_scaled_physics_loss_is_finite():
    pointers = torch.tensor([[[2.0, 2.0]]], requires_grad=True)
    daughters = torch.tensor([[[1.0, 0.0, 0.0, 1.2], [-1.0, 0.0, 0.0, 1.2]]])
    target = daughters.sum(1, keepdim=True)
    loss = p4_sum_consistency_loss(
        pointers,
        daughters,
        target,
        component_scales=(1.0, 0.5, 0.5, 0.5),
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pointers.grad).all()
