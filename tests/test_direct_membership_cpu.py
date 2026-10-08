import torch

from hypertagging.models.direct_membership import DirectMembershipHead, direct_membership_loss


def test_group_loss_is_invariant_to_truth_group_names_and_padding():
    torch.manual_seed(17)
    model = DirectMembershipHead(8)
    x, mask = torch.randn(2, 5, 8), torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]).bool()
    target = torch.tensor([[1, 1, 2, 2, -1], [0, 0, 0, -1, -1]])
    logits, objects = model(x, mask)
    original = direct_membership_loss(logits, objects, target, mask)
    renamed = torch.where(target > 0, 3 - target, target)
    torch.testing.assert_close(original, direct_membership_loss(logits, objects, renamed, mask))
    changed = x.clone()
    changed[~mask] = 1000
    other, other_objects = model(changed, mask)
    torch.testing.assert_close(logits[mask], other[mask])
    torch.testing.assert_close(objects, other_objects)
    original.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_empty_continuum_targets_reward_background_and_absent_groups():
    target, mask = torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool)
    correct = torch.tensor([[[8., -8., -8.]]]).expand(1, 4, 3)
    wrong = correct.flip(-1)
    assert direct_membership_loss(correct, torch.full((1, 2), -8.), target, mask) < .01
    assert direct_membership_loss(wrong, torch.full((1, 2), 8.), target, mask) > 10


def test_membership_head_is_equivariant_to_detector_order():
    torch.manual_seed(42)
    model = DirectMembershipHead(8).eval()
    x, mask = torch.randn(1, 6, 8), torch.ones(1, 6, dtype=torch.bool)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    a, presence_a = model(x, mask)
    b, presence_b = model(x[:, order], mask[:, order])
    torch.testing.assert_close(a[:, order], b)
    torch.testing.assert_close(presence_a, presence_b)
