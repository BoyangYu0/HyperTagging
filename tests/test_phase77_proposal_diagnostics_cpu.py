import torch
from scripts.diagnose_phase77_proposals import (
    describe,
    loss_parts,
    logit_loss_audit,
    auc,
    hard_mapping,
)
from hypertagging.models.direct_membership import direct_membership_loss


def test_native_loss_parts_and_target_permutation_gradients():
    torch.manual_seed(13)
    z = torch.randn(7, 3, requires_grad=True)
    o = torch.randn(2, requires_grad=True)
    target = torch.tensor([1, 1, 2, 2, 0, 0, -1])
    mask = torch.ones(1, 7, dtype=torch.bool)
    part, _, alternatives = loss_parts(z, o, target)
    actual = direct_membership_loss(z[None], o[None], target[None], mask)
    assert torch.allclose(actual, alternatives.min())
    swapped = torch.where(target > 0, 3 - target, target)
    other = direct_membership_loss(z[None], o[None], swapped[None], mask)
    assert torch.equal(actual, other)
    g = torch.autograd.grad(actual, z, retain_graph=True)[0]
    assert g[-1].abs().sum() == 0
    assert torch.equal(g, torch.autograd.grad(other, z)[0])
    assert torch.allclose(part["ce"] + part["overlap"] + part["presence"], actual)


def test_three_way_and_binary_decisions_are_not_equivalent():
    p = torch.tensor(
        [[0.4, 0.35, 0.25], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]
    )
    c, _, _, _, _ = describe(p.log(), torch.ones(2), torch.tensor([1, 2, 0, 0]))
    assert c["B_to_unassigned"] == 1 and c["background_to_B"] == 0
    assert c["aggregate_B_to_unassigned"] == 0 and c["aggregate_background_to_B"] == 1
    assert c["B_unassigned_but_aggregate_gt_half"] == 1
    assert c["oracle_foreground_raw_exact"] == 2
    assert c["aggregate_half_raw_exact"] == 1


def test_unknown_support_and_native_loss_gradient_accounting():
    z = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    y = torch.tensor([0, 1, 2, -1])
    o = torch.zeros(2)
    c, _, _, _, _ = describe(z, o, y)
    a = logit_loss_audit(z, o, y)
    assert c["unknown_nodes"] == 1 and c["background_nodes"] == 1 and c["B_nodes"] == 2
    assert a["classes"]["unknown"]["ce_gradient_squared_norm"] == 0
    assert a["classes"]["unknown"]["overlap_gradient_squared_norm"] == 0
    assert (
        abs(
            sum(a["classes"][k]["ce_event_contribution"] for k in a["classes"])
            - a["ce"]
        )
        < 1e-6
    )
    none = logit_loss_audit(z, o, torch.full_like(y, -1))
    assert none["total"] == 0 and none["valid_nodes"] == 0


def test_presence_does_not_change_raw_membership_and_canonical_ties():
    z = torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0]])
    y = torch.tensor([1, 2, 0])
    swapped = torch.where(y > 0, 3 - y, y)
    assert torch.equal(
        hard_mapping(z.softmax(-1), y)[0], hard_mapping(z.softmax(-1), swapped)[0]
    )
    c1, _, _, _, _ = describe(z, torch.ones(2) * 20, y)
    c2, _, _, _, _ = describe(z, torch.ones(2) * -20, swapped)
    assert c1 == c2
    assert c1["native_raw_exact"] == 0


def test_auc_ties_and_unavailable_support():
    assert auc([0.1, 0.2], [0, 1])["auc"] == 1
    assert auc([0.2, 0.1], [0, 1])["auc"] == 0
    assert auc([0.5, 0.5], [0, 1])["auc"] == 0.5
    assert auc([0.5], [0])["auc"] is None
