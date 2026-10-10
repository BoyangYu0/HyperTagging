"""Bounded synthetic mechanism regressions, not scientific performance evidence."""

import pytest
import torch
from hypertagging.models.assembly_development import AssemblyMembershipDecoder
from hypertagging.models.pair_conditioned_membership import (
    PairConditionedMembershipDecoder,
    semantic_pair_loss,
)
from hypertagging.training.capacity_development import membership_loss


def model(on=True):
    torch.manual_seed(91)
    return PairConditionedMembershipDecoder(
        connection_enabled=on, addition_seed=22
    ).eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    return torch.randn(6, 128, generator=g), torch.eye(6, dtype=torch.bool)


def test_native_initial_state_and_rng_exact():
    torch.manual_seed(91)
    native = AssemblyMembershipDecoder().eval()
    rng = torch.get_rng_state().clone()
    h, s = inputs()
    expected = native(h, s)
    for on in (False, True):
        m = model(on)
        assert torch.equal(torch.get_rng_state(), rng)
        for k, v in native.state_dict().items():
            assert torch.equal(v, m.state_dict()[k])
        actual = m(h, s)
        for key in ["logits", "objects", "proposal_logits", "proposal_objects"]:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert sum(p.numel() for p in model().parameters()) == sum(
        p.numel() for p in model(False).parameters()
    )


def test_source_unknown_and_slot_masks():
    h, s = inputs()
    s[1] = s[0]
    s[5] = False
    r = model()(h, s)
    assert [0, 1] not in r["edge_pairs"].T.tolist()
    assert 5 not in r["edge_pairs"].flatten().tolist()
    t = torch.tensor([1, 1, 2, 0, -1, 0])
    a, support = semantic_pair_loss(r["edge_logits"], r["edge_pairs"], t)
    b, other = semantic_pair_loss(
        r["edge_logits"], r["edge_pairs"], torch.where(t > 0, 3 - t, t)
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert support == other
    assert support["eligible_pairs"] == 9
    assert support["source_excluded_pairs"] == 6
    assert support["unknown_pairs"] == 4
    assert (
        sum(support[k] for k in ["same_b", "cross_b", "background", "unknown_pairs"])
        == 9
    )


def test_messages_permutation_and_truth_not_input():
    h, s = inputs()
    m = model()
    with torch.no_grad():
        m.message[-1].weight.fill_(0.003)
    perm = torch.tensor([3, 0, 5, 2, 1, 4])
    original = m.semantic_message(h, s)[0]
    reordered = m.semantic_message(h[perm], s[perm])[0]
    torch.testing.assert_close(reordered, original[perm], atol=1e-6, rtol=1e-5)
    # Disable native partial merge context only for the equivariance fixture:
    # native tie-breaking is unchanged and is not the semantic edge mechanism.
    m.partial_state_conditioning = False
    a = m(h, s)
    b = m(h[perm], s[perm])
    torch.testing.assert_close(b["logits"], a["logits"][:, perm], atol=1e-6, rtol=1e-5)
    with pytest.raises(TypeError):
        m(h, s, targets=torch.ones(6, dtype=torch.long))


def test_membership_gradient_opens_after_zero_projection_update():
    h, s = inputs()
    target = torch.tensor([1, 1, 2, 2, 0, 0])
    for on in [False, True]:
        m = model(on)
        # One controlled membership update: initially only final message projection can learn.
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        membership_loss(m(h, s), target).backward()
        assert all(
            p.grad is None or not p.grad.any() for p in m.semantic_edges.parameters()
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        membership_loss(m(h, s), target).backward()
        magnitude = sum(
            float(p.grad.abs().sum())
            for p in m.semantic_edges.parameters()
            if p.grad is not None
        )
        assert (magnitude > 0) == on
        m.zero_grad(set_to_none=True)
        r = m(h, s)
        semantic_pair_loss(r["edge_logits"], r["edge_pairs"], target)[0].backward()
        assert (
            sum(
                float(p.grad.abs().sum())
                for p in m.semantic_edges.parameters()
                if p.grad is not None
            )
            > 0
        )


def test_edge_perturbation_changes_refinement_only():
    h, s = inputs()
    m = model()
    with torch.no_grad():
        torch.nn.init.normal_(m.message[-1].weight, std=0.02)
    a = m(h, s)
    with torch.no_grad():
        m.semantic_edges[-1].weight.mul_(8)
    b = m(h, s)
    assert not torch.allclose(a["logits"], b["logits"])
    torch.testing.assert_close(
        a["proposal_logits"], b["proposal_logits"], rtol=0, atol=0
    )


def test_no_pairs_and_unsupported_message_zero():
    h, s = inputs()
    s[:] = False
    m = model()
    with torch.no_grad():
        m.message[-1].bias.fill_(1)
    message, logits, pairs = m.semantic_message(h, s)
    assert not message.any() and pairs.shape == (2, 0)
    loss, support = semantic_pair_loss(
        logits, pairs, torch.full((6,), -1, dtype=torch.long)
    )
    assert torch.isfinite(loss) and loss == 0 and support["eligible_pairs"] == 0
    loss.backward()
    assert torch.isfinite(m(h, s)["logits"]).all()


@pytest.mark.parametrize("shape", [(3, 256), (0, 128), (257, 128)])
def test_bad_capacity_width(shape):
    with pytest.raises(ValueError):
        model()(torch.zeros(shape), torch.eye(shape[0], dtype=torch.bool))


def test_bad_flags_and_pairs():
    with pytest.raises(ValueError):
        PairConditionedMembershipDecoder(connection_enabled=1, addition_seed=22)
    with pytest.raises(ValueError):
        semantic_pair_loss(
            torch.zeros(2, 3), torch.tensor([[0, 0], [1, 1]]), torch.tensor([1, 2])
        )


def test_refinement_equivariance_with_singleton_latent_context(monkeypatch):
    h, s = inputs()
    m = model()
    with torch.no_grad():
        torch.nn.init.normal_(m.message[-1].weight, std=0.02)

    def singleton_states(features, sources):
        return [
            {
                "groups": tuple(frozenset([i]) for i in range(len(features))),
                "pairs": (),
                "logits": features.new_empty((0, 3)),
            }
        ]

    monkeypatch.setattr(m, "generate", singleton_states)
    perm = torch.tensor([2, 5, 1, 0, 4, 3])
    a = m(h, s)
    b = m(h[perm], s[perm])
    torch.testing.assert_close(b["logits"], a["logits"][:, perm], atol=1e-6, rtol=1e-5)
    before = a["logits"].detach().clone()
    for targets in [
        torch.tensor([1, 1, 2, 0, -1, 0]),
        torch.zeros(6, dtype=torch.long),
    ]:
        semantic_pair_loss(a["edge_logits"], a["edge_pairs"], targets)
    torch.testing.assert_close(m(h, s)["logits"], before, rtol=0, atol=0)


def test_semantic_loss_is_present_class_mean_not_pair_mean():
    targets = torch.tensor([1, 1, 2, 0])
    pairs = torch.triu_indices(4, 4, 1)
    logits = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 7
    loss, counts = semantic_pair_loss(logits, pairs, targets)
    labels = torch.tensor([0, 1, 2, 1, 2, 2])
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(logits[labels == c], labels[labels == c])
            for c in range(3)
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert [counts[k] for k in ["same_b", "cross_b", "background"]] == [1, 2, 3]
