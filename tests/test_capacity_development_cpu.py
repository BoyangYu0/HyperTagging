import copy
import pytest
import torch
from hypertagging.models.assembly_development import (
    AssemblyMembershipDecoder,
    assembly_relation_loss,
    relation_targets,
)
from hypertagging.training.capacity_development import (
    fresh_decoder,
    fresh_model,
    baseline_loss,
)
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events


def supervision():
    return {
        "node_sets": [frozenset([i]) for i in range(4)]
        + [frozenset([0, 1]), frozenset([2, 3]), frozenset(range(4))],
        "parents": [4, 4, 5, 5, 6, 6, -1],
        "b_groups": [frozenset(range(4))],
    }


@pytest.mark.parametrize("width", [128, 256])
def test_common_lossless_interface_and_consumed_group_gradients(width):
    h = torch.randn(4, width, requires_grad=True)
    model = fresh_decoder()
    padded = model.interface(h)
    torch.testing.assert_close(padded[:, :width], h, rtol=0, atol=0)
    assert padded.shape[-1] == 256
    result = model(h, torch.eye(4, dtype=torch.bool))
    result["logits"].square().mean().backward()
    assert h.grad.abs().sum() > 0
    assert model.proposal.nodes[1].weight.grad.abs().sum() > 0
    assert model.condition[0].weight.grad.abs().sum() > 0
    without = model(
        h.detach(), torch.eye(4, dtype=torch.bool), group_conditioning=False
    )
    assert not torch.equal(result["logits"], without["logits"])


def test_generation_does_not_consume_truth_and_preserves_sources():
    model = fresh_decoder()
    h = torch.randn(5, 128)
    sources = torch.eye(5, dtype=torch.bool)
    sources[1] = sources[0]
    a = model(h, sources)
    unrelated_truth = copy.deepcopy(supervision())
    unrelated_truth["parents"] = [-1] * 7
    # Supervision may change loss, never generation/conditioning/ranking.
    assembly_relation_loss(a["states"], unrelated_truth)
    b = model(h, sources)
    torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    for state in a["states"]:
        for group in state["groups"]:
            assert int(sources[list(group)].sum()) == int(
                sources[list(group)].any(0).sum()
            )


def test_parent_and_sibling_discrimination_is_not_global_repulsion():
    s = supervision()
    groups = tuple(frozenset([i]) for i in range(4))
    assert relation_targets(groups, [(0, 1), (0, 2)], s) == [0, 2]
    s["node_sets"][4] = frozenset([0, 1, 2])
    assert relation_targets(groups, [(0, 1)], s) == [1]
    assert relation_targets(
        [frozenset([0, 1]), frozenset([2, 3])], [(0, 1)], supervision()
    ) == [0]


@pytest.mark.parametrize("width", [128, 256])
def test_both_widths_relation_objective_has_real_gradients(width):
    h = torch.randn(4, width, requires_grad=True)
    model = fresh_decoder()
    states = model(h, torch.eye(4, dtype=torch.bool))["states"]
    loss, counts = assembly_relation_loss(states, supervision())
    assert counts[0] > 0 and counts[2] > 0
    loss.backward()
    assert model.relations[0].weight.grad.abs().sum() > 0 and h.grad.abs().sum() > 0


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("step", [0, 250, 500, 800])
def test_native_baseline_objective_phases_are_finite_with_encoder_gradients(
    width, step
):
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(e) for e in tiny_level_events()[:2]]
    )
    model = fresh_model(width, RuntimeFeatureNormalizer.identity(12, 13))
    loss, parts = baseline_loss(model, batch, step)
    assert torch.isfinite(loss)
    assert {
        "lca",
        "parent",
        "tree_distance",
        "depth",
        "channel",
        "var",
        "cov",
        "pid",
        "corruption",
        "correctness",
        "hard_negative",
    } <= parts.keys()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_initializations_are_matched_within_width_and_decoder_across_widths():
    for width in (128, 256):
        a = fresh_model(width, RuntimeFeatureNormalizer.identity(12, 13))
        b = fresh_model(width, RuntimeFeatureNormalizer.identity(12, 13))
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    a = fresh_decoder()
    b = fresh_decoder()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    with pytest.raises(ValueError):
        fresh_model(512, None)
    with pytest.raises(ValueError):
        AssemblyMembershipDecoder.interface(torch.zeros(2, 64))
