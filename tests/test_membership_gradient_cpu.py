import torch
from hypertagging.training.membership_gradient import (
    relation_correction,
    encoder_correction,
)


def test_conflict_projection_preserves_membership_and_removes_only_opposition():
    a = [torch.tensor([2.0, 0.0]), None]
    b = [torch.tensor([-3.0, 4.0]), torch.tensor([9.0])]
    correction, info = relation_correction(a, b)
    torch.testing.assert_close(b[0] + correction[0], torch.tensor([0.0, 4.0]))
    assert correction[1] is None and info["dot_before"] == -6 and info["dot_after"] == 0
    torch.testing.assert_close(a[0], torch.tensor([2.0, 0.0]))


def test_aligned_gradients_are_unchanged():
    delta, info = relation_correction([torch.tensor([2.0])], [torch.tensor([3.0])])
    assert delta[0].item() == 0 and info["conflict"] == 0


def test_projection_changes_encoder_gradient_not_decoder_gradient():
    encoder = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    head = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    member = (encoder * head).sum()
    relation = (-encoder * head).sum() + encoder[1] * head[1]
    correction, info = encoder_correction(member, relation, (encoder,))
    total = member + relation
    expected_head = torch.autograd.grad(total, head, retain_graph=True)[0]
    total.backward()
    encoder.grad.add_(correction[0])
    torch.testing.assert_close(head.grad, expected_head)
    assert info["conflict"] == 1 and info["dot_after"] >= 0


def test_zero_membership_gradient_is_finite():
    delta, info = relation_correction([torch.zeros(3)], [torch.ones(3)])
    assert torch.isfinite(delta[0]).all() and info["dot_after"] == 0


def test_successor_contract_rejects_extra_arms_resources_and_history():
    import copy
    import pytest
    from scripts.phase75_development import settings, validate

    c = {
        "arms": ["joint", "project_conflicting_relation"],
        "settings": settings(),
        "stage": "development",
        "automatic_successor": False,
        "sealed_test_access": False,
        "resources": {
            "cpus": 2,
            "memory_gib": 32,
            "hours": 8,
            "gpus": 0,
            "max_jobs": 2,
            "requeue": False,
        },
        "training_count": 1536,
        "heldout_count": 600,
        "context_width": 128,
    }
    validate(c)
    for key, value in [
        ("arms", ["joint"]),
        ("context_width", 256),
        ("automatic_successor", True),
        ("heldout_count", 60),
    ]:
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            validate(bad)
    bad = copy.deepcopy(c)
    bad["settings"]["pretraining_updates"] = 1000
    with pytest.raises(ValueError):
        validate(bad)
    bad = copy.deepcopy(c)
    bad["resources"]["gpus"] = 1
    with pytest.raises(ValueError):
        validate(bad)
