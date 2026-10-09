import copy
import pytest
import torch
from hypertagging.models.assembly_development import AssemblyMembershipDecoder
from hypertagging.models.direct_membership import direct_membership_loss
from scripts.phase76_development import (
    ARMS,
    INITIAL_SHA256,
    settings,
    arm_settings,
    validate,
)


def test_partial_context_ablation_preserves_proposal_and_generated_states():
    torch.manual_seed(7)
    on = AssemblyMembershipDecoder().eval()
    off = copy.deepcopy(on)
    off.partial_state_conditioning = False
    h = torch.randn(9, 128, requires_grad=True)
    sources = torch.eye(9, dtype=torch.bool)
    a = on(h, sources)
    b = off(h, sources)
    assert torch.equal(a["proposal_logits"], b["proposal_logits"])
    assert not torch.equal(a["logits"], b["logits"])
    for x, y in zip(a["states"], b["states"]):
        assert x["groups"] == y["groups"] and x["pairs"] == y["pairs"]
        assert torch.equal(x["logits"], y["logits"])
    target = torch.tensor([[1, 1, 2, 2, 0, 0, 0, 0, 0]])
    loss = direct_membership_loss(
        b["logits"], b["objects"], target, torch.ones(1, 9, dtype=torch.bool)
    )
    loss.backward()
    assert h.grad.abs().sum() > 0
    # Refinement loss still reaches proposal via soft predicted memberships.
    assert off.proposal.queries.grad.abs().sum() > 0
    w = off.condition[0].weight.grad
    assert w[:, :512].abs().sum() > 0 and w[:, 512:].abs().sum() == 0
    assert off.refinement.queries.grad.abs().sum() > 0
    assert len(list(on.parameters())) == len(list(off.parameters()))


def test_default_context_is_historical_and_flag_is_strict():
    torch.manual_seed(2)
    a = AssemblyMembershipDecoder()
    torch.manual_seed(2)
    b = AssemblyMembershipDecoder(partial_state_conditioning=True)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)
    with pytest.raises(ValueError):
        AssemblyMembershipDecoder(partial_state_conditioning="false")


def contract():
    return dict(
        initial_checkpoint={"sha256": INITIAL_SHA256},
        arms=list(ARMS),
        settings=settings(),
        arm_settings={a: arm_settings(a) for a in ARMS},
        stage="development",
        automatic_successor=False,
        sealed_test_access=False,
        resources=dict(
            cpus=2, memory_gib=32, hours=8, gpus=0, max_jobs=2, requeue=False
        ),
        training_count=1536,
        heldout_count=600,
        context_width=128,
    )


def test_contract_controls_and_unknown_arm():
    c = contract()
    validate(c)
    for key, value in [
        ("arms", ["joint", "project_conflicting_relation"]),
        ("context_width", 256),
        ("training_count", 70000),
        ("heldout_count", 12000),
        ("sealed_test_access", True),
    ]:
        d = copy.deepcopy(c)
        d[key] = value
        with pytest.raises(ValueError):
            validate(d)
    d = copy.deepcopy(c)
    d["arm_settings"][ARMS[1]]["partial_state_conditioning"] = True
    with pytest.raises(ValueError):
        validate(d)
    d = copy.deepcopy(c)
    d["settings"]["pretraining_updates"] = 1000
    with pytest.raises(ValueError):
        validate(d)
    with pytest.raises(ValueError):
        arm_settings("projection")


def test_history_and_resource_changes_rejected():
    c = contract()
    c["initial_checkpoint"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="history"):
        validate(c)
    c = contract()
    c["resources"]["gpus"] = 1
    with pytest.raises(ValueError, match="Resource"):
        validate(c)
