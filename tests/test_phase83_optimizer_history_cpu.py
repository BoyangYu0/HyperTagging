import copy
import torch
import pytest
from scripts.diagnose_phase83_optimizer_history import (
    use_current_gradient,
    qualifies,
    validate_contract,
)


def test_current_gradient_removes_first_history_preserves_second_moment_and_settings():
    p = torch.nn.Parameter(torch.tensor([0.7, -0.3]))
    opt = torch.optim.AdamW([p], lr=0.001, weight_decay=0.0001)
    for _ in range(4):
        p.grad = torch.tensor([0.2, -0.1])
        opt.step()
    before = p.detach().clone()
    state = copy.deepcopy(opt.state_dict())
    g = torch.tensor([-0.4, 0.2])
    p.grad = g.clone()
    opt.step()
    second = opt.state[p]["exp_avg_sq"].clone()
    step = opt.state[p]["step"].clone()
    native = p.detach().clone()
    with torch.no_grad():
        p.copy_(before)
    opt.load_state_dict(copy.deepcopy(state))
    original = {
        k: v for k, v in opt.param_groups[0].items() if k not in ("params", "betas")
    }
    use_current_gradient(opt)
    p.grad = g.clone()
    opt.step()
    torch.testing.assert_close(opt.state[p]["exp_avg"], g)
    assert torch.equal(opt.state[p]["exp_avg_sq"], second) and torch.equal(
        opt.state[p]["step"], step
    )
    assert {
        k: v for k, v in opt.param_groups[0].items() if k not in ("params", "betas")
    } == original
    assert not torch.equal(native, p)
    with pytest.raises(ValueError):
        use_current_gradient(opt)


def test_loss_only_or_smaller_damage_does_not_qualify():
    stage = {
        "change": {"B_correct": 2, "background_to_B": 0, "native_raw_exact": 0},
        "conditional_change": {"conditional_optimal_correct": 1},
    }
    r = {
        "non_minibatch_audit": {
            "risk_change": {"member": -0.01, "total": -0.02},
            "stages": {
                "proposal": copy.deepcopy(stage),
                "refinement": copy.deepcopy(stage),
            },
        },
        "accepted_continuum_change": 0,
    }
    assert qualifies(r)
    for target, key, value in [
        ("risk", "total", 0.001),
        ("risk", "member", 0.001),
        ("change", "B_correct", 0),
        ("change", "background_to_B", 1),
        ("change", "native_raw_exact", -1),
        ("conditional", "conditional_optimal_correct", -1),
    ]:
        bad = copy.deepcopy(r)
        if target == "risk":
            bad["non_minibatch_audit"]["risk_change"][key] = value
        else:
            bad["non_minibatch_audit"]["stages"]["proposal"][
                "conditional_change" if target == "conditional" else "change"
            ][key] = value
        assert not qualifies(bad)
    bad = copy.deepcopy(r)
    bad["accepted_continuum_change"] = 1
    assert not qualifies(bad)


def test_diagnostic_scope_and_resource_rejections():
    c = {
        "stage": "training_only_history_diagnostic",
        "batches": 12,
        "batch_size": 8,
        "audit_events": 96,
        "gradient_presentations": 96,
        "optimizer_steps": 24,
        "resources": {
            "cpus": 2,
            "memory_gib": 16,
            "hours": 1,
            "gpus": 0,
            "requeue": False,
        },
        "validation_events": 0,
        "automatic_successor": False,
    }
    validate_contract(c)
    for key, value in [
        ("batches", 24),
        ("validation_events", 600),
        ("automatic_successor", True),
        ("optimizer_steps", 100),
        ("resources", {"gpus": 1}),
    ]:
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_contract(bad)


def test_native_replay_uses_same_fp32_delta_reconstruction_in_both_conditions():
    from scripts.diagnose_phase80_optimizer_steps import interpolate

    initial = torch.tensor([1.0], dtype=torch.float32)
    updated = torch.tensor([1e-8], dtype=torch.float32)
    delta = updated - initial
    p = torch.nn.Parameter(updated.clone())
    interpolate([p], [initial], [delta], 1.0)
    assert torch.equal(p, initial + delta)
    assert not torch.equal(p, updated)
    import inspect
    from scripts.diagnose_phase83_optimizer_history import main

    source = inspect.getsource(main)
    assert "interpolate(parameters, initial, native_delta, 1.0)" in source
    assert "interpolate(parameters, initial, current_delta, 1.0)" in source
