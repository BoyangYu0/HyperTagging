import pytest
import torch
from scripts.diagnose_phase80_optimizer_steps import (
    interpolate,
    detached_result,
    trace_json,
)


def test_step_scaling_matches_one_adamw_step_including_decay():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0.1)
    initial = [p.detach().clone()]
    p.grad = torch.tensor([0.2, -0.3])
    opt.step()
    delta = [p.detach().clone() - initial[0]]
    q = torch.nn.Parameter(initial[0].clone())
    other = torch.optim.AdamW([q], lr=0.005, weight_decay=0.1)
    q.grad = torch.tensor([0.2, -0.3])
    other.step()
    interpolate([p], initial, delta, 0.5)
    torch.testing.assert_close(p, q)
    with pytest.raises(ValueError):
        interpolate([p], initial, delta, 0.25)


def test_trace_detached_and_contains_no_targets():
    logits = torch.randn(1, 3, 3, requires_grad=True)
    result = {
        "logits": logits,
        "states": [
            {
                "groups": (frozenset([0]),),
                "pairs": (),
                "logits": torch.zeros(0, 3, requires_grad=True),
            }
        ],
    }
    copied = detached_result(result)
    assert not copied["logits"].requires_grad
    with torch.no_grad():
        logits.add_(1)
    assert not torch.equal(logits, copied["logits"])
    record = trace_json(copied)
    assert set(record) == {"logits", "states"} and record["states"][0]["groups"] == [
        [0]
    ]
