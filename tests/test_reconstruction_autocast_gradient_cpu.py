"""No-grad context generation must not suppress later mixed-precision updates."""
import torch
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.training.reconstruction_trainer import _gradient_safe_autocast
from test_two_pass_leaf_pid_reconstruction_cpu import _batch


def test_pid_head_updates_after_no_grad_context_forward():
    torch.manual_seed(4)
    model = LevelAutoregressiveReconstructor(n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4, n_queries=3)
    batch = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.leaf_pid_head.weight.detach().clone()
    with _gradient_safe_autocast(device_type='cpu', dtype=torch.bfloat16):
        with torch.no_grad():
            model(batch, target_level=1)
        output = model(batch, target_level=1)
        loss = output.leaf_pid_logits.float().square().mean()
    loss.backward()
    assert model.leaf_pid_head.weight.grad is not None
    assert torch.isfinite(model.leaf_pid_head.weight.grad).all()
    assert model.leaf_pid_head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.leaf_pid_head.weight)
