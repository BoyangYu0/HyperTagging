"""The development graph adapts the encoder while preserving frozen PID parameters."""
import copy
import torch
from hypertagging.models.direct_membership import DirectMembershipHead, direct_membership_loss
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
from hypertagging.reconstruction.beam_search import _truth_free_model_view
from tests.test_hierarchical_inference_cpu import _normalized_native_v4_batch


def test_membership_gradient_reaches_adapted_encoder_and_keeps_pid_parameters_frozen():
    torch.manual_seed(13)
    native = _normalized_native_v4_batch()
    n_features = native['node_features'].shape[-1]
    batch = _truth_free_model_view(project_schema_v4_fsps(native).batch)
    base = LevelAutoregressiveReconstructor(n_features=n_features,
        n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4, n_queries=2,
        encoder_mode='heterogeneous').eval()
    initial = copy.deepcopy(base.state_dict())
    outputs = []
    for adapted in (False, True):
        model = copy.deepcopy(base)
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.encoder.parameters(): p.requires_grad_(adapted)
        torch.manual_seed(17)
        head = DirectMembershipHead(16, hidden_dim=128)
        features = model(batch, target_level=1, pid_kinematics_mode_override='soft_expectation',
                         pid_temperature_override=.5).node_embeddings
        logits, objects = head(features, batch['node_mask'])
        outputs.append(logits.detach())
        target = torch.ones(batch['node_mask'].shape, dtype=torch.long)
        loss = direct_membership_loss(logits, objects, target, batch['node_mask'])
        loss.backward()
        gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert bool(gradients) == adapted
        if adapted:
            assert all(torch.isfinite(g).all() for g in gradients)
            assert any(g.ne(0).any() for g in gradients)
        optimizer = torch.optim.AdamW([p for p in [*model.parameters(), *head.parameters()] if p.requires_grad], lr=.001)
        optimizer.step()
        assert all(torch.equal(v, initial['leaf_pid_head.'+k]) for k,v in model.leaf_pid_head.state_dict().items())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
