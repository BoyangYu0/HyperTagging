from argparse import Namespace
import os

import pytest
import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.level_dataset import LevelReconstructionDataset
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import radius_depth_loss
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.losses.set_matching import hungarian_or_greedy
from hypertagging.models.hyperbolic import distance, expmap0, logmap0
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor, construct_mother_p4
from hypertagging.models.stair_masks import context_mask_for_level, stair_attention_mask
from hypertagging.training.hyperbolic_pretrain import run_hyperbolic_pretrain_dry_run
from hypertagging.training.level_reconstruction_train import run_level_reconstruction_dry_run
from hypertagging.utils import gpu_safety


def test_level_dataset_and_collator_variable_size():
    dataset = LevelReconstructionDataset(tiny=True)
    batch = collate_level_events([dataset[0], dataset[1]], max_query_slots=4)

    assert batch.node_features.shape[0] == 2
    assert batch.node_mask.dtype is torch.bool
    assert batch.daughter_adjacency.shape[-1] == batch.node_features.shape[1]
    assert batch.same_mother[0, 0, 1]
    assert batch.lca_depth[0, 0, 1] == 1


def test_hyperbolic_ops_are_stable_and_round_trip():
    v = torch.tensor([[0.01, -0.02], [0.03, 0.04]], dtype=torch.float32)
    z = expmap0(v)
    back = logmap0(z)
    d = distance(z[:, None, :], z[None, :, :])

    torch.testing.assert_close(back, v, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d, d.T, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(d).all()


def test_radius_depth_loss_has_gradient():
    z = expmap0(torch.randn(1, 3, 2, requires_grad=True) * 0.01)
    z.retain_grad()
    levels = torch.tensor([[0, 1, 2]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    loss = radius_depth_loss(z, levels, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None


def test_stair_masks_prevent_target_leakage():
    batch = collate_level_events([tiny_level_events()[1]])
    context = context_mask_for_level(batch.level_ids, batch.node_mask, target_level=2)
    assert context[0, batch.level_ids[0] < 2].all()
    assert not context[0, batch.level_ids[0] >= 2].any()
    mask = stair_attention_mask(batch.level_ids, batch.node_mask)
    child = int((batch.level_ids[0] == 0).nonzero()[0])
    parent = int((batch.level_ids[0] == 2).nonzero()[0])
    assert not mask[0, child, parent]
    assert mask[0, parent, child]


def test_set_matching_order_invariant_for_tiny_cost():
    cost = torch.tensor([[2.0, 0.1], [0.2, 3.0]])
    assert sorted(hungarian_or_greedy(cost)) == [(0, 1), (1, 0)]


def test_level_model_loss_and_daughter_sum_p4():
    batch = collate_level_events([tiny_level_events()[0]], max_query_slots=3).to_dict()
    model = LevelAutoregressiveReconstructor(n_features=batch["node_features"].shape[-1], n_types=4096, hidden_dim=16, hyper_dim=4, n_queries=3)
    output = model(batch, target_level=1)
    loss = level_reconstruction_loss(output.pointer, batch, target_level=1)
    loss.total.backward()
    p4 = construct_mother_p4(output.pointer.pointer_logits, batch["p4"], hard=False)
    assert torch.isfinite(loss.total)
    assert p4.shape[-1] == 4


def test_cpu_training_dry_runs():
    hyp = run_hyperbolic_pretrain_dry_run(max_steps=1, batch_size=2)
    rec = run_level_reconstruction_dry_run(max_steps=1, batch_size=2)
    assert hyp.steps == 1 and hyp.loss >= 0
    assert rec.steps == 1 and rec.matches > 0


def test_gpu_safety_refuses_local_full_cuda(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = Namespace(device="cuda", tiny=False, max_steps=100, batch_size=16, allow_local_tiny_gpu_test=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        gpu_safety.assert_full_training_requires_slurm(args)


def test_gpu_safety_inside_slurm_allows_cuda(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = Namespace(device="cuda", tiny=False, max_steps=100, batch_size=16, allow_local_tiny_gpu_test=False)
    gpu_safety.assert_full_training_requires_slurm(args)


def test_gpu_safety_tiny_requires_explicit_flag(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = Namespace(device="cuda", tiny=True, max_steps=2, batch_size=1, allow_local_tiny_gpu_test=False)
    with pytest.raises(RuntimeError, match="allow-local"):
        gpu_safety.assert_local_gpu_tiny_test_allowed(args)
