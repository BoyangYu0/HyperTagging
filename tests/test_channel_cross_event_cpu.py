import pytest
import torch

from hypertagging.losses.hyperbolic_pretraining import cross_event_channel_metric_loss
from hypertagging.preprocessing.channels import (
    canonical_decay_signature,
    find_resonance_b_branches,
    unordered_b_pair_signature,
)
from hypertagging.preprocessing.mdst_tree_builder import EventTree, FourVector, TreeNode
from hypertagging.training.pretrain_trainer import ChannelMemoryBank
from hypertagging.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)


def _tree(include_second=True, bs=False):
    tree = EventTree(1)
    tree.add_node(TreeNode(0, 300553, 0, FourVector(0, 0, 0, 0), daughter_ids=[1, 2]))
    tree.add_node(TreeNode(1, 531 if bs else 521, 1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[3]))
    tree.add_node(TreeNode(2, -521, -1, FourVector(0, 0, 0, 0), parent_id=0, daughter_ids=[4]))
    tree.add_node(TreeNode(3, 211, 1, FourVector(0, 0, 0, 1), parent_id=1))
    tree.add_node(TreeNode(4, -211, -1, FourVector(0, 0, 0, 1), parent_id=2))
    if include_second:
        tree.nodes[1].daughter_ids.append(5)
        tree.add_node(TreeNode(5, 22, 0, FourVector(0, 0, 0, 1), parent_id=1, copied_from=None))
    return tree


def test_full_and_reconstructable_signatures_and_y4s_pair():
    full = canonical_decay_signature(_tree(True), 1)
    reco = canonical_decay_signature(_tree(False), 1)
    assert full != reco
    assert unordered_b_pair_signature([full, reco]) == unordered_b_pair_signature([reco, full])
    assert len(find_resonance_b_branches(_tree())) == 2
    try:
        find_resonance_b_branches(_tree(bs=True))
    except ValueError:
        pass
    else:
        raise AssertionError("Upsilon(4S) B_s branch was not rejected")


def test_cross_event_channel_pairs_not_same_event_shortcut():
    embedding = torch.randn(2, 2, 8, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    full_ids = torch.tensor([[1, 2], [1, 3]])
    loss, diagnostics = cross_event_channel_metric_loss(embedding, mask, full_ids)
    loss.backward()
    assert diagnostics["channel_positive_pairs"] == 1
    assert diagnostics["channel_negative_pairs"] > 0
    assert torch.isfinite(embedding.grad).all()


def test_memory_bank_can_supply_a_cross_batch_positive():
    embedding = torch.randn(1, 2, 8, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    full_ids = torch.tensor([[7, 8]])
    memory = torch.randn(1, 8)
    loss, diagnostics = cross_event_channel_metric_loss(
        embedding,
        mask,
        full_ids,
        memory_embeddings=memory,
        memory_full_truth_channel_ids=torch.tensor([7]),
        memory_reconstructable_channel_ids=torch.tensor([0]),
    )
    loss.backward()
    assert diagnostics["channel_positive_pairs"] > 0
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()


def test_structured_cross_event_similarity_is_trainable_without_exact_id_match():
    embedding = torch.randn(2, 2, 8, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    full_ids = torch.tensor([[1, 2], [3, 4]])
    counts = torch.zeros(2, 2, 41)
    counts[0, 0, 5] = 2
    counts[1, 0, 5] = 2
    counts[0, 1, 8] = 1
    counts[1, 1, 9] = 1
    loss, diagnostics = cross_event_channel_metric_loss(
        embedding,
        mask,
        full_ids,
        branch_count_arrays=counts,
    )
    loss.backward()
    assert diagnostics["channel_positive_pairs"] > 0
    assert diagnostics["channel_structured_regression"] >= 0
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()


def test_structured_regression_pairs_are_channel_loss_support_without_positives():
    embedding = torch.randn(1, 2, 8, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    counts = torch.zeros(1, 2, 41)
    counts[0, 0, 3] = 1
    counts[0, 1, 9] = 1
    loss, diagnostics = cross_event_channel_metric_loss(
        embedding,
        mask,
        torch.tensor([[101, 202]]),
        reconstructable_channel_ids=torch.tensor([[301, 402]]),
        branch_count_arrays=counts,
    )
    loss.backward()
    assert diagnostics["channel_positive_pairs"] == 0
    assert diagnostics["channel_active_anchors"] == 0
    assert diagnostics["channel_loss_support_terms"] == 1
    assert loss > 0
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()


def test_channel_memory_bank_has_resume_stable_state_shape():
    bank = ChannelMemoryBank(capacity=3, embedding_dim=4)
    bank.enqueue(
        torch.randn(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[7, 8]]),
        torch.tensor([[17, 18]]),
    )
    restored = ChannelMemoryBank(capacity=3, embedding_dim=4)
    restored.load_state_dict(bank.state_dict())
    embeddings, full_ids, reco_ids = restored.contents()
    assert embeddings.shape == (2, 4)
    assert full_ids.tolist() == [7, 8]
    assert reco_ids.tolist() == [17, 18]


class _MemoryModel(torch.nn.Module):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.channel_memory = ChannelMemoryBank(capacity=capacity, embedding_dim=4)


def test_empty_zero_capacity_channel_memory_can_expand_on_explicit_resume(tmp_path):
    source = _MemoryModel(0)
    path = save_training_checkpoint(
        tmp_path / "empty-memory.pt", model=source, step=2188,
        config={"channel_memory_size": 0},
    )
    target = _MemoryModel(3)
    payload = restore_training_checkpoint(
        path,
        model=target,
        allow_empty_channel_memory_expansion=True,
    )
    assert target.channel_memory.capacity == 3
    assert target.channel_memory.contents()[0].shape == (0, 4)
    assert payload["training_state"]["checkpoint_load_migrations"] == [
        {
            "kind": "empty_channel_memory_expansion_v1",
            "checkpoint_step": 2188,
            "source_capacity": 0,
            "target_capacity": 3,
            "preserved_entries": 0,
        }
    ]


def test_zero_capacity_channel_memory_shape_mismatch_remains_fail_closed(tmp_path):
    path = save_training_checkpoint(
        tmp_path / "empty-memory.pt", model=_MemoryModel(0), step=2188,
        config={"channel_memory_size": 0},
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        restore_training_checkpoint(path, model=_MemoryModel(3))
