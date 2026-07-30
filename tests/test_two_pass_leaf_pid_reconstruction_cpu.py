import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.training.checkpointing import save_training_checkpoint
from hypertagging.training.pretrain_trainer import ContextualPretrainingModel
from hypertagging.training.pretrained_transfer import load_pretrained_encoder


def _batch():
    event = heterogeneous_from_level_event(tiny_level_events()[0])
    batch = collate_heterogeneous_events([event])
    leaves = batch["level_ids"] == 0
    leaf_indices = leaves[0].nonzero(as_tuple=False).flatten()
    batch["charge"][0, leaf_indices] = torch.tensor([1.0, -1.0])
    batch["node_kind_ids"][leaves] = 1
    batch["pid_labels"][leaves] = 0
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    return batch


def test_leaf_pid_refinement_precedes_level_one_pointer_and_receives_gradient():
    torch.manual_seed(4)
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=3,
    )
    batch = _batch()
    first = model(batch, target_level=1)
    with torch.no_grad():
        model.leaf_pid_head.weight.zero_()
        model.leaf_pid_head.bias.fill_(-20)
        model.leaf_pid_head.bias[11] = 20
        model.leaf_pid_head.bias[28] = 20
    second = model(batch, target_level=1)
    assert not torch.allclose(first.relation_bias, second.relation_bias)
    assert not torch.allclose(first.pointer.pointer_logits, second.pointer.pointer_logits)

    model.zero_grad(set_to_none=True)
    second.pointer.pointer_logits.square().mean().backward()
    assert model.leaf_pid_head.weight.grad is not None
    assert model.leaf_pid_head.weight.grad.abs().sum() > 0


def test_fixed_hypothesis_nodes_are_not_refined():
    batch = _batch()
    batch["leaf_kinematics_mode_ids"][:] = LEAF_MODE_TO_ID[
        "fixed_hypothesis_candidate"
    ]
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4
    )
    output = model(batch, target_level=1)
    assert torch.equal(output.current_pid_tokens, batch["pid_labels"])


def test_leaf_pid_head_transfers_separately(tmp_path):
    source = ContextualPretrainingModel(d_model=16, hyper_dim=4)
    with torch.no_grad():
        source.leaf_pid_head.bias.fill_(0.75)
    checkpoint = save_training_checkpoint(
        tmp_path / "pretrain.pt", model=source, encoder=source.encoder
    )
    target = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
    )
    report = load_pretrained_encoder(
        target.encoder,
        checkpoint,
        leaf_pid_head=target.leaf_pid_head,
        transfer_leaf_pid_head=True,
    )
    assert report.leaf_pid_loaded_keys == ("bias", "weight")
    torch.testing.assert_close(target.leaf_pid_head.bias, source.leaf_pid_head.bias)
