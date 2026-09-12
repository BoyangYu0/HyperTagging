"""Physics supervision must follow the current PID-dependent daughter state."""

import pytest
import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.level_reconstruction import (
    level_reconstruction_loss,
    targets_for_level,
)
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.reconstruction_trainer import ReconstructionConfig, _optimization_loss
from hypertagging.training.scheduled_sampling import TeacherForcingSchedule


@pytest.mark.parametrize("aligned_override", [False, True])
def test_exact_current_daughters_have_zero_physics_loss_after_pid_refinement(aligned_override):
    batch = collate_heterogeneous_events([
        heterogeneous_from_level_event(tiny_level_events()[0])
    ])
    leaves = batch["level_ids"] == 0
    batch["charge"][0, leaves[0]] = torch.tensor([1.0, -1.0])
    batch["pid_labels"][leaves] = 0
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    model = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1], n_types=len(PDG_TOKENS),
        hidden_dim=16, hyper_dim=4, n_queries=1,
    )
    with torch.no_grad():
        refined = model(batch, target_level=1)
    loss_batch = dict(batch, p4=refined.current_p4)
    targets = targets_for_level(batch, 1)
    # The original stored target p4 is deliberately retained to reproduce
    # both teacher targets and scheduled targets captured before this forward.
    target_p4_before_refinement = targets[2][0].clone()
    current_daughter_sum = refined.current_p4[0, leaves[0]].sum(dim=0)
    assert not torch.allclose(target_p4_before_refinement[0], current_daughter_sum)
    types = torch.full((1, 1, len(PDG_TOKENS)), -30.0)
    types[0, 0, targets[0][0][0]] = 30.0
    pointer = torch.where(leaves[:, None], 30.0, -30.0).requires_grad_()
    output = MotherPointerOutput(
        object_logits=torch.full((1, 1), 30.0), type_logits=types,
        pointer_logits=pointer, cardinality_logits=torch.zeros((1, 1, 4)),
        confidence_logits=torch.zeros((1, 1)),
    )
    loss = level_reconstruction_loss(
        output, loss_batch, target_level=1,
        target_override=targets if aligned_override else None,
    )
    assert float(loss.components["physics"]) < 1e-10
    loss.total.backward()
    assert torch.isfinite(pointer.grad).all()


def test_flat_contextual_encoder_receives_decoder_gradients():
    batch = collate_heterogeneous_events([
        heterogeneous_from_level_event(tiny_level_events()[1])
    ])
    model = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1], n_types=len(PDG_TOKENS),
        hidden_dim=16, hyper_dim=4, n_queries=2, encoder_mode="flat",
        use_contextual_encoder=True, use_relation_bias=True,
    )
    output = model(batch, target_level=1)
    output.pointer.object_logits.square().sum().backward()
    contextual_gradients = [
        p.grad for p in model.flat_contextualizer.parameters() if p.grad is not None
    ]
    assert contextual_gradients
    assert sum(float(g.abs().sum()) for g in contextual_gradients) > 0
    relation_gradients = [
        p.grad for p in model.flat_relation_bias.parameters() if p.grad is not None
    ]
    assert relation_gradients
    assert sum(float(g.abs().sum()) for g in relation_gradients) > 0


def test_flat_context_uses_requested_geometry_and_attention_configuration():
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
        encoder_mode="flat", curvature=0.7, ffn_dim=48, dropout=0.1,
    )
    assert model.encoder.curvature == 0.7
    assert model.flat_relation_bias.curvature == 0.7
    layer = model.flat_contextualizer.layers[0]
    assert layer.feedforward[0].out_features == 48
    assert layer.attention.dropout == 0.1


@pytest.mark.parametrize("ablation, enabled", [("heterogeneous_only", False), ("full_revised", True)])
def test_reconstruction_leaf_pid_objective_respects_ablation(ablation, enabled):
    batch = collate_heterogeneous_events([
        heterogeneous_from_level_event(tiny_level_events()[0])
    ])
    leaves = batch["level_ids"] == 0
    batch["charge"][0, leaves[0]] = torch.tensor([1.0, -1.0])
    batch["pid_labels"][leaves] = 0
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    batch["truth_pid_available"][leaves] = True
    model = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1], n_types=len(PDG_TOKENS),
        hidden_dim=16, hyper_dim=4, n_queries=2,
    )
    _, leaf_loss, *_ = _optimization_loss(
        model, batch, valid_levels=[1],
        config=ReconstructionConfig(data="unused", output_dir="unused", ablation=ablation),
        schedule=TeacherForcingSchedule(), step=1, use_scheduled_sampling=False,
        allowed_types_by_level={}, constraint_policy=ReconstructionConstraintPolicy(),
    )
    assert bool(float(leaf_loss) > 0) is enabled
