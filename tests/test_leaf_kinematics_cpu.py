import pytest
import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, TOKENIZE_DICT
from hypertagging.reconstruction.kinematics import (
    cluster_reco_p4,
    composite_p4_from_daughters,
    hard_reconstructed_p4_from_leaf_pid,
    hard_track_p4_from_pid_token,
    soft_reconstructed_p4_from_leaf_pid,
    soft_track_p4_from_pid_logits,
    track_energy_hypotheses,
    validate_recursive_p4_closure,
)


def test_track_hypotheses_are_deterministic_and_soft_path_is_differentiable():
    p3 = torch.tensor([[0.3, -0.2, 0.4]], requires_grad=True)
    first = track_energy_hypotheses(p3)
    second = track_energy_hypotheses(p3.detach())
    assert torch.allclose(first.detach(), second)
    assert torch.all(first[:, 1:] > first[:, :-1])
    logits = torch.zeros((1, 41), requires_grad=True)
    soft = soft_track_p4_from_pid_logits(p3, logits)
    soft.sum().backward()
    assert torch.isfinite(p3.grad).all() and torch.isfinite(logits.grad).all()


def test_hard_cluster_and_recursive_composite_p4():
    p3 = torch.tensor([[0.3, 0.0, 0.0]])
    hard = hard_track_p4_from_pid_token(p3, TOKENIZE_DICT[321])
    assert hard[0, 3] > torch.linalg.vector_norm(p3)
    cluster = cluster_reco_p4(energy=torch.tensor([0.5]), direction=torch.tensor([[0.0, 0.0, 1.0]]))
    daughters = torch.stack([hard[0], cluster[0]])
    mother = composite_p4_from_daughters(daughters)
    assert torch.equal(mother, daughters.sum(0))
    result = validate_recursive_p4_closure(
        {0: mother.tolist(), 1: hard[0].tolist(), 2: cluster[0].tolist()},
        {0: [1, 2], 1: [], 2: []},
        atol=1e-6,
    )
    assert result["nodes"] == 3
    with pytest.raises(ValueError):
        hard_track_p4_from_pid_token(p3, TOKENIZE_DICT[22])


def test_raw_track_soft_and_hard_pid_flow_rebuilds_composites():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    leaves = (
        batch["node_mask"]
        & (batch["level_ids"] == 0)
        & (batch["node_kind_ids"] == 1)
    )
    batch["pid_labels"][leaves] = 0
    logits = torch.zeros(
        (*batch["pid_labels"].shape, len(PDG_TOKENS)),
        requires_grad=True,
    )
    soft = soft_reconstructed_p4_from_leaf_pid(batch, logits)
    soft[..., 3].sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    hard = hard_reconstructed_p4_from_leaf_pid(batch, logits.detach())
    for mother in (4, 5, 6):
        daughters = batch["daughter_adjacency"][0, mother]
        torch.testing.assert_close(hard[0, mother], hard[0, daughters].sum(dim=0))
