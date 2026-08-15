from types import SimpleNamespace

import pytest
import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.hyperbolic import distance, expmap0
from hypertagging.models.level_autoregressive import _runtime_reconstruction_batch
from hypertagging.models.relations import PhysicalRelationBias
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.kinematics import stable_invariant_mass


@pytest.mark.parametrize("separation", [0.0, 1.0e-8])
def test_coincident_hyperbolic_distance_is_finite_under_bf16_autocast(
    separation: float,
):
    source = torch.full((2, 4, 8), 0.01, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        left = expmap0(source)
        right = expmap0(source + separation)
        value = distance(left, right)
        loss = value.square().mean()
    assert value.dtype == torch.float32
    assert torch.isfinite(value).all()
    loss.backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_stable_invariant_mass_has_exact_resolved_values_and_finite_lightcone_gradient():
    p4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mass = stable_invariant_mass(p4)
    torch.testing.assert_close(mass, torch.tensor([0.0, 0.0, 2.0]))
    assert mass.dtype == torch.float32
    mass.sum().backward()
    assert p4.grad is not None
    assert torch.isfinite(p4.grad).all()
    torch.testing.assert_close(p4.grad[:2], torch.zeros_like(p4.grad[:2]))


def test_pid_runtime_mass_and_physical_relation_loss_path_has_finite_backward():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    batch_size, nodes = batch["node_mask"].shape
    link = torch.zeros((), requires_grad=True)
    spatial = torch.zeros(batch_size, nodes, 3)
    spatial[..., 0] = 1.0
    energy = torch.ones(batch_size, nodes, 1) + link - link.detach()
    p4 = torch.cat([spatial, energy], dim=-1)
    runtime = SimpleNamespace(
        probabilities=torch.zeros(batch_size, nodes, len(PDG_TOKENS)),
        current_tokens=torch.zeros(batch_size, nodes, dtype=torch.long),
        available=batch["node_mask"].clone(),
        p4=p4,
        daughter_input_histograms=batch["daughter_input_pid_histogram"].float(),
        daughter_histogram_available=batch[
            "daughter_input_pid_histogram_available"
        ].clone(),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        second = _runtime_reconstruction_batch(
            batch,
            runtime,
            normalizer=RuntimeFeatureNormalizer.identity(
                batch["common_features"].shape[-1],
                batch["composite_features"].shape[-1],
            ),
            canonical_batch=batch,
            use_canonical=False,
        )
        relation = PhysicalRelationBias(hidden_dim=8)
        bias = relation(
            p4=second["p4"],
            charge=second["charge"],
            level_ids=second["level_ids"],
            node_mask=second["node_mask"],
            node_kind_ids=second["node_kind_ids"],
            copied=second["copied"],
            source_node_ids=second["source_node_ids"],
            recursive_leaf_source_mask=second["recursive_leaf_source_mask"],
            reco_ids=second["reco_ids"],
        )
        loss = second["common_features"][..., 4].square().mean() + bias.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert link.grad is not None and torch.isfinite(link.grad)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in relation.parameters()
    )
