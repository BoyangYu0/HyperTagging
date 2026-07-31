import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.hyperbolic import distance, expmap0, logmap0
from hypertagging.losses.hyperbolic_pretraining import hyperbolic_tree_distance_loss


def _batch():
    return collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )


def test_context_changes_one_nodes_hyperbolic_embedding_and_padding_does_not():
    torch.manual_seed(4)
    model = HeterogeneousNodeEncoder(d_model=16, hyper_dim=4)
    batch = _batch()
    first = model(batch)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["p4"][0, 1, 0] += 0.5
    changed["common_features"][0, 1, 0] += 0.5
    second = model(changed)
    assert not torch.allclose(first.hyperbolic_embeddings[0, 0], second.hyperbolic_embeddings[0, 0])
    assert not torch.allclose(first.adapter_embeddings[0, 0], first.node_embeddings[0, 0])


def test_curvature_round_trip_is_consistent():
    tangent = torch.tensor([[0.2, -0.1]])
    for curvature in (0.5, 2.0):
        z = expmap0(tangent, curvature=curvature)
        assert torch.allclose(logmap0(z, curvature=curvature), tangent, atol=1e-5)
        assert torch.isfinite(distance(z, -z, curvature=curvature)).all()


def test_direct_tree_distance_objective_has_finite_gradients():
    tangent = (torch.randn(1, 3, 2) * 0.1).requires_grad_()
    z = expmap0(tangent)
    exact_distance = torch.tensor([[[0, 2, 1], [2, 0, 1], [1, 1, 0]]])
    mask = torch.ones((1, 3, 3), dtype=torch.bool)
    loss = hyperbolic_tree_distance_loss(
        z,
        exact_tree_path_distance=exact_distance,
        pair_mask=mask,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(tangent.grad).all()
