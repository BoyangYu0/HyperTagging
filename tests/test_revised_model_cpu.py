import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import (
    covariance_regularization,
    parent_child_margin_loss,
    radius_targets,
    variance_regularization,
)
from hypertagging.models.heterogeneous import (
    ClusterNodeEncoder,
    CommonNodeEncoder,
    CompositeNodeEncoder,
    HeterogeneousNodeEncoder,
    TrackNodeEncoder,
    masked_mean_pool,
)
from hypertagging.models.hyperbolic import distance, expmap0, logmap0
from hypertagging.models.relation_attention import RelationAwareSelfAttention
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.ablation import ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS


def _heterogeneous_batch():
    events = [heterogeneous_from_level_event(event) for event in tiny_level_events()[:2]]
    return collate_heterogeneous_events(events)


def test_all_adapters_produce_same_d_model_and_masked_nan_is_safe():
    d_model = 12
    common = CommonNodeEncoder(d_model)
    track = TrackNodeEncoder(d_model)
    cluster = ClusterNodeEncoder(d_model)
    composite = CompositeNodeEncoder(d_model)

    common_values = torch.full((2, 3, 12), float("nan"))
    common_mask = torch.zeros_like(common_values, dtype=torch.bool)
    assert common(common_values, common_mask).shape == (2, 3, d_model)
    assert track(torch.zeros(2, 3, 6), torch.zeros(2, 3, 6, dtype=torch.bool)).shape[-1] == d_model
    assert cluster(torch.zeros(2, 3, 9), torch.zeros(2, 3, 9, dtype=torch.bool)).shape[-1] == d_model
    composite_output = composite(
        torch.zeros(2, 3, 9),
        torch.zeros(2, 3, 9, dtype=torch.bool),
        torch.zeros(2, 3, d_model),
        torch.zeros(2, 3, 41),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert composite_output.shape[-1] == d_model
    assert torch.isfinite(common(common_values, common_mask)).all()


def test_all_node_kinds_share_one_hyperbolic_space():
    batch = _heterogeneous_batch()
    encoder = HeterogeneousNodeEncoder(d_model=16, hyper_dim=5)
    output = encoder(batch)

    assert output.node_embeddings.shape[-1] == 16
    assert output.hyperbolic_embeddings.shape[-1] == 5
    assert output.hyperbolic_embeddings.shape[:2] == batch["node_kind_ids"].shape
    assert len({id(encoder.hyper_projection)}) == 1
    assert torch.isfinite(output.hyperbolic_embeddings).all()


def test_composite_daughter_summary_is_permutation_invariant():
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]])
    adjacency = torch.tensor([[[True, True, False], [False, False, False], [False, False, False]]])
    original = masked_mean_pool(embeddings, adjacency)[0, 0]

    permutation = torch.tensor([1, 0, 2])
    permuted_embeddings = embeddings[:, permutation]
    permuted_adjacency = adjacency[:, :, permutation]
    permuted = masked_mean_pool(permuted_embeddings, permuted_adjacency)[0, 0]
    torch.testing.assert_close(original, permuted)


def test_relation_bias_changes_context_and_receives_gradients():
    torch.manual_seed(3)
    attention = RelationAwareSelfAttention(d_model=8, n_heads=2)
    x = torch.randn(1, 3, 8, requires_grad=True)
    allowed = torch.ones((1, 3, 3), dtype=torch.bool)
    node_mask = torch.ones((1, 3), dtype=torch.bool)
    zero_bias = torch.zeros((1, 3, 3), requires_grad=True)
    changed_bias = torch.zeros((1, 3, 3), requires_grad=True)
    changed_bias.data[0, 0, 1] = 4.0

    output_zero, _ = attention(
        x,
        relation_bias=zero_bias,
        attention_mask=allowed,
        node_mask=node_mask,
    )
    output_changed, _ = attention(
        x,
        relation_bias=changed_bias,
        attention_mask=allowed,
        node_mask=node_mask,
    )
    assert not torch.allclose(output_zero, output_changed)
    output_changed.sum().backward()
    assert changed_bias.grad is not None
    assert changed_bias.grad.abs().sum() > 0


def test_relation_bias_network_parameters_receive_context_gradient():
    batch = _heterogeneous_batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=64,
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
        n_heads=4,
        n_context_layers=1,
    )
    output = model(batch, target_level=1)
    output.pointer.object_logits.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.relation_bias.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_padded_queries_and_keys_do_not_contribute():
    attention = RelationAwareSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(1, 3, 4)
    node_mask = torch.tensor([[True, True, False]])
    allowed = node_mask[:, :, None] & node_mask[:, None, :]
    output, weights = attention(
        x,
        relation_bias=torch.zeros(1, 3, 3),
        attention_mask=allowed,
        node_mask=node_mask,
    )
    assert torch.count_nonzero(output[0, 2]) == 0
    assert torch.count_nonzero(weights[0, :, :, 2]) == 0


def test_radius_target_direction_has_leaves_outside_roots():
    levels = torch.tensor([[0, 1, 3]])
    mask = torch.ones_like(levels, dtype=torch.bool)
    target = radius_targets(levels, mask, r_min=0.2, r_max=1.4)
    assert target[0, 0] > target[0, 1] > target[0, 2]


def test_parent_ranking_matches_actual_hyperbolic_distance_formula():
    z = expmap0(
        torch.tensor(
            [[[1.2, 0.0], [0.7, 0.1], [-0.3, 0.9]]],
            dtype=torch.float32,
        )
    )
    parents = torch.tensor([[1, -1, -1]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    margin = 0.3
    actual = parent_child_margin_loss(z, parents, mask, margin=margin)
    expected = torch.relu(
        distance(z[0, 0], z[0, 1]) - distance(z[0, 0], z[0, 2]) + margin
    )
    torch.testing.assert_close(actual, expected)


def test_variance_and_covariance_penalize_collapse_and_redundancy():
    mask = torch.ones((1, 8), dtype=torch.bool)
    collapsed = torch.zeros(1, 8, 4)
    dispersed = torch.eye(4).repeat(2, 1).unsqueeze(0) * 4
    assert variance_regularization(collapsed, mask) > variance_regularization(dispersed, mask)

    base = torch.linspace(-1, 1, 8)
    redundant = torch.stack([base, base, torch.zeros_like(base), torch.zeros_like(base)], dim=-1).unsqueeze(0)
    decorrelated = torch.stack(
        [base, torch.roll(base, 2), torch.roll(base, 4), torch.roll(base, 6)],
        dim=-1,
    ).unsqueeze(0)
    assert covariance_regularization(redundant, mask) > 0
    assert covariance_regularization(decorrelated, mask) >= 0


def test_hyperbolic_losses_have_finite_gradients_near_boundary():
    tangent = torch.tensor([[[5.0, 0.0], [0.0, 5.0], [1.0, 1.0]]], requires_grad=True)
    z = expmap0(tangent)
    z.retain_grad()
    mask = torch.ones((1, 3), dtype=torch.bool)
    loss = variance_regularization(logmap0(z), mask) + covariance_regularization(logmap0(z), mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_masked_nodes_do_not_change_hyperbolic_regularizers_or_parent_loss():
    tangent = torch.tensor([[[0.1, 0.0], [0.2, 0.1], [9.0, -9.0]]])
    z = expmap0(tangent)
    mask = torch.tensor([[True, True, False]])
    parents = torch.tensor([[1, -1, 0]])
    base_tangent = logmap0(z[:, :2])
    base_mask = torch.ones((1, 2), dtype=torch.bool)
    torch.testing.assert_close(
        variance_regularization(logmap0(z), mask),
        variance_regularization(base_tangent, base_mask),
    )
    torch.testing.assert_close(
        covariance_regularization(logmap0(z), mask),
        covariance_regularization(base_tangent, base_mask),
    )
    torch.testing.assert_close(
        parent_child_margin_loss(z, parents, mask),
        parent_child_margin_loss(z[:, :2], parents[:, :2], base_mask),
    )


def test_ablation_surface_includes_requested_stages_and_is_cpu_forwardable():
    assert list(ABLATIONS) == [
        "flat_baseline",
        "heterogeneous_only",
        "contextual_euclidean",
        "contextual_hyperbolic_parent_lca",
        "plus_radius_depth",
        "plus_variance_covariance",
        "plus_cross_event_channel",
        "plus_hyperbolic_relation_attention",
        "plus_leaf_pid",
        "plus_scheduled_sampling",
        "full_revised",
    ]
    batch = _heterogeneous_batch()
    assert ABLATIONS["plus_hyperbolic_relation_attention"] != ABLATIONS["full_revised"]
    heterogeneous_only = build_ablation_model(
        "heterogeneous_only",
        n_features=batch["node_features"].shape[-1],
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
    )
    contextual = build_ablation_model(
        "contextual_euclidean",
        n_features=batch["node_features"].shape[-1],
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
    )
    assert not heterogeneous_only.encoder.use_contextual_encoder
    assert contextual.encoder.use_contextual_encoder
    for name in ("flat_baseline", "full_revised"):
        model = build_ablation_model(
            name,
            n_features=batch["node_features"].shape[-1],
            n_types=64,
            hidden_dim=16,
            hyper_dim=4,
            n_queries=2,
        )
        output = model(batch, target_level=1)
        assert torch.isfinite(output.pointer.object_logits).all()
