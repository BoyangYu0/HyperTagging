import math

import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.level_collate import build_lca_depth
from hypertagging.data.tree_geometry import build_exact_tree_geometry
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.evaluation.hierarchical_metrics import p4_closure_rate
from hypertagging.losses.hyperbolic_pretraining import (
    build_tree_relation_targets,
    collapse_diagnostics,
    dimension_aware_tangent_variance_target,
    hyperbolic_pretraining_loss,
    parent_child_margin_loss,
    parent_negative_coverage_statistics,
    topology_safe_parent_negative_mask,
    tree_distance_targets,
)
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.hyperbolic import logmap0, radius
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.relations import (
    PHYSICAL_RELATION_FEATURE_NAMES,
    _physical_features,
)
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.preprocessing.schema_v4 import (
    RUNTIME_MODEL_CONTRACTS_V4,
    feature_spec_v4,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    RolloutConfig,
    bounded_beam_proposal_sets,
    level_rollout,
)
from hypertagging.training.model_config import MODEL_PRESETS


def _connected_upsilon_topology():
    # node 8 is Upsilon, nodes 6/7 are the two B roots.  Every node is in the
    # one connected retained event; there is no disconnected unmatched root.
    parents = torch.tensor([4, 4, 0, 7, 6, 6, 8, 8, -1])
    levels = torch.tensor([1, 0, 0, 0, 2, 0, 3, 1, 4])
    sides = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, -1])
    mask = torch.ones(9, dtype=torch.bool)
    geometry = build_exact_tree_geometry(parents)
    lca_depth = build_lca_depth(parents, levels)
    targets, relation_mask = build_tree_relation_targets(
        parent_ids=parents.unsqueeze(0),
        lca_depth=lca_depth.unsqueeze(0),
        level_ids=levels.unsqueeze(0),
        node_mask=mask.unsqueeze(0),
        b_side=sides.unsqueeze(0),
        lca_node_id=geometry.lca_node_id.unsqueeze(0),
        edges_to_lca_from_i=geometry.edges_to_lca_from_i.unsqueeze(0),
        edges_to_lca_from_j=geometry.edges_to_lca_from_j.unsqueeze(0),
    )
    return parents, levels, sides, mask, geometry, lca_depth, targets, relation_mask


def test_connected_upsilon_different_b_negatives_survive_explicit_lca_contract():
    parents, _levels, sides, mask, _geometry, lca_depth, targets, _ = (
        _connected_upsilon_topology()
    )
    allowed = topology_safe_parent_negative_mask(
        parents,
        mask,
        0,
        lca_depth=lca_depth,
        tree_relation_targets=targets[0],
        b_side=sides,
    )
    assert allowed[3] and allowed[7]  # safe B2 leaf/root candidates
    for excluded in (0, 1, 2, 4, 5, 6, 8):
        assert not allowed[excluded]
    assert set(targets[0, 0, allowed].tolist()) == {4}
    stats = parent_negative_coverage_statistics(
        parents.unsqueeze(0),
        mask.unsqueeze(0),
        lca_depth=lca_depth.unsqueeze(0),
        tree_relation_targets=targets,
        b_side=sides.unsqueeze(0),
    )
    assert stats["parent_children_with_eligible_negative"] > 0
    assert stats["parent_children_with_different_b_negative"] > 0
    assert stats["parent_loss_active_fraction"] > 0
    assert stats["parent_ranking_accuracy_denominator"] == stats[
        "parent_children_with_eligible_negative"
    ]
    z = torch.randn(1, 9, 3) * 0.02
    loss = parent_child_margin_loss(
        z,
        parents.unsqueeze(0),
        mask.unsqueeze(0),
        lca_depth=lca_depth.unsqueeze(0),
        tree_relation_targets=targets,
        b_side=sides.unsqueeze(0),
    )
    assert torch.isfinite(loss)


def test_unbalanced_reconstruction_height_is_not_exact_edge_distance():
    # root(level 4) -> direct leaf(level 0) and composite(level 3) -> leaf.
    parents = torch.tensor([3, 2, 3, -1])
    levels = torch.tensor([0, 0, 3, 4])
    geometry = build_exact_tree_geometry(parents)
    lca_height = build_lca_depth(parents, levels)
    assert geometry.lca_node_id[0, 3] == 3
    assert geometry.edges_to_lca_from_i[0, 3] == 1
    assert geometry.edges_to_lca_from_j[0, 3] == 0
    assert geometry.exact_tree_path_distance[0, 3] == 1
    assert lca_height[0, 3] - levels[0] == 4  # the rejected old synonym
    assert geometry.depth_from_retained_root.tolist() == [1, 2, 1, 0]
    assert torch.equal(
        geometry.depth_from_retained_root,
        geometry.distance_to_nearest_retained_root,
    )
    # Without explicit relation labels, the directed-negative fallback still
    # rejects this exact one-edge ancestor despite the height difference.
    fallback = topology_safe_parent_negative_mask(
        parents,
        torch.ones_like(parents, dtype=torch.bool),
        0,
    )
    assert not fallback[3]


def test_runtime_geometry_scale_and_relation_contracts_are_named_in_model_spec():
    assert feature_spec_v4()["runtime_model_contracts"] == RUNTIME_MODEL_CONTRACTS_V4
    for preset in MODEL_PRESETS.values():
        assert preset.tree_geometry_contract_version == RUNTIME_MODEL_CONTRACTS_V4[
            "tree_geometry"
        ]
        assert preset.tree_distance_contract_version == RUNTIME_MODEL_CONTRACTS_V4[
            "tree_distance"
        ]
        assert preset.hyperbolic_scale_contract_version == RUNTIME_MODEL_CONTRACTS_V4[
            "hyperbolic_scale"
        ]
        assert preset.relation_feature_contract_version == RUNTIME_MODEL_CONTRACTS_V4[
            "physical_relation_features"
        ]


def test_fixed_tree_distance_scaling_is_not_changed_by_an_outlier_pair():
    base = torch.tensor([[[0, 2], [2, 0]]])
    base_mask = torch.ones_like(base, dtype=torch.bool)
    base_target = tree_distance_targets(
        exact_tree_path_distance=base,
        pair_mask=base_mask,
        target_scale_edges=8.0,
    )
    extended = torch.tensor([[[0, 2, 20], [2, 0, 19], [20, 19, 0]]])
    extended_target = tree_distance_targets(
        exact_tree_path_distance=extended,
        pair_mask=torch.ones_like(extended, dtype=torch.bool),
        target_scale_edges=8.0,
    )
    assert torch.allclose(base_target[0, :2, :2], extended_target[0, :2, :2])


def _preset_batch():
    return collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )


def test_all_model_presets_have_finite_dimension_aware_initial_geometry_and_gradients():
    batch = _preset_batch()
    for name, preset in MODEL_PRESETS.items():
        torch.manual_seed(100 + preset.hyper_dim)
        model = HeterogeneousNodeEncoder(
            d_model=preset.d_model,
            hyper_dim=preset.hyper_dim,
            n_heads=preset.n_heads,
            n_context_layers=preset.n_context_layers,
            ffn_dim=preset.ffn_dim,
            dropout=0.0,
            curvature=preset.curvature,
            hyper_projection_init_scale=preset.hyper_projection_init_scale,
            tangent_scale_mode=preset.tangent_scale_mode,
        )
        output = model(batch)
        z = output.hyperbolic_embeddings
        assert torch.isfinite(z).all(), name
        radii = radius(z, curvature=preset.curvature)[batch["node_mask"]]
        assert torch.isfinite(radii).all() and float(radii.quantile(0.95).detach()) < 1.5, name
        diagnostics = collapse_diagnostics(
            z,
            batch["node_mask"],
            curvature=preset.curvature,
        )
        assert float(diagnostics["boundary_fraction"]) <= 0.01, name
        tangent = logmap0(z, curvature=preset.curvature)[batch["node_mask"]]
        assert torch.isfinite(tangent.std(dim=0, unbiased=False)).all(), name
        expected_floor = dimension_aware_tangent_variance_target(preset.hyper_dim)
        assert expected_floor < 1.0 and math.isclose(
            expected_floor * math.sqrt(preset.hyper_dim), 0.5
        )
        loss_output = hyperbolic_pretraining_loss(
            z=z,
            parent_ids=batch["parent_ids"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
            exact_tree_path_distance=batch["exact_tree_path_distance"],
            tangent_variance_target=preset.tangent_variance_target,
            curvature=preset.curvature,
        )
        assert all(torch.isfinite(value) for value in loss_output.components.values()), name
        assert all(0 <= float(value.detach()) < 100 for value in loss_output.components.values()), name
        loss_output.total.backward()
        assert model.hyper_projection.weight.grad is not None, name
        assert torch.isfinite(model.hyper_projection.weight.grad).all(), name


def _raw_pid_rollout_batch():
    batch = _preset_batch()
    leaves = batch["node_mask"] & (batch["level_ids"] == 0)
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    batch["pid_labels"][leaves] = 0
    return batch


def test_rollout_pid_modes_describe_forward_relations_decisions_and_construction():
    torch.manual_seed(31)
    batch = _raw_pid_rollout_batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=41,
        hidden_dim=16,
        hyper_dim=4,
        n_queries=8,
        n_context_layers=1,
    ).eval()
    policy = ReconstructionConstraintPolicy(mother_charge_compatibility="off")
    expectations = {
        "soft_decision_hard_construction": ("soft_expectation", "hard"),
        "hard": ("hard", "hard"),
        "temperature_softmax": ("temperature_softmax", "temperature_softmax"),
        "straight_through_hard": ("straight_through_hard", "straight_through_hard"),
    }
    for rollout_mode, (forward_mode, construction_mode) in expectations.items():
        config = RolloutConfig(
            max_level=3,
            root_types=(),
            constraint_policy=policy,
            rollout_pid_kinematics_mode=rollout_mode,
            seed=9,
        )
        first = level_rollout(model, batch, mode="teacher_forced", config=config)
        second = level_rollout(model, batch, mode="teacher_forced", config=config)
        assert first.steps and first.valid
        assert first.steps[0].model_output.relation_pid_kinematics_mode == forward_mode
        assert first.steps[0].model_output.decision_pid_kinematics_mode == forward_mode
        assert first.steps[0].appended_mother_p4_pid_kinematics_mode == construction_mode
        assert p4_closure_rate(first.batch) == 1.0
        assert torch.equal(first.batch["p4"], second.batch["p4"])
        assert torch.equal(first.batch["daughter_adjacency"], second.batch["daughter_adjacency"])

    hard = model(batch, target_level=1, pid_kinematics_mode_override="hard")
    changed = {name: value.clone() for name, value in batch.items()}
    changed["pid_target_labels"].remainder_(41).add_(1).remainder_(41)
    changed["truth_pid_labels"].remainder_(41).add_(2).remainder_(41)
    changed_hard = model(changed, target_level=1, pid_kinematics_mode_override="hard")
    assert torch.allclose(hard.relation_bias, changed_hard.relation_bias)
    assert torch.allclose(hard.pointer.pointer_logits, changed_hard.pointer.pointer_logits)


def test_overlap_aware_physical_relations_withhold_double_counted_mass_and_energy():
    p4 = torch.tensor(
        [[[0.1, 0.0, 0.0, 0.2], [0.0, 0.1, 0.0, 0.2],
          [0.1, 0.1, 0.0, 0.4], [0.2, 0.1, 0.0, 0.5],
          [0.0, 0.0, 0.2, 0.3]]]
    )
    sources = torch.tensor(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
          [1, 1, 0, 0], [0, 0, 1, 1]]],
        dtype=torch.bool,
    )
    parents = torch.tensor([[2, 2, -1, -1, -1]])
    features = _physical_features(
        p4=p4,
        charge=torch.zeros(1, 5),
        level_ids=torch.tensor([[0, 0, 1, 1, 1]]),
        node_kind_ids=torch.tensor([[1, 1, 3, 3, 3]]),
        copied=torch.tensor([[False, False, False, True, False]]),
        source_node_ids=torch.tensor([[10, 11, 12, 12, 14]]),
        recursive_leaf_source_mask=sources,
        parent_ids=parents,
        reco_ids=torch.tensor([[100, 101, -1, -1, -1]]),
    )
    index = {name: position for position, name in enumerate(PHYSICAL_RELATION_FEATURE_NAMES)}
    for left, right in ((2, 0), (2, 3)):
        assert features[0, left, right, index["recursive_source_overlap"]] == 1
        assert features[0, left, right, index["disjoint_source_pair"]] == 0
        assert features[0, left, right, index["disjoint_pair_mass"]] == 0
        assert features[0, left, right, index["disjoint_pair_energy"]] == 0
        assert features[0, left, right, index["pair_mass_energy_available"]] == 0
    assert features[0, 2, 0, index["ancestor_descendant_relation"]] == 1
    assert features[0, 2, 4, index["disjoint_source_pair"]] == 1
    assert features[0, 2, 4, index["pair_mass_energy_available"]] == 1
    assert features[0, 2, 4, index["disjoint_pair_energy"]] > 0
    assert features[0, 3, 2, index["copied_source_conflict"]] == 1


def test_bounded_beam_keeps_top_k_distinct_partial_candidate_sets():
    proposals = [
        CompositeProposal(0, 4, (0, 1), 1.0, 0.9),
        CompositeProposal(1, 4, (0,), 1.0, 0.6),
        CompositeProposal(2, 4, (1,), 1.0, 0.6),
        CompositeProposal(3, 5, (2,), 1.0, 0.4),
    ]
    hypotheses = bounded_beam_proposal_sets(
        proposals,
        recursive_leaf_source_mask=torch.eye(3, dtype=torch.bool),
        beam_width=3,
        max_proposals=8,
    )
    assert len(hypotheses) == 3
    assert tuple(item.query_id for item in hypotheses[0]) == (1, 2, 3)
    assert len({tuple(item.query_id for item in hypothesis) for hypothesis in hypotheses}) == 3
