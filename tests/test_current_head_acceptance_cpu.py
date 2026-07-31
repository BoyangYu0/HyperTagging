from __future__ import annotations

from unittest.mock import patch

import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import (
    build_topology_safe_parent_negative_mask,
    build_tree_relation_targets,
    hyperbolic_pretraining_loss,
    pool_b_branch_embeddings,
    radius_depth_loss,
)
from hypertagging.models.mother_pointer import source_conflict_penalty
from hypertagging.models.relations import PhysicalRelationBias
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy


def _batch():
    return collate_heterogeneous_events(
        [heterogeneous_from_level_event(event) for event in tiny_level_events()]
    )


def test_exact_geometry_is_built_once_per_event_during_collation_and_reused():
    events = [heterogeneous_from_level_event(event) for event in tiny_level_events()]
    from hypertagging.data import heterogeneous as module

    original = module.build_exact_tree_geometry
    calls = []

    def counted(parent_ids):
        calls.append(parent_ids.numel())
        return original(parent_ids)

    with patch.object(module, "build_exact_tree_geometry", side_effect=counted):
        batch = collate_heterogeneous_events(events)
    assert len(calls) == len(events)
    with patch(
        "hypertagging.losses.hyperbolic_pretraining.build_exact_tree_geometry",
        side_effect=AssertionError("normal loss path rebuilt geometry"),
    ):
        targets, mask = build_tree_relation_targets(
            parent_ids=batch["parent_ids"],
            lca_depth=batch["lca_depth"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
            b_side=batch["b_side"],
            lca_node_id=batch["lca_node_id"],
            edges_to_lca_from_i=batch["edges_to_lca_from_i"],
            edges_to_lca_from_j=batch["edges_to_lca_from_j"],
        )
        assert targets.shape == mask.shape


def test_precomputed_and_cpu_fallback_targets_and_gradients_match():
    batch = _batch()
    precomputed = build_tree_relation_targets(
        parent_ids=batch["parent_ids"], lca_depth=batch["lca_depth"],
        level_ids=batch["level_ids"], node_mask=batch["node_mask"],
        b_side=batch["b_side"], lca_node_id=batch["lca_node_id"],
        edges_to_lca_from_i=batch["edges_to_lca_from_i"],
        edges_to_lca_from_j=batch["edges_to_lca_from_j"],
    )
    fallback = build_tree_relation_targets(
        parent_ids=batch["parent_ids"], lca_depth=batch["lca_depth"],
        level_ids=batch["level_ids"], node_mask=batch["node_mask"],
        b_side=batch["b_side"],
    )
    assert torch.equal(precomputed[0], fallback[0])
    assert torch.equal(precomputed[1], fallback[1])
    negatives = build_topology_safe_parent_negative_mask(
        precomputed[0], batch["node_mask"], batch["ancestor_descendant_relation"]
    )
    base = torch.randn((*batch["parent_ids"].shape, 4)) * 0.02
    gradients = []
    for explicit in (negatives, None):
        z = base.clone().requires_grad_(True)
        output = hyperbolic_pretraining_loss(
            z=z, parent_ids=batch["parent_ids"], level_ids=batch["level_ids"],
            node_mask=batch["node_mask"], tree_relation_targets=precomputed[0],
            tree_relation_mask=precomputed[1], lca_depth=batch["lca_depth"],
            exact_tree_path_distance=batch["exact_tree_path_distance"],
            parent_negative_mask=explicit, b_side=batch["b_side"],
            weights={"lca": 0, "depth": 0, "tree_distance": 0, "channel": 0,
                     "var": 0, "cov": 0},
        )
        output.total.backward()
        gradients.append(z.grad)
    assert torch.allclose(gradients[0], gradients[1], atol=1e-6, rtol=1e-5)


def test_normal_relation_and_parent_paths_need_no_python_scalar_conversion():
    batch = _batch()
    targets, mask = build_tree_relation_targets(
        parent_ids=batch["parent_ids"], lca_depth=batch["lca_depth"],
        level_ids=batch["level_ids"], node_mask=batch["node_mask"],
        b_side=batch["b_side"], lca_node_id=batch["lca_node_id"],
        edges_to_lca_from_i=batch["edges_to_lca_from_i"],
        edges_to_lca_from_j=batch["edges_to_lca_from_j"],
    )
    negatives = build_topology_safe_parent_negative_mask(
        targets, batch["node_mask"], batch["ancestor_descendant_relation"]
    )
    z = (torch.randn((*batch["parent_ids"].shape, 4)) * 0.02).requires_grad_(True)
    result = hyperbolic_pretraining_loss(
        z=z, parent_ids=batch["parent_ids"], level_ids=batch["level_ids"],
        node_mask=batch["node_mask"], tree_relation_targets=targets,
        tree_relation_mask=mask, parent_negative_mask=negatives,
        exact_tree_path_distance=batch["exact_tree_path_distance"],
        lca_depth=batch["lca_depth"], b_side=batch["b_side"],
    )
    result.total.backward()
    relation = PhysicalRelationBias(8)
    bias = relation(
        p4=batch["p4"], charge=batch["charge"], level_ids=batch["level_ids"],
        node_mask=batch["node_mask"], node_kind_ids=batch["node_kind_ids"],
        recursive_leaf_source_mask=batch["recursive_leaf_source_mask"],
        ancestor_descendant_relation=batch["ancestor_descendant_relation"],
    )
    assert torch.isfinite(bias[batch["node_mask"][:, :, None] & batch["node_mask"][:, None, :]]).all()

    # CPU tensors stand in for CUDA-shaped inputs while scalar extraction is
    # forbidden. The normal precomputed path must remain entirely tensorized.
    def forbidden_scalar(*_args, **_kwargs):
        raise AssertionError("normal device path attempted Python scalar extraction")

    with patch.object(torch.Tensor, "item", forbidden_scalar), patch.object(
        torch.Tensor, "__int__", forbidden_scalar
    ):
        rebuilt_targets, rebuilt_mask = build_tree_relation_targets(
            parent_ids=batch["parent_ids"], lca_depth=batch["lca_depth"],
            level_ids=batch["level_ids"], node_mask=batch["node_mask"],
            b_side=batch["b_side"], lca_node_id=batch["lca_node_id"],
            edges_to_lca_from_i=batch["edges_to_lca_from_i"],
            edges_to_lca_from_j=batch["edges_to_lca_from_j"],
        )
        rebuilt_negatives = build_topology_safe_parent_negative_mask(
            rebuilt_targets, batch["node_mask"],
            batch["ancestor_descendant_relation"],
        )
        tensorized = hyperbolic_pretraining_loss(
            z=z.detach(), parent_ids=batch["parent_ids"],
            level_ids=batch["level_ids"], node_mask=batch["node_mask"],
            tree_relation_targets=rebuilt_targets,
            tree_relation_mask=rebuilt_mask,
            parent_negative_mask=rebuilt_negatives,
            exact_tree_path_distance=batch["exact_tree_path_distance"],
            lca_depth=batch["lca_depth"], b_side=batch["b_side"],
        )
        assert tensorized.total.shape == ()
        relation(
            p4=batch["p4"], charge=batch["charge"],
            level_ids=batch["level_ids"], node_mask=batch["node_mask"],
            node_kind_ids=batch["node_kind_ids"],
            recursive_leaf_source_mask=batch["recursive_leaf_source_mask"],
            ancestor_descendant_relation=batch["ancestor_descendant_relation"],
        )


def test_source_conflict_normalization_is_capacity_invariant_and_finite():
    conflict = torch.tensor([[[False, True], [True, False]]])
    active_logits = torch.tensor([[[3.0, 3.0]]], requires_grad=True)
    object_logits = torch.tensor([[3.0]])
    reference = source_conflict_penalty(
        active_logits, conflict, object_logits=object_logits
    )
    padded_logits = torch.cat(
        [active_logits, torch.tensor([[[-2.0, 4.0], [8.0, -8.0]]])], dim=1
    )
    padded_objects = torch.tensor([[3.0, -100.0, -100.0]])
    padded = source_conflict_penalty(
        padded_logits, conflict, object_logits=padded_objects
    )
    assert torch.allclose(reference, padded, atol=1e-6)
    duplicated = source_conflict_penalty(
        active_logits.expand(1, 2, 2), conflict,
        active_query_mask=torch.ones(1, 2),
    )
    assert torch.allclose(reference, duplicated, atol=1e-6)
    zero = source_conflict_penalty(
        active_logits, torch.zeros_like(conflict), object_logits=object_logits
    )
    assert zero == 0
    (reference + padded + duplicated + zero).backward()
    assert torch.isfinite(active_logits.grad).all()


def test_source_conflict_different_query_capacities_have_same_scale():
    conflict = torch.tensor([[[False, True], [True, False]]])
    for queries in (1, 4, 9):
        logits = torch.full((1, queries, 2), 1.25)
        active = torch.ones((1, queries), dtype=torch.bool)
        loss = source_conflict_penalty(
            logits, conflict, active_query_mask=active
        )
        if queries == 1:
            reference = loss
        else:
            assert torch.allclose(loss, reference)


def test_radius_fsp_pooling_and_upsilon_initial_state_ablations():
    batch = _batch()
    z = (torch.randn((*batch["parent_ids"].shape, 5)) * 0.03).requires_grad_(True)
    losses = [
        radius_depth_loss(
            z, batch["level_ids"], batch["node_mask"], target_mode=mode,
            depth_from_retained_root=batch["depth_from_retained_root"],
        )
        for mode in (
            "generation_height_radius", "exact_root_depth_radius",
            "weak_or_learned_radius",
        )
    ]
    sum(losses).backward()
    assert torch.isfinite(z.grad).all()
    embeddings = torch.randn((*batch["node_mask"].shape, 6))
    pooled, available = pool_b_branch_embeddings(
        embeddings, batch["b_side"], batch["node_mask"], mode="fsp_only",
        level_ids=batch["level_ids"],
    )
    assert pooled.shape == (batch["node_mask"].shape[0], 2, 6)
    assert available.shape == (batch["node_mask"].shape[0], 2)
    unknown, _ = ReconstructionConstraintPolicy().type_constraints(3, device=torch.device("cpu"))
    upsilon, _ = ReconstructionConstraintPolicy(
        initial_state_policy="upsilon4s"
    ).type_constraints(3, device=torch.device("cpu"))
    assert unknown[23] and unknown[40]
    assert not upsilon[23] and not upsilon[40]
    assert all(upsilon[token] for token in (21, 22, 38, 39))
