import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.data.capacity import production_capacity_report
from hypertagging.losses.hyperbolic_pretraining import (
    pool_b_branch_embeddings,
    topology_safe_parent_negative_mask,
)
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    compare_pid_kinematics_modes,
)
from hypertagging.models.relations import (
    HyperbolicRelationBias,
    PhysicalRelationBias,
)
from hypertagging.models.mother_pointer import MotherPointerDecoder
from hypertagging.preprocessing.pid_filter import (
    MOTHER_ONTOLOGY_VERSION,
    STATIC_MOTHER_TOKENS,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    proposal_ambiguity_metrics,
    resolve_weighted_set_packing,
)
from hypertagging.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def _batch():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    leaves = batch["node_mask"] & (batch["level_ids"] == 0)
    batch["node_kind_ids"][leaves] = 1
    batch["pid_labels"][leaves] = 0
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    return batch


def test_reconstruction_projection_and_query_decoder_receive_loss_gradients():
    torch.manual_seed(5)
    batch = _batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=41, hidden_dim=16, hyper_dim=4,
        n_queries=8, n_context_layers=1,
    )
    output = model(batch, target_level=1)
    loss = level_reconstruction_loss(output.pointer, batch, target_level=1).total
    loss.backward()
    intended = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.reconstruction_head.")
        or name.startswith("decoder.")
        or name.startswith("leaf_pid_head.")
    }
    for name, parameter in intended.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0, name


def test_topology_safe_parent_mask_excludes_child_grandparent_descendant_and_sibling():
    # 0,1 are siblings under 2; 2 is child of 3; 4 is descendant of 0; 5 is unrelated.
    parents = torch.tensor([2, 2, 3, -1, 0, -1])
    mask = torch.ones(6, dtype=torch.bool)
    allowed = topology_safe_parent_negative_mask(parents, mask, 0)
    assert not allowed[0]  # child itself
    assert not allowed[2]  # true parent
    assert not allowed[1]  # sibling
    assert not allowed[3]  # grandparent/ancestor
    assert not allowed[4]  # descendant
    assert allowed[5]


def test_static_mother_ontology_is_checkpointable_and_independent_of_empirical_prior(tmp_path):
    policy = ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=((1, (4,)),),
        empirical_type_prior_mode="soft",
    )
    allowed, bias = policy.type_constraints(1, device=torch.device("cpu"))
    assert policy.mother_ontology_version == MOTHER_ONTOLOGY_VERSION
    assert not allowed[0]
    assert not allowed[2]  # gamma is leaf-only
    assert not allowed[8]  # charged kaon is leaf-only
    assert allowed[5] and bias[5] < 0  # legitimate unseen J/psi remains reachable
    assert ReconstructionConstraintPolicy.from_dict(policy.to_dict()) == policy
    checkpoint = save_training_checkpoint(
        tmp_path / "policy.pt",
        model=torch.nn.Linear(1, 1),
        feature_contract={"reconstruction_constraint_policy": policy.to_dict()},
    )
    loaded = load_training_checkpoint(checkpoint)
    assert ReconstructionConstraintPolicy.from_dict(
        loaded["feature_contract"]["reconstruction_constraint_policy"]
    ) == policy
    off = ReconstructionConstraintPolicy(empirical_type_prior_mode="off")
    off_allowed, _ = off.type_constraints(99, device=torch.device("cpu"))
    assert set(off_allowed.nonzero().flatten().tolist()) == set(STATIC_MOTHER_TOKENS)
    decoder = MotherPointerDecoder(hidden_dim=8, n_types=41, n_queries=2)
    decoded = decoder(
        torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.bool), target_level=1
    )
    emitted = set(decoded.type_logits.argmax(dim=-1).flatten().tolist())
    assert emitted <= set(STATIC_MOTHER_TOKENS)


def test_relation_features_are_finite_symmetric_scaled_and_all_parameters_receive_gradients():
    torch.manual_seed(7)
    p4 = torch.tensor([[[1e6, -2e6, 3e6, 4e6], [0.2, 0.1, 0.0, 0.5]]])
    charge = torch.tensor([[1.0, -1.0]])
    levels = torch.tensor([[0, 1]])
    kinds = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])
    physical = PhysicalRelationBias(hidden_dim=8)
    bias = physical(
        p4=p4, charge=charge, level_ids=levels, node_mask=mask,
        node_kind_ids=kinds, copied=torch.zeros_like(mask),
        source_node_ids=torch.tensor([[1, 2]]),
    )
    assert torch.isfinite(bias).all()
    assert not torch.allclose(bias, bias.transpose(1, 2))  # directed level feature
    bias.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in physical.parameters()
    )
    z = torch.tensor([[[0.1, 0.0], [0.0, 0.2]]], requires_grad=True)
    hyper = HyperbolicRelationBias(hidden_dim=8)
    hyper_bias = hyper(z_hyperbolic=z, node_mask=mask)
    assert torch.isfinite(hyper_bias).all()
    assert not torch.allclose(hyper_bias, hyper_bias.transpose(1, 2))  # ordered radii
    hyper_bias.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in hyper.parameters()
    )


def test_duplicate_metrics_and_bounded_set_packing_can_beat_greedy_choice():
    sources = torch.eye(3, dtype=torch.bool)
    proposals = [
        CompositeProposal(0, 4, (0, 1), 1.0, 0.9),
        CompositeProposal(1, 4, (0,), 1.0, 0.6),
        CompositeProposal(2, 4, (1,), 1.0, 0.6),
        CompositeProposal(3, 4, (0,), 1.0, 0.5),
    ]
    packed = resolve_weighted_set_packing(
        proposals, recursive_leaf_source_mask=sources, max_proposals=8
    )
    assert [item.query_id for item in packed] == [1, 2]
    metrics = proposal_ambiguity_metrics(
        proposals, packed, total_queries=6, recursive_leaf_source_mask=sources
    )
    assert metrics["duplicate_daughter_set_rate"] > 0
    assert metrics["unused_query_fraction"] > 0
    assert metrics["overlap_rate_before_exclusive_resolution"] > 0
    assert metrics["overlap_rate_after_exclusive_resolution"] == 0


def test_query_self_attention_keeps_ambiguous_query_slots_differentiable_and_distinct():
    torch.manual_seed(19)
    decoder = MotherPointerDecoder(hidden_dim=8, n_types=41, n_queries=4)
    context = torch.zeros(1, 3, 8, requires_grad=True)
    output = decoder(context, torch.ones(1, 3, dtype=torch.bool), target_level=1)
    # Identical context does not force query collapse because learned slots
    # interact before cross-attention.
    assert not torch.allclose(
        output.object_logits[:, 0], output.object_logits[:, 1]
    )
    output.object_logits.square().mean().backward()
    assert decoder.query_self_attention.in_proj_weight.grad is not None
    assert float(decoder.query_self_attention.in_proj_weight.grad.abs().sum()) > 0


def test_pid_soft_hard_diagnostics_and_channel_pooling_modes_are_cpu_finite():
    batch = _batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=41, hidden_dim=16, hyper_dim=4,
        n_queries=8, n_context_layers=1,
    )
    diagnostics = compare_pid_kinematics_modes(model, batch, target_level=1)
    assert set(diagnostics) == {
        "soft_hard_energy_difference",
        "soft_hard_mother_mass_difference",
        "pid_entropy",
        "soft_hard_relation_bias_change",
        "soft_hard_pointer_logit_change",
        "annealed_soft_pointer_logit_change",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
    embeddings = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    side = torch.tensor([[0, 0, 0, 1, 1, 1]])
    node_mask = torch.ones_like(side, dtype=torch.bool)
    levels = torch.tensor([[0, 1, 2, 0, 1, 2]])
    for mode in ("mean_all", "b_root", "level_weighted"):
        pooled, available = pool_b_branch_embeddings(
            embeddings, side, node_mask, mode=mode, level_ids=levels
        )
        assert pooled.shape == (1, 2, 4) and available.all()
    pooled, _ = pool_b_branch_embeddings(
        embeddings, side, node_mask, mode="learned_attention",
        level_ids=levels, attention_logits=torch.zeros_like(side, dtype=torch.float32),
    )
    assert torch.isfinite(pooled).all()


def test_production_capacity_report_covers_every_indexed_level_and_flags_overflow():
    index = {
        "target_policy": "complete_only",
        "mother_count_histograms_by_level": {
            "1": {"2": 4}, "4": {"5": 1},
        },
        "daughter_cardinality_histograms_by_level": {
            "1": {"2": 8}, "4": {"13": 1},
        },
    }
    report = production_capacity_report(
        index, global_n_queries=4, global_max_cardinality=12,
        n_queries_by_level={1: 3}, max_cardinality_by_level={1: 8},
        target_policy="complete_only",
    )
    assert [row["level"] for row in report["levels"]] == [1, 4]
    assert report["levels"][1]["configured_queries"] == 4
    assert report["levels"][1]["cardinality_overflow_count"] == 1
    assert not report["production_training_allowed"]
