from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hypertagging.data.heterogeneous import (
    MODEL_INPUT_SOURCE_TO_ID,
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelReconstructionOutput
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.hierarchical_inference import (
    FULL_ROOT_TOKEN,
    HierarchicalInferenceConfig,
    project_schema_v4_fsps,
    reconstruct_full_tree_from_fsps,
)
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig,
    batched_free_rollout,
    batched_decode_level,
    hard_decode_proposals,
    level_rollout,
)


def _normalized_native_v4_batch() -> dict[str, torch.Tensor]:
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    native = MODEL_INPUT_SOURCE_TO_ID["native_v4_reconstructed"]
    batch["model_input_source_ids"].fill_(native)
    batch["daughter_input_pid_source_ids"].fill_(native)
    batch["runtime_features_are_raw"] = torch.tensor(True)
    return batch


def _interleave_truth_mothers(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Place truth mothers between FSPs instead of relying on an FSP prefix."""

    permutation = torch.tensor([0, 4, 1, 5, 2, 6, 3])
    node_count = batch["node_mask"].shape[1]
    output: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            output[name] = value
        elif value.ndim >= 3 and value.shape[1:3] == (node_count, node_count):
            output[name] = value[:, permutation][:, :, permutation].clone()
        elif value.ndim >= 2 and value.shape[1] == node_count:
            output[name] = value[:, permutation].clone()
        else:
            output[name] = value.clone()
    return output


class _TwoBThenUpsilonModel:
    """Deterministic CPU oracle for scope and information-boundary tests."""

    def __init__(self) -> None:
        self.inference_mode_flags: list[bool] = []
        self.evaluation_metadata_flags: list[bool] = []

    def __call__(
        self,
        batch,
        *,
        target_level,
        pid_kinematics_mode_override=None,
        pid_temperature_override=None,
        return_attention=False,
    ):
        del pid_kinematics_mode_override, pid_temperature_override, return_attention
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        self.evaluation_metadata_flags.append(
            "evaluation_leaf_source_keys" in batch
        )
        batch_size, node_count = batch["node_mask"].shape
        query_count = 2
        context = batch["node_mask"] & (batch["level_ids"] < target_level)
        pointer_logits = torch.full(
            (batch_size, query_count, node_count), -20.0
        )
        object_logits = torch.full((batch_size, query_count), -20.0)
        type_logits = torch.full(
            (batch_size, query_count, len(PDG_TOKENS)), -20.0
        )
        cardinality_logits = torch.full(
            (batch_size, query_count, 7), -20.0
        )
        cardinality_logits[..., 2] = 20.0
        if target_level == 1:
            leaves = batch["node_mask"] & (batch["level_ids"] == 0)
            rank = leaves.long().cumsum(dim=-1) - 1
            first = leaves & (rank < 2)
            second = leaves & (rank >= 2) & (rank < 4)
            pointer_logits[:, 0] = torch.where(
                first, torch.full_like(rank, 20.0, dtype=torch.float32), pointer_logits[:, 0]
            )
            pointer_logits[:, 1] = torch.where(
                second, torch.full_like(rank, 20.0, dtype=torch.float32), pointer_logits[:, 1]
            )
            object_logits[:, 0] = torch.where(
                leaves.sum(dim=-1) >= 2,
                torch.full_like(object_logits[:, 0], 20.0),
                object_logits[:, 0],
            )
            object_logits[:, 1] = torch.where(
                leaves.sum(dim=-1) >= 4,
                torch.full_like(object_logits[:, 1], 20.0),
                object_logits[:, 1],
            )
            type_logits[:, 0, 21] = 20.0  # B0
            type_logits[:, 1, 22] = 20.0  # B+
        elif target_level == 2:
            previous = batch["node_mask"] & (batch["level_ids"] == 1)
            pointer_logits[:, 0] = torch.where(
                previous,
                torch.full_like(previous, 20.0, dtype=torch.float32),
                pointer_logits[:, 0],
            )
            object_logits[:, 0] = torch.where(
                previous.sum(dim=-1) >= 2,
                torch.full_like(object_logits[:, 0], 20.0),
                object_logits[:, 0],
            )
            type_logits[..., FULL_ROOT_TOKEN] = 20.0
        else:
            type_logits[..., FULL_ROOT_TOKEN] = 20.0
        pointer = MotherPointerOutput(
            object_logits=object_logits,
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=cardinality_logits,
            confidence_logits=torch.full_like(object_logits, 20.0),
        )
        hidden = torch.zeros(batch_size, node_count, 4)
        hyper = torch.zeros(batch_size, node_count, 2)
        relation = torch.zeros(batch_size, node_count, node_count)
        return LevelReconstructionOutput(
            target_level=target_level,
            pointer=pointer,
            node_embeddings=hidden,
            hyperbolic_embeddings=hyper,
            context_mask=context,
            relation_bias=relation,
            attention_weights=None,
            physical_relation_bias=relation,
            physical_attention_weights=None,
            hyperbolic_relation_bias=None,
            hyperbolic_attention_weights=None,
            final_contextual_embeddings=hidden,
            tree_projection=hidden,
            reconstruction_projection=hidden,
            channel_projection=hidden,
        )


class _EmptyLevelThenUpsilonModel:
    """Emit no object at level one and an Upsilon at level two."""

    def __init__(self) -> None:
        self.forward_pid_modes: list[str | None] = []
        self.constraint_snapshots: list[dict[str, torch.Tensor]] = []

    def __call__(
        self,
        batch,
        *,
        target_level,
        pid_kinematics_mode_override=None,
        pid_temperature_override=None,
        return_attention=False,
    ):
        self.forward_pid_modes.append(pid_kinematics_mode_override)
        self.constraint_snapshots.append(
            {
                name: batch[name].detach().clone()
                for name in (
                    "allowed_type_mask",
                    "type_logit_bias",
                    "pointer_validity_mask",
                )
                if name in batch
            }
        )
        del pid_temperature_override, return_attention
        batch_size, node_count = batch["node_mask"].shape
        context = batch["node_mask"] & (batch["level_ids"] < target_level)
        object_logits = torch.full((batch_size, 1), -20.0)
        pointer_logits = torch.full((batch_size, 1, node_count), -20.0)
        if target_level == 2:
            roots = batch["node_mask"] & (batch["parent_ids"] < 0)
            rank = roots.long().cumsum(dim=-1) - 1
            selected = roots & (rank < 2)
            object_logits.fill_(20.0)
            pointer_logits[:, 0] = torch.where(
                selected,
                torch.full_like(pointer_logits[:, 0], 20.0),
                pointer_logits[:, 0],
            )
        type_logits = torch.full(
            (batch_size, 1, len(PDG_TOKENS)), -20.0
        )
        type_logits[..., FULL_ROOT_TOKEN] = 20.0
        cardinality_logits = torch.full((batch_size, 1, 7), -20.0)
        cardinality_logits[..., 2] = 20.0
        pointer = MotherPointerOutput(
            object_logits=object_logits,
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=cardinality_logits,
            confidence_logits=torch.full_like(object_logits, 20.0),
        )
        hidden = torch.zeros(batch_size, node_count, 4)
        hyper = torch.zeros(batch_size, node_count, 2)
        relation = torch.zeros(batch_size, node_count, node_count)
        return LevelReconstructionOutput(
            target_level=target_level,
            pointer=pointer,
            node_embeddings=hidden,
            hyperbolic_embeddings=hyper,
            context_mask=context,
            relation_bias=relation,
            attention_weights=None,
            physical_relation_bias=relation,
            physical_attention_weights=None,
            hyperbolic_relation_bias=None,
            hyperbolic_attention_weights=None,
            final_contextual_embeddings=hidden,
            tree_projection=hidden,
            reconstruction_projection=hidden,
            channel_projection=hidden,
        )


def _rollout_config() -> RolloutConfig:
    return RolloutConfig(
        max_level=3,
        exclusive_final=True,
        constraint_policy=ReconstructionConstraintPolicy(
            mother_charge_compatibility="off",
            empirical_type_prior_mode="off",
        ),
    )


def test_projection_physically_compacts_fsps_and_scrubs_truth_topology():
    source = _normalized_native_v4_batch()
    projection = project_schema_v4_fsps(source)
    batch = projection.batch

    assert projection.audit.fsp_counts == (4,)
    assert projection.audit.discarded_higher_level_node_counts == (3,)
    assert batch["node_mask"].shape == (1, 4)
    assert set(batch["node_kind_ids"][batch["node_mask"]].tolist()) == {
        NODE_KIND_TO_ID["track"]
    }
    assert torch.equal(batch["node_ids"][0], torch.arange(4))
    assert torch.equal(batch["reco_ids"][0], torch.arange(4))
    assert torch.equal(batch["source_node_ids"][0], torch.arange(4))
    assert torch.equal(
        projection.evaluation_leaf_source_keys[0],
        torch.tensor(projection.audit.evaluation_fsp_source_keys[0]),
    )
    assert "evaluation_leaf_source_keys" not in batch
    assert (batch["parent_ids"] == -1).all()
    assert not batch["daughter_adjacency"].any()
    assert (batch["b_side"] == -1).all()
    assert not batch["daughter_truth_pid_histogram_available"].any()
    assert batch["daughter_truth_pid_histogram"].count_nonzero() == 0
    for forbidden in ("pid_target_labels", "truth_pid_labels", "truth_pid_available"):
        assert forbidden not in batch


def test_projection_preserves_fsp_detector_source_conflicts():
    source = _normalized_native_v4_batch()
    # Model a KLM/ECL-style association: FSP 3 carries its own detector source
    # and source 0, so the pair must not be consumed by competing mothers.
    source["recursive_leaf_source_mask"][0, 3, 0] = True

    projection = project_schema_v4_fsps(source)

    assert projection.batch["recursive_leaf_source_mask"][0, 3, 0]
    assert projection.batch["source_conflict_matrix"][0, 0, 3]
    assert projection.batch["source_conflict_matrix"][0, 3, 0]
    assert projection.audit.detector_source_conflict_pair_counts == (1,)


def test_truth_and_higher_node_perturbations_cannot_change_projection_or_rollout():
    source = _interleave_truth_mothers(_normalized_native_v4_batch())
    changed = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in source.items()
    }
    higher = changed["level_ids"] > 0
    changed["common_features"][higher] = 1.0e6
    changed["p4"][higher] = -1234.0
    changed["charge"][higher] = 99.0
    changed["pid_target_labels"].remainder_(len(PDG_TOKENS) - 1).add_(1)
    changed["truth_pid_labels"].remainder_(len(PDG_TOKENS) - 1).add_(1)
    changed["truth_pid_available"].logical_not_()
    changed["node_ids"].add_(10_000)
    changed["b_side"].fill_(1)
    changed["parent_ids"].fill_(6)
    changed["parent_ids"][higher] = -1
    changed["daughter_adjacency"][higher] = True
    changed["daughter_truth_pid_histogram"].fill_(123.0)
    changed["daughter_truth_pid_histogram_available"].fill_(True)
    changed["truth_supervision_source_ids"].fill_(777)
    changed["daughter_truth_pid_source_ids"].fill_(888)

    first_projection = project_schema_v4_fsps(source)
    second_projection = project_schema_v4_fsps(changed)
    assert first_projection.audit.original_fsp_positions == ((0, 2, 4, 6),)
    assert first_projection.audit.original_fsp_node_ids == ((0, 1, 2, 3),)
    assert first_projection.audit.original_fsp_reco_ids == (
        tuple(int(source["reco_ids"][0, position]) for position in (0, 2, 4, 6)),
    )
    assert first_projection.batch.keys() == second_projection.batch.keys()
    for name in first_projection.batch:
        assert torch.equal(first_projection.batch[name], second_projection.batch[name]), name

    first_model = _TwoBThenUpsilonModel()
    second_model = _TwoBThenUpsilonModel()
    config = HierarchicalInferenceConfig(
        scope="full", rollout_config=_rollout_config()
    )
    first = reconstruct_full_tree_from_fsps(first_model, source, config=config)
    second = reconstruct_full_tree_from_fsps(second_model, changed, config=config)
    for name in ("p4", "pid_labels", "parent_ids", "daughter_adjacency"):
        torch.testing.assert_close(first.batch[name], second.batch[name])
    assert first_model.inference_mode_flags and all(first_model.inference_mode_flags)
    assert second_model.inference_mode_flags and all(second_model.inference_mode_flags)
    assert not any(first_model.evaluation_metadata_flags)
    assert not any(second_model.evaluation_metadata_flags)
    assert "evaluation_leaf_source_keys" in first.batch


def test_full_scope_stops_at_upsilon_while_half_scope_evaluates_b_multiplicity():
    source = _normalized_native_v4_batch()
    full = reconstruct_full_tree_from_fsps(
        _TwoBThenUpsilonModel(),
        source,
        config=HierarchicalInferenceConfig(
            scope="full", rollout_config=_rollout_config()
        ),
    )
    half = reconstruct_full_tree_from_fsps(
        _TwoBThenUpsilonModel(),
        source,
        config=HierarchicalInferenceConfig(
            scope="half", rollout_config=_rollout_config()
        ),
    )

    assert full.rollout.root_completed_mask.tolist() == [True]
    assert full.rollout.stop_code.tolist() == [2]
    assert full.evaluation_slice_multiplicity.tolist() == [1]
    assert half.rollout.root_completed_mask.tolist() == [False]
    assert half.rollout.stop_code.tolist() == [3]  # evaluator scans through max level
    assert half.rollout.empty_level_counts.tolist() == [1]
    assert half.evaluation_slice_multiplicity.tolist() == [2]
    assert int(half.b_root_mask.sum()) == 2
    b_positions = half.b_root_mask[0].nonzero(as_tuple=False).flatten()
    assert (half.batch["parent_ids"][0, b_positions] >= 0).all()


def test_strict_inference_rejects_nonexclusive_rollout():
    with pytest.raises(ValueError, match="exclusive_final=True"):
        HierarchicalInferenceConfig(
            rollout_config=RolloutConfig(exclusive_final=False)
        )


def test_batched_decode_never_reuses_an_already_parented_node():
    state = project_schema_v4_fsps(_normalized_native_v4_batch()).batch
    state["parent_ids"][0, 0] = 2
    node_count = state["node_mask"].shape[1]
    pointer = MotherPointerOutput(
        object_logits=torch.tensor([[20.0]]),
        type_logits=torch.nn.functional.one_hot(
            torch.tensor([[4]]), num_classes=len(PDG_TOKENS)
        ).float()
        * 20.0,
        pointer_logits=torch.tensor([[[20.0, 20.0, -20.0, -20.0]]]),
        cardinality_logits=torch.zeros(1, 1, 7),
        confidence_logits=torch.tensor([[20.0]]),
    )
    output = SimpleNamespace(
        target_level=1,
        context_mask=torch.ones(1, node_count, dtype=torch.bool),
        pointer=pointer,
    )
    daughters, accepted, _ = batched_decode_level(
        output,
        state,
        active_event_mask=torch.tensor([True]),
        config=RolloutConfig(
            use_cardinality=False,
            min_daughters=1,
            exclusive_final=False,
            constraint_policy=ReconstructionConstraintPolicy(
                minimum_daughters=1,
                daughter_cardinality_policy="threshold",
                mother_charge_compatibility="off",
                empirical_type_prior_mode="off",
            ),
        ),
    )
    assert accepted.item()
    assert daughters[0, 0].tolist() == [False, True, False, False]
    proposals = hard_decode_proposals(
        output,
        state,
        RolloutConfig(
            use_cardinality=False,
            min_daughters=1,
            exclusive_final=False,
            constraint_policy=ReconstructionConstraintPolicy(
                minimum_daughters=1,
                daughter_cardinality_policy="threshold",
                mother_charge_compatibility="off",
                empirical_type_prior_mode="off",
            ),
        ),
    )
    assert [proposal.daughter_positions for proposal in proposals] == [(1,)]


def test_threshold_decode_rejects_recursive_source_conflicts_without_cardinality():
    state = project_schema_v4_fsps(_normalized_native_v4_batch()).batch
    sources = torch.zeros((1, 4, 3), dtype=torch.bool)
    sources[0, 0, 0] = True
    sources[0, 1, 0] = True  # associated detector source conflicts with node zero
    sources[0, 2, 1] = True
    sources[0, 3, 2] = True
    state["recursive_leaf_source_mask"] = sources
    overlap = (sources.float() @ sources.float().transpose(1, 2)).bool()
    state["source_conflict_matrix"] = overlap & ~torch.eye(
        4, dtype=torch.bool
    ).unsqueeze(0)
    pointer = MotherPointerOutput(
        object_logits=torch.tensor([[20.0]]),
        type_logits=torch.nn.functional.one_hot(
            torch.tensor([[4]]), num_classes=len(PDG_TOKENS)
        ).float()
        * 20.0,
        pointer_logits=torch.tensor([[[20.0, 19.0, 18.0, -20.0]]]),
        cardinality_logits=torch.zeros(1, 1, 7),
        confidence_logits=torch.tensor([[20.0]]),
    )
    output = SimpleNamespace(
        target_level=1,
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        pointer=pointer,
    )
    config = RolloutConfig(
        use_cardinality=False,
        exclusive_final=False,
        constraint_policy=ReconstructionConstraintPolicy(
            minimum_daughters=2,
            daughter_cardinality_policy="threshold",
            reject_recursive_source_conflicts=True,
            mother_charge_compatibility="off",
            empirical_type_prior_mode="off",
        ),
    )
    daughters, accepted, _ = batched_decode_level(
        output,
        state,
        active_event_mask=torch.tensor([True]),
        config=config,
    )
    assert accepted.item()
    assert daughters[0, 0].tolist() == [True, False, True, False]
    proposals = hard_decode_proposals(output, state, config)
    assert [proposal.daughter_positions for proposal in proposals] == [(0, 2)]


def test_offline_hierarchy_continues_across_empty_target_level():
    source = _normalized_native_v4_batch()
    projected = project_schema_v4_fsps(source).batch
    policy = ReconstructionConstraintPolicy(
        mother_charge_compatibility="off",
        empirical_type_prior_mode="off",
    )
    historical_model = _EmptyLevelThenUpsilonModel()
    historical = batched_free_rollout(
        historical_model,
        projected,
        config=RolloutConfig(
            max_level=3,
            root_types=(FULL_ROOT_TOKEN,),
            constraint_policy=policy,
        ),
    )
    assert historical.stop_code.tolist() == [1]
    assert historical.levels_completed.tolist() == [1]
    assert historical.empty_level_counts.tolist() == [1]
    assert not historical.root_completed_mask.item()
    assert historical_model.forward_pid_modes
    assert set(historical_model.forward_pid_modes) == {"soft_expectation"}

    offline_model = _EmptyLevelThenUpsilonModel()
    offline = reconstruct_full_tree_from_fsps(
        offline_model,
        source,
        config=HierarchicalInferenceConfig(
            scope="full",
            rollout_config=RolloutConfig(
                max_level=3,
                constraint_policy=policy,
            ),
        ),
    )
    assert offline.rollout.root_completed_mask.item()
    assert offline.rollout.stop_code.tolist() == [2]
    assert offline.rollout.levels_completed.tolist() == [2]
    assert offline.rollout.empty_level_counts.tolist() == [1]
    assert offline_model.forward_pid_modes
    assert set(offline_model.forward_pid_modes) == {"soft_expectation"}

    reference = level_rollout(
        _EmptyLevelThenUpsilonModel(),
        source,
        mode="predicted",
        config=RolloutConfig(
            max_level=3,
            root_types=(FULL_ROOT_TOKEN,),
            constraint_policy=policy,
            continue_through_empty_levels=True,
        ),
    )
    assert reference.stop_reason == "configured_root_reconstructed"
    assert reference.empty_level_count == 1
    assert [step.target_level for step in reference.steps] == [1, 2]


def test_batched_rollout_injects_soft_type_and_pointer_constraints_before_model():
    projected = project_schema_v4_fsps(_normalized_native_v4_batch()).batch
    model = _EmptyLevelThenUpsilonModel()
    policy = ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=((1, (4,)),),
        empirical_type_prior_mode="soft",
        empirical_type_soft_penalty=2.0,
        mother_charge_compatibility="off",
    )

    batched_free_rollout(
        model,
        projected,
        config=RolloutConfig(
            max_level=1,
            constraint_policy=policy,
        ),
    )

    snapshot = model.constraint_snapshots[0]
    allowed, expected_bias = policy.type_constraints(
        1, device=projected["node_mask"].device
    )
    assert torch.equal(snapshot["allowed_type_mask"], allowed)
    assert torch.equal(snapshot["type_logit_bias"], expected_bias)
    assert snapshot["type_logit_bias"][4] == 0
    assert bool((snapshot["type_logit_bias"] < 0).any())
    assert torch.equal(
        snapshot["pointer_validity_mask"],
        policy.pointer_validity_mask(projected, 1)
        & (projected["parent_ids"] < 0),
    )


def test_projection_rejects_compatibility_adapter_and_non_normalized_batches():
    compatibility = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    compatibility["runtime_features_are_raw"] = torch.tensor(True)
    with pytest.raises(ValueError, match="native direct-mdst-tree-v4"):
        project_schema_v4_fsps(compatibility)

    native = _normalized_native_v4_batch()
    native["runtime_features_are_raw"] = torch.tensor(False)
    with pytest.raises(ValueError, match="training data-module contract"):
        project_schema_v4_fsps(native)

    wrong_dtype = _normalized_native_v4_batch()
    wrong_dtype["p4"] = wrong_dtype["p4"].double()
    with pytest.raises(TypeError, match="p4 must have dtype float32"):
        project_schema_v4_fsps(wrong_dtype)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    (
        ("common_features", "append", r"common_features must have shape \[B,N,12\]"),
        ("track_features", "truncate", r"track_features must have shape \[B,N,16\]"),
        ("cluster_availability", "append", r"cluster_availability must have shape \[B,N,9\]"),
        ("klm_features", "truncate", r"klm_features must have shape \[B,N,9\]"),
        ("composite_features", "append", r"composite_features must have shape \[B,N,13\]"),
        (
            "daughter_input_pid_histogram",
            "truncate",
            r"daughter_input_pid_histogram must have shape \[B,N,41\]",
        ),
        (
            "daughter_input_pid_histogram_available",
            "append",
            r"daughter_input_pid_histogram_available must have shape \[B,N\]",
        ),
        ("pid_labels", "append", r"pid_labels must have shape \[B,N\]"),
        ("p4", "truncate", r"p4 must have shape \[B,N,4\]"),
        (
            "daughter_adjacency",
            "append",
            r"daughter_adjacency must have shape \[B,N,N\]",
        ),
        (
            "recursive_leaf_source_mask",
            "empty_width",
            r"recursive_leaf_source_mask must have shape \[B,N,S>0\]",
        ),
        (
            "recursive_leaf_source_mask",
            "append",
            r"recursive_leaf_source_mask must have shape \[B,N,S>0\]",
        ),
        (
            "runtime_features_are_raw",
            "append",
            "training data-module contract",
        ),
    ),
)
def test_projection_rejects_non_native_tensor_ranks_and_widths(
    field, mutation, message
):
    batch = _normalized_native_v4_batch()
    value = batch[field]
    if mutation == "append":
        batch[field] = value.unsqueeze(-1)
    elif mutation == "truncate":
        batch[field] = value[..., :-1]
    else:
        batch[field] = value[..., :0]

    with pytest.raises(ValueError, match=message):
        project_schema_v4_fsps(batch)
