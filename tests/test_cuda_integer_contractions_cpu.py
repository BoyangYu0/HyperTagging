from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.embedding_losses import connection_loss_from_predictions
from hypertagging.losses.level_reconstruction import query_proposal_repulsion_loss
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    _upgrade_flat_batch,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, STATIC_MOTHER_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    RolloutConfig,
    _batched_initial_leaf_state,
    append_composite_proposals,
    batched_decode_level,
    batched_level_step,
)
from hypertagging.training.pretraining_curriculum import (
    _rebuild_corrupted_derived_fields,
)
from hypertagging.training.reconstruction_trainer import _recursive_source_conflicts
from hypertagging.utils.tensor_contractions import boolean_matmul


@pytest.fixture
def contraction_calls(monkeypatch):
    observed = []
    original_bmm = torch.bmm
    original_matmul = torch.matmul

    def checked_bmm(left, right):
        observed.append(
            ("bmm", left.dtype, right.dtype, torch.is_autocast_enabled("cpu"))
        )
        return original_bmm(left, right)

    def checked_matmul(left, right):
        observed.append(
            ("matmul", left.dtype, right.dtype, torch.is_autocast_enabled("cpu"))
        )
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "bmm", checked_bmm)
    monkeypatch.setattr(torch, "matmul", checked_matmul)
    return observed


def _assert_safe_calls(
    observed: list[tuple[str, torch.dtype, torch.dtype, bool]],
) -> None:
    assert observed
    assert all(
        left == right == torch.float32 and not autocast
        for _operation, left, right, autocast in observed
    )


def _leaf_state(event_index: int = 0) -> dict[str, torch.Tensor]:
    batch = collate_level_events(
        [tiny_level_events()[event_index]], max_query_slots=4
    ).to_dict()
    upgraded = _upgrade_flat_batch(batch)
    keep = upgraded["node_mask"][0] & (upgraded["level_ids"][0] == 0)
    indices = keep.nonzero(as_tuple=False).flatten()
    count = upgraded["node_mask"].shape[1]
    result = {}
    for name, value in upgraded.items():
        if value.ndim >= 3 and value.shape[1:3] == (count, count):
            result[name] = value[:, indices][:, :, indices]
        elif value.ndim >= 2 and value.shape[1] == count:
            result[name] = value[:, indices]
        else:
            result[name] = value
    result["parent_ids"] = torch.full_like(result["parent_ids"], -1)
    leaf_count = indices.numel()
    result["recursive_leaf_source_mask"] = torch.eye(
        leaf_count, dtype=torch.bool
    ).unsqueeze(0)
    result["source_conflict_matrix"] = torch.zeros(
        (1, leaf_count, leaf_count), dtype=torch.bool
    )
    return result


def _model() -> LevelAutoregressiveReconstructor:
    return LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
        n_heads=2,
        n_context_layers=1,
        max_cardinality=4,
        dropout=0.0,
    ).eval()


def test_boolean_matmul_uses_fp32_outside_autocast_for_wide_bool_and_int_masks(
    contraction_calls,
):
    width = 65_537
    sources = torch.zeros((1, 3, width), dtype=torch.int32)
    sources[0, 0] = 7
    sources[0, 1, 0] = -2
    sources[0, 2, -1] = 1
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        overlap = boolean_matmul(sources, sources.transpose(1, 2))
        two_dimensional = boolean_matmul(sources[0], sources[0].T)

    expected = torch.tensor(
        [[[True, True, True], [True, True, False], [True, False, True]]]
    )
    assert torch.equal(overlap, expected)
    assert torch.equal(two_dimensional, expected[0])
    assert {operation for operation, *_rest in contraction_calls} == {"bmm", "matmul"}
    _assert_safe_calls(contraction_calls)


def test_collation_overlap_accepts_integer_sources_and_uses_safe_contraction(
    contraction_calls,
):
    event = heterogeneous_from_level_event(tiny_level_events()[0])
    event = replace(
        event,
        recursive_leaf_source_mask=event.recursive_leaf_source_mask.to(torch.int16),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        batch = collate_heterogeneous_events([event])

    sources = batch["recursive_leaf_source_mask"].bool()
    expected = (sources[:, :, None, :] & sources[:, None, :, :]).any(dim=-1)
    expected &= ~torch.eye(expected.shape[-1], dtype=torch.bool).unsqueeze(0)
    assert torch.equal(batch["source_conflict_matrix"], expected)
    _assert_safe_calls(contraction_calls)


def test_corrupted_field_rebuild_unions_integer_sources_and_builds_overlap_safely(
    contraction_calls,
):
    batch = {
        "daughter_adjacency": torch.tensor(
            [[[False, False, False], [False, False, False], [True, True, False]]]
        ),
        "node_mask": torch.ones((1, 3), dtype=torch.bool),
        "p4": torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0] * 4]]),
        "charge": torch.tensor([[1.0, -1.0, 0.0]]),
        "recursive_leaf_source_mask": torch.tensor(
            [[[3, 0], [0, -4], [0, 0]]], dtype=torch.int32
        ),
        "level_ids": torch.tensor([[0, 0, 1]]),
        "pid_labels": torch.ones((1, 3), dtype=torch.long),
        "composite_features": torch.zeros((1, 3, 9)),
        "common_features": torch.zeros((1, 3, 12)),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _rebuild_corrupted_derived_fields(batch)

    expected_sources = torch.tensor([[[True, False], [False, True], [True, True]]])
    expected_conflicts = torch.tensor(
        [[[False, False, True], [False, False, True], [True, True, False]]]
    )
    assert torch.equal(batch["recursive_leaf_source_mask"].bool(), expected_sources)
    assert torch.equal(batch["source_conflict_matrix"], expected_conflicts)
    _assert_safe_calls(contraction_calls)


def test_reference_rollout_append_unions_and_overlaps_integer_sources_safely(
    contraction_calls,
):
    state = _leaf_state()
    state["recursive_leaf_source_mask"] = state["recursive_leaf_source_mask"].to(
        torch.int16
    )
    proposal = CompositeProposal(
        0,
        int(STATIC_MOTHER_TOKENS[0]),
        tuple(range(state["node_mask"].shape[1])),
        1.0,
        1.0,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result, _new_ids = append_composite_proposals(state, [proposal], target_level=1)

    expected_union = state["recursive_leaf_source_mask"].bool().any(dim=1)
    assert torch.equal(
        result["recursive_leaf_source_mask"][:, -1].bool(), expected_union
    )
    assert result["source_conflict_matrix"][0, -1, :-1].any()
    _assert_safe_calls(contraction_calls)


def test_batched_step_updates_closure_union_and_overlap_safely(contraction_calls):
    state = _leaf_state()
    model = _model()
    output = model(state, target_level=1)
    contraction_calls.clear()
    state["recursive_leaf_source_mask"] = state["recursive_leaf_source_mask"].to(
        torch.int64
    )
    daughters = torch.zeros_like(output.pointer.pointer_logits, dtype=torch.bool)
    daughters[:, 0] = state["node_mask"]
    accepted = torch.zeros_like(output.pointer.object_logits, dtype=torch.bool)
    accepted[:, 0] = True
    mother_types = torch.full_like(
        output.pointer.object_logits, int(STATIC_MOTHER_TOKENS[0]), dtype=torch.long
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = batched_level_step(
            state,
            output,
            daughter_mask=daughters,
            accepted_query_mask=accepted,
            mother_types=mother_types,
            target_level=1,
        )

    old_count = state["node_mask"].shape[1]
    expected_union = state["recursive_leaf_source_mask"].bool().any(dim=1)
    assert torch.equal(
        result["recursive_leaf_source_mask"][:, old_count].bool(), expected_union
    )
    assert result["ancestor_descendant_relation"][0, old_count, :old_count].all()
    assert result["source_conflict_matrix"][0, old_count, :old_count].any()
    _assert_safe_calls(contraction_calls)


def test_initial_leaf_overlap_accepts_integer_sources_and_contracts_safely(
    contraction_calls,
):
    full_batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    full_batch["recursive_leaf_source_mask"] = full_batch[
        "recursive_leaf_source_mask"
    ].to(torch.int8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        leaves = _batched_initial_leaf_state(full_batch)

    sources = leaves["recursive_leaf_source_mask"].bool()
    expected = (sources[:, :, None, :] & sources[:, None, :, :]).any(dim=-1)
    expected &= ~torch.eye(expected.shape[-1], dtype=torch.bool).unsqueeze(0)
    assert torch.equal(leaves["source_conflict_matrix"], expected)
    _assert_safe_calls(contraction_calls)


def test_exclusive_decode_uses_integer_source_unions_safely(contraction_calls):
    state = _leaf_state()
    model = _model()
    output = model(state, target_level=1)
    contraction_calls.clear()
    pointer = replace(
        output.pointer,
        object_logits=torch.full_like(output.pointer.object_logits, 10.0),
        pointer_logits=torch.full_like(output.pointer.pointer_logits, 10.0),
        type_logits=torch.nn.functional.one_hot(
            torch.full_like(
                output.pointer.object_logits, int(STATIC_MOTHER_TOKENS[0])
            ).long(),
            num_classes=len(PDG_TOKENS),
        ).to(output.pointer.type_logits.dtype)
        * 20.0,
    )
    output = replace(output, pointer=pointer)
    state["node_kind_ids"] = torch.full_like(
        state["node_kind_ids"], NODE_KIND_TO_ID["track"]
    )
    state["recursive_leaf_source_mask"] = state["recursive_leaf_source_mask"].to(
        torch.int32
    )
    config = RolloutConfig(
        object_threshold=0.0,
        pointer_threshold=0.0,
        confidence_threshold=0.0,
        min_daughters=1,
        use_cardinality=False,
        exclusive_final=True,
        constraint_policy=ReconstructionConstraintPolicy(
            minimum_pointer_probability=0.0,
            minimum_daughters=1,
            daughter_cardinality_policy="threshold",
            mother_charge_compatibility="off",
        ),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        selected, accepted, _mother_types = batched_decode_level(
            output,
            state,
            active_event_mask=torch.ones(1, dtype=torch.bool),
            config=config,
        )

    assert accepted.sum() == 1
    assert selected[accepted].all()
    _assert_safe_calls(contraction_calls)


def test_target_overlap_and_padding_outer_product_use_safe_bool_contractions(
    contraction_calls,
):
    logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]], requires_grad=True)
    targets = torch.tensor([[[5, 0], [-3, 0]]], dtype=torch.int16)
    predictions = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    dataset = {
        "padding_mask": torch.tensor([[True, False]]),
        "links": torch.tensor([[0, -1]]),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        repulsion = query_proposal_repulsion_loss(
            logits,
            active_query_mask=torch.ones((1, 2), dtype=torch.bool),
            matched_pointer_targets=targets,
        )
        connection_loss, _accuracy = connection_loss_from_predictions(
            predictions, dataset, already_scaled=True
        )

    assert repulsion == 0
    assert torch.isfinite(connection_loss)
    _assert_safe_calls(contraction_calls)


def test_recursive_conflict_counter_uses_safe_integer_matmul(contraction_calls):
    batch = {
        "recursive_leaf_source_mask": torch.tensor(
            [[[4, 0, 0], [-2, 0, 0], [0, 0, 9]]], dtype=torch.int32
        ),
        "node_mask": torch.ones((1, 3), dtype=torch.bool),
        "level_ids": torch.ones((1, 3), dtype=torch.long),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        conflicts = _recursive_source_conflicts(batch)

    assert conflicts == 1
    assert {operation for operation, *_rest in contraction_calls} == {"matmul"}
    _assert_safe_calls(contraction_calls)
