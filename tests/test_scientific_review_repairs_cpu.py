import torch
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import _constrained_rollout_model_batch
from hypertagging.training.scheduled_sampling import aligned_level_targets
from hypertagging.training.pretraining_diagnostics import (
    channel_retrieval_metrics,
    radial_cap_diagnostics,
)


def contexts():
    n = 6
    truth = {
        "node_mask": torch.ones(1, n, dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 0, 0, 1, 2]]),
        "valid_reconstruction_target": torch.tensor(
            [[0, 0, 0, 0, 1, 1]], dtype=torch.bool
        ),
        "recursive_reconstructable_complete": torch.ones(1, n, dtype=torch.bool),
        "daughter_adjacency": torch.zeros(1, n, n, dtype=torch.bool),
        "recursive_leaf_source_mask": torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 1, 0],
                    [1, 1, 1, 0],
                ]
            ],
            dtype=torch.bool,
        ),
        "pid_labels": torch.tensor([[11, 11, 11, 11, 4, 2]]),
    }
    truth["daughter_adjacency"][0, 4, [1, 2]] = True
    truth["daughter_adjacency"][0, 5, [0, 4]] = True
    state = {
        "node_mask": torch.ones(1, n, dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 0, 0, 1, 1]]),
        "parent_ids": torch.tensor([[5, 4, 4, 5, -1, -1]]),
        "daughter_adjacency": torch.zeros(1, n, n, dtype=torch.bool),
        "recursive_leaf_source_mask": torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ],
            dtype=torch.bool,
        ),
        "node_kind_ids": torch.tensor(
            [[NODE_KIND_TO_ID["track"]] * 4 + [NODE_KIND_TO_ID["composite"]] * 2]
        ),
        "p4": torch.zeros(1, n, 4),
        "charge": torch.zeros(1, n),
    }
    state["daughter_adjacency"][0, 4, [1, 2]] = True
    state["daughter_adjacency"][0, 5, [0, 3]] = True
    return truth, state


def test_consumed_target_is_unrepresentable_and_context_is_preserved():
    truth, state = contexts()
    policy = ReconstructionConstraintPolicy()
    legal = policy.forest_pointer_validity_mask(state, 2)
    aligned = aligned_level_targets(
        truth, state, target_level=2, pointer_eligibility=legal
    )
    assert aligned.truth_target_count == 1 and aligned.representable_count == 0
    assert legal.tolist() == [[False, False, False, False, True, True]]
    inference = _constrained_rollout_model_batch(state, target_level=2, policy=policy)
    assert torch.equal(legal, inference["pointer_validity_mask"])
    assert inference["node_mask"].all()


def test_teacher_future_parents_do_not_hide_legal_targets():
    truth, state = contexts()
    truth.update(
        node_kind_ids=state["node_kind_ids"],
        p4=state["p4"],
        charge=state["charge"],
        parent_ids=torch.tensor([[5, 4, 4, -1, 5, -1]]),
    )
    legal = ReconstructionConstraintPolicy().forest_pointer_validity_mask(
        truth, 2, teacher_prefix=True
    )
    aligned = aligned_level_targets(
        truth, truth, target_level=2, pointer_eligibility=legal
    )
    assert aligned.representable_count == 1
    assert aligned.target_override[1][0].tolist() == [
        [True, False, False, False, True, False]
    ]


def test_alignment_selects_eligible_later_alias():
    truth, state = contexts()
    state["recursive_leaf_source_mask"][0, 3] = state["recursive_leaf_source_mask"][
        0, 0
    ]
    legal = torch.tensor([[False, False, False, True, True, False]])
    aligned = aligned_level_targets(
        truth, state, target_level=2, pointer_eligibility=legal
    )
    assert aligned.representable_count == 1
    assert (
        aligned.target_override[1][0][0, 3] and not aligned.target_override[1][0][0, 0]
    )


def test_committed_sources_block_parentless_alias_in_batched_states():
    _, state = contexts()
    state["parent_ids"][0, 0] = -1  # alias is parentless but its source was consumed
    batch = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in state.items()}
    legal = ReconstructionConstraintPolicy().forest_pointer_validity_mask(batch, 2)
    assert not legal[:, 0].any()
    assert legal[:, 4:].all()


def test_repeated_views_do_not_supply_channel_peers():
    result = channel_retrieval_metrics(
        torch.eye(2).repeat(2, 1),
        torch.tensor([1, 2, 1, 2]),
        torch.tensor([0, 1, 0, 1]),
        view_ids=torch.tensor([1, 1, 2, 2]),
    )
    assert result["validation_channel_retrieval_queries"] == 0
    assert "validation_channel_retrieval_accuracy" not in result


def test_cross_event_retrieval_has_explicit_support():
    result = channel_retrieval_metrics(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([1, 1, 2]),
        torch.arange(3),
    )
    assert result["validation_channel_retrieval_queries"] == 2
    assert result["validation_channel_retrieval_correct"] == 2


def test_radial_diagnostic_detects_cap_not_ball_boundary():
    result = radial_cap_diagnostics(
        torch.tensor([[[30.0, 0.0], [31.0, 0.0]]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.long),
        1.5,
    )
    assert result["radial_saturated_fraction"] == 1
    assert result["radial_derivative_mean"] == 0
    assert result["level_0_radius_variance"] == 0


def test_confidence_uses_constrained_threshold_and_cardinality():
    from hypertagging.losses.level_reconstruction import level_reconstruction_loss
    from hypertagging.models.mother_pointer import MotherPointerOutput

    batch = {
        "node_features": torch.zeros(1, 3, 1),
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
        "level_ids": torch.zeros(1, 3, dtype=torch.long),
        "node_kind_ids": torch.zeros(1, 3, dtype=torch.long),
        "pointer_validity_mask": torch.ones(1, 3, dtype=torch.bool),
        "p4": torch.zeros(1, 3, 4),
        "charge": torch.zeros(1, 3),
        "pid_labels": torch.ones(1, 3, dtype=torch.long),
    }
    output = MotherPointerOutput(
        object_logits=torch.ones(1, 1),
        type_logits=torch.nn.functional.one_hot(torch.tensor([[4]]), 41).float() * 8,
        pointer_logits=torch.logit(torch.tensor([[[0.45, 0.40, 0.36]]])),
        cardinality_logits=torch.tensor([[[0.0, 0.0, 8.0, 0.0]]]),
        confidence_logits=torch.zeros(1, 1),
    )
    targets = (
        [torch.tensor([4])],
        [torch.tensor([[True, True, False]])],
        [torch.zeros(1, 4)],
        [torch.zeros(1)],
    )
    policy = ReconstructionConstraintPolicy(
        minimum_pointer_probability=0.35, mother_charge_compatibility="off"
    )
    loss = level_reconstruction_loss(
        output, batch, target_level=1, target_override=targets, constraint_policy=policy
    )
    assert float(loss.confidence_targets[0, 0]) == 1.0


def test_zero_cardinality_never_selects_one_daughter():
    from hypertagging.models.mother_pointer import constrained_daughter_decode

    selected, valid = constrained_daughter_decode(
        torch.ones(3),
        cardinality=0,
        pointer_mask=torch.ones(3, dtype=torch.bool),
        source_conflict=torch.zeros(3, 3, dtype=torch.bool),
    )
    assert valid and not selected.any()


def test_uid_filter_runs_after_role_check_before_tensor_construction(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from hypertagging.training import data_module as module
    from hypertagging.data.splitting import SourceAwareSplitConfig

    dm = module.RealDataModule(
        input_paths=(),
        normalizers={},
        split_manifest={},
        split_manifest_hash="x",
        overflow_counters={},
        seed=1,
        split_config=SourceAwareSplitConfig(),
        source_split_overrides={"dev": "validation", "train": "train"},
        selection_manifest_hash="bound",
    )
    records = [
        {"event_uid": "a", "source_file": "dev"},
        {"event_uid": "b", "source_file": "dev"},
        {"event_uid": "c", "source_file": "train"},
    ]
    monkeypatch.setattr(dm, "_records", lambda: iter(records))
    converted = []

    def convert(record):
        converted.append(record["event_uid"])
        return SimpleNamespace(event_uid=record["event_uid"])

    monkeypatch.setattr(module, "heterogeneous_event_from_record", convert)
    assert [
        e.event_uid for e in dm.iter_events("validation", event_uids=["b", "c"])
    ] == ["b"]
    assert converted == ["b"]
    records.append({"event_uid": "unrequested", "source_file": "unbound"})
    with pytest.raises(ValueError, match="training-selection roles"):
        list(dm.iter_events("validation", event_uids=["b"]))


def test_recovery_has_no_gradient_on_already_matched_query():
    from hypertagging.training.scheduled_sampling import unmatched_object_recovery_loss
    logits = torch.tensor([9., -1., -2.], requires_grad=True)
    loss = unmatched_object_recovery_loss(logits, [(0, 0)], 1)
    loss.backward()
    assert logits.grad[0] == logits.grad[2] == 0
    assert logits.grad[1] < 0
    assert unmatched_object_recovery_loss(logits, [(0, 0), (1, 1), (2, 2)], 2) == 0


def test_indexed_uid_reads_skip_unrequested_json_and_check_identity(tmp_path, monkeypatch):
    import json
    import pytest
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from hypertagging.training import data_module as module
    from hypertagging.data.splitting import SourceAwareSplitConfig
    path = tmp_path / 'indexed.parquet'
    def write(source='dev', uid='b'):
        table = pa.table({'event_uid':['a','b'], 'source_file':[source,'dev'],
                          'event_json':['not decoded', json.dumps({'event_uid':uid,'source_file':'dev'})]})
        pq.write_table(table, path, row_group_size=1)
    write()
    dm = module.RealDataModule(input_paths=(str(path),), normalizers={}, split_manifest={}, split_manifest_hash='x',
                               overflow_counters={}, seed=1, split_config=SourceAwareSplitConfig(),
                               source_split_overrides={'dev':'validation'}, selection_manifest_hash='bound', dataset_index={})
    monkeypatch.setattr(module, 'heterogeneous_event_from_record', lambda r: SimpleNamespace(event_uid=r['event_uid']))
    assert [e.event_uid for e in dm.iter_events('validation', event_uids=['b'])] == ['b']
    write(uid='wrong')
    with pytest.raises(ValueError, match='identity columns'):
        list(dm.iter_events('validation', event_uids=['b']))
    write(source='unbound')
    with pytest.raises(ValueError, match='training-selection roles'):
        list(dm.iter_events('validation', event_uids=['b']))
