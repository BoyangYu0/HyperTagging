"""Native greedy trace information boundary and post-hoc source-axis accounting."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import RolloutConfig
from scripts.phase84_hierarchy_trace import (
    capture_native_trace,
    evaluate_native_trace,
    proposal_signature,
)


def native_fixture():
    path = Path(__file__).with_name("test_hierarchical_inference_cpu.py")
    spec = importlib.util.spec_from_file_location("native_trace_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._normalized_native_v4_batch(), module._TwoBThenUpsilonModel


def test_truth_target_mutations_cannot_change_generated_trace():
    batch, model_class = native_fixture()
    config = RolloutConfig(
        max_level=3,
        root_types=(),
        continue_through_empty_levels=True,
        constraint_policy=ReconstructionConstraintPolicy(),
    )
    projection, result, baseline = capture_native_trace(model_class(), batch, config)
    changed = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    for key in (
        "truth_pid_labels",
        "pid_target_labels",
        "b_side",
        "parent_ids",
        "daughter_adjacency",
    ):
        if key in changed:
            changed[key].zero_()
    _, altered, trace = capture_native_trace(model_class(), changed, config)
    assert trace == baseline
    assert [s.accepted for s in result.steps] == [s.accepted for s in altered.steps]
    assert projection.batch["node_mask"].shape[1] < batch["node_mask"].shape[1]
    json.dumps(trace, allow_nan=False)
    assert trace["truth_used_for_generation"] is False
    assert all(not s.used_teacher_forcing for s in result.steps)


def join_fixture():
    target = PDG_TOKENS.index(-521)
    sources = torch.tensor(
        [[[True, False, False], [False, False, True], [True, False, True]]]
    )
    truth = {
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 3]]),
        "pid_labels": torch.tensor([[0, 0, target]]),
        "pid_target_labels": torch.tensor([[0, 0, target]]),
        "daughter_adjacency": torch.tensor(
            [[[False, False, False], [False, False, False], [True, True, False]]]
        ),
        "recursive_leaf_source_mask": sources,
        "valid_reconstruction_target": torch.tensor([[False, False, True]]),
        "recursive_reconstructable_complete": torch.ones(1, 3, dtype=torch.bool),
    }
    projection = SimpleNamespace(
        audit=SimpleNamespace(original_fsp_positions=((0, 1),)),
        batch={"recursive_leaf_source_mask": sources[:, :2][:, :, [0, 2]]},
    )
    proposal = {
        "query_id": 0,
        "mother_type": target,
        "daughter_positions": [0, 1],
        "object_score": 0.9,
        "confidence": 0.8,
    }
    probabilities = [0.0] * len(PDG_TOKENS)
    probabilities[target] = 1.0
    trace = {
        "truth_used_for_generation": False,
        "mode": "predicted",
        "steps": [
            {
                "height": 3,
                "state_before": {
                    "recursive_leaf_source_mask": [[True, False], [False, True]]
                },
                "eligible_positions": [0, 1],
                "proposals": [proposal],
                "accepted": [deepcopy(proposal)],
                "probabilities": {"type": [probabilities]},
            }
        ],
        "final_state": {
            "recursive_leaf_source_mask": [[True, False], [False, True], [True, True]]
        },
    }
    return truth, projection, trace


def evaluate(t, p, s):
    return evaluate_native_trace(
        t, p, s, target_policy="complete_only", minimum_daughters=2
    )


def test_actual_generated_daughters_conditional_signed_pid_and_generation_height():
    t, p, s = join_fixture()
    report = evaluate(t, p, s)
    row = report["targets"][0]
    assert report["source_columns_kept"] == [0, 2]
    assert row["height"] == 3 and row["signed_pdg"] == -521
    assert row["first_error"] == "accepted_exact_daughters_and_pid"
    assert (
        row["conditional_pid"]["correct"]
        and row["conditional_pid"]["multiclass_brier"] == 0
    )
    assert report["counts"]["conditional_pid_trials"] == 1
    assert report["counts"]["all_retained_mothers"] == 1
    assert report["counts"]["accepted_exact_daughters_and_pid"] == 1
    assert report["first_error_counts"] == {"accepted_exact_daughters_and_pid": 1}
    assert sum(report["first_error_counts"].values()) == len(report["targets"])
    assert (
        sum(
            cell.get("accepted_exact_daughters_and_pid", 0)
            for cell in report["cells"].values()
        )
        == report["counts"]["accepted_exact_daughters_and_pid"]
    )
    assert row["legal_unvisited_group_reachability"] == "unavailable_not_checked"


@pytest.mark.parametrize(
    "failure",
    ["wrong_pid", "retention_rejected", "no_proposal", "unavailable_sources"],
)
def test_first_error_accounting_does_not_add_boolean_successes(failure):
    truth, projection, trace = join_fixture()
    step = trace["steps"][0]
    step["accepted"] = []
    if failure == "wrong_pid":
        step["proposals"][0]["mother_type"] = PDG_TOKENS.index(521)
    elif failure == "no_proposal":
        step["proposals"] = []
    elif failure == "unavailable_sources":
        truth["recursive_leaf_source_mask"][0, 2, 1] = True
    report = evaluate(truth, projection, trace)
    reason = report["targets"][0]["first_error"]
    assert reason != "accepted_exact_daughters_and_pid"
    assert report["first_error_counts"] == {reason: 1}
    assert reason not in report["counts"]
    assert report["counts"]["accepted_exact_daughters_and_pid"] == 0
    for key in (
        "generated_exact_daughters",
        "generated_exact_daughters_and_pid",
        "accepted_exact_daughters",
        "accepted_exact_daughters_and_pid",
        "eligible_daughter_source_coverage",
    ):
        assert report["counts"][key] == sum(
            int(row.get(key, False)) for row in report["targets"]
        )
    assert (
        sum(report["first_error_counts"].values())
        == report["counts"]["all_retained_mothers"]
    )
    assert sum(cell[reason] for cell in report["cells"].values()) == 1


def test_wrong_generated_pid_is_separate_from_retention_and_absent_group():
    t, p, s = join_fixture()
    s["steps"][0]["proposals"][0]["mother_type"] = PDG_TOKENS.index(521)
    s["steps"][0]["accepted"] = []
    row = evaluate(t, p, s)["targets"][0]
    assert row["first_error"] == "mother_pid_error_on_generated_correct_daughters"
    assert not row["conditional_pid"]["correct"]
    s["steps"][0]["proposals"][0]["mother_type"] = PDG_TOKENS.index(-521)
    assert (
        evaluate(t, p, s)["targets"][0]["first_error"]
        == "correct_generated_proposal_retention_rejected"
    )
    s["steps"][0]["proposals"] = []
    row = evaluate(t, p, s)["targets"][0]
    assert (
        row["first_error"]
        == "exact_group_not_generated_object_pointer_type_or_constraint_unresolved"
    )
    assert "conditional_pid" not in row
    s["steps"][0]["eligible_positions"] = [0]
    assert (
        evaluate(t, p, s)["targets"][0]["first_error"]
        == "eligible_daughter_source_group_absent"
    )


def test_source_axis_mismatch_and_aliases_fail_closed():
    t, p, s = join_fixture()
    p.batch["recursive_leaf_source_mask"] = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="source-column"):
        evaluate(t, p, s)
    assert (
        proposal_signature(
            {"daughter_positions": [0, 1]}, [frozenset({0}), frozenset({0})]
        )
        is None
    )
    assert proposal_signature({"daughter_positions": [0, 0]}, [frozenset({0})]) is None
    t, p, s = join_fixture()
    t["recursive_leaf_source_mask"][0, 2, 1] = True
    assert (
        evaluate(t, p, s)["targets"][0]["first_error"] == "source_support_unavailable"
    )


def test_unknown_pid_and_target_policy_remain_unavailable():
    t, p, s = join_fixture()
    t["pid_target_labels"][0, 2] = 0
    row = evaluate(t, p, s)["targets"][0]
    assert row["first_error"] == "mother_pid_target_unavailable"
    assert "conditional_pid" not in row
    t, p, s = join_fixture()
    t["recursive_reconstructable_complete"][0, 2] = False
    assert (
        evaluate(t, p, s)["targets"][0]["first_error"]
        == "outside_requested_target_policy"
    )
    partial = evaluate_native_trace(
        t, p, s, target_policy="reconstructable_partial", minimum_daughters=2
    )
    assert partial["targets"][0]["first_error"] == "accepted_exact_daughters_and_pid"
    assert partial["targets"][0]["population"] == "partial"


def test_prejoin_serializer_scrubs_targets_and_accepts_native_source_axis():
    from scripts.phase84_hierarchy_trace import serialize_native_rollout
    from dataclasses import replace

    batch, model_class = native_fixture()
    config = RolloutConfig(
        max_level=2,
        root_types=(),
        continue_through_empty_levels=True,
        constraint_policy=ReconstructionConstraintPolicy(),
    )
    _, rollout, _ = capture_native_trace(model_class(), batch, config)
    baseline = serialize_native_rollout(rollout)
    changed_states = []
    for height, state in rollout.cached_states:
        changed = dict(state)
        changed["b_side"] = torch.tensor([[888]])
        changed["truth_pid_labels"] = torch.tensor([[999]])
        changed["evaluation_leaf_source_keys"] = torch.tensor([[777]])
        changed_states.append((height, changed))
    altered = replace(rollout, cached_states=tuple(changed_states))
    assert serialize_native_rollout(altered) == baseline
    assert baseline["source_axis"] == "native_rollout_source_columns"
    assert all(step["eligible_positions"] is None for step in baseline["steps"])
    assert "truth_pid_labels" not in json.dumps(baseline)
    json.dumps(baseline, allow_nan=False)
    if rollout.steps:
        forbidden = replace(
            rollout, steps=(replace(rollout.steps[0], used_teacher_forcing=True),)
        )
        with pytest.raises(ValueError, match="Truth-guided"):
            serialize_native_rollout(forbidden)
