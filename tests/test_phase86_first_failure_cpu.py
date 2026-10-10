"""Causal audit regressions: delayed recovery, constraints, joint support failures."""

from dataclasses import asdict

from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import RolloutConfig
from scripts.phase86_first_failure import legal_group, lifecycle_target, query_support


def fixture():
    pid = PDG_TOKENS.index(111)
    policy = ReconstructionConstraintPolicy()
    state = dict(
        node_mask=[True, True],
        node_ids=[0, 1],
        level_ids=[0, 0],
        pid_labels=[0, 0],
        parent_ids=[-1, -1],
        node_kind_ids=[0, 0],
        recursive_leaf_source_mask=[[True, False], [False, True]],
        source_conflict_matrix=[[False, False], [False, False]],
        p4=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        charge=[0.0, 0.0],
        daughter_adjacency=[[False, False], [False, False]],
    )
    # Use accepted node kinds independent of representation enum order.
    state["node_kind_ids"] = [policy.valid_leaf_node_kinds[0]] * 2
    ty = [0.0] * len(PDG_TOKENS)
    ty[pid] = 1.0
    d = dict(
        state=state,
        hard_decode_context_mask=[True, True],
        context_mask=[True, True],
        pointer_validity_mask=[True, True],
        forest_pointer_validity_mask=[True, True],
        raw_proposals=[],
    )
    step = dict(
        height=1,
        decode_trace=d,
        probabilities=dict(
            object=[0.9],
            type=[ty],
            pointer=[[0.8, 0.8]],
            cardinality=[[0.0, 0.0, 1.0]],
            confidence=[0.9],
        ),
        accepted=[],
    )
    config = asdict(RolloutConfig(constraint_policy=policy))
    return pid, policy, step, config


def test_current_charge_constraint_is_separate_from_source_cover():
    pid, policy, step, _ = fixture()
    step["decode_trace"]["state"]["charge"] = [1.0, 1.0]
    assert "mother_charge_incompatible" in legal_group(
        step["decode_trace"]["state"], [0, 1], pid, policy, 1
    )


def test_query_retains_multiple_failures_without_changing_proposals():
    _, policy, step, config = fixture()
    step["probabilities"]["object"] = [0.1]
    step["probabilities"]["pointer"] = [[0.1, 0.9]]
    step["probabilities"]["cardinality"] = [[0.0, 1.0, 0.0]]
    rows = query_support(step, [0, 1], policy, config)
    assert {
        "object_rejection",
        "cardinality_support",
        "pointer_threshold_support",
        "pointer_ranking_or_source_selection",
    } <= set(rows[0]["failures"])
    assert step["decode_trace"]["raw_proposals"] == []


def test_delayed_round_proposal_counts_as_all_round_recovery():
    pid, policy, step, config = fixture()
    step["height"] = 3
    p = dict(daughter_positions=[0, 1], mother_type=pid, query_id=0)
    step["decode_trace"]["raw_proposals"] = [p]
    step["accepted"] = [p]
    rounds, loss = lifecycle_target(
        [frozenset([0]), frozenset([1])], pid, dict(config=config, steps=[step]), policy
    )
    assert (
        rounds[0]["accepted_correct_type"] == 1
        and rounds[0]["legal_correct_type_group"]
    )
    assert loss is None


def test_one_root_split_is_only_an_oracle_opportunity():
    pid, policy, step, config = fixture()
    s = step["decode_trace"]["state"]
    s.update(
        node_mask=[True] * 3,
        node_ids=[0, 1, 2],
        level_ids=[0, 0, 1],
        pid_labels=[0, 0, pid],
        parent_ids=[2, 2, -1],
        node_kind_ids=[policy.valid_leaf_node_kinds[0]] * 2
        + [policy.valid_composite_node_kinds[0]],
        recursive_leaf_source_mask=[[True, False], [False, True], [True, True]],
        source_conflict_matrix=[[False] * 3 for _ in range(3)],
        p4=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]],
        charge=[0.0] * 3,
        daughter_adjacency=[[False] * 3, [False] * 3, [True, True, False]],
    )
    step["height"] = 2
    step["decode_trace"]["context_mask"] = [True] * 3
    step["decode_trace"]["forest_pointer_validity_mask"] = [False, False, True]
    rounds, loss = lifecycle_target(
        [frozenset([0]), frozenset([1])], pid, dict(config=config, steps=[step]), policy
    )
    assert loss is not None and rounds[0]["single_root_split_oracle"]
    assert rounds[0]["accepted_correct_type"] == 0
    assert s["parent_ids"] == [2, 2, -1]
