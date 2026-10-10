from copy import deepcopy
import pytest
from scripts.recursive_target_witness_audit import (
    RecursiveGraph,
    GraphInvalid,
    bounded_tuples,
    inherited_blockers,
    matches,
    terminal_adjacency,
)


def graph(pids=None):
    return RecursiveGraph(
        [[], [], [0], [2, 1], [0, 1]],
        [False, False, True, True, True],
        [10, 11, None, None, None],
        pids or [1, 2, 3, 4, 4],
    )


def test_unary_and_equal_source_are_not_equal_topology():
    g = graph()
    assert g.leaves[0] == g.leaves[2]
    assert g.topology[0] != g.topology[2]
    assert g.leaves[3] == g.leaves[4]
    assert g.topology[3] != g.topology[4]


def test_descendant_pid_and_missing_pid():
    a, b = graph(), graph([2, 2, 3, 4, 4])
    assert matches(a, [3], b, [3])
    assert not matches(a, [3], b, [3], pid=True)
    assert matches(a, [3], graph([0, 2, 3, 4, 4]), [3], pid=True) is None


def test_multiplicity_alias_and_lowest_outside_policy_blocker():
    g = RecursiveGraph(
        [[], [0, 0], [1]], [False, True, True], [10, None, None], [1, 2, 3]
    )
    assert g.leaves[1] == (10, 10)
    assert inherited_blockers(g, {1: ["duplicate"], 2: ["unary"]})[2] == [1]


def test_bound_counts_rejected_tuples_without_backtracking():
    stream, truncated = bounded_tuples([list(range(20))] * 2)
    tuples = list(stream)
    assert len(tuples) == 256 and truncated and tuples[0] == (0, 0)
    assert len(set(tuples)) == 256


@pytest.mark.parametrize(
    "children,reason", [([[1], [0]], "cycle"), ([[2], []], "dangling")]
)
def test_invalid_graph(children, reason):
    with pytest.raises(GraphInvalid, match=reason):
        RecursiveGraph(children, [True] * 2, [None] * 2, [1] * 2)


def test_last_step_adjacency_and_delayed_round():
    state = dict(node_ids=[0, 1], daughter_adjacency=[[False] * 2 for _ in range(2)])
    trace = dict(
        steps=[
            dict(
                height=3,
                decode_trace=dict(state=state),
                accepted=[dict(daughter_positions=[0, 1])],
                appended_node_ids=[2],
            )
        ],
        final_state=dict(node_ids=[0, 1, 2]),
    )
    assert terminal_adjacency(trace)["daughter_adjacency"][2] == [True, True, False]
    changed = deepcopy(trace)
    changed["steps"].append(
        dict(
            decode_trace=dict(
                state=dict(
                    node_ids=[0, 1, 2],
                    daughter_adjacency=[[False] * 3 for _ in range(3)],
                )
            ),
            accepted=[],
            appended_node_ids=[],
        )
    )
    with pytest.raises(GraphInvalid, match="later_adjacency"):
        terminal_adjacency(changed)


def test_native_fixture_audit_and_unknown_track_pid():
    from dataclasses import replace
    from tests.test_phase86_trace_cpu import fixture, config
    from scripts.phase84_hierarchy_trace import (
        capture_native_trace,
        evaluate_native_trace,
    )
    from scripts.recursive_target_witness_audit import audit_event

    batch, model = fixture()
    cfg = replace(config(), capture_decode_trace=True)
    projection, _, trace = capture_native_trace(model(), batch, cfg)
    historical = evaluate_native_trace(
        batch, projection, trace, target_policy="complete_only", minimum_daughters=2
    )
    result = audit_event(batch, projection, trace, historical, cfg.constraint_policy)
    assert len(result["targets"]) == len(historical["targets"])
    assert result["attempted_tuples"] <= len(result["targets"]) * 6 * 256
    assert all(
        t["historical"] == h for t, h in zip(result["targets"], historical["targets"])
    )
