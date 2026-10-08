"""Trace parity, source grouping and first irreversible-loss regressions."""
import pytest

from hypertagging.evaluation.search_survival import membership_possible, analyze_search_trace
from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
from hypertagging.reconstruction.beam_search import full_depth_beam_rollout, canonical_forest_key
from tests.test_full_depth_beam_cpu import _ScriptedModel, _beam, _fsp_batch, _native_batch, _query, _rollout


def test_trace_does_not_change_candidates_scores_or_model_inputs():
    batch = _fsp_batch()
    def program(level, state):
        return [_query((0, 1)), _query((2, 3))]
    plain_model, traced_model = _ScriptedModel(program), _ScriptedModel(program)
    config, beam = _rollout(max_level=2, continue_through_empty_levels=True), _beam()
    plain = full_depth_beam_rollout(plain_model, batch, config=config, beam_config=beam)
    trace = []
    observed = full_depth_beam_rollout(traced_model, batch, config=config, beam_config=beam, trace=trace)
    assert [canonical_forest_key(h.batch) for h in plain.candidates] == [canonical_forest_key(h.batch) for h in observed.candidates]
    assert [h.score for h in plain.candidates] == [h.score for h in observed.candidates]
    assert plain.diagnostics == observed.diagnostics
    assert plain_model.inputs == traced_model.inputs
    assert trace[-1]["stage"] == "final"
    assert {"decode", "expanded", "retained_level"} <= {r["stage"] for r in trace}
    assert all("evaluation_leaf_source_keys" not in keys for keys in traced_model.inputs)


def test_trace_records_candidates_removed_before_forest_search():
    trace = []
    model = _ScriptedModel(lambda *_: [_query(types={4: 12., 5: 11.})])
    full_depth_beam_rollout(model, _fsp_batch(), config=_rollout(),
                           beam_config=_beam(max_candidates_per_query=1), trace=trace)
    record = next(r for r in trace if r["stage"] == "decode")
    assert len(record["generated"]) > len(record["query_retained"])
    assert len(record["query_retained"]) == 1


def test_clean_root_cover_detects_wrong_cross_group_merge():
    state = {"memberships": [[0], [1], [2], [0, 2]], "roots": [1, 3], "composites": [3]}
    assert not membership_possible(state, frozenset({0, 1}))
    # The other B may still be represented by pure roots; do not count all
    # event targets lost because one branch has made a bad merge.
    assert membership_possible(state, frozenset({1}))


def test_previously_formed_group_survives_consumption_by_larger_composite():
    state = {"memberships": [[0], [1], [2], [0, 1], [0, 1, 2]],
             "roots": [4], "composites": [3, 4]}
    assert membership_possible(state, frozenset({0, 1}))


def test_width_one_trace_fails_explicitly_instead_of_claiming_full_coverage():
    with pytest.raises(ValueError, match="beam_width"):
        full_depth_beam_rollout(_ScriptedModel(lambda *_: []), _fsp_batch(),
                               config=_rollout(), beam_config=_beam(beam_width=1), trace=[])


def test_posthoc_analysis_joins_original_identities_after_projection():
    truth = _native_batch()
    projection = project_schema_v4_fsps(truth)
    trace = []
    full_depth_beam_rollout(_ScriptedModel(lambda *_: [_query((0, 1))]), projection.batch,
                           config=_rollout(), beam_config=_beam(), trace=trace)
    result = analyze_search_trace(truth, projection, trace, target_policy="complete_only",
                                  minimum_daughters=2, object_threshold=.5, pointer_threshold=.5)
    assert sum(c["truth_targets"] for c in result["by_level"].values()) > 0
    assert all(c.get("identity_unavailable", 0) == 0 for c in result["by_level"].values())
