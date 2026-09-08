"""Regressions for bounded full-depth beam-search review findings."""

from __future__ import annotations

import math

import pytest
import torch

from hypertagging.reconstruction.beam_search import (
    _daughter_combinations,
    canonical_forest_key,
    full_depth_beam_rollout,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    append_composite_proposals,
)
from tests.test_full_depth_beam_cpu import (
    _ScriptedModel,
    _beam,
    _fsp_batch,
    _query,
    _rollout,
)


def test_canonical_key_distinguishes_confidence_model_state():
    batch = _fsp_batch()
    low_confidence, _ = append_composite_proposals(
        batch,
        [CompositeProposal(0, 4, (0, 1), 0.9, 0.2)],
        target_level=1,
    )
    high_confidence, _ = append_composite_proposals(
        batch,
        [CompositeProposal(0, 4, (0, 1), 0.9, 0.8)],
        target_level=1,
    )

    # The topology and particle identities are identical. Confidence is stored
    # in the reconstructed feature state consumed by the next model call.
    torch.testing.assert_close(
        low_confidence["daughter_adjacency"],
        high_confidence["daughter_adjacency"],
    )
    torch.testing.assert_close(
        low_confidence["pid_labels"], high_confidence["pid_labels"]
    )
    assert low_confidence["common_features"][0, -1, 11] == pytest.approx(0.2)
    assert high_confidence["common_features"][0, -1, 11] == pytest.approx(0.8)
    assert canonical_forest_key(low_confidence) != canonical_forest_key(high_confidence)


def test_canonical_key_distinguishes_daughter_pid_model_state():
    batch, _ = append_composite_proposals(
        _fsp_batch(),
        [CompositeProposal(0, 4, (0, 1), 0.9, 0.5)],
        target_level=1,
    )
    changed_histogram = dict(batch)
    changed_histogram["daughter_input_pid_histogram"] = batch[
        "daughter_input_pid_histogram"
    ].clone()
    changed_histogram["daughter_input_pid_histogram"][0, -1, 0] += 0.25
    assert canonical_forest_key(batch) != canonical_forest_key(changed_histogram)

    changed_availability = dict(batch)
    changed_availability["daughter_input_pid_histogram_available"] = batch[
        "daughter_input_pid_histogram_available"
    ].clone()
    changed_availability["daughter_input_pid_histogram_available"][0, -1] = False
    assert canonical_forest_key(batch) != canonical_forest_key(changed_availability)


def test_cross_query_proposal_cap_ranks_gain_over_no_object():
    # Query 0 has the better raw candidate likelihood, but query 1 has the
    # larger gain relative to omitting its high-probability object decision.
    query_zero = _query((0, 1), pointers={0: 12.0, 1: 12.0})
    query_zero["object_logit"] = math.log(0.6 / 0.4)
    query_one = _query((2, 3), pointers={2: 0.04, 3: 0.04})
    query_one["object_logit"] = math.log(0.9 / 0.1)
    pointer_one = float(torch.sigmoid(torch.tensor(0.04)))
    assert math.log(0.6) > math.log(0.9 * pointer_one)
    assert math.log(0.6) - math.log(0.4) < (math.log(0.9 * pointer_one) - math.log(0.1))

    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [query_zero, query_one]),
        _fsp_batch(),
        config=_rollout(),
        beam_config=_beam(
            max_proposals_per_level=1,
            max_candidates_per_query=1,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    proposals = result.candidates[0].result.steps[0].proposals
    assert len(proposals) == 1
    assert proposals[0].query_id == 1


def test_global_proposal_cap_preserves_a_completed_root_candidate():
    root = _query(
        (0, 1),
        types={1: 12.0},
        pointers={0: 0.05, 1: 0.05},
    )
    root["object_logit"] = 0.05
    regular_zero = _query(
        (0, 1),
        types={4: 12.0},
        pointers={0: 4.0, 1: 4.0},
    )
    regular_zero["object_logit"] = 4.0
    regular_one = _query(
        (2, 3),
        types={4: 12.0},
        pointers={2: 4.0, 3: 4.0},
    )
    regular_one["object_logit"] = 4.0

    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [root, regular_zero, regular_one]),
        _fsp_batch(),
        config=_rollout(root_types=(1,)),
        beam_config=_beam(
            max_proposals_per_level=2,
            max_candidates_per_query=1,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    assert all(
        len(step.proposals) <= 2
        for candidate in result.candidates
        for step in candidate.result.steps
    )
    assert any(
        proposal.mother_type == 1
        for proposal in result.candidates[0].result.steps[0].proposals
    )
    assert any(
        candidate.result.stop_reason == "configured_root_reconstructed"
        for candidate in result.candidates
    )


def test_pointer_subsets_follow_mean_pointer_objective():
    # Only (0, 1) and (2, 3) are source-disjoint. Log-odds favored by the old
    # enumerator ranks (0, 1), while the inference objective's mean probability
    # ranks (2, 3).
    probabilities = torch.tensor([0.9, 0.4, 0.7, 0.7])
    source_sets = [
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({0, 2}),
        frozenset({1, 3}),
    ]
    subsets, expanded, exhausted, source_rejections = _daughter_combinations(
        [0, 2, 3, 1],
        probabilities,
        2,
        source_sets=source_sets,
        budget=100,
    )

    assert subsets == [(2, 3), (0, 1)]
    means = [float(probabilities[list(subset)].mean()) for subset in subsets]
    assert means == sorted(means, reverse=True)
    assert 0 < expanded <= 100
    assert not exhausted
    assert source_rejections > 0


def test_invalid_type_and_cardinality_options_backfill_valid_alternatives():
    batch = _fsp_batch()
    batch["p4"][0, :, :3] = 0
    batch["p4"][0, :, 3] = torch.tensor([0.1, 0.2, 2.0, 3.0])
    policy = ReconstructionConstraintPolicy(empirical_type_prior_mode="off")
    specification = _query(
        types={22: 2.0, 4: 1.0},
        pointers={0: 12.0, 1: 12.0},
    )
    specification["cardinalities"] = {1: 2.0, 2: 1.0}

    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [specification]),
        batch,
        config=_rollout(constraint_policy=policy),
        beam_config=_beam(
            max_cardinality_options=1,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    proposals = result.candidates[0].result.steps[0].proposals
    assert [
        (proposal.mother_type, proposal.daughter_positions) for proposal in proposals
    ] == [(4, (0, 1))]
    assert result.diagnostics["charge_rejected"] > 0


def test_source_conflicts_do_not_consume_daughter_output_capacity():
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][
        0, 0
    ]
    sources = batch["recursive_leaf_source_mask"][0]
    width = sources.shape[0]
    batch["source_conflict_matrix"] = (
        (sources.float() @ sources.float().T > 0) & ~torch.eye(width, dtype=torch.bool)
    ).unsqueeze(0)
    specification = _query(
        (0, 1, 2),
        pointers={0: 5.0, 1: 4.0, 2: 2.0},
        cardinality=2,
    )

    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [specification]),
        batch,
        config=_rollout(),
        beam_config=_beam(
            max_cardinality_options=1,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    proposals = result.candidates[0].result.steps[0].proposals
    assert len(proposals) == 1
    assert proposals[0].daughter_positions == (0, 2)
    selected_sources = sources[list(proposals[0].daughter_positions)]
    assert not bool((selected_sources.sum(dim=0) > 1).any())


def test_max_nodes_rejects_oversized_input_and_stops_at_saturation():
    oversized_model = _ScriptedModel(lambda _level, _batch: [_query()])
    with pytest.raises(ValueError, match="max_nodes_per_hypothesis"):
        full_depth_beam_rollout(
            oversized_model,
            _fsp_batch(),
            config=_rollout(),
            beam_config=_beam(max_nodes_per_hypothesis=3),
        )
    assert not oversized_model.calls

    model = _ScriptedModel(lambda _level, _batch: [_query((0, 1)), _query((2, 3))])
    result = full_depth_beam_rollout(
        model,
        _fsp_batch(),
        config=_rollout(max_level=2),
        beam_config=_beam(
            max_nodes_per_hypothesis=5,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    assert any(
        candidate.result.stop_reason == "node_limit_reached"
        for candidate in result.candidates
    )
    assert all(
        int(candidate.batch["node_mask"].sum()) <= 5 for candidate in result.candidates
    )
    assert result.diagnostics["max_nodes_observed"] == 5
    assert result.diagnostics["states_saturated_at_node_limit"] > 0
    assert result.diagnostics["node_limit_hits"] > 0
    assert model.calls == [1]


def test_width_one_stops_at_node_saturation_before_calling_model():
    model = _ScriptedModel(lambda _level, _batch: [_query()])
    result = full_depth_beam_rollout(
        model,
        _fsp_batch(),
        config=_rollout(max_level=2),
        beam_config=_beam(beam_width=1, max_nodes_per_hypothesis=4),
    )

    assert not model.calls
    assert len(result.candidates) == 1
    assert result.candidates[0].result.stop_reason == "node_limit_reached"
    assert int(result.candidates[0].batch["node_mask"].sum()) == 4
    assert result.diagnostics["node_limit_hits"] == 1
    assert result.diagnostics["max_nodes_observed"] == 4


def test_direct_api_rejects_known_truth_and_hides_unknown_payloads():
    unsanitized = _fsp_batch()
    unsanitized["truth_pid_labels"] = torch.ones_like(unsanitized["pid_labels"])
    rejected_model = _ScriptedModel(lambda _level, _batch: [])
    with pytest.raises(ValueError, match="evaluation/target metadata"):
        full_depth_beam_rollout(
            rejected_model,
            unsanitized,
            config=_rollout(),
            beam_config=_beam(),
        )
    assert not rejected_model.calls

    hidden = _fsp_batch()
    hidden["unlisted_truth_payload"] = torch.ones_like(hidden["node_mask"])
    observing_model = _ScriptedModel(lambda _level, _batch: [])
    full_depth_beam_rollout(
        observing_model,
        hidden,
        config=_rollout(),
        beam_config=_beam(),
    )
    assert observing_model.inputs
    assert all("unlisted_truth_payload" not in keys for keys in observing_model.inputs)


def test_width_one_alias_mask_marks_greedy_compatibility_false():
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][
        0, 0
    ]
    sources = batch["recursive_leaf_source_mask"][0]
    width = sources.shape[0]
    batch["source_conflict_matrix"] = (
        (sources.float() @ sources.float().T > 0) & ~torch.eye(width, dtype=torch.bool)
    ).unsqueeze(0)

    def program(level, _batch):
        return [_query((0, 2))] if level == 1 else [_query((1, 3))]

    result = full_depth_beam_rollout(
        _ScriptedModel(program),
        batch,
        config=_rollout(max_level=2),
        beam_config=_beam(
            beam_width=1,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )

    assert result.diagnostics["source_alias_masks_applied"] > 0
    assert result.diagnostics["width_one_source_safe_greedy"] is True
    assert result.diagnostics["width_one_greedy_compatibility"] is False


def test_selected_empty_and_no_candidate_have_distinct_stop_reasons():
    low_gain = _query((0, 1), pointers={0: 0.04, 1: 0.04})
    low_gain["object_logit"] = 0.04
    with_candidate = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [low_gain]),
        _fsp_batch(),
        config=_rollout(),
        beam_config=_beam(
            beam_width=2,
            max_type_options=1,
            max_daughter_options=1,
        ),
    )
    selected_empty = [
        candidate
        for candidate in with_candidate.candidates
        if candidate.result.stop_reason == "beam_selected_empty"
    ]
    assert len(selected_empty) == 1
    assert selected_empty[0].result.steps[0].proposals
    assert not selected_empty[0].accepted_by_level[0]
    assert all(
        candidate.result.stop_reason != "all_no_object"
        for candidate in with_candidate.candidates
    )

    without_candidate = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: []),
        _fsp_batch(),
        config=_rollout(),
        beam_config=_beam(beam_width=2),
    )
    assert len(without_candidate.candidates) == 1
    no_object = without_candidate.candidates[0]
    assert no_object.result.stop_reason == "all_no_object"
    assert not no_object.result.steps[0].proposals
