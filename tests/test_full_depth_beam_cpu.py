"""Behavioral tests for bounded, truth-free full-depth candidate search."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Callable

import pytest
import torch

from hypertagging.data.heterogeneous import (
    MODEL_INPUT_SOURCE_TO_ID,
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    LevelReconstructionOutput,
)
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.beam_search import (
    BeamSearchConfig,
    canonical_forest_key,
    full_depth_beam_rollout,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    RolloutConfig,
    _state_fingerprint,
    append_composite_proposals,
    bounded_beam_rollout,
    evaluation_reference_rollout,
)


def _native_batch() -> dict[str, torch.Tensor]:
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    native = MODEL_INPUT_SOURCE_TO_ID["native_v4_reconstructed"]
    batch["model_input_source_ids"].fill_(native)
    batch["daughter_input_pid_source_ids"].fill_(native)
    batch["runtime_features_are_raw"] = torch.tensor(True)
    return batch


def _fsp_batch() -> dict[str, torch.Tensor]:
    return project_schema_v4_fsps(_native_batch()).batch


def _rollout(**kwargs) -> RolloutConfig:
    defaults = {
        "max_level": 1,
        "root_types": (),
        "constraint_policy": ReconstructionConstraintPolicy(
            mother_charge_compatibility="off",
            empirical_type_prior_mode="off",
        ),
    }
    defaults.update(kwargs)
    return RolloutConfig(**defaults)


def _beam(**kwargs) -> BeamSearchConfig:
    defaults = {
        "beam_width": 8,
        "max_candidates_per_query": 4,
        "max_type_options": 2,
        "max_daughter_options": 2,
        "max_cardinality_options": 1,
    }
    defaults.update(kwargs)
    return BeamSearchConfig(**defaults)


def _query(
    daughters=(0, 1),
    *,
    types=None,
    pointers=None,
    cardinality=2,
) -> dict:
    return {
        "types": types or {4: 12.0},
        "pointers": pointers or {position: 12.0 for position in daughters},
        "cardinality": cardinality,
    }


class _ScriptedModel:
    """Scores reconstructed features/topology only; never receives targets."""

    def __init__(self, program: Callable) -> None:
        self.program = program
        self.calls: list[int] = []
        self.inputs: list[set[str]] = []

    def __call__(self, batch, *, target_level, **kwargs):
        del kwargs
        self.calls.append(target_level)
        self.inputs.append(set(batch))
        specifications = self.program(target_level, batch)
        size, width = batch["node_mask"].shape
        count = max(len(specifications), 1)
        objects = torch.full((size, count), -20.0)
        types = torch.full((size, count, len(PDG_TOKENS)), -20.0)
        pointers = torch.full((size, count, width), -20.0)
        cardinalities = torch.full((size, count, 7), -20.0)
        for query, specification in enumerate(specifications):
            objects[:, query] = specification.get("object_logit", 12.0)
            for token, logit in specification["types"].items():
                types[:, query, token] = logit
            for position, logit in specification["pointers"].items():
                pointers[:, query, position] = logit
            for cardinality, logit in specification.get(
                "cardinalities", {specification["cardinality"]: 12.0}
            ).items():
                cardinalities[:, query, cardinality] = logit
        pointer = MotherPointerOutput(
            object_logits=objects,
            type_logits=types,
            pointer_logits=pointers,
            cardinality_logits=cardinalities,
            confidence_logits=torch.full_like(objects, 12.0),
        )
        hidden = torch.zeros(size, width, 4)
        relation = torch.zeros(size, width, width)
        return LevelReconstructionOutput(
            target_level=target_level,
            pointer=pointer,
            node_embeddings=hidden,
            hyperbolic_embeddings=torch.zeros(size, width, 2),
            context_mask=batch["node_mask"] & (batch["level_ids"] < target_level),
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


def _tree_key(batch) -> tuple:
    mask = batch["node_mask"][0]
    adjacency = batch["daughter_adjacency"][0]
    tokens = batch.get("current_pid_tokens", batch["pid_labels"])[0]

    def node_key(position):
        daughters = (adjacency[position] & mask).nonzero().flatten().tolist()
        if not daughters:
            return ("leaf", position, int(tokens[position]))
        return (
            "mother",
            int(tokens[position]),
            int(batch["level_ids"][0, position]),
            tuple(sorted(node_key(daughter) for daughter in daughters)),
        )

    roots = mask & ~adjacency.any(dim=0)
    return tuple(
        sorted(node_key(position) for position in roots.nonzero().flatten().tolist())
    )


def test_one_query_retains_multiple_mother_types_and_daughter_sets():
    model = _ScriptedModel(
        lambda _level, _batch: [
            _query(types={4: 1.0, 3: 0.8}, pointers={0: 4.0, 1: 3.0, 2: 2.5})
        ]
    )
    result = full_depth_beam_rollout(
        model,
        _fsp_batch(),
        config=_rollout(),
        beam_config=_beam(),
    )
    accepted = [
        proposal
        for item in result.candidates
        for level in item.accepted_by_level
        for proposal in level
    ]
    assert {proposal.mother_type for proposal in accepted} >= {3, 4}
    assert len({proposal.daughter_positions for proposal in accepted}) >= 2
    assert all(
        len(level) <= 1
        for item in result.candidates
        for level in item.accepted_by_level
    )


def test_competing_partial_events_compose_beyond_two_levels():
    def program(level, batch):
        if level == 1:
            return [_query(types={4: 1.0, 3: 0.8})]
        previous = (
            ((batch["level_ids"][0] == level - 1) & batch["node_mask"][0])
            .nonzero()
            .flatten()
            .tolist()
        )
        if not previous:
            return []
        mother = previous[0]
        if level == 2 and int(batch["pid_labels"][0, mother]) != 3:
            return []
        return [_query((mother, level), types={5 if level == 2 else 1: 12.0})]

    config = _rollout(max_level=3, root_types=(1,))
    model = _ScriptedModel(program)
    result = full_depth_beam_rollout(
        model,
        _fsp_batch(),
        config=config,
        beam_config=_beam(),
    )
    completed = [
        item
        for item in result.candidates
        if item.result.stop_reason == "configured_root_reconstructed"
    ]
    assert completed, "the lower-ranked first-level type must remain recoverable"
    assert max(model.calls) == 3
    successful = completed[0]
    assert [level[0].mother_type for level in successful.accepted_by_level] == [3, 5, 1]
    assert max(successful.batch["level_ids"][0]).item() == 3
    assert not any(step.used_teacher_forcing for step in successful.result.steps)


def test_lower_ranked_cardinality_remains_a_candidate():
    specification = _query((0, 1, 2), types={4: 12.0})
    specification["cardinalities"] = {2: 1.0, 3: 0.8}
    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [specification]),
        _fsp_batch(),
        config=_rollout(),
        beam_config=_beam(max_cardinality_options=2, max_type_options=1),
    )
    cardinalities = {
        len(proposal.daughter_positions)
        for item in result.candidates
        for level in item.accepted_by_level
        for proposal in level
    }
    assert cardinalities >= {2, 3}


def test_reduce_cardinality_policy_handles_recursive_source_conflicts():
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][
        0, 0
    ]
    sources = batch["recursive_leaf_source_mask"][0]
    batch["source_conflict_matrix"] = (
        (sources.float() @ sources.float().T > 0) & ~torch.eye(4, dtype=torch.bool)
    ).unsqueeze(0)
    policy = ReconstructionConstraintPolicy(
        mother_charge_compatibility="off",
        empirical_type_prior_mode="off",
        cardinality_insufficient_policy="reduce",
    )
    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [_query((0, 1, 2), cardinality=3)]),
        batch,
        config=_rollout(constraint_policy=policy),
        beam_config=_beam(max_type_options=1),
    )
    accepted = [
        proposal
        for item in result.candidates
        for level in item.accepted_by_level
        for proposal in level
    ]
    assert accepted
    assert {proposal.daughter_positions for proposal in accepted} <= {(0, 2), (1, 2)}


def test_reduced_cardinality_retains_the_likelihood_of_the_predicted_option():
    policy = ReconstructionConstraintPolicy(
        mother_charge_compatibility="off",
        empirical_type_prior_mode="off",
        cardinality_insufficient_policy="reduce",
    )
    result = full_depth_beam_rollout(
        _ScriptedModel(lambda _level, _batch: [_query((0, 1), cardinality=3)]),
        _fsp_batch(),
        config=_rollout(constraint_policy=policy),
        beam_config=_beam(max_type_options=1),
    )
    best = result.candidates[0]
    assert best.accepted_by_level[0], "valid reduced candidate should beat no-object"
    assert best.accepted_by_level[0][0].daughter_positions == (0, 1)
    assert best.score > -0.01


def test_equivalent_query_assignments_deduplicate_with_stable_ordering():
    model = _ScriptedModel(
        lambda _level, _batch: [
            _query(),
            _query(),
            _query((2, 3)),
            _query((2, 3)),
        ]
    )
    config = _beam(max_type_options=1, max_daughter_options=1)
    first = full_depth_beam_rollout(
        model, _fsp_batch(), config=_rollout(), beam_config=config
    )
    second = full_depth_beam_rollout(
        model, _fsp_batch(), config=_rollout(), beam_config=config
    )
    first_keys = [_tree_key(item.batch) for item in first.candidates]
    second_keys = [_tree_key(item.batch) for item in second.candidates]
    assert len(first_keys) == len(set(first_keys))
    assert first_keys == second_keys
    assert [item.score for item in first.candidates] == [
        item.score for item in second.candidates
    ]
    assert any(
        len(level) == 2 for item in first.candidates for level in item.accepted_by_level
    )


def test_canonical_tree_key_preserves_daughters_with_identical_source_memberships():
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][
        0, 0
    ]
    first, _ = append_composite_proposals(
        batch,
        [CompositeProposal(0, 4, (0, 2), 0.9, 0.8)],
        target_level=1,
    )
    second, _ = append_composite_proposals(
        batch,
        [CompositeProposal(0, 4, (1, 2), 0.9, 0.8)],
        target_level=1,
    )
    # The older key flattened each node into (level, PID, sources), hiding
    # which detector candidate among associated aliases was actually used.
    assert _state_fingerprint(first) == _state_fingerprint(second)
    assert canonical_forest_key(first) != canonical_forest_key(second)


def test_legacy_beam_entry_point_defaults_to_complete_configured_depth():
    def program(level, batch):
        if level == 1:
            return [_query(types={4: 12.0})]
        previous = (batch["level_ids"][0] == level - 1).nonzero().flatten().tolist()
        return (
            [_query((previous[0], level), types={5 if level == 2 else 1: 12.0})]
            if previous
            else []
        )

    model = _ScriptedModel(program)
    hypotheses = bounded_beam_rollout(
        model,
        _native_batch(),
        config=_rollout(max_level=3, root_types=(1,)),
    )
    assert max(model.calls) == 3
    assert any(
        hypothesis.result.stop_reason == "configured_root_reconstructed"
        and len(hypothesis.result.steps) == 3
        for hypothesis in hypotheses
    )


def test_real_model_beam_updates_hard_leaf_pid_and_preserves_recursive_p4_closure():
    torch.manual_seed(23)
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=1,
        n_heads=4,
        n_context_layers=1,
    ).eval()
    # Hold discrete head preferences fixed while exercising the real encoder,
    # two-pass PID state, decoder, composite construction and later contexts.
    with torch.no_grad():
        for head in (
            model.decoder.object_head,
            model.decoder.type_head,
            model.decoder.cardinality_head,
            model.leaf_pid_head,
        ):
            head.weight.zero_()
        model.decoder.object_head.bias.fill_(12.0)
        model.decoder.type_head.bias.fill_(-20.0)
        model.decoder.type_head.bias[4] = 12.0
        model.decoder.cardinality_head.bias.fill_(-20.0)
        model.decoder.cardinality_head.bias[2] = 12.0
        for head in (model.decoder.pointer_query, model.decoder.pointer_key):
            head.weight.zero_()
            head.bias.zero_()
        model.leaf_pid_head.bias.zero_()
        model.leaf_pid_head.bias[8] = 12.0  # charge-compatible K+
        model.leaf_pid_head.bias[26] = 12.0  # charge-compatible pi-
    batch = _fsp_batch()
    batch["leaf_kinematics_mode_ids"].fill_(LEAF_MODE_TO_ID["raw_track_predicted_pid"])
    batch["pid_labels"].zero_()
    input_energy = batch["p4"][0, :, 3].clone()
    result = full_depth_beam_rollout(
        model,
        batch,
        config=_rollout(max_level=3),
        beam_config=_beam(beam_width=2, max_type_options=1),
    )
    deepest = max(
        result.candidates, key=lambda candidate: int(candidate.batch["level_ids"].max())
    )
    assert int(deepest.batch["level_ids"].max()) == 3
    assert not torch.equal(deepest.batch["p4"][0, :4, 3], input_energy)
    assert all(
        step.model_output.leaf_pid_logits is not None for step in deepest.result.steps
    )
    assert not model.training
    for candidate in result.candidates:
        assert math.isfinite(candidate.score)
        adjacency = candidate.batch["daughter_adjacency"][0]
        p4 = candidate.batch["p4"][0]
        for mother in adjacency.any(dim=-1).nonzero().flatten().tolist():
            torch.testing.assert_close(p4[mother], p4[adjacency[mother]].sum(dim=0))


def test_query_candidate_proposal_and_global_beam_limits_are_enforced():
    model = _ScriptedModel(
        lambda _level, _batch: [
            _query(
                types={4: 1.0, 3: 0.8, 5: 0.6},
                pointers={0: 3.0, 1: 3.0, 2: 3.0, 3: 3.0},
            )
            for _ in range(5)
        ]
    )
    config = _beam(
        beam_width=2,
        max_candidates_per_query=2,
        max_proposals_per_level=3,
        max_candidate_expansions_per_query=8,
    )
    result = full_depth_beam_rollout(
        model, _fsp_batch(), config=_rollout(), beam_config=config
    )
    assert 1 <= len(result.candidates) <= 2
    assert len(model.calls) == 1
    for item in result.candidates:
        for step in item.result.steps:
            assert len(step.proposals) <= 3
            for query in {proposal.query_id for proposal in step.proposals}:
                assert (
                    sum(proposal.query_id == query for proposal in step.proposals) <= 2
                )
    assert result.diagnostics["max_live_states"] <= config.beam_width
    assert (
        result.diagnostics["max_proposals_retained"] <= config.max_proposals_per_level
    )
    assert (
        result.diagnostics["daughter_sets_considered"]
        <= 5 * config.max_candidate_expansions_per_query
    )
    assert result.diagnostics["candidates_pruned_per_query"] > 0
    assert result.diagnostics["candidates_pruned_per_level"] > 0
    assert all(
        row["live_states"] <= config.beam_width
        and row["finished_states"] <= config.beam_width
        for row in result.diagnostics["per_level"]
    )
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize("beam_width", [1, 8])
def test_recursive_sources_are_exclusive_within_and_across_levels(beam_width):
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][
        0, 0
    ]
    sources = batch["recursive_leaf_source_mask"][0]
    batch["source_conflict_matrix"] = (
        (sources.float() @ sources.float().T > 0) & ~torch.eye(4, dtype=torch.bool)
    ).unsqueeze(0)

    def program(level, _batch):
        if level == 1:
            return [_query((0, 2)), _query((0, 1))]
        return [_query((1, 3))]

    result = full_depth_beam_rollout(
        _ScriptedModel(program),
        batch,
        config=_rollout(max_level=2),
        beam_config=_beam(
            beam_width=beam_width, max_type_options=1, max_daughter_options=1
        ),
    )
    assert any(item.accepted_by_level[0] for item in result.candidates)
    for item in result.candidates:
        adjacency = item.batch["daughter_adjacency"][0]
        sources = item.batch["recursive_leaf_source_mask"][0]
        for mother in adjacency.any(dim=-1).nonzero().flatten().tolist():
            assert not (sources[adjacency[mother]].sum(dim=0) > 1).any()
        composite_roots = adjacency.any(dim=-1) & ~adjacency.any(dim=0)
        assert not (sources[composite_roots].sum(dim=0) > 1).any()


@pytest.mark.parametrize("beam_width", [1, 8])
def test_beam_rejects_policy_that_disables_recursive_source_exclusivity(beam_width):
    policy = ReconstructionConstraintPolicy(reject_recursive_source_conflicts=False)
    model = _ScriptedModel(lambda _level, _batch: [_query()])
    with pytest.raises(ValueError, match="source"):
        full_depth_beam_rollout(
            model,
            _fsp_batch(),
            config=_rollout(constraint_policy=policy),
            beam_config=_beam(beam_width=beam_width),
        )
    assert not model.calls


def test_alternative_types_and_daughters_respect_charge_and_physical_constraints():
    batch = _fsp_batch()
    batch["p4"][0, :, :3] = 0
    batch["p4"][0, :, 3] = torch.tensor([0.1, 0.2, 2.0, 3.0])
    policy = ReconstructionConstraintPolicy(
        empirical_type_prior_mode="off",
        loose_physical_constraints=(("maximum_mother_mass", 1.0),),
    )
    model = _ScriptedModel(
        lambda _level, _batch: [
            _query(types={22: 1.0, 4: 0.8}, pointers={0: 4.0, 1: 3.0, 2: 2.5}),
        ]
    )
    result = full_depth_beam_rollout(
        model,
        batch,
        config=_rollout(constraint_policy=policy),
        beam_config=_beam(),
    )
    accepted = [
        proposal
        for item in result.candidates
        for level in item.accepted_by_level
        for proposal in level
    ]
    assert accepted, "valid lower-ranked neutral mother must be generated"
    assert 4 in {proposal.mother_type for proposal in accepted}
    assert all(proposal.mother_type != 22 for proposal in accepted)
    assert all(proposal.daughter_positions == (0, 1) for proposal in accepted)


def test_width_one_preserves_reference_greedy_tree_and_stop_contract():
    model = _ScriptedModel(
        lambda level, _batch: (
            [_query(types={4: 1.0, 3: 0.8}), _query((2, 3))] if level == 1 else []
        )
    )
    config = _rollout(max_level=3, continue_through_empty_levels=True)
    batch = _fsp_batch()
    greedy = evaluation_reference_rollout(model, batch, config=config)
    beam = full_depth_beam_rollout(
        model,
        batch,
        config=config,
        beam_config=_beam(beam_width=1),
    )
    assert len(beam.candidates) == 1
    candidate = beam.candidates[0]
    assert _tree_key(candidate.batch) == _tree_key(greedy.batch)
    assert candidate.result.stop_reason == greedy.stop_reason
    assert candidate.result.empty_level_count == greedy.empty_level_count
    assert [
        [(p.query_id, p.mother_type, p.daughter_positions) for p in step.accepted]
        for step in candidate.result.steps
    ] == [
        [(p.query_id, p.mother_type, p.daughter_positions) for p in step.accepted]
        for step in greedy.steps
    ]


def test_truth_payload_changes_cannot_influence_projected_beam():
    original = _native_batch()
    changed = {key: value.clone() for key, value in original.items()}
    for key in ("pid_target_labels", "truth_pid_labels", "truth_pid_available"):
        if key in changed:
            changed[key].zero_()
    changed["parent_ids"].fill_(-1)
    changed["daughter_adjacency"].zero_()
    changed["b_side"].fill_(1)
    mothers = changed["level_ids"] > 0
    changed["pid_labels"][mothers] = 1
    model = _ScriptedModel(lambda _level, _batch: [_query(types={4: 1.0, 3: 0.8})])
    results = [
        full_depth_beam_rollout(
            model,
            project_schema_v4_fsps(source).batch,
            config=_rollout(),
            beam_config=_beam(),
        )
        for source in (original, changed)
    ]
    assert [(_tree_key(item.batch), item.score) for item in results[0].candidates] == [
        (_tree_key(item.batch), item.score) for item in results[1].candidates
    ]
    assert all(
        not {
            "pid_target_labels",
            "truth_pid_labels",
            "truth_pid_available",
            "evaluation_leaf_source_keys",
        }
        & keys
        for keys in model.inputs
    )


@pytest.mark.parametrize("continue_empty", [False, True])
def test_empty_generation_uses_the_explicit_rollout_stop_policy(continue_empty):
    model = _ScriptedModel(
        lambda level, _batch: [_query(types={1: 12.0})] if level == 2 else []
    )
    result = full_depth_beam_rollout(
        model,
        _fsp_batch(),
        config=_rollout(
            max_level=3, root_types=(1,), continue_through_empty_levels=continue_empty
        ),
        beam_config=_beam(max_type_options=1),
    )
    best = result.candidates[0].result
    if continue_empty:
        assert best.stop_reason == "configured_root_reconstructed"
        assert [step.target_level for step in best.steps] == [1, 2]
        assert best.empty_level_count == 1
    else:
        assert best.stop_reason == "all_no_object"
        assert model.calls == [1]


@pytest.mark.parametrize("name", ["p4", "charge"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_detector_state_is_rejected_before_search(name, value):
    batch = _fsp_batch()
    batch[name].flatten()[0] = value
    model = _ScriptedModel(lambda _level, _batch: [_query()])
    with pytest.raises(ValueError, match="finite"):
        full_depth_beam_rollout(model, batch, config=_rollout(), beam_config=_beam())
    assert not model.calls


@pytest.mark.parametrize(
    "name",
    [
        "object_logits",
        "type_logits",
        "pointer_logits",
        "cardinality_logits",
        "confidence_logits",
        "leaf_pid_logits",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf])
@pytest.mark.parametrize("width", [1, 2])
def test_nonfinite_model_scores_fail_instead_of_ranking_arbitrarily(name, value, width):
    class InvalidOutputModel(_ScriptedModel):
        def __call__(self, batch, **kwargs):
            output = super().__call__(batch, **kwargs)
            if name == "leaf_pid_logits":
                return replace(
                    output,
                    leaf_pid_logits=torch.full(
                        (*batch["node_mask"].shape, len(PDG_TOKENS)),
                        value,
                    ),
                )
            getattr(output.pointer, name).flatten()[0] = value
            return output

    with pytest.raises(ValueError, match="finite"):
        full_depth_beam_rollout(
            InvalidOutputModel(lambda _level, _batch: [_query()]),
            _fsp_batch(),
            config=_rollout(),
            beam_config=_beam(beam_width=width),
        )


@pytest.mark.parametrize(
    "name",
    [
        "beam_width",
        "max_candidates_per_query",
        "max_proposals_per_level",
        "max_daughter_options",
        "max_type_options",
        "max_cardinality_options",
        "max_candidate_expansions_per_query",
        "max_nodes_per_hypothesis",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_beam_count_limits_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        BeamSearchConfig(**{name: value})


@pytest.mark.parametrize("name", ["score_length_normalization", "empty_level_penalty"])
@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_beam_score_options_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        BeamSearchConfig(**{name: value})
