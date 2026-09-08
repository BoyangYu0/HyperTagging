"""Full-depth, truth-free bounded candidate and coherent forest search.

The network is unchanged. Candidate construction, PID updates and hard physics
policy are shared with level_rollout; only discrete decoding/search changes.
Use hierarchical_inference.reconstruct_beam_from_fsps for stored event batches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import heapq
import math
from itertools import count

import torch

from hypertagging.data.heterogeneous import TRUTH_SUPERVISION_SOURCE_TO_ID
from hypertagging.reconstruction.bounded_search import (
    bounded_conflict_free_sets,
    stable_bounded_unique,
)
from hypertagging.reconstruction.level_rollout import (
    BeamRolloutHypothesis,
    CompositeProposal,
    LevelRolloutResult,
    RolloutConfig,
    RolloutStep,
    _constrained_rollout_model_batch,
    _resolved_rollout_constraint_policy,
    _select_nodes,
    _with_predicted_leaf_p4,
    append_composite_proposals,
    evaluation_reference_rollout,
)

BEAM_SEARCH_POLICY_VERSION = "full-depth-query-log-score-v2"

_MODEL_INPUT_KEYS = frozenset(
    {
        "common_features",
        "common_availability",
        "track_features",
        "track_availability",
        "cluster_features",
        "cluster_availability",
        "klm_features",
        "klm_availability",
        "composite_features",
        "composite_availability",
        "daughter_input_pid_histogram",
        "daughter_input_pid_histogram_available",
        "daughter_pid_histogram",
        "daughter_pid_histogram_available",
        "node_kind_ids",
        "leaf_kinematics_mode_ids",
        "pid_labels",
        "level_ids",
        "p4",
        "charge",
        "parent_ids",
        "daughter_adjacency",
        "node_mask",
        "active",
        "copied",
        "node_ids",
        "reco_ids",
        "source_node_ids",
        "recursive_leaf_source_mask",
        "source_conflict_matrix",
        "copied_from",
        "runtime_composite_type_source_ids",
        "allowed_type_mask",
        "type_logit_bias",
        "pointer_validity_mask",
        "model_input_source_ids",
        "daughter_input_pid_source_ids",
        # The real model requires explicit proof that truth supervision is
        # unavailable; these two arrays are validated as constant sentinels.
        "truth_supervision_source_ids",
        "daughter_truth_pid_source_ids",
        "current_pid_probabilities",
        "current_pid_tokens",
        "current_pid_available",
        "runtime_features_are_raw",
    }
)


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 4
    max_candidates_per_query: int = 4
    max_proposals_per_level: int = 12
    max_daughter_options: int = 4
    max_type_options: int = 3
    max_cardinality_options: int = 2
    max_candidate_expansions_per_query: int = 128
    max_nodes_per_hypothesis: int = 256
    score_length_normalization: float = 1.0
    empty_level_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name.startswith("max_") or name == "beam_width":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
        if (
            not math.isfinite(self.score_length_normalization)
            or not 0 <= self.score_length_normalization <= 1
        ):
            raise ValueError("score_length_normalization must be finite and in [0,1]")
        if not math.isfinite(self.empty_level_penalty) or self.empty_level_penalty < 0:
            raise ValueError("empty_level_penalty must be finite and nonnegative")


@dataclass(frozen=True)
class BeamSearchResult:
    candidates: tuple[BeamRolloutHypothesis, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _Candidate:
    proposal: CompositeProposal
    log_score: float
    sources: frozenset[int]


def _log_probability(value: float) -> float:
    return math.log(min(1.0, max(1e-12, value)))


def _proposal_key(proposal: CompositeProposal) -> tuple:
    return proposal.mother_type, proposal.daughter_positions


def _proposal_state_key(proposal: CompositeProposal) -> tuple:
    return (*_proposal_key(proposal), float(proposal.confidence))


def _canonical_forest_key(
    batch: dict[str, torch.Tensor], *, include_model_state: bool
) -> tuple:
    """Recursive topology identity independent of query/allocated mother IDs.

    Leaf runtime IDs distinguish detector candidates sharing source aliases.
    Leaf PID and p4 distinguish physically different runtime interpretations.
    Composite feature state is part of the identity because proposal confidence
    and daughter summaries are consumed by later model calls. Only hypotheses
    with identical future model inputs may therefore be deduplicated.
    """
    adjacency = batch["daughter_adjacency"][0]
    tokens = batch.get("current_pid_tokens", batch["pid_labels"])[0]
    keys: dict[int, tuple] = {}
    active = batch["node_mask"][0].nonzero(as_tuple=False).flatten().tolist()
    # Construction requires all daughters to pre-exist their mother.
    for position in active:
        daughters = adjacency[position].nonzero(as_tuple=False).flatten().tolist()
        if daughters:
            current_probabilities = batch.get("current_pid_probabilities")
            model_state = (
                tuple(float(value) for value in batch["common_features"][0, position]),
                tuple(
                    bool(value) for value in batch["common_availability"][0, position]
                ),
                tuple(
                    float(value) for value in batch["composite_features"][0, position]
                ),
                tuple(
                    bool(value)
                    for value in batch["composite_availability"][0, position]
                ),
                tuple(
                    float(value)
                    for value in batch["daughter_input_pid_histogram"][0, position]
                ),
                bool(batch["daughter_input_pid_histogram_available"][0, position]),
                tuple(
                    float(value)
                    for value in (
                        current_probabilities[0, position]
                        if current_probabilities is not None
                        else ()
                    )
                ),
            )
            keys[position] = (
                1,
                int(batch["level_ids"][0, position]),
                int(tokens[position]),
                model_state if include_model_state else (),
                tuple(sorted(keys[d] for d in daughters)),
            )
        else:
            keys[position] = (
                0,
                int(batch["node_ids"][0, position]),
                int(tokens[position]),
                tuple(float(x) for x in batch["p4"][0, position]),
            )
    return tuple(sorted(keys[p] for p in active if int(batch["parent_ids"][0, p]) < 0))


def canonical_forest_key(batch: dict[str, torch.Tensor]) -> tuple:
    """Identity of a live forest, including all future-visible dynamic state."""

    return _canonical_forest_key(batch, include_model_state=True)


def _terminal_forest_key(batch: dict[str, torch.Tensor]) -> tuple:
    """Physics/topology identity after a hypothesis will receive no model call."""

    return _canonical_forest_key(batch, include_model_state=False)


def _source_sets(state: dict[str, torch.Tensor]) -> list[frozenset[int]]:
    return [
        frozenset(row.nonzero(as_tuple=False).flatten().tolist())
        for row in state["recursive_leaf_source_mask"][0]
    ]


def _daughter_combinations(
    positions: list[int],
    probabilities: torch.Tensor,
    cardinality: int,
    *,
    source_sets: list[frozenset[int]],
    budget: int,
) -> tuple[list[tuple[int, ...]], int, bool, int]:
    """Enumerate high-mean-pointer, source-disjoint subsets with bounded A*.

    For a fixed cardinality, mean pointer probability and summed pointer
    probability have identical ordering. The optimistic bound ignores conflicts
    among future nodes but never enqueues a node conflicting with the selected
    prefix, so invalid complete subsets cannot consume the output capacity.
    ``budget`` limits popped partial states, including unsuccessful branches.
    """

    if cardinality < 1 or cardinality > len(positions) or budget < 1:
        return [], 0, False, 0
    values = [float(probabilities[position]) for position in positions]

    def optimistic_bound(
        cursor: int, selected_score: float, selected_count: int, used: frozenset[int]
    ) -> float | None:
        needed = cardinality - selected_count
        compatible = [
            values[index]
            for index in range(cursor, len(positions))
            if source_sets[positions[index]]
            and not (used & source_sets[positions[index]])
        ]
        if len(compatible) < needed:
            return None
        return selected_score + sum(sorted(compatible, reverse=True)[:needed])

    initial_bound = optimistic_bound(0, 0.0, 0, frozenset())
    if initial_bound is None:
        return [], 0, False, 0
    serial = count()
    heap: list[
        tuple[
            float,
            int,
            int,
            tuple[int, ...],
            frozenset[int],
            float,
        ]
    ] = [(-initial_bound, next(serial), 0, (), frozenset(), 0.0)]
    complete: list[tuple[int, ...]] = []
    expanded = 0
    source_rejections = 0
    while heap and expanded < budget:
        _negative_bound, _serial, cursor, selected, used, selected_score = (
            heapq.heappop(heap)
        )
        expanded += 1
        if len(selected) == cardinality:
            complete.append(tuple(positions[index] for index in selected))
            continue
        if cursor >= len(positions):
            continue

        source = source_sets[positions[cursor]]
        if source and not (used & source):
            include_selected = selected + (cursor,)
            include_score = selected_score + values[cursor]
            include_used = used | source
            include_bound = optimistic_bound(
                cursor + 1, include_score, len(include_selected), include_used
            )
            if include_bound is not None:
                heapq.heappush(
                    heap,
                    (
                        -include_bound,
                        next(serial),
                        cursor + 1,
                        include_selected,
                        include_used,
                        include_score,
                    ),
                )
        else:
            source_rejections += 1
        skip_bound = optimistic_bound(cursor + 1, selected_score, len(selected), used)
        if skip_bound is not None:
            heapq.heappush(
                heap,
                (
                    -skip_bound,
                    next(serial),
                    cursor + 1,
                    selected,
                    used,
                    selected_score,
                ),
            )
    return complete, expanded, bool(heap), source_rejections


def _decode_candidates(output, state, config, search, diagnostics, *, max_mothers: int):
    policy = _resolved_rollout_constraint_policy(config)
    sources = _source_sets(state)
    allowed, _ = policy.type_constraints(output.target_level, device=state["p4"].device)
    eligible = (
        output.context_mask[0]
        & policy.pointer_validity_mask(state, output.target_level)[0]
        & (state["parent_ids"][0] < 0)
    )
    committed = [
        p
        for p in (state["node_mask"][0] & (state["parent_ids"][0] < 0))
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
        if bool(state["daughter_adjacency"][0, p].any())
    ]
    object_probabilities = torch.sigmoid(output.pointer.object_logits[0])
    candidates = []
    for query in range(len(object_probabilities)):
        object_score = float(object_probabilities[query])
        if object_score < config.object_threshold:
            continue
        probabilities = torch.sigmoid(output.pointer.pointer_logits[0, query])
        positions = []
        for position in eligible.nonzero(as_tuple=False).flatten().tolist():
            if float(probabilities[position]) < config.pointer_threshold:
                continue
            # Once a composite is committed as a forest root, selecting one of
            # its detector aliases directly can never be source-exclusive. Drop
            # such positions before bounded subset enumeration.
            if any(
                position != prior and sources[position] & sources[prior]
                for prior in committed
            ):
                diagnostics["source_conflicts_rejected"] += 1
                continue
            positions.append(position)
        positions.sort(key=lambda p: (-float(probabilities[p]), p))
        type_probs = torch.softmax(output.pointer.type_logits[0, query], dim=-1)
        ranked_types = sorted(
            allowed.nonzero(as_tuple=False).flatten().tolist(),
            key=lambda t: (-float(type_probs[t]), t),
        )
        cardinality_probs = torch.softmax(
            output.pointer.cardinality_logits[0, query], dim=-1
        )
        predicted_cardinality = (
            config.use_cardinality and policy.daughter_cardinality_policy == "predicted"
        )
        if predicted_cardinality:
            predicted_counts = sorted(
                range(len(cardinality_probs)),
                key=lambda k: (-float(cardinality_probs[k]), k),
            )
        else:
            # Threshold decoding retains its maximal source-disjoint size.
            chosen_sources: set[int] = set()
            count = 0
            for p in positions:
                if not sources[p] & chosen_sources:
                    chosen_sources.update(sources[p])
                    count += 1
            predicted_counts = [count]
        per_query: list[_Candidate] = []
        expansions = 0
        valid_cardinality_options = 0
        expansion_limit_hit = False
        for predicted_count in predicted_counts:
            if predicted_count < policy.minimum_daughters:
                continue
            if policy.cardinality_insufficient_policy == "reduce":
                actual_counts = range(
                    min(predicted_count, len(positions)),
                    policy.minimum_daughters - 1,
                    -1,
                )
            else:
                actual_counts = (predicted_count,)
            cardinality_produced = False
            for cardinality in actual_counts:
                if cardinality > len(positions):
                    continue
                remaining_budget = (
                    search.max_candidate_expansions_per_query - expansions
                )
                if remaining_budget <= 0:
                    expansion_limit_hit = True
                    break
                (
                    daughter_sets,
                    used_expansions,
                    exhausted,
                    source_rejections,
                ) = _daughter_combinations(
                    positions,
                    probabilities,
                    cardinality,
                    source_sets=sources,
                    budget=remaining_budget,
                )
                expansions += used_expansions
                diagnostics["daughter_partial_states_expanded"] += used_expansions
                diagnostics["source_conflicts_rejected"] += source_rejections
                expansion_limit_hit |= exhausted
                valid_daughter_options = 0
                for daughters in daughter_sets:
                    diagnostics["daughter_sets_considered"] += 1
                    used = frozenset().union(*(sources[p] for p in daughters))
                    p4 = state["p4"][0, list(daughters)].sum(dim=0)
                    if not bool(
                        torch.isfinite(p4).all()
                    ) or not policy.rollout_physical_valid(p4):
                        diagnostics["physics_rejected"] += 1
                        continue
                    charge = float(state["charge"][0, list(daughters)].sum())
                    produced_for_daughters = False
                    valid_type_options = 0
                    for mother_type in ranked_types:
                        type_probability = float(type_probs[mother_type])
                        if (
                            config.type_probability_threshold is not None
                            and type_probability < config.type_probability_threshold
                        ):
                            break
                        expected = dict(config.mother_charge_by_token).get(
                            mother_type, policy.expected_charge(mother_type)
                        )
                        if (
                            policy.mother_charge_compatibility
                            in {"hard", "soft_train_hard_rollout"}
                            and abs(charge - expected) > policy.mother_charge_tolerance
                        ):
                            diagnostics["charge_rejected"] += 1
                            continue
                        pointer_quality = float(probabilities[list(daughters)].mean())
                        confidence = (
                            float(
                                torch.sigmoid(
                                    output.pointer.confidence_logits[0, query]
                                )
                            )
                            if config.use_learned_confidence
                            else object_score * type_probability * pointer_quality
                        )
                        if confidence < config.confidence_threshold:
                            continue
                        proposal = CompositeProposal(
                            query,
                            mother_type,
                            daughters,
                            object_score,
                            confidence,
                        )
                        # Keep existing confidence feature semantics. Cardinality
                        # likelihood belongs to search, not composite features.
                        likelihood = confidence * (
                            float(cardinality_probs[predicted_count])
                            if predicted_cardinality
                            else 1.0
                        )
                        if config.use_learned_confidence:
                            likelihood *= (
                                object_score * type_probability * pointer_quality
                            )
                        per_query.append(
                            _Candidate(proposal, _log_probability(likelihood), used)
                        )
                        diagnostics["candidates_generated"] += 1
                        produced_for_daughters = True
                        valid_type_options += 1
                        if valid_type_options >= search.max_type_options:
                            break
                    if produced_for_daughters:
                        cardinality_produced = True
                        valid_daughter_options += 1
                        if valid_daughter_options >= search.max_daughter_options:
                            break
                if cardinality_produced:
                    # Under the reduce policy, retain the highest feasible
                    # actual size for this predicted option and then backfill
                    # lower-ranked predicted options.
                    break
                if expansion_limit_hit:
                    break
            if cardinality_produced:
                valid_cardinality_options += 1
                if valid_cardinality_options >= search.max_cardinality_options:
                    break
            if expansion_limit_hit:
                break
        if expansion_limit_hit:
            diagnostics["query_expansion_limit_hits"] += 1
        # Deterministic ordering does not depend on GPU topk tie behavior.
        per_query, duplicate, pruned = stable_bounded_unique(
            per_query,
            key=lambda c: _proposal_key(c.proposal),
            order=lambda c: (
                c.proposal.mother_type not in config.root_types,
                -c.log_score,
                _proposal_key(c.proposal),
            ),
            limit=search.max_candidates_per_query,
        )
        diagnostics["candidates_deduplicated"] += duplicate
        diagnostics["candidates_pruned_per_query"] += pruned
        candidates.extend(per_query)
    omitted = [_log_probability(1 - float(p)) for p in object_probabilities]
    candidates.sort(
        key=lambda c: (
            c.proposal.mother_type not in config.root_types,
            -(c.log_score - omitted[c.proposal.query_id]),
            _proposal_key(c.proposal),
            c.proposal.query_id,
        )
    )
    diagnostics["candidates_pruned_per_level"] += max(
        0, len(candidates) - search.max_proposals_per_level
    )
    candidates = candidates[: search.max_proposals_per_level]
    diagnostics["max_proposals_retained"] = max(
        diagnostics["max_proposals_retained"], len(candidates)
    )
    groups = [
        [c for c in candidates if c.proposal.query_id == q]
        for q in range(len(object_probabilities))
    ]
    sets, counts = bounded_conflict_free_sets(
        groups,
        sources=lambda c: c.sources,
        identity=lambda c: _proposal_state_key(c.proposal),
        chosen_score=lambda c: c.log_score,
        omitted_scores=omitted,
        width=search.beam_width,
        singleton=lambda c: c.proposal.mother_type in config.root_types,
        max_items=max_mothers,
    )
    for key, value in counts.items():
        diagnostics[key] += value
    return candidates, sets, len(object_probabilities)


def _score(total: float, decisions: int, search: BeamSearchConfig) -> float:
    return total / max(decisions, 1) ** search.score_length_normalization


def _order(hypothesis: BeamRolloutHypothesis) -> tuple:
    completed = hypothesis.result.stop_reason == "configured_root_reconstructed"
    return (not completed, -hypothesis.score, canonical_forest_key(hypothesis.batch))


def _truth_free_model_view(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expose only the documented reconstructed-data surface to a model."""

    return {name: value for name, value in batch.items() if name in _MODEL_INPUT_KEYS}


def _validate_inputs(state, config, search):
    if state["node_mask"].shape[0] != 1:
        raise ValueError("full-depth beam requires batch size one")
    if (
        isinstance(config.max_level, bool)
        or not isinstance(config.max_level, int)
        or config.max_level < 1
    ):
        raise ValueError("max_level must be a positive integer")
    if not config.exclusive_final or config.exclusive_resolution != "greedy":
        raise ValueError(
            "beam requires exclusive_final=True and the greedy base policy"
        )
    if not _resolved_rollout_constraint_policy(
        config
    ).reject_recursive_source_conflicts:
        raise ValueError("beam requires reject_recursive_source_conflicts=True")
    if config.rollout_pid_kinematics_mode not in {
        "soft_decision_hard_construction",
        "hard",
        "temperature_softmax",
        "straight_through_hard",
    }:
        raise ValueError("invalid rollout_pid_kinematics_mode")
    if (
        not math.isfinite(config.rollout_pid_temperature)
        or config.rollout_pid_temperature <= 0
    ):
        raise ValueError("rollout_pid_temperature must be finite and positive")
    for name in (
        "object_threshold",
        "pointer_threshold",
        "confidence_threshold",
        "type_probability_threshold",
    ):
        value = getattr(config, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and in [0,1]")
    if config.use_learned_confidence and not config.confidence_trained:
        raise ValueError("learned confidence requires a trained confidence head")
    active = state["node_mask"].bool()
    if int(active.sum()) > search.max_nodes_per_hypothesis:
        raise ValueError("projected FSP count exceeds max_nodes_per_hypothesis")
    if (
        bool((state["level_ids"][active] != 0).any())
        or bool(state["daughter_adjacency"].any())
        or bool((state["parent_ids"][active] >= 0).any())
    ):
        raise ValueError(
            "beam input must be projected FSPs; use reconstruct_beam_from_fsps"
        )
    if "evaluation_leaf_source_keys" in state or any(
        k in state
        for k in ("pid_target_labels", "truth_pid_labels", "truth_pid_available")
    ):
        raise ValueError(
            "beam input contains evaluation/target metadata; project FSPs first"
        )
    if "b_side" in state and bool((state["b_side"][active] != -1).any()):
        raise ValueError("beam input contains unsanitized B-side truth labels")
    unavailable = TRUTH_SUPERVISION_SOURCE_TO_ID["unavailable"]
    for name in ("truth_supervision_source_ids", "daughter_truth_pid_source_ids"):
        if name in state and bool((state[name][active] != unavailable).any()):
            raise ValueError("beam input contains available truth provenance")
    if "daughter_truth_pid_histogram_available" in state and bool(
        state["daughter_truth_pid_histogram_available"].any()
    ):
        raise ValueError("beam input contains available truth daughter PID state")
    if "daughter_truth_pid_histogram" in state and bool(
        state["daughter_truth_pid_histogram"].count_nonzero()
    ):
        raise ValueError("beam input contains truth daughter PID values")
    for name in (
        "complete_truth_decay",
        "complete_reconstructable_decay",
        "recursive_reconstructable_complete",
        "partial_missing_daughters",
        "contracted_intermediate",
        "valid_reconstruction_target",
    ):
        if name in state and bool(state[name][active].any()):
            raise ValueError(f"beam input contains unsanitized target field: {name}")
    for name in (
        "full_truth_daughter_count",
        "retained_truth_daughter_count_expected",
        "retained_daughter_count",
        "truth_root_distance",
        "full_event_max_level",
        "depth_from_retained_root",
        "distance_to_nearest_retained_root",
    ):
        if name in state and bool((state[name][active] != -1).any()):
            raise ValueError(f"beam input contains unsanitized target field: {name}")
    for name, sentinel in (
        ("ancestor_descendant_relation", False),
        ("lca_node_id", -1),
        ("edges_to_lca_from_i", -1),
        ("edges_to_lca_from_j", -1),
        ("exact_tree_path_distance", -1),
        ("lca_depth", -1),
    ):
        if name in state and bool((state[name] != sentinel).any()):
            raise ValueError(f"beam input contains unsanitized truth geometry: {name}")
    if "recursive_leaf_source_mask" not in state:
        raise ValueError("beam requires recursive leaf-source masks")
    for name in ("p4", "charge"):
        if not bool(torch.isfinite(state[name][active]).all()):
            raise ValueError(f"beam FSP {name} must be finite")
    if not bool(state["recursive_leaf_source_mask"][active].any(dim=-1).all()):
        raise ValueError("beam FSPs require nonempty detector provenance")


def _validate_model_output(output) -> None:
    for name in (
        "object_logits",
        "type_logits",
        "pointer_logits",
        "cardinality_logits",
        "confidence_logits",
    ):
        logits = getattr(output.pointer, name)
        if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
            raise ValueError(f"nonfinite beam model output: {name}")
    for name in ("type_logits", "cardinality_logits"):
        if not bool(
            torch.isfinite(torch.softmax(getattr(output.pointer, name), -1)).all()
        ):
            raise ValueError(f"beam model has no finite probability: {name}")
    if output.leaf_pid_logits is not None and not bool(
        torch.isfinite(output.leaf_pid_logits).all()
    ):
        raise ValueError("nonfinite beam model output: leaf_pid_logits")


@torch.inference_mode()
def full_depth_beam_rollout(
    model,
    full_batch: dict[str, torch.Tensor],
    *,
    config: RolloutConfig | None = None,
    beam_config: BeamSearchConfig | None = None,
) -> BeamSearchResult:
    """Search all configured levels from an already truth-scrubbed FSP batch.

    Scores sum chosen-query log likelihood and unused-query log no-object
    probability, normalized by cumulative query decisions (exponent alpha).
    Completed roots rank before unfinished forests; oracle selection is never
    performed here. Width one delegates to a source-safe greedy reference.
    """
    config = config or RolloutConfig()
    search = beam_config or BeamSearchConfig()
    _validate_inputs(full_batch, config, search)
    if isinstance(model, torch.nn.Module) and model.training:
        raise ValueError("beam requires a separate model in eval mode")
    state = _select_nodes(full_batch, full_batch["node_mask"][0])
    counters = (
        "model_calls",
        "daughter_sets_considered",
        "daughter_partial_states_expanded",
        "source_conflicts_rejected",
        "physics_rejected",
        "charge_rejected",
        "candidates_generated",
        "candidates_deduplicated",
        "candidates_pruned_per_query",
        "candidates_pruned_per_level",
        "query_expansion_limit_hits",
        "proposal_sets_expanded",
        "proposal_sets_pruned",
        "proposal_sets_deduplicated",
        "states_expanded",
        "states_deduplicated",
        "states_pruned",
        "node_limit_hits",
        "states_saturated_at_node_limit",
        "max_live_states",
        "max_proposals_retained",
        "max_nodes_observed",
        "levels_processed",
    )
    diagnostics: dict[str, object] = {key: 0 for key in counters}
    diagnostics["max_live_states"] = 1  # Includes the initial FSP forest.
    diagnostics["max_nodes_observed"] = int(state["node_mask"].sum())
    diagnostics.update(
        policy_version=BEAM_SEARCH_POLICY_VERSION,
        configuration=asdict(search),
        per_level=[],
        proposal_diagnostics_available=search.beam_width > 1,
    )
    if search.beam_width == 1:
        source_alias_masks_applied = 0

        def checked_model(*args, **kwargs):
            nonlocal source_alias_masks_applied
            model_batch = dict(args[0])
            # Legacy greedy can consume a different detector alias at a later
            # generation. Such a tree is outside the source-exclusive overlap
            # of the two policies. Mask the alias before scoring/decoding;
            # valid greedy decisions, ordering and construction stay intact.
            sources = _source_sets(model_batch)
            valid = model_batch["pointer_validity_mask"].clone()
            committed = [
                p
                for p in range(len(sources))
                if bool(model_batch["node_mask"][0, p])
                and int(model_batch["parent_ids"][0, p]) < 0
                and bool(model_batch["daughter_adjacency"][0, p].any())
            ]
            for p in valid[0].nonzero(as_tuple=False).flatten().tolist():
                if any(sources[p] & sources[c] for c in committed if p != c):
                    valid[0, p] = False
            source_alias_masks_applied += int(
                not torch.equal(valid, model_batch["pointer_validity_mask"])
            )
            model_batch["pointer_validity_mask"] = valid
            output = model(_truth_free_model_view(model_batch), **kwargs)
            _validate_model_output(output)
            return replace(output, context_mask=output.context_mask & valid)

        result = evaluation_reference_rollout(
            checked_model,
            state,
            config=config,
            max_nodes=search.max_nodes_per_hypothesis,
        )
        observed_nodes = int(result.batch["node_mask"].sum())
        total = 0.0
        count = decisions = 0
        for step in result.steps:
            objects = torch.sigmoid(step.model_output.pointer.object_logits[0])
            accepted = {p.query_id: p for p in step.accepted}
            for q, probability in enumerate(objects):
                likelihood = 1 - float(probability)
                if q in accepted:
                    proposal = accepted[q]
                    likelihood = proposal.confidence
                    policy = _resolved_rollout_constraint_policy(config)
                    if (
                        config.use_cardinality
                        and policy.daughter_cardinality_policy == "predicted"
                    ):
                        card_probs = torch.softmax(
                            step.model_output.pointer.cardinality_logits[0, q], -1
                        )
                        likelihood *= float(card_probs.max())
                    if config.use_learned_confidence:
                        pointer = step.model_output.pointer
                        likelihood *= (
                            float(probability)
                            * float(
                                torch.softmax(pointer.type_logits[0, q], -1)[
                                    proposal.mother_type
                                ]
                            )
                            * float(
                                torch.sigmoid(
                                    pointer.pointer_logits[
                                        0, q, list(proposal.daughter_positions)
                                    ]
                                ).mean()
                            )
                        )
                total += _log_probability(likelihood)
            decisions += len(objects)
            count += len(accepted)
            if not accepted:
                total -= search.empty_level_penalty
        hypothesis = BeamRolloutHypothesis(
            result.batch,
            _score(total, decisions, search),
            tuple(step.accepted for step in result.steps),
            result,
            total,
            count,
            decisions,
        )
        diagnostics.update(
            model_calls=len(result.steps),
            levels_processed=len(result.steps),
            max_live_states=1,
            max_nodes_observed=observed_nodes,
            node_limit_hits=int(result.stop_reason == "node_limit_reached"),
            states_saturated_at_node_limit=int(
                observed_nodes == search.max_nodes_per_hypothesis
            ),
            returned_candidates=1,
            completed_candidates=int(
                result.stop_reason == "configured_root_reconstructed"
            ),
            width_one_greedy_compatibility=(source_alias_masks_applied == 0),
            width_one_source_safe_greedy=True,
            source_alias_masks_applied=source_alias_masks_applied,
        )
        return BeamSearchResult((hypothesis,), diagnostics)
    policy = _resolved_rollout_constraint_policy(config)
    result = LevelRolloutResult(state, (), "maximum_level", True, False)
    beam = [BeamRolloutHypothesis(state, 0.0, (), result)]
    finished: list[BeamRolloutHypothesis] = []
    forward_mode = (
        "soft_expectation"
        if config.rollout_pid_kinematics_mode == "soft_decision_hard_construction"
        else config.rollout_pid_kinematics_mode
    )
    construction_mode = (
        "hard"
        if config.rollout_pid_kinematics_mode == "soft_decision_hard_construction"
        else forward_mode
    )
    for level in range(1, config.max_level + 1):
        expanded = []
        before = {k: diagnostics[k] for k in counters}
        for hypothesis in beam:
            state = hypothesis.batch
            if not bool(state["node_mask"].any()):
                finished.append(
                    replace(
                        hypothesis,
                        result=replace(hypothesis.result, stop_reason="no_context"),
                    )
                )
                continue
            active_nodes = int(state["node_mask"].sum())
            diagnostics["max_nodes_observed"] = max(
                diagnostics["max_nodes_observed"], active_nodes
            )
            remaining_nodes = search.max_nodes_per_hypothesis - active_nodes
            if remaining_nodes <= 0:
                diagnostics["node_limit_hits"] += 1
                finished.append(
                    replace(
                        hypothesis,
                        result=replace(
                            hypothesis.result, stop_reason="node_limit_reached"
                        ),
                    )
                )
                continue
            model_batch = _constrained_rollout_model_batch(
                state, target_level=level, policy=policy
            )
            output = model(
                _truth_free_model_view(model_batch),
                target_level=level,
                pid_kinematics_mode_override=forward_mode,
                pid_temperature_override=config.rollout_pid_temperature,
            )
            diagnostics["model_calls"] += 1
            _validate_model_output(output)
            if output.leaf_pid_logits is not None:
                state = _with_predicted_leaf_p4(
                    state,
                    output.leaf_pid_logits,
                    mode=construction_mode,
                    temperature=config.rollout_pid_temperature,
                )
            candidates, proposal_sets, decisions = _decode_candidates(
                output,
                state,
                config,
                search,
                diagnostics,
                max_mothers=remaining_nodes,
            )
            for stage_score, chosen in proposal_sets:
                accepted = tuple(
                    sorted((c.proposal for c in chosen), key=lambda p: p.query_id)
                )
                next_state, appended = (
                    append_composite_proposals(
                        state, list(accepted), target_level=level
                    )
                    if accepted
                    else (state, [])
                )
                complete = any(p.mother_type in config.root_types for p in accepted)
                reason = (
                    "configured_root_reconstructed" if complete else "maximum_level"
                )
                if not accepted and not config.continue_through_empty_levels:
                    reason = "beam_selected_empty" if candidates else "all_no_object"
                step = RolloutStep(
                    level,
                    output,
                    tuple(c.proposal for c in candidates),
                    accepted,
                    False,
                    tuple(appended),
                    construction_mode,
                )
                next_result = LevelRolloutResult(
                    next_state,
                    hypothesis.result.steps + (step,),
                    reason,
                    True,
                    False,
                    empty_level_count=hypothesis.result.empty_level_count
                    + int(not accepted),
                )
                total = (
                    hypothesis.log_score_sum
                    + stage_score
                    - (search.empty_level_penalty if not accepted else 0)
                )
                n_decisions = hypothesis.scored_decisions + decisions
                item = BeamRolloutHypothesis(
                    next_state,
                    _score(total, n_decisions, search),
                    hypothesis.accepted_by_level + (accepted,),
                    next_result,
                    total,
                    hypothesis.scored_candidates + len(accepted),
                    n_decisions,
                )
                diagnostics["states_expanded"] += 1
                next_nodes = int(next_state["node_mask"].sum())
                diagnostics["max_nodes_observed"] = max(
                    diagnostics["max_nodes_observed"], next_nodes
                )
                diagnostics["states_saturated_at_node_limit"] += int(
                    next_nodes == search.max_nodes_per_hypothesis
                )
                if complete or reason in {"all_no_object", "beam_selected_empty"}:
                    finished.append(item)
                else:
                    expanded.append(item)
        beam, duplicate, pruned = stable_bounded_unique(
            expanded,
            key=lambda h: canonical_forest_key(h.batch),
            order=_order,
            limit=search.beam_width,
        )
        finished, finished_duplicate, finished_pruned = stable_bounded_unique(
            finished,
            key=lambda h: _terminal_forest_key(h.batch),
            order=_order,
            limit=search.beam_width,
        )
        diagnostics["states_deduplicated"] += duplicate + finished_duplicate
        diagnostics["states_pruned"] += pruned + finished_pruned
        diagnostics["max_live_states"] = max(diagnostics["max_live_states"], len(beam))
        diagnostics["levels_processed"] = level
        diagnostics["per_level"].append(
            {
                "level": level,
                "live_states": len(beam),
                "finished_states": len(finished),
                **{
                    k: diagnostics[k] - before[k]
                    for k in counters
                    if not k.startswith("max_") and k != "levels_processed"
                },
            }
        )
        if not beam:
            break
    retained, duplicate, pruned = stable_bounded_unique(
        finished + beam,
        key=lambda h: _terminal_forest_key(h.batch),
        order=_order,
        limit=search.beam_width,
    )
    diagnostics["states_deduplicated"] += duplicate
    diagnostics["states_pruned"] += pruned
    diagnostics["returned_candidates"] = len(retained)
    diagnostics["completed_candidates"] = sum(
        h.result.stop_reason == "configured_root_reconstructed" for h in retained
    )
    diagnostics["width_one_greedy_compatibility"] = False
    diagnostics["width_one_source_safe_greedy"] = False
    diagnostics["source_alias_masks_applied"] = 0
    return BeamSearchResult(tuple(retained), diagnostics)


__all__ = [
    "BEAM_SEARCH_POLICY_VERSION",
    "BeamSearchConfig",
    "BeamSearchResult",
    "canonical_forest_key",
    "full_depth_beam_rollout",
]
