"""Complete level-by-level teacher-forced and free reconstruction rollout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import torch

from hypertagging.models.heterogeneous import composite_token_from_daughters
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    LevelReconstructionOutput,
    _upgrade_flat_batch,
)
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.kinematics import (
    hard_reconstructed_p4_from_leaf_pid,
)
from hypertagging.reconstruction.pid_state import (
    COMPOSITE_TYPE_SOURCE_TO_ID,
    rebuild_runtime_pid_state,
)
from hypertagging.models.mother_pointer import constrained_daughter_decode
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy


@dataclass(frozen=True)
class RolloutConfig:
    max_level: int = 8
    object_threshold: float = 0.5
    pointer_threshold: float = 0.5
    confidence_threshold: float = 0.0
    min_daughters: int = 2
    root_types: tuple[int, ...] = (1,)  # reduced Upsilon(4S) token by default
    use_cardinality: bool = True
    allow_competing: bool = True
    exclusive_final: bool = True
    scheduled_sampling_probability: float = 0.0
    seed: int = 17
    use_learned_confidence: bool = False
    confidence_trained: bool = False
    cardinality_insufficient_policy: str = "invalid"
    mother_charge_by_token: tuple[tuple[int, float], ...] = ()
    allowed_daughter_node_kinds: tuple[int, ...] = (
        NODE_KIND_TO_ID["track"],
        NODE_KIND_TO_ID["ecl_cluster"],
        NODE_KIND_TO_ID["composite"],
        NODE_KIND_TO_ID["other"],
    )
    constraint_policy: ReconstructionConstraintPolicy | None = None


@dataclass(frozen=True)
class CompositeProposal:
    query_id: int
    mother_type: int
    daughter_positions: tuple[int, ...]
    object_score: float
    confidence: float
    truth_node_id: int | None = None


@dataclass(frozen=True)
class RolloutStep:
    target_level: int
    model_output: LevelReconstructionOutput
    proposals: tuple[CompositeProposal, ...]
    accepted: tuple[CompositeProposal, ...]
    used_teacher_forcing: bool
    appended_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class LevelRolloutResult:
    batch: dict[str, torch.Tensor]
    steps: tuple[RolloutStep, ...]
    stop_reason: str
    valid: bool
    teacher_forced: bool
    cached_states: tuple[tuple[int, dict[str, torch.Tensor]], ...] = ()


def hard_decode_proposals(
    output: LevelReconstructionOutput,
    batch: dict[str, torch.Tensor],
    config: RolloutConfig,
) -> list[CompositeProposal]:
    """Decode unordered query slots; daughter order never enters the result."""

    if output.pointer.object_logits.shape[0] != 1:
        raise ValueError("Tiny rollout currently requires batch size 1")
    policy = config.constraint_policy or ReconstructionConstraintPolicy(
        minimum_pointer_probability=config.pointer_threshold,
        minimum_daughters=config.min_daughters,
        cardinality_insufficient_policy=config.cardinality_insufficient_policy,
        valid_leaf_node_kinds=tuple(
            kind for kind in config.allowed_daughter_node_kinds
            if kind != NODE_KIND_TO_ID["composite"]
        ),
        valid_composite_node_kinds=(NODE_KIND_TO_ID["composite"],),
    )
    context = output.context_mask[0]
    context_positions = context.nonzero(as_tuple=False).flatten()
    proposals: list[CompositeProposal] = []
    for query_id in range(output.pointer.object_logits.shape[1]):
        object_score = float(torch.sigmoid(output.pointer.object_logits[0, query_id]).detach())
        if config.use_learned_confidence and not config.confidence_trained:
            raise RuntimeError(
                "learned confidence was requested for decoding but the checkpoint "
                "does not mark the confidence head as trained"
            )
        learned_confidence = float(
            torch.sigmoid(output.pointer.confidence_logits[0, query_id]).detach()
        )
        if object_score < config.object_threshold:
            continue
        probabilities = torch.sigmoid(
            output.pointer.pointer_logits[0, query_id, context_positions]
        )
        if config.use_cardinality and policy.daughter_cardinality_policy == "predicted":
            cardinality = int(output.pointer.cardinality_logits[0, query_id].argmax())
            if cardinality > context_positions.numel():
                # This is an invalid prediction, not a training-target
                # overflow. Training truth overflows raise explicitly.
                continue
            conflict = (
                batch.get("source_conflict_matrix")
                if policy.reject_recursive_source_conflicts else None
            )
            context_conflict = (
                conflict[0][
                    context_positions[:, None],
                    context_positions[None, :],
                ]
                if conflict is not None
                else torch.zeros(
                    (context_positions.numel(), context_positions.numel()),
                    dtype=torch.bool,
                    device=probabilities.device,
                )
            )
            selected_bool, valid_selection = constrained_daughter_decode(
                probabilities,
                cardinality=cardinality,
                pointer_mask=torch.ones_like(probabilities, dtype=torch.bool),
                source_conflict=context_conflict,
                min_probability=policy.minimum_pointer_probability,
                insufficient_policy=policy.cardinality_insufficient_policy,
            )
            if not valid_selection:
                continue
            selected_local = selected_bool.nonzero(as_tuple=False).flatten()
        else:
            selected_local = (probabilities >= policy.minimum_pointer_probability).nonzero(
                as_tuple=False
            ).flatten()
        daughter_positions = tuple(
            sorted(int(context_positions[index]) for index in selected_local.tolist())
        )
        if len(daughter_positions) < policy.minimum_daughters:
            continue
        policy_valid = policy.pointer_validity_mask(batch, output.target_level)[0]
        if any(not bool(policy_valid[position]) for position in daughter_positions):
            continue
        mother_type = int(output.pointer.type_logits[0, query_id].argmax())
        charge_contract = dict(config.mother_charge_by_token)
        expected_charge = charge_contract.get(mother_type, policy.expected_charge(mother_type))
        if policy.mother_charge_compatibility in {"hard", "soft_train_hard_rollout"}:
            daughter_charge = float(
                batch["charge"][0, list(daughter_positions)].sum()
            )
            if abs(daughter_charge - expected_charge) > policy.mother_charge_tolerance:
                continue
        if policy.loose_physical_constraints and not policy.rollout_physical_valid(
            batch["p4"][0, list(daughter_positions)].sum(dim=0)
        ):
            continue
        pointer_quality = (
            float(probabilities[selected_local].mean().detach())
            if selected_local.numel()
            else 0.0
        )
        type_probability = float(
            torch.softmax(output.pointer.type_logits[0, query_id], dim=-1).max().detach()
        )
        confidence = (
            learned_confidence
            if config.use_learned_confidence
            else object_score * type_probability * pointer_quality
        )
        if confidence < config.confidence_threshold:
            continue
        proposals.append(
            CompositeProposal(
                query_id=query_id,
                mother_type=mother_type,
                daughter_positions=daughter_positions,
                object_score=object_score,
                confidence=confidence,
            )
        )
    return proposals


def resolve_exclusive_proposals(
    proposals: list[CompositeProposal],
    source_node_ids: torch.Tensor | None = None,
    *,
    recursive_leaf_source_mask: torch.Tensor | None = None,
) -> list[CompositeProposal]:
    """Resolve reuse using recursive underlying leaf provenance."""

    accepted: list[CompositeProposal] = []
    used_sources: set[int] = set()
    ranked = sorted(
        proposals,
        key=lambda proposal: (
            -proposal.confidence,
            -proposal.object_score,
            proposal.mother_type,
            proposal.daughter_positions,
            proposal.query_id,
        ),
    )
    for proposal in ranked:
        if recursive_leaf_source_mask is not None:
            sources = set(
                recursive_leaf_source_mask[list(proposal.daughter_positions)]
                .any(dim=0)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
        elif source_node_ids is not None:
            sources = {
                int(source_node_ids[position])
                for position in proposal.daughter_positions
            }
        else:
            raise ValueError("recursive leaf sources or legacy source_node_ids are required")
        if sources & used_sources:
            continue
        accepted.append(proposal)
        used_sources.update(sources)
    return sorted(accepted, key=lambda proposal: proposal.query_id)


def validate_proposals(
    proposals: list[CompositeProposal],
    *,
    existing_node_count: int,
) -> bool:
    """Reject invalid links and cycles by requiring every daughter to pre-exist."""

    return all(
        proposal.daughter_positions
        and len(set(proposal.daughter_positions)) == len(proposal.daughter_positions)
        and all(0 <= daughter < existing_node_count for daughter in proposal.daughter_positions)
        for proposal in proposals
    )


def level_rollout(
    model: LevelAutoregressiveReconstructor,
    full_batch: dict[str, torch.Tensor],
    *,
    mode: Literal["predicted", "teacher_forced", "scheduled"] = "predicted",
    config: RolloutConfig | None = None,
) -> LevelRolloutResult:
    """Re-encode all current nodes, decode, append composites, and stop safely."""

    config = config or RolloutConfig()
    if full_batch["node_mask"].shape[0] != 1:
        raise ValueError("Tiny rollout currently requires batch size 1")
    full_batch = _upgrade_flat_batch(full_batch)
    truth_batch = {key: value for key, value in full_batch.items()}
    state = _select_nodes(full_batch, full_batch["node_mask"][0] & (full_batch["level_ids"][0] == 0))
    generator = torch.Generator(device=state["p4"].device).manual_seed(config.seed)
    seen_states = {_state_fingerprint(state)}
    cached_states: list[tuple[int, dict[str, torch.Tensor]]] = [(0, state)]
    steps: list[RolloutStep] = []
    stop_reason = "maximum_level"
    valid = True

    for target_level in range(1, config.max_level + 1):
        if not state["node_mask"].any():
            stop_reason = "no_context"
            break
        output = model(state, target_level=target_level)
        if output.leaf_pid_logits is not None:
            state = _with_hard_predicted_leaf_p4(state, output.leaf_pid_logits)
        predicted = hard_decode_proposals(output, state, config)
        if not config.allow_competing:
            predicted = resolve_exclusive_proposals(
                predicted,
                state["source_node_ids"][0],
                recursive_leaf_source_mask=state.get("recursive_leaf_source_mask", None)[0]
                if "recursive_leaf_source_mask" in state
                else None,
            )

        use_truth = mode == "teacher_forced"
        if mode == "scheduled":
            use_truth = bool(
                torch.rand((), generator=generator, device=state["p4"].device)
                >= config.scheduled_sampling_probability
            )
        proposals = _truth_proposals(truth_batch, state, target_level) if use_truth else predicted
        if not proposals:
            stop_reason = "all_no_object" if not predicted else "no_valid_new_mother"
            steps.append(
                RolloutStep(
                    target_level,
                    output,
                    tuple(predicted),
                    (),
                    use_truth,
                    (),
                )
            )
            break
        accepted = (
            resolve_exclusive_proposals(
                proposals,
                state["source_node_ids"][0],
                recursive_leaf_source_mask=state.get("recursive_leaf_source_mask", None)[0]
                if "recursive_leaf_source_mask" in state
                else None,
            )
            if config.exclusive_final
            else proposals
        )
        if not accepted:
            stop_reason = "no_valid_new_mother"
            break
        if not validate_proposals(
            accepted,
            existing_node_count=state["p4"].shape[1],
        ):
            stop_reason = "invalid_or_cyclic_state"
            valid = False
            break
        state, appended = append_composite_proposals(
            state,
            accepted,
            target_level=target_level,
        )
        cached_states.append((target_level, state))
        steps.append(
            RolloutStep(
                target_level,
                output,
                tuple(predicted),
                tuple(accepted),
                use_truth,
                tuple(appended),
            )
        )
        fingerprint = _state_fingerprint(state)
        if fingerprint in seen_states:
            stop_reason = "repeated_reconstruction_state"
            valid = False
            break
        seen_states.add(fingerprint)
        if config.root_types and any(
            proposal.mother_type in config.root_types for proposal in accepted
        ):
            stop_reason = "configured_root_reconstructed"
            break
    return LevelRolloutResult(
        batch=state,
        steps=tuple(steps),
        stop_reason=stop_reason,
        valid=valid,
        teacher_forced=mode == "teacher_forced",
        cached_states=tuple(cached_states),
    )


def cached_context_for_level(
    result: LevelRolloutResult, target_level: int
) -> dict[str, torch.Tensor]:
    """Return the cached state after level ``target_level - 1``."""

    desired = max(int(target_level) - 1, 0)
    candidates = [item for item in result.cached_states if item[0] <= desired]
    return candidates[-1][1] if candidates else result.batch


def _with_hard_predicted_leaf_p4(
    batch: dict[str, torch.Tensor],
    pid_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Update raw tracks and existing composites without touching fixed leaves."""

    result = dict(batch)
    runtime = rebuild_runtime_pid_state(batch, pid_logits, hard=True)
    result["p4"] = runtime.p4
    result["current_pid_probabilities"] = torch.nn.functional.one_hot(
        runtime.current_tokens, num_classes=runtime.probabilities.shape[-1]
    ).to(runtime.probabilities.dtype)
    result["current_pid_tokens"] = runtime.current_tokens
    result["current_pid_available"] = runtime.available
    result["daughter_input_pid_histogram"] = runtime.daughter_input_histograms
    result["daughter_pid_histogram"] = runtime.daughter_input_histograms
    result["daughter_input_pid_histogram_available"] = (
        runtime.daughter_histogram_available
    )
    result["daughter_pid_histogram_available"] = runtime.daughter_histogram_available
    common = batch["common_features"].clone()
    common[..., :4] = result["p4"]
    mass2 = result["p4"][..., 3].square() - result["p4"][..., :3].square().sum(dim=-1)
    common[..., 4] = mass2.clamp_min(0).sqrt()
    result["common_features"] = common
    composite = batch["composite_features"].clone()
    has_daughters = batch["daughter_adjacency"].any(dim=-1)
    composite[..., :4] = torch.where(
        has_daughters.unsqueeze(-1),
        result["p4"],
        composite[..., :4],
    )
    result["composite_features"] = composite
    return result


def append_composite_proposals(
    batch: dict[str, torch.Tensor],
    proposals: list[CompositeProposal],
    *,
    target_level: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Append mothers using the same daughter-derived construction in all modes."""

    if batch["node_mask"].shape[0] != 1:
        raise ValueError("append_composite_proposals requires batch size 1")
    old_count = batch["node_mask"].shape[1]
    proposal_count = len(proposals)
    device = batch["p4"].device
    daughter_masks = torch.zeros(
        (proposal_count, old_count),
        dtype=torch.bool,
        device=device,
    )
    for row, proposal in enumerate(proposals):
        daughter_masks[row, list(proposal.daughter_positions)] = True
    pointer_confidence = torch.zeros(
        (proposal_count, old_count),
        dtype=batch["p4"].dtype,
        device=device,
    )
    for row, proposal in enumerate(proposals):
        pointer_confidence[row, list(proposal.daughter_positions)] = proposal.confidence
    construction = composite_token_from_daughters(
        daughter_mask=daughter_masks,
        p4=batch["p4"].expand(proposal_count, -1, -1),
        charge=batch["charge"].expand(proposal_count, -1),
        pid_labels=batch["pid_labels"].expand(proposal_count, -1),
        pid_probabilities=(
            batch["current_pid_probabilities"].expand(proposal_count, -1, -1)
            if "current_pid_probabilities" in batch
            else None
        ),
        daughter_embeddings=batch.get(
            "_last_node_embeddings",
            torch.zeros(
                (1, old_count, getattr(batch, "hidden_dim", 1)),
                device=device,
            ),
        ).expand(proposal_count, -1, -1),
        pointer_confidence=pointer_confidence,
        copied=batch["copied"].expand(proposal_count, -1),
    )
    # Daughter pooling is recomputed by HeterogeneousNodeEncoder after append;
    # only reco-derived structural values are persisted in the state.
    generated_ids = _next_node_ids(batch["node_ids"][0], proposal_count)
    new_ids = torch.tensor(
        [
            proposal.truth_node_id
            if proposal.truth_node_id is not None
            else int(generated_ids[index])
            for index, proposal in enumerate(proposals)
        ],
        device=device,
        dtype=torch.long,
    )
    if len(set(new_ids.tolist())) != proposal_count or any(
        int(node_id) in set(batch["node_ids"][0].tolist()) for node_id in new_ids
    ):
        new_ids = generated_ids
    result = dict(batch)
    tensor_node_fields = [
        "common_features",
        "common_availability",
        "track_features",
        "track_availability",
        "cluster_features",
        "cluster_availability",
        "composite_features",
        "composite_availability",
        "daughter_pid_histogram",
        "daughter_input_pid_histogram",
        "daughter_truth_pid_histogram",
    ]
    common = batch["common_features"].new_zeros(
        (1, proposal_count, batch["common_features"].shape[-1])
    )
    common[0, :, :4] = construction["p4"]
    mass2 = construction["p4"][:, 3].square() - construction["p4"][:, :3].square().sum(dim=-1)
    common[0, :, 4] = mass2.clamp_min(0).sqrt()
    common[0, :, 5] = construction["charge"]
    common[0, :, 6] = torch.tensor(
        [proposal.mother_type for proposal in proposals],
        device=device,
    )
    common[0, :, 7] = target_level
    common[0, :, 8] = 1
    common[0, :, 9] = 0
    common[0, :, 10] = daughter_masks.sum(dim=-1)
    common[0, :, 11] = torch.tensor(
        [proposal.confidence for proposal in proposals],
        device=device,
    )
    additions: dict[str, torch.Tensor] = {
        "common_features": common,
        "common_availability": torch.ones_like(common, dtype=torch.bool),
        "track_features": batch["track_features"].new_zeros(
            (1, proposal_count, batch["track_features"].shape[-1])
        ),
        "track_availability": torch.zeros(
            (1, proposal_count, batch["track_features"].shape[-1]),
            dtype=torch.bool,
            device=device,
        ),
        "cluster_features": batch["cluster_features"].new_zeros(
            (1, proposal_count, batch["cluster_features"].shape[-1])
        ),
        "cluster_availability": torch.zeros(
            (1, proposal_count, batch["cluster_features"].shape[-1]),
            dtype=torch.bool,
            device=device,
        ),
        "composite_features": batch["composite_features"].new_zeros(
            (1, proposal_count, batch["composite_features"].shape[-1])
        ),
        "composite_availability": torch.zeros(
            (1, proposal_count, batch["composite_features"].shape[-1]),
            dtype=torch.bool,
            device=device,
        ),
        "daughter_pid_histogram": construction["daughter_pid_histogram"].unsqueeze(0),
        "daughter_input_pid_histogram": construction[
            "daughter_pid_histogram"
        ].unsqueeze(0),
        "daughter_truth_pid_histogram": torch.zeros_like(
            construction["daughter_pid_histogram"]
        ).unsqueeze(0),
    }
    copied_width = min(
        construction["features"].shape[-1],
        additions["composite_features"].shape[-1],
    )
    additions["composite_features"][0, :, :copied_width] = construction["features"][:, :copied_width]
    additions["composite_availability"][0, :, :copied_width] = construction[
        "availability"
    ][:, :copied_width]
    for field in tensor_node_fields:
        result[field] = torch.cat([batch[field], additions[field]], dim=1)
    vector_additions = {
        "daughter_pid_histogram_available": construction[
            "daughter_pid_histogram_available"
        ].unsqueeze(0),
        "daughter_input_pid_histogram_available": construction[
            "daughter_pid_histogram_available"
        ].unsqueeze(0),
        "daughter_truth_pid_histogram_available": torch.zeros(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "node_kind_ids": torch.full(
            (1, proposal_count),
            NODE_KIND_TO_ID["composite"],
            dtype=torch.long,
            device=device,
        ),
        "leaf_kinematics_mode_ids": torch.full(
            (1, proposal_count),
            LEAF_MODE_TO_ID["composite"],
            dtype=torch.long,
            device=device,
        ),
        "runtime_composite_type_source_ids": torch.full(
            (1, proposal_count),
            COMPOSITE_TYPE_SOURCE_TO_ID["predicted"],
            dtype=torch.long,
            device=device,
        ),
        "pid_labels": torch.tensor(
            [[proposal.mother_type for proposal in proposals]],
            dtype=torch.long,
            device=device,
        ),
        "level_ids": torch.full(
            (1, proposal_count),
            target_level,
            dtype=torch.long,
            device=device,
        ),
        "charge": construction["charge"].unsqueeze(0),
        "parent_ids": torch.full(
            (1, proposal_count),
            -1,
            dtype=torch.long,
            device=device,
        ),
        "active": torch.ones((1, proposal_count), dtype=torch.bool, device=device),
        "copied": torch.zeros((1, proposal_count), dtype=torch.bool, device=device),
        "node_ids": new_ids.unsqueeze(0),
        "reco_ids": torch.full(
            (1, proposal_count),
            -1,
            dtype=torch.long,
            device=device,
        ),
        "source_node_ids": new_ids.unsqueeze(0),
        "copied_from": torch.full(
            (1, proposal_count),
            -1,
            dtype=torch.long,
            device=device,
        ),
        "b_side": torch.stack(
            [
                _proposal_b_side(batch["b_side"][0], proposal)
                for proposal in proposals
            ]
        ).unsqueeze(0),
        "full_truth_daughter_count": daughter_masks.sum(dim=-1).unsqueeze(0),
        "retained_truth_daughter_count_expected": daughter_masks.sum(
            dim=-1
        ).unsqueeze(0),
        "retained_daughter_count": daughter_masks.sum(dim=-1).unsqueeze(0),
        "reconstructed_daughter_count": daughter_masks.sum(dim=-1).unsqueeze(0),
        "complete_truth_decay": torch.zeros(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "complete_reconstructable_decay": torch.ones(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "recursive_reconstructable_complete": torch.ones(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "partial_missing_daughters": torch.zeros(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "contracted_intermediate": torch.zeros(
            (1, proposal_count), dtype=torch.bool, device=device
        ),
        "valid_reconstruction_target": daughter_masks.sum(dim=-1).ge(2).unsqueeze(0),
        "truth_root_distance": torch.zeros(
            (1, proposal_count), dtype=torch.long, device=device
        ),
        "full_event_max_level": torch.full(
            (1, proposal_count), target_level, dtype=torch.long, device=device
        ),
    }
    for optional_pid_field in ("pid_target_labels", "truth_pid_labels"):
        if optional_pid_field in batch:
            vector_additions[optional_pid_field] = vector_additions["pid_labels"].clone()
    if "truth_pid_available" in batch:
        vector_additions["truth_pid_available"] = torch.zeros(
            (1, proposal_count),
            dtype=torch.bool,
            device=device,
        )
    for field, addition in vector_additions.items():
        result[field] = torch.cat([batch[field], addition], dim=1)
    result["p4"] = torch.cat([batch["p4"], construction["p4"].unsqueeze(0)], dim=1)
    old_adjacency = batch["daughter_adjacency"]
    adjacency = torch.zeros(
        (1, old_count + proposal_count, old_count + proposal_count),
        dtype=torch.bool,
        device=device,
    )
    adjacency[:, :old_count, :old_count] = old_adjacency
    adjacency[0, old_count:, :old_count] = daughter_masks
    result["daughter_adjacency"] = adjacency
    if "recursive_leaf_source_mask" in batch:
        old_sources = batch["recursive_leaf_source_mask"]
        new_sources = torch.einsum(
            "qn,bns->bqs",
            daughter_masks.float(),
            old_sources.float(),
        ).bool()
        result["recursive_leaf_source_mask"] = torch.cat(
            [old_sources, new_sources],
            dim=1,
        )
        overlap = torch.einsum(
            "bns,bms->bnm",
            result["recursive_leaf_source_mask"].to(torch.int32),
            result["recursive_leaf_source_mask"].to(torch.int32),
        ) > 0
        diagonal = torch.eye(
            overlap.shape[-1], dtype=torch.bool, device=device
        ).unsqueeze(0)
        result["source_conflict_matrix"] = overlap & ~diagonal
    if "current_pid_probabilities" in batch:
        mother_probabilities = torch.nn.functional.one_hot(
            vector_additions["pid_labels"],
            num_classes=batch["current_pid_probabilities"].shape[-1],
        ).to(batch["current_pid_probabilities"].dtype)
        result["current_pid_probabilities"] = torch.cat(
            [batch["current_pid_probabilities"], mother_probabilities], dim=1
        )
        result["current_pid_tokens"] = torch.cat(
            [batch["current_pid_tokens"], vector_additions["pid_labels"]], dim=1
        )
        result["current_pid_available"] = torch.cat(
            [
                batch["current_pid_available"],
                torch.ones((1, proposal_count), dtype=torch.bool, device=device),
            ],
            dim=1,
        )
    result["node_mask"] = result["active"]
    result["node_features"] = result["common_features"]
    result.pop("allowed_type_mask", None)
    result.pop("pointer_validity_mask", None)
    # Record a unique parent only after exclusive resolution.
    for row, proposal in enumerate(proposals):
        for daughter in proposal.daughter_positions:
            if result["parent_ids"][0, daughter] < 0:
                result["parent_ids"][0, daughter] = old_count + row
    return result, [int(value) for value in new_ids.tolist()]


def _select_nodes(
    batch: dict[str, torch.Tensor],
    selection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    indices = selection.nonzero(as_tuple=False).flatten()
    selected: dict[str, torch.Tensor] = {}
    n_nodes = batch["node_mask"].shape[1]
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key == "recursive_leaf_source_mask":
            selected[key] = value[:, indices]
        elif value.ndim >= 3 and value.shape[1:3] == (n_nodes, n_nodes):
            selected[key] = value[:, indices][:, :, indices]
        elif value.ndim >= 2 and value.shape[1] == n_nodes:
            selected[key] = value[:, indices]
        else:
            selected[key] = value
    id_to_new = {
        int(batch["node_ids"][0, old]): new
        for new, old in enumerate(indices.tolist())
    }
    remapped_parent = torch.full_like(selected["parent_ids"], -1)
    for new, old in enumerate(indices.tolist()):
        old_parent_position = int(batch["parent_ids"][0, old])
        if old_parent_position >= 0:
            parent_node_id = int(batch["node_ids"][0, old_parent_position])
            remapped_parent[0, new] = id_to_new.get(parent_node_id, -1)
    selected["parent_ids"] = remapped_parent
    selected["node_mask"] = selected["active"]
    selected["node_features"] = selected["common_features"]
    return selected


def _truth_proposals(
    truth: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    target_level: int,
) -> list[CompositeProposal]:
    truth_positions = (
        truth["node_mask"][0] & (truth["level_ids"][0] == target_level)
    ).nonzero(as_tuple=False).flatten()
    state_by_node_id = {
        int(node_id): position
        for position, node_id in enumerate(state["node_ids"][0].tolist())
    }
    proposals: list[CompositeProposal] = []
    for query_id, truth_position in enumerate(truth_positions.tolist()):
        daughter_truth_positions = truth["daughter_adjacency"][0, truth_position].nonzero(
            as_tuple=False
        ).flatten()
        daughters = []
        for daughter_position in daughter_truth_positions.tolist():
            node_id = int(truth["node_ids"][0, daughter_position])
            if node_id in state_by_node_id:
                daughters.append(state_by_node_id[node_id])
        if not daughters:
            continue
        proposals.append(
            CompositeProposal(
                query_id=query_id,
                mother_type=int(
                    truth.get("pid_target_labels", truth["pid_labels"])[0, truth_position]
                ),
                daughter_positions=tuple(sorted(daughters)),
                object_score=1.0,
                confidence=1.0,
                truth_node_id=int(truth["node_ids"][0, truth_position]),
            )
        )
    return proposals


def _next_node_ids(existing: torch.Tensor, count: int) -> torch.Tensor:
    start = int(existing.max()) + 1 if existing.numel() else 0
    return torch.arange(start, start + count, device=existing.device, dtype=torch.long)


def _proposal_b_side(b_side: torch.Tensor, proposal: CompositeProposal) -> torch.Tensor:
    sides = b_side[list(proposal.daughter_positions)]
    valid = sides[sides >= 0]
    if not valid.numel() or not torch.all(valid == valid[0]):
        return torch.tensor(-1, dtype=torch.long, device=b_side.device)
    return valid[0]


def _state_fingerprint(batch: dict[str, torch.Tensor]) -> str:
    records = []
    for position in range(batch["node_mask"].shape[1]):
        if not bool(batch["node_mask"][0, position]):
            continue
        daughters = batch["daughter_adjacency"][0, position].nonzero(
            as_tuple=False
        ).flatten()
        if "recursive_leaf_source_mask" in batch:
            sources = (
                batch["recursive_leaf_source_mask"][0, position]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
        else:
            sources = sorted(
                int(batch["source_node_ids"][0, daughter])
                for daughter in daughters.tolist()
            )
        records.append(
            (
                int(batch["level_ids"][0, position]),
                int(batch["pid_labels"][0, position]),
                tuple(sources),
            )
        )
    return hashlib.sha256(repr(sorted(records)).encode("utf-8")).hexdigest()


__all__ = [
    "CompositeProposal",
    "LevelRolloutResult",
    "RolloutConfig",
    "RolloutStep",
    "append_composite_proposals",
    "hard_decode_proposals",
    "level_rollout",
    "cached_context_for_level",
    "resolve_exclusive_proposals",
    "validate_proposals",
]
