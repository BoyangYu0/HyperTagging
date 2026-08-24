"""Fail-closed read-only diagnostics for reconstruction query activation."""

from __future__ import annotations

import math
from typing import Any

import torch

from hypertagging.losses.level_reconstruction import targets_for_level
from hypertagging.losses.set_matching import matching_cost
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    LevelReconstructionOutput,
)
from hypertagging.preprocessing.pid_filter import DETOKENIZE_DICT
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig,
    _resolve_with_config,
    hard_decode_proposals,
)


QUERY_ACTIVATION_DIAGNOSTIC_SCHEMA_VERSION = (
    "hypertagging-query-activation-diagnostic-v1"
)
ALLOWED_DIAGNOSTIC_ROLES = frozenset({"train", "validation"})


def require_diagnostic_role(role: str, *, split_counts: dict[str, int]) -> None:
    """Reject sealed/test roles and any data module that retained test rows."""

    if role not in ALLOWED_DIAGNOSTIC_ROLES:
        raise ValueError("query-activation diagnostics permit train/validation roles only")
    if int(split_counts.get("test", 0)) != 0:
        raise ValueError("query-activation diagnostic data module contains sealed-test rows")


def require_finite_json(value: Any, *, path: str = "diagnostic") -> None:
    """Recursively reject non-finite numeric diagnostic output."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite diagnostic value at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"unsupported diagnostic value at {path}: {type(value).__name__}")


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    detached = values.detach().float().flatten().cpu()
    if detached.numel() == 0 or not torch.isfinite(detached).all():
        raise ValueError("diagnostic distribution must be non-empty and finite")
    quantiles = torch.quantile(
        detached, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    )
    return {
        "count": int(detached.numel()),
        "minimum": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "mean": float(detached.mean()),
    }


def per_query_probability_distributions(
    object_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    """Return object and null probability summaries for every decoder slot."""

    if object_logits.ndim != 2 or object_logits.shape[0] == 0:
        raise ValueError("object logits must have non-empty shape [events, queries]")
    probability = torch.sigmoid(object_logits)
    return [
        {
            "query_id": query_id,
            "object_probability": _distribution(probability[:, query_id]),
            "null_probability": _distribution(1.0 - probability[:, query_id]),
        }
        for query_id in range(probability.shape[1])
    ]


def matching_and_margin_diagnostics(
    output: LevelReconstructionOutput,
    batch: dict[str, torch.Tensor],
    matches: list[list[tuple[int, int]]],
    *,
    target_level: int,
    target_policy: str,
) -> dict[str, Any]:
    """Summarize the exact Hungarian cost matrix and class margins."""

    target_types, target_masks, _target_p4, _target_charge = targets_for_level(
        batch, target_level, target_policy=target_policy
    )
    selected_costs: list[float] = []
    unmatched_query_best_costs: list[float] = []
    matched_target_margins: list[float] = []
    matched_costs_by_family: dict[str, list[float]] = {}
    all_top_two_margins: list[float] = []
    matched_queries = 0
    unmatched_queries = 0
    truth_targets = 0
    for event_index, types in enumerate(target_types):
        context = batch["node_mask"][event_index] & (
            batch["level_ids"][event_index] < target_level
        )
        logits = output.pointer.type_logits[event_index]
        allowed_logits = logits > -9_999.0
        for query_id in range(logits.shape[0]):
            values = logits[query_id, allowed_logits[query_id]]
            margin = (
                values.topk(2).values.diff().abs()[0]
                if values.numel() >= 2
                else values.new_zeros(())
            )
            all_top_two_margins.append(float(margin.detach().cpu()))
        event_matches = matches[event_index]
        matched_ids = {query_id for query_id, _target_id in event_matches}
        matched_queries += len(event_matches)
        unmatched_queries += logits.shape[0] - len(event_matches)
        truth_targets += int(types.numel())
        if types.numel() == 0:
            continue
        cost = matching_cost(
            type_logits=logits,
            pointer_logits=output.pointer.pointer_logits[event_index, :, context],
            target_types=types,
            target_masks=target_masks[event_index],
            object_logits=output.pointer.object_logits[event_index],
            cardinality_logits=output.pointer.cardinality_logits[event_index],
        )
        if not torch.isfinite(cost).all():
            raise ValueError("Hungarian diagnostic cost contains non-finite values")
        for query_id, target_id in event_matches:
            selected_cost = float(cost[query_id, target_id].detach().cpu())
            selected_costs.append(selected_cost)
            family = str(DETOKENIZE_DICT[int(types[target_id])])
            matched_costs_by_family.setdefault(family, []).append(selected_cost)
            target_logit = logits[query_id, types[target_id]]
            alternatives = logits[query_id, allowed_logits[query_id]].clone()
            allowed_ids = allowed_logits[query_id].nonzero(as_tuple=False).flatten()
            alternatives[allowed_ids == types[target_id]] = -torch.inf
            alternative = alternatives.max()
            matched_target_margins.append(
                float(
                    (target_logit - alternative).detach().cpu()
                    if torch.isfinite(alternative)
                    else 0.0
                )
            )
        for query_id in range(logits.shape[0]):
            if query_id not in matched_ids:
                unmatched_query_best_costs.append(
                    float(cost[query_id].min().detach().cpu())
                )
    result = {
        "truth_target_count": truth_targets,
        "matched_query_count": matched_queries,
        "unmatched_query_count": unmatched_queries,
        "matched_cost": _distribution(torch.tensor(selected_costs))
        if selected_costs
        else {"count": 0},
        "unmatched_query_best_cost": _distribution(
            torch.tensor(unmatched_query_best_costs)
        )
        if unmatched_query_best_costs
        else {"count": 0},
        "matched_target_class_logit_margin": _distribution(
            torch.tensor(matched_target_margins)
        )
        if matched_target_margins
        else {"count": 0},
        "top_two_class_logit_margin": _distribution(torch.tensor(all_top_two_margins)),
        "matched_assignment_cost_by_target_family_pdg": {
            family: _distribution(torch.tensor(values))
            for family, values in sorted(matched_costs_by_family.items())
        },
    }
    require_finite_json(result)
    return result


def inference_decision_diagnostics(
    output: LevelReconstructionOutput,
    batch: dict[str, torch.Tensor],
    config: RolloutConfig,
) -> dict[str, Any]:
    """Trace counts around the actual, unchanged hard inference decision."""

    if output.pointer.object_logits.shape[0] != 1:
        raise ValueError("inference decision diagnostics require one event")
    object_probability = torch.sigmoid(output.pointer.object_logits[0])
    object_active = object_probability >= config.object_threshold
    context = output.context_mask[0]
    pointer_probability = torch.sigmoid(
        output.pointer.pointer_logits[0, :, context]
    )
    pointer_threshold_active = (
        (pointer_probability >= config.pointer_threshold).sum(dim=-1)
        >= config.min_daughters
    )
    proposals = hard_decode_proposals(output, batch, config)
    accepted = _resolve_with_config(proposals, batch, config)
    null_probability = 1.0 - object_probability
    derived_stop_probability = null_probability.prod()
    pointer_validity = batch.get("pointer_validity_mask")
    if pointer_validity is None:
        valid_context_count = int(context.sum())
    else:
        valid_context_count = int((pointer_validity[0].bool() & context).sum())
    result = {
        "query_count": int(object_probability.numel()),
        "query_mask_present": False,
        "context_node_count": int(context.sum()),
        "pointer_valid_context_count": valid_context_count,
        "object_active_query_count_before_decision": int(object_active.sum()),
        "pointer_active_query_count_before_decision": int(
            (object_active & pointer_threshold_active).sum()
        ),
        "active_query_count_after_unchanged_inference_decision": len(proposals),
        "active_query_count_after_exclusive_pruning": len(accepted),
        "predicted_node_count_before_pruning": len(proposals),
        "predicted_node_count_after_pruning": len(accepted),
        "predicted_depth_before_pruning": int(output.target_level if proposals else 0),
        "predicted_depth_after_pruning": int(output.target_level if accepted else 0),
        "derived_continue_probability": float(
            (1.0 - derived_stop_probability).detach().cpu()
        ),
        "derived_stop_probability": float(derived_stop_probability.detach().cpu()),
        "depth_continue_head_present": False,
        "depth_probability_semantics": (
            "derived_from_product_of_query_null_probabilities; no learned depth head"
        ),
    }
    require_finite_json(result)
    return result


def gradient_reachability_diagnostics(
    model: LevelAutoregressiveReconstructor,
    loss: torch.Tensor,
    *,
    target_level: int,
) -> dict[str, Any]:
    """Backpropagate without an optimizer step and report decoder gradient paths."""

    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("gradient diagnostic loss must be a finite scalar")
    model.zero_grad(set_to_none=True)
    loss.backward()
    decoder = (
        model.level_decoders[str(target_level)]
        if str(target_level) in model.level_decoders
        else model.decoder
    )

    def report(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
        gradients = [parameter.grad for parameter in parameters]
        reached = [gradient for gradient in gradients if gradient is not None]
        finite = bool(reached) and all(torch.isfinite(gradient).all() for gradient in reached)
        squared_norm = sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in reached
        )
        norm = math.sqrt(squared_norm)
        return {
            "parameter_tensor_count": len(parameters),
            "gradient_tensor_count": len(reached),
            "all_reached_gradients_finite": finite,
            "gradient_norm": norm,
            "gradient_reaches_parameters": bool(reached) and finite and norm > 0.0,
        }

    result = {
        "query_embeddings": report([decoder.query]),
        "objectness_head": report(list(decoder.object_head.parameters())),
        "pointer_heads": report(
            list(decoder.pointer_query.parameters())
            + list(decoder.pointer_key.parameters())
        ),
        "depth_continue_head": {
            "present": False,
            "gradient_reaches_parameters": False,
            "reason": "architecture derives continuation from query decisions",
        },
    }
    model.zero_grad(set_to_none=True)
    require_finite_json(result)
    return result


__all__ = [
    "ALLOWED_DIAGNOSTIC_ROLES",
    "QUERY_ACTIVATION_DIAGNOSTIC_SCHEMA_VERSION",
    "gradient_reachability_diagnostics",
    "inference_decision_diagnostics",
    "matching_and_margin_diagnostics",
    "per_query_probability_distributions",
    "require_diagnostic_role",
    "require_finite_json",
]
