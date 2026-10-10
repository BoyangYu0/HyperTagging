"""Observational native greedy traces and strictly post-hoc TRAIN target joins."""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict
import math

import torch
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
from hypertagging.reconstruction.level_rollout import (
    level_rollout,
    cached_context_for_level,
)

STATE_KEYS = (
    "node_mask",
    "node_ids",
    "level_ids",
    "pid_labels",
    "parent_ids",
    "node_kind_ids",
    "recursive_leaf_source_mask",
    "p4",
    "charge",
)


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return value


def state_record(state):
    return {key: json_safe(state[key][0]) for key in STATE_KEYS}


def serialize_native_rollout(rollout, config=None):
    """Serialize reconstructed state/model outputs only, before supervision joins.

    Source columns retain the native rollout axis. Config is optional for a
    trainer wrapper; without it eligibility is explicitly unavailable. No target
    fields, target geometry, evaluation source keys, or truth model inputs are read.
    """
    if rollout.teacher_forced:
        raise ValueError("Truth-guided rollout in generated trace")
    steps = []
    for step in rollout.steps:
        if step.used_teacher_forcing or any(
            p.truth_node_id is not None for p in (*step.proposals, *step.accepted)
        ):
            raise ValueError("Truth-guided proposal in generated trace")
        context = cached_context_for_level(rollout, step.target_level)
        pointer = step.model_output.pointer
        probabilities = {
            "object": pointer.object_logits.sigmoid(),
            "type": pointer.type_logits.softmax(-1),
            "pointer": pointer.pointer_logits.sigmoid(),
            "cardinality": pointer.cardinality_logits.softmax(-1),
            "confidence": pointer.confidence_logits.sigmoid(),
        }
        if not all(torch.isfinite(v).all() for v in probabilities.values()):
            raise ValueError("Nonfinite generated probabilities")
        eligible = (
            config.constraint_policy.forest_pointer_validity_mask(
                context, step.target_level, teacher_prefix=False
            )
            if config is not None
            else None
        )
        steps.append(
            {
                "height": step.target_level,
                "state_before": state_record(context),
                "eligible_positions": (
                    torch.where(eligible[0])[0].tolist()
                    if eligible is not None
                    else None
                ),
                "proposals": [json_safe(asdict(p)) for p in step.proposals],
                "accepted": [json_safe(asdict(p)) for p in step.accepted],
                "appended_node_ids": list(step.appended_node_ids),
                "probabilities": {k: json_safe(v[0]) for k, v in probabilities.items()},
                "pointer_output": {
                    k: json_safe(v[0]) if isinstance(v, torch.Tensor) else None
                    for k, v in vars(pointer).items()
                },
                "construction_pid_mode": step.appended_mother_p4_pid_kinematics_mode,
            }
        )
        if step.decode_trace is not None:
            record = step.decode_trace
            steps[-1]["decode_trace"] = {
                "version": "post-pid-native-decode-v1",
                "state_semantics": "Exact post-PID state consumed by hard decoding; no truth joins.",
                "state": {key: json_safe(value[0])
                          for key, value in record["state"].items()},
                "context_mask": json_safe(record["context_mask"][0]),
                "hard_decode_context_mask": json_safe(record["hard_decode_context_mask"][0]),
                "pointer_validity_mask": json_safe(record["pointer_validity_mask"][0]),
                "forest_pointer_validity_mask": json_safe(record["forest_pointer_validity_mask"][0]),
                "raw_proposals": [json_safe(asdict(p)) for p in record["raw_proposals"]],
            }
    trace = {
        "version": "native-greedy-detector-only-trace-v1",
        "mode": "predicted",
        "source_axis": "native_rollout_source_columns",
        "state_before_semantics": "Cached previous-generation state; native current-level PID reconstruction occurs within rollout.",
        "config": json_safe(asdict(config)) if config is not None else None,
        "steps": steps,
        "cached_states": [
            {"after_height": h, "state": state_record(s)}
            for h, s in rollout.cached_states
        ],
        "final_state": state_record(rollout.batch),
        "stop_reason": rollout.stop_reason,
        "valid": rollout.valid,
        "empty_level_count": rollout.empty_level_count,
        "truth_used_for_generation": False,
        "search": "native unbatched greedy; no beam pool",
    }
    return trace


def capture_native_trace(model, full_batch, config):
    """Return projection, unchanged native rollout, and detached generated trace.

    Persist trace BEFORE evaluate_native_trace. Clone cached tensors outside
    inference_mode before any gradient observer.
    """
    if full_batch["node_mask"].shape[0] != 1 or config.constraint_policy is None:
        raise ValueError("One native event and explicit checkpoint policy required")
    projection = project_schema_v4_fsps(full_batch)
    if hasattr(model, "eval"):
        model.eval()
    with torch.inference_mode():
        rollout = level_rollout(
            model, projection.batch, mode="predicted", config=config
        )
        trace = serialize_native_rollout(rollout, config)
        trace["source_axis"] = "strict_projection_compact_detector_columns"
    return projection, rollout, trace


def source_rows(state):
    return [
        frozenset(i for i, present in enumerate(row) if present)
        for row in state["recursive_leaf_source_mask"]
    ]


def proposal_signature(proposal, memberships):
    positions = proposal["daughter_positions"]
    if (
        not positions
        or len(set(positions)) != len(positions)
        or any(p < 0 or p >= len(memberships) for p in positions)
    ):
        return None
    values = [memberships[p] for p in positions]
    if any(not v for v in values) or sum(map(len, values)) != len(set().union(*values)):
        return None
    return tuple(sorted(tuple(sorted(v)) for v in values))


def evaluate_native_trace(
    truth, projection, trace, *, target_policy, minimum_daughters
):
    """After-the-fact target join; never creates, reranks, or changes proposals.

    Eligible daughter coverage is only structural availability, never a claim of
    legal unvisited proposals. Generation and acceptance count actual proposals.
    """
    if (
        trace.get("truth_used_for_generation") is not False
        or trace.get("mode") != "predicted"
    ):
        raise ValueError("Only detached detector-generated trace accepted")
    if target_policy not in (
        "complete_only",
        "reconstructable_partial",
        "diagnostic_all",
    ):
        raise ValueError("Unknown target policy")
    original_sources = truth["recursive_leaf_source_mask"][0].bool()
    kept = torch.tensor(projection.audit.original_fsp_positions[0], dtype=torch.long)
    columns = original_sources[kept].any(0)
    compact = original_sources[:, columns]
    if not torch.equal(
        compact[kept], projection.batch["recursive_leaf_source_mask"][0]
    ):
        raise ValueError("Evaluation/source-column projection mismatch")
    source_width = int(columns.sum())
    for step in trace["steps"]:
        if any(
            len(r) != source_width
            for r in step["state_before"]["recursive_leaf_source_mask"]
        ):
            raise ValueError("Trace source-axis mismatch")
    steps = {step["height"]: step for step in trace["steps"]}
    final_sources = source_rows(trace["final_state"])
    counts, first_error_counts, cells, rows = (
        Counter(),
        Counter(),
        defaultdict(Counter),
        [],
    )
    labels = truth.get("pid_target_labels", truth["pid_labels"])[0]
    for node in torch.where(truth["node_mask"][0] & (truth["level_ids"][0] > 0))[
        0
    ].tolist():
        height = int(truth["level_ids"][0, node])
        pid = int(labels[node])
        daughters = torch.where(
            truth["daughter_adjacency"][0, node]
            & truth["node_mask"][0]
            & (truth["level_ids"][0] < height)
        )[0].tolist()
        complete_key = (
            "recursive_reconstructable_complete"
            if "recursive_reconstructable_complete" in truth
            else "complete_reconstructable_decay"
        )
        complete = bool(truth[complete_key][0, node]) if complete_key in truth else None
        valid_target = bool(
            truth.get("valid_reconstruction_target", truth["node_mask"])[0, node]
        )
        population = (
            "complete"
            if complete is True
            else ("partial" if complete is False else "completion_unavailable")
        )
        out = {
            "node_position": node,
            "height": height,
            "pid_token": pid,
            "signed_pdg": PDG_TOKENS[pid] if 0 <= pid < len(PDG_TOKENS) else None,
            "pid_family": f"abs_pdg_{abs(PDG_TOKENS[pid])}"
            if 0 < pid < len(PDG_TOKENS)
            else "unknown",
            "daughter_count": len(daughters),
            "source_size": int(compact[node].sum()),
            "population": population,
            "target_policy": target_policy,
        }
        eligible = (target_policy == "diagnostic_all" or valid_target) and (
            target_policy != "complete_only" or complete is not False
        )
        support = [frozenset(torch.where(compact[d])[0].tolist()) for d in daughters]
        missing_sources = (
            bool(original_sources[[node, *daughters]][:, ~columns].any())
            or not support
            or any(not v for v in support)
        )
        alias = bool(support) and sum(map(len, support)) != len(set().union(*support))
        signature = tuple(sorted(tuple(sorted(v)) for v in support))
        counts["all_retained_mothers"] += 1
        if not eligible:
            out["first_error"] = "outside_requested_target_policy"
        elif len(daughters) < minimum_daughters:
            out["first_error"] = "below_minimum_daughter_support"
        elif missing_sources:
            out["first_error"] = "source_support_unavailable"
        elif alias:
            out["first_error"] = "target_daughter_sources_overlap"
        elif height not in steps:
            out["first_error"] = "height_not_visited"
        else:
            step = steps[height]
            memberships = source_rows(step["state_before"])
            coverage = all(
                any(memberships[p] == d for p in step["eligible_positions"])
                for d in support
            )
            generated = [
                p
                for p in step["proposals"]
                if proposal_signature(p, memberships) == signature
            ]
            accepted = [
                p
                for p in step["accepted"]
                if proposal_signature(p, memberships) == signature
            ]
            correct_type = [p for p in generated if p["mother_type"] == pid]
            accepted_type = [p for p in accepted if p["mother_type"] == pid]
            out.update(
                eligible_daughter_source_coverage=coverage,
                generated_exact_daughters=bool(generated),
                generated_exact_daughters_and_pid=bool(correct_type),
                accepted_exact_daughters=bool(accepted),
                accepted_exact_daughters_and_pid=bool(accepted_type),
                final_constructed_source_membership=frozenset(
                    torch.where(compact[node])[0].tolist()
                )
                in final_sources,
                legal_unvisited_group_reachability="unavailable_not_checked",
            )
            if generated and 0 < pid < len(PDG_TOKENS):
                # Select one existing proposal by model-only score/query tie-break.
                p = sorted(
                    generated,
                    key=lambda p: (-p["object_score"], -p["confidence"], p["query_id"]),
                )[0]
                probability = step["probabilities"]["type"][p["query_id"]]
                predicted = p["mother_type"]
                target_probability = probability[pid]
                out["conditional_pid"] = {
                    "target_token": pid,
                    "predicted_token": predicted,
                    "target_signed_pdg": PDG_TOKENS[pid],
                    "predicted_signed_pdg": PDG_TOKENS[predicted],
                    "correct": predicted == pid,
                    "query_id": p["query_id"],
                    "target_probability": target_probability,
                    "predicted_probability": probability[predicted],
                    "negative_log_likelihood_clipped_1e30": -math.log(
                        max(target_probability, 1e-30)
                    ),
                    "multiclass_brier": sum(
                        (v - float(k == pid)) ** 2 for k, v in enumerate(probability)
                    ),
                }
            if accepted_type:
                reason = "accepted_exact_daughters_and_pid"
            elif correct_type:
                reason = "correct_generated_proposal_retention_rejected"
            elif generated:
                reason = (
                    "mother_pid_error_on_generated_correct_daughters"
                    if 0 < pid < len(PDG_TOKENS)
                    else "mother_pid_target_unavailable"
                )
            elif not coverage:
                reason = "eligible_daughter_source_group_absent"
            else:
                reason = "exact_group_not_generated_object_pointer_type_or_constraint_unresolved"
            out["first_error"] = reason
        out["policy_eligible"] = eligible
        first_error_counts[out["first_error"]] += 1
        for key in (
            "generated_exact_daughters",
            "generated_exact_daughters_and_pid",
            "accepted_exact_daughters",
            "accepted_exact_daughters_and_pid",
            "eligible_daughter_source_coverage",
        ):
            counts[key] += int(out.get(key, False))
        if "conditional_pid" in out:
            counts["conditional_pid_trials"] += 1
            counts["conditional_pid_correct"] += int(out["conditional_pid"]["correct"])
        key = f"height={height}|pdg={out['signed_pdg']}|daughters={len(daughters)}|sources={out['source_size']}|population={population}"
        cells[key]["targets"] += 1
        cells[key][out["first_error"]] += 1
        rows.append(out)
    return {
        "counts": dict(counts),
        "first_error_counts": dict(first_error_counts),
        "cells": {k: dict(v) for k, v in cells.items()},
        "targets": rows,
        "source_columns_kept": torch.where(columns)[0].tolist(),
        "limitations": [
            "Eligible source-set coverage is not legal decoder reachability.",
            "Native post-hard-decode proposals cannot resolve every missing proposal into object/pointer/type/constraint causes; unresolved remains explicit.",
            "Conditional PID is measured only on actually generated exact daughter groups; unknown target PID remains unavailable.",
            "No beam or retained-pool endpoint is defined by this greedy trace.",
        ],
    }
