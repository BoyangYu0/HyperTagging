"""Post-inference, two-trial tag accounting and retained-channel coverage.

These are exact retained-tree/source/reduced-PID diagnostics. They do not
implement Belle II MC matching and must not be labelled physical FEI efficiency.
No helper in this module is called while generating or ranking hypotheses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping, Sequence

import torch

from hypertagging.evaluation.full_decay_metrics import (
    B_ROOT_TOKENS, RatioMetric, _component_structurally_valid,
    _is_continuum_category, _root_positions, _tree_view, _truth_b_roots,
)
from hypertagging.preprocessing.channels import deterministic_channel_id
from hypertagging.preprocessing.pid_filter import LEGACY_PARTICLE_NAMES, PDG_TOKENS


VERSION = "two-trial-retained-tag-efficiency-v1"
_NAMES = dict(zip(PDG_TOKENS[1:], LEGACY_PARTICLE_NAMES))
_LIMITATIONS = ["retained_tree_and_reduced_PID_not_full_Belle_II_MC_matching", "no_common_FEI_candidate_selection_or_kinematic_working_point"]
_SLOT_POLICY = "first_two_source_disjoint_outermost_B_subtrees_in_model_construction_order_per_hypothesis"


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return RatioMetric(float(numerator), float(denominator)).as_dict()


def _signature(view, root: int, *, sources: bool, require_truth: bool = False):
    """Canonical unordered PID tree; source labels appear only on leaves."""
    visiting = set()

    def visit(node):
        if node in visiting:
            raise ValueError("cyclic decay tree")
        if require_truth and (view.truth_pid_available is None
                              or not bool(view.truth_pid_available[node])):
            raise ValueError("truth_pid_unavailable")
        pid = int(view.pid[node])
        if not 0 < pid < len(PDG_TOKENS):
            raise ValueError("unknown_pid_token")
        visiting.add(node)
        children = view.children(node)
        daughters = tuple(sorted((visit(child) for child in children), key=repr))
        visiting.remove(node)
        if sources:
            return (pid, tuple(sorted(view.source_set(node))) if not children else (), daughters)
        return (pid, daughters)

    return visit(root)


def _safe_signature(view, root, *, require_truth=False):
    if not _component_structurally_valid(view, root):
        return None
    try:
        signature = _signature(view, root, sources=True, require_truth=require_truth)
    except ValueError:
        return None
    return signature if view.source_set(root) else None


def _selected_b_roots(view) -> list[int]:
    """Truth-blind two-slot policy: earliest constructed disjoint B subtrees.

    Tensor positions are construction order, independent of truth alignment.
    Only outermost B subtrees qualify. Each coherent hypothesis has its own
    slots; slots are never carried across competing beam states.
    """
    b_nodes = [node for node in view.positions if int(view.pid[node]) in B_ROOT_TOKENS]
    candidates = []
    for node in b_nodes:
        parents = [int(p) for p in (view.adjacency[:, node] & view.active).nonzero().flatten()]
        seen = set()
        while parents:
            parent = parents.pop()
            if parent in seen:
                continue
            seen.add(parent)
            parents.extend(int(p) for p in (view.adjacency[:, parent] & view.active).nonzero().flatten())
        if not (set(b_nodes) & seen) and view.children(node):
            candidates.append(node)
    selected, used = [], set()
    for node in sorted(candidates):
        detector_sources = view.detector_sources
        keys = (set(detector_sources[node].nonzero().flatten().tolist())
                if detector_sources is not None else set(view.source_set(node)))
        if not keys or keys & used or not _component_structurally_valid(view, node):
            continue
        selected.append(node)
        used.update(keys)
        if len(selected) == 2:
            break
    return selected


def _channel(view, root, event) -> dict[str, Any]:
    try:
        signature = _signature(view, root, sources=False, require_truth=True)
    except ValueError:
        signature = None
    signature_json = json.dumps(signature, separators=(",", ":")) if signature is not None else None
    side = int(view.b_side[root]) if view.b_side is not None else -1
    side = side if side in (0, 1) else None

    def identifier(suffix):
        value = getattr(event, f"b{side + 1}_{suffix}", 0) if event is not None and side is not None else 0
        return int(value) if value else None

    def description(tree):
        token, daughters = tree
        name = _NAMES.get(PDG_TOKENS[token], str(PDG_TOKENS[token]))
        return name if not daughters else name + " -> (" + ", ".join(description(child) for child in daughters) + ")"

    return {
        "b_side": side,
        "full_truth_channel_id": identifier("full_truth_channel_id"),
        "reconstructable_channel_id": identifier("reconstructable_channel_id"),
        "stored_channel_id": identifier("channel_id"),
        "stored_channel_ids_available": any(identifier(field) is not None for field in ("full_truth_channel_id", "reconstructable_channel_id", "channel_id")),
        "retained_signed_channel_id": deterministic_channel_id(signature_json) if signature_json else None,
        "retained_signed_pid_signature": signature_json,
        "retained_decay_description": description(signature) if signature is not None else None,
        "retained_pid_signature_available": signature is not None,
        "signature_scope": "retained_reduced_pid_tree_signed_no_charge_conjugation_normalization",
        "full_generator_decay_description_available": False,
    }


def evaluate_tag_efficiency_event(
    truth_batch: Mapping[str, torch.Tensor],
    candidate_batches: Sequence[Mapping[str, torch.Tensor]],
    *, source_category: str, event: Any = None, event_index: int = 0,
    oracle_ks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed ranked hypothesis list, without truth-based reranking.

    A one-element list represents greedy inference. Unknown B truth remains
    in the two-trial denominator and is reported separately; proven-success
    fractions are explicitly lower bounds while any truth slot is unknown.
    """
    batches = tuple(candidate_batches)
    if not batches:
        raise ValueError("tag evaluation requires at least one ranked hypothesis")
    ks = tuple(sorted(set(oracle_ks if oracle_ks is not None else (1, len(batches)))))
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("oracle_ks must contain positive integers")
    truth = _tree_view(truth_batch, event_index, truth=True)
    predictions = [_tree_view(batch, event_index, truth=False) for batch in batches]
    normalized = source_category.lower().replace("_", "").replace("-", "")
    continuum = _is_continuum_category(source_category)
    roots = [] if continuum else _truth_b_roots(truth)
    expected_bb = len(roots) == 2 or normalized in {"charged", "mixed", "generic", "bbbar", "bbar", "genericbbbar"}
    sample_kind = "continuum" if continuum else "bbbar" if expected_bb else "other_or_unknown"
    roots = sorted(roots, key=lambda root: (tuple(sorted(truth.source_set(root))), root)) if len(roots) <= 2 else []
    selected = [_selected_b_roots(view) for view in predictions]
    signatures = [[_safe_signature(view, root) for root in nodes]
                  for view, nodes in zip(predictions, selected)]
    truth_signatures = [_safe_signature(truth, root, require_truth=True) for root in roots]
    successes = [[bool(signature is not None and signature in candidates) for signature in truth_signatures]
                 for candidates in signatures]
    known = sum(signature is not None for signature in truth_signatures)
    units = []
    for index, root in enumerate(roots):
        channel = _channel(truth, root, event)
        units.append({
            "truth_root_position": root, "truth_known": truth_signatures[index] is not None,
            "truth_source_count": len(truth.source_set(root)), "channel": channel,
            "top1_correct": successes[0][index],
            "first_correct_rank": next((rank for rank, correct in enumerate(successes, 1) if correct[index]), None),
            "pool_at_k_correct": {str(k): any(correct[index] for correct in successes[:k]) for k in ks},
        })
    if expected_bb and not continuum:
        # Missing retained B roots are unknown targets, not measured failures.
        # Keep their channel rows so the coverage table reconciles with 2N.
        while len(units) < 2:
            units.append({"truth_root_position": None, "truth_known": False,
                "truth_source_count": None, "channel": {
                    "b_side": None, "full_truth_channel_id": None,
                    "reconstructable_channel_id": None, "stored_channel_id": None,
                    "stored_channel_ids_available": False, "retained_signed_channel_id": None,
                    "retained_signed_pid_signature": None, "retained_decay_description": None,
                    "retained_pid_signature_available": False,
                    "signature_scope": "unavailable_missing_or_ambiguous_B_root",
                    "full_generator_decay_description_available": False},
                "top1_correct": False, "first_correct_rank": None,
                "pool_at_k_correct": {str(k): False for k in ks}})

    def b_metrics(correct):
        count = sum(correct)
        return {"per_b_correct": _ratio(count, 2), "event_any_correct": _ratio(int(count > 0), 1),
                "event_both_correct": _ratio(int(count == 2), 1), "known_truth_trials": known,
                "unknown_truth_trials": 2 - known, "known_failed_trials": known - count,
                "is_lower_bound_due_to_unknown_truth": known != 2}

    pool = {}
    for k in ks:
        pool_correct = [any(candidate[index] for candidate in successes[:k]) for index in range(len(roots))]
        item = {"requested_k": k, "evaluated_hypothesis_count": min(k, len(batches)),
                "oracle_uses_truth": True, "oracle_is_deployable_performance": False,
                "any_b_event_acceptance": _ratio(int(any(selected[:k])), 1),
                "unique_selected_b_candidate_count": len({signature for candidate in signatures[:k] for signature in candidate if signature is not None}),
                "two_slot_acceptance": {"value": None, "reason": "slots_not_identified_across_competing_hypotheses"}}
        if expected_bb and not continuum:
            item["b_reconstruction"] = b_metrics(pool_correct)
            item["b_reconstruction"]["event_both_correct"]["meaning"] = "both_truth_Bs_recovered_possibly_in_different_hypotheses"
            item["b_reconstruction"]["coherent_event_both_correct"] = _ratio(int(any(sum(correct) == 2 for correct in successes[:k])), 1)
        pool[str(k)] = item

    # Parton truth is absent in the reduced ontology. Keep observed retained
    # component reconstruction separate from q/anti-q reconstruction.
    continuum_record = None
    if continuum:
        components = [root for root in _root_positions(truth) if truth.children(root)]
        component_signatures = [_safe_signature(truth, root, require_truth=True) for root in components]
        predicted_components = [[_safe_signature(view, root) for root in _root_positions(view) if view.children(root)] for view in predictions]
        component_known = sum(signature is not None for signature in component_signatures)
        def component_count(k):
            return sum(signature is not None and any(signature in candidates for candidates in predicted_components[:k]) for signature in component_signatures)
        continuum_record = {
            "type": source_category, "q_reconstruction": {"value": None, "numerator": None, "denominator": None,
                "available": False, "reason": "schema_has_no_quark_roots_or_parton_ancestry"},
            "retained_component_count": len(components), "known_retained_component_count": component_known,
            "unknown_retained_component_count": len(components) - component_known,
            "is_lower_bound_due_to_unknown_truth": component_known != len(components),
            "retained_component_top1_correct": _ratio(component_count(1), len(components)),
            "retained_component_pool_at_k_correct": {str(k): _ratio(component_count(k), len(components)) for k in ks},
            "retained_component_scope": "explicit_top_level_composites_not_quark_hemispheres",
            "top1_fake_b_slot_acceptance": _ratio(len(selected[0]), 2),
            "top1_fake_b_event_acceptance": _ratio(int(bool(selected[0])), 1),
            "top1_fake_b_event_rejection": _ratio(int(not selected[0]), 1),
        }
    record = {
        "version": VERSION, "source_category": source_category, "sample_kind": sample_kind,
        "event_count": 1, "tag_trial_count": 2, "hypothesis_count": len(batches),
        "truth_used_for_inference": False,
        "correctness": "exact_detector_sources_and_unordered_retained_tree_with_all_reduced_PID_tokens",
        "physical_fei_comparison_ready": False,
        "physical_fei_comparison_limitations": list(_LIMITATIONS),
        "slot_policy": _SLOT_POLICY,
        "top1": {"selected_b_candidate_count": len(selected[0]), "tag_slot_acceptance": _ratio(len(selected[0]), 2),
                 "tag_event_acceptance": _ratio(int(bool(selected[0])), 1)},
        "pool_at_k": pool, "b_units": units, "continuum": continuum_record,
    }
    if expected_bb and not continuum:
        record["top1"]["b_reconstruction"] = b_metrics(successes[0])
    validate_tag_efficiency_event(record)
    return record


def _sum_ratios(values):
    values = list(values)
    return _ratio(sum(value["numerator"] for value in values), sum(value["denominator"] for value in values))


def _summary(records):
    rows = list(records)
    bb = [row for row in rows if row["sample_kind"] == "bbbar"]
    keys = sorted(set(key for row in rows for key in row["pool_at_k"]), key=int)
    if any(set(row["pool_at_k"]) != set(keys) for row in rows):
        raise ValueError("tag summaries require identical requested K values")
    summary = {"event_count": len(rows), "tag_trial_count": 2 * len(rows),
               "bbbar_event_count": len(bb), "bbbar_trial_count": 2 * len(bb),
               "sample_kind_counts": dict(sorted(Counter(row["sample_kind"] for row in rows).items())),
               "top1": {}, "pool_at_k": {}}
    for name in ("tag_slot_acceptance", "tag_event_acceptance"):
        summary["top1"][name] = _sum_ratios(row["top1"][name] for row in rows)
    def reconstruction(items):
        items = list(items)
        result = {name: _sum_ratios(item[name] for item in items) for name in ("per_b_correct", "event_any_correct", "event_both_correct")}
        for name in ("known_truth_trials", "unknown_truth_trials", "known_failed_trials"):
            result[name] = sum(item[name] for item in items)
        result["is_lower_bound_due_to_unknown_truth"] = result["unknown_truth_trials"] > 0
        return result
    summary["top1"]["b_reconstruction"] = reconstruction(row["top1"]["b_reconstruction"] for row in bb)
    for key in keys:
        items = [row["pool_at_k"][key] for row in rows]
        b_items = [row["pool_at_k"][key]["b_reconstruction"] for row in bb]
        result = reconstruction(b_items)
        result["coherent_event_both_correct"] = _sum_ratios(item["coherent_event_both_correct"] for item in b_items)
        summary["pool_at_k"][key] = {"b_reconstruction": result,
            "any_b_event_acceptance": _sum_ratios(item["any_b_event_acceptance"] for item in items),
            "unique_selected_b_candidate_count": sum(item["unique_selected_b_candidate_count"] for item in items),
            "two_slot_acceptance": {"value": None, "reason": "slots_not_identified_across_competing_hypotheses"}}
    return summary


def summarize_tag_efficiency_events(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum fixed-trial counts, retaining zero-success channels and categories."""
    rows = list(records)
    for row in rows:
        validate_tag_efficiency_event(row)
    categories = defaultdict(list)
    channels = {}
    continuum_types = defaultdict(list)
    for row in rows:
        categories[row["source_category"]].append(row)
        if row["continuum"] is not None:
            continuum_types[row["source_category"]].append(row["continuum"])
        for unit in row["b_units"]:
            channel = unit["channel"]
            if channel is None:
                continue
            key = json.dumps([channel["full_truth_channel_id"], channel["reconstructable_channel_id"], channel["stored_channel_id"], channel["retained_signed_channel_id"], channel["signature_scope"]])
            if key not in channels:
                channels[key] = {**channel, "evaluated_b_trials": 0, "top1_correct_count": 0,
                                 "unknown_truth_trials": 0,
                                 "pool_at_k_correct_count": {k: 0 for k in row["pool_at_k"]}}
                channels[key].pop("b_side")
            result = channels[key]
            result["evaluated_b_trials"] += 1
            result["unknown_truth_trials"] += int(not unit["truth_known"])
            result["top1_correct_count"] += int(unit["top1_correct"])
            for k, correct in unit["pool_at_k_correct"].items():
                result["pool_at_k_correct_count"][k] += int(correct)
    for channel in channels.values():
        denominator = channel["evaluated_b_trials"]
        channel["top1_efficiency"] = _ratio(channel["top1_correct_count"], denominator)
        channel["covered_top1"] = channel["top1_correct_count"] > 0
        channel["is_lower_bound_due_to_unknown_truth"] = channel["unknown_truth_trials"] > 0
        channel["unavailable_count"] = channel["unknown_truth_trials"]
        channel["known_failed_top1_count"] = denominator - channel["unknown_truth_trials"] - channel["top1_correct_count"]
        channel["efficiency_definition"] = "proven_exact_retained_B_successes_over_all_evaluated_B_trials_in_channel"
        channel["pool_at_k_efficiency"] = {key: _ratio(value, denominator) for key, value in channel["pool_at_k_correct_count"].items()}
        channel["covered_pool_at_k"] = {key: value > 0 for key, value in channel["pool_at_k_correct_count"].items()}
    types = {}
    for category, records in sorted(continuum_types.items()):
        keys = records[0]["retained_component_pool_at_k_correct"]
        types[category] = {"event_count": len(records), "tag_trial_count": 2 * len(records),
            "q_reconstruction": records[0]["q_reconstruction"],
            "retained_component_count": sum(record["retained_component_count"] for record in records),
            "unknown_retained_component_count": sum(record["unknown_retained_component_count"] for record in records),
            "retained_component_pool_at_k_correct": {key: _sum_ratios(record["retained_component_pool_at_k_correct"][key] for record in records) for key in keys}}
        for name in ("retained_component_top1_correct", "top1_fake_b_slot_acceptance", "top1_fake_b_event_acceptance", "top1_fake_b_event_rejection"):
            types[category][name] = _sum_ratios(record[name] for record in records)
        types[category]["covered_retained_component_top1"] = types[category]["retained_component_top1_correct"]["numerator"] > 0
        types[category]["covered_retained_component_pool_at_k"] = {key: value["numerator"] > 0 for key, value in types[category]["retained_component_pool_at_k_correct"].items()}
        types[category]["is_lower_bound_due_to_unknown_truth"] = types[category]["unknown_retained_component_count"] > 0
    result = {"version": VERSION, "truth_used_for_inference": False,
        "physical_fei_comparison_ready": False, "physical_fei_comparison_limitations": list(_LIMITATIONS),
        "slot_policy": _SLOT_POLICY, "correctness": "exact_detector_sources_and_unordered_retained_tree_with_all_reduced_PID_tokens",
        "summary": _summary(rows),
        "by_source_category": {key: _summary(value) for key, value in sorted(categories.items())},
        "b_channel_coverage": sorted(channels.values(), key=lambda channel: (channel["full_truth_channel_id"] or 0, channel["retained_signed_channel_id"] or 0)),
        "unknown_b_truth_trials": sum(row["top1"].get("b_reconstruction", {}).get("unknown_truth_trials", 0) for row in rows),
        "continuum_type_coverage": types}
    if sum(channel["evaluated_b_trials"] for channel in channels.values()) != result["summary"]["bbbar_trial_count"]:
        raise ValueError("channel coverage does not account for every B trial")
    json.dumps(result, allow_nan=False)
    return result


def validate_tag_efficiency_event(record: Mapping[str, Any]) -> bool:
    """Validate finite ratios and fixed-trial accounting before serialization."""
    if record.get("version") != VERSION or record.get("tag_trial_count") != 2:
        raise ValueError("invalid two-trial tag evaluation contract")
    if record.get("truth_used_for_inference") is not False or record.get("physical_fei_comparison_ready") is not False:
        raise ValueError("invalid tag truth boundary or FEI comparability claim")
    def check(value):
        if isinstance(value, Mapping):
            if {"numerator", "denominator", "value"} <= value.keys() and value["denominator"] is not None:
                numerator, denominator = value["numerator"], value["denominator"]
                if not 0 <= numerator <= denominator:
                    raise ValueError("tag ratio outside its denominator")
                expected = numerator / denominator if denominator else None
                if value["value"] != expected:
                    raise ValueError("tag ratio value inconsistent with counts")
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
    check(record)
    for metrics in [record["top1"], *record["pool_at_k"].values()]:
        b = metrics.get("b_reconstruction")
        if b is not None and (b["known_truth_trials"] + b["unknown_truth_trials"] != 2
                              or b["known_failed_trials"] + b["per_b_correct"]["numerator"] != b["known_truth_trials"]):
            raise ValueError("B truth coverage accounting mismatch")
    json.dumps(record, allow_nan=False)
    return True


def validate_tag_efficiency_report(report: Mapping[str, Any]) -> bool:
    """Require every emitted method/event to appear in the matching summary."""
    contract = report.get("tag_efficiency", {})
    if contract.get("version") != "tag-efficiency-study-v1" or contract.get("replaces_preregistered_metrics") is not False:
        raise ValueError("missing tag-efficiency study contract")
    grouped = defaultdict(list)
    for event in report.get("events", []):
        for scope, record in event["scopes"].items():
            metrics = record.get("tag_efficiency", {})
            if "greedy" not in metrics:
                raise ValueError("event/scope lacks greedy tag-efficiency coverage")
            if "beam" in record and "full_depth_beam" not in metrics:
                raise ValueError("event/scope lacks full-depth beam tag-efficiency coverage")
            for method, evaluation in metrics.items():
                validate_tag_efficiency_event(evaluation)
                grouped[f"{scope}/{method}"].append(evaluation)
    expected = {key: summarize_tag_efficiency_events(value) for key, value in sorted(grouped.items())}
    if expected != contract.get("summaries"):
        raise ValueError("tag-efficiency summaries do not match emitted event records")
    return True


__all__ = ["evaluate_tag_efficiency_event", "summarize_tag_efficiency_events", "validate_tag_efficiency_event", "validate_tag_efficiency_report"]
