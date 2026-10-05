"""Post-inference, two-trial tag accounting and retained-channel coverage.

These are exact retained-tree/source/reduced-PID diagnostics. They do not
implement Belle II MC matching and must not be labelled physical FEI efficiency.
No helper in this module is called while generating or ranking hypotheses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from typing import Any, Iterable, Mapping, Sequence

import torch

from hypertagging.evaluation.full_decay_metrics import (
    B_ROOT_TOKENS, RatioMetric, _component_structurally_valid,
    _is_continuum_category, _root_positions, _tree_view, _truth_b_roots,
)
from hypertagging.preprocessing.channels import deterministic_channel_id
from hypertagging.preprocessing.pid_filter import LEGACY_PARTICLE_NAMES, PDG_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID


VERSION = "two-trial-retained-tag-efficiency-v1"
_NAMES = dict(zip(PDG_TOKENS[1:], LEGACY_PARTICLE_NAMES))
_LIMITATIONS = ["retained_tree_and_reduced_PID_not_full_Belle_II_MC_matching", "no_common_FEI_candidate_selection_or_kinematic_working_point"]
_SLOT_POLICY = "first_two_source_disjoint_outermost_B_subtrees_in_model_construction_order_per_hypothesis"
_RECOVERY_POLICY = "all_qualifying_source_disjoint_outermost_B_subtrees_per_ranked_hypothesis_each_truth_B_at_most_once"
_INCLUSIVE_POLICY = "all_explicit_reconstructed_structurally_valid_composite_subtrees_any_PID_no_invented_unions_or_cuts"


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


def _accepted_b_roots(view) -> list[int]:
    """Return all qualifying disjoint B subtrees in a coherent hypothesis.

    Tensor positions are construction order, independent of truth alignment.
    Only outermost B subtrees qualify. Recovery uses the full accepted set;
    the separate nominal two-slot acceptance statistic uses its first two.
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
    explicit_b_present = any(int(truth.pid[node]) in B_ROOT_TOKENS for node in truth.positions)
    if not continuum and not roots and explicit_b_present:
        # Side labels may be incomplete when only one retained B survives.
        # Discover explicit trees independently, but never create hemispheres.
        roots = _truth_b_roots(replace(truth, b_side=None))
    if len(roots) == 2 and truth.source_set(roots[0]) & truth.source_set(roots[1]):
        roots = []
    expected_bb = explicit_b_present or normalized in {"charged", "mixed", "generic", "bbbar", "bbar", "genericbbbar"}
    expected_bb = expected_bb or (not continuum and truth.b_side is not None and bool((truth.b_side >= 0).any()))
    sample_kind = "continuum" if continuum else "bbbar" if expected_bb else "other_or_unknown"
    roots = sorted(roots, key=lambda root: (tuple(sorted(truth.source_set(root))), root)) if len(roots) <= 2 else []
    accepted = [_accepted_b_roots(view) for view in predictions]
    selected = [nodes[:2] for nodes in accepted]
    signatures = [[_safe_signature(view, root) for root in nodes]
                  for view, nodes in zip(predictions, accepted)]
    selected_signatures = [candidate[:2] for candidate in signatures]
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
                "unique_selected_b_candidate_count": len({signature for candidate in selected_signatures[:k] for signature in candidate if signature is not None}),
                "unique_accepted_b_candidate_count": len({signature for candidate in signatures[:k] for signature in candidate if signature is not None}),
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
        "recovery_candidate_policy": _RECOVERY_POLICY,
        "top1": {"selected_b_candidate_count": len(selected[0]), "accepted_b_candidate_count": len(accepted[0]), "tag_slot_acceptance": _ratio(len(selected[0]), 2),
                 "tag_event_acceptance": _ratio(int(bool(selected[0])), 1)},
        "pool_at_k": pool, "b_units": units, "continuum": continuum_record,
    }
    if expected_bb and not continuum:
        record["top1"]["b_reconstruction"] = b_metrics(successes[0])
    record["inclusive_fsp_grouping"] = _evaluate_inclusive(
        truth, predictions, batches, roots, source_category=source_category,
        sample_kind=sample_kind, event=event, event_index=event_index, ks=ks,
    )
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
    for name in ("selected_b_candidate_count", "accepted_b_candidate_count"):
        summary["top1"][name] = sum(row["top1"][name] for row in rows)
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
            "unique_accepted_b_candidate_count": sum(item["unique_accepted_b_candidate_count"] for item in items),
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
        "recovery_candidate_policy": _RECOVERY_POLICY,
        "summary": _summary(rows),
        "by_source_category": {key: _summary(value) for key, value in sorted(categories.items())},
        "b_channel_coverage": sorted(channels.values(), key=lambda channel: (channel["full_truth_channel_id"] or 0, channel["retained_signed_channel_id"] or 0)),
        "unknown_b_truth_trials": sum(row["top1"].get("b_reconstruction", {}).get("unknown_truth_trials", 0) for row in rows),
        "continuum_type_coverage": types}
    result["inclusive_fsp_grouping"] = _summarize_inclusive([row["inclusive_fsp_grouping"] for row in rows])
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
    inclusive = record.get("inclusive_fsp_grouping")
    if inclusive is None or inclusive.get("candidate_group_policy") != _INCLUSIVE_POLICY:
        raise ValueError("missing inclusive FSP assignment contract")
    for metrics in [inclusive["top1"], *inclusive["pool_at_k"].values()]:
        b = metrics.get("b_reconstruction")
        if b is not None and (b["known_truth_trials"] + b["unknown_truth_trials"] != 2
                              or b["known_failed_trials"] + b["per_b_correct"]["numerator"] != b["known_truth_trials"]):
            raise ValueError("inclusive B membership accounting mismatch")
    json.dumps(record, allow_nan=False)
    return True


def _unknown_channel(side=None, event=None):
    def identifier(name):
        value = getattr(event, f"b{side + 1}_{name}", 0) if side is not None and event is not None else 0
        return int(value) if value else None
    return {"b_side": side, "full_truth_channel_id": identifier("full_truth_channel_id"),
            "reconstructable_channel_id": identifier("reconstructable_channel_id"),
            "stored_channel_id": identifier("channel_id"),
            "retained_signed_channel_id": None, "retained_signed_pid_signature": None,
            "retained_decay_description": None, "retained_pid_signature_available": False,
            "signature_scope": "unavailable_PID_or_retained_root",
            "full_generator_decay_description_available": False,
            "stored_channel_ids_available": any(identifier(name) is not None for name in ("full_truth_channel_id", "reconstructable_channel_id", "channel_id"))}


def _inclusive_truth_units(truth, roots, event):
    """Membership availability is independent of every truth-PID mask."""
    units = []
    if truth.b_side is not None and bool((truth.b_side >= 0).any()):
        for side in (0, 1):
            nodes = {node for node in truth.positions if int(truth.b_side[node]) == side}
            side_roots = [node for node in nodes if not any(int(parent) in nodes for parent in (truth.adjacency[:, node] & truth.active).nonzero().flatten())]
            root = side_roots[0] if len(side_roots) == 1 else None
            sources = frozenset().union(*(truth.source_set(node) for node in nodes))
            units.append({"root": root, "sources": sources or None,
                          "channel": _channel(truth, root, event) if root is not None else _unknown_channel(side, event)})
    else:
        for root in roots:
            sources = truth.source_set(root) if _component_structurally_valid(truth, root) else None
            units.append({"root": root, "sources": sources or None, "channel": _channel(truth, root, event)})
    while len(units) < 2:
        units.append({"root": None, "sources": None, "channel": _unknown_channel()})
    if len(units) != 2:
        units = [{"root": None, "sources": None, "channel": _unknown_channel()} for _ in range(2)]
    if units[0]["sources"] and units[1]["sources"] and units[0]["sources"] & units[1]["sources"]:
        for unit in units:
            unit["sources"] = None
    return units


def _inclusive_groups(view, batch, event_index):
    """Use actual model-produced composite nodes, never truth-created cuts."""
    kinds = batch.get("node_kind_ids")
    if kinds is not None:
        kinds = kinds[event_index] if kinds.ndim == 2 else kinds
    groups = []
    for node in view.positions:
        if not view.children(node) or (kinds is not None and int(kinds[node]) != NODE_KIND_TO_ID["composite"]):
            continue
        if not _component_structurally_valid(view, node):
            continue
        sources = view.source_set(node)
        if not sources:
            continue
        detector = (frozenset(view.detector_sources[node].nonzero().flatten().tolist())
                    if view.detector_sources is not None else sources)
        groups.append({"root": node, "sources": sources, "detector_sources": detector})
    return groups


def _evaluate_inclusive(truth, predictions, batches, roots, *, source_category, sample_kind, event, event_index, ks):
    groups = [_inclusive_groups(view, batch, event_index) for view, batch in zip(predictions, batches)]
    units = _inclusive_truth_units(truth, roots, event) if sample_kind == "bbbar" else []
    successes = [[unit["sources"] is not None and any(group["sources"] == unit["sources"] for group in candidate) for unit in units] for candidate in groups]
    known = sum(unit["sources"] is not None for unit in units)
    def coherent(candidate):
        if known != 2:
            return False
        left = [group for group in candidate if group["sources"] == units[0]["sources"]]
        right = [group for group in candidate if group["sources"] == units[1]["sources"]]
        return any(not (a["sources"] & b["sources"] or a["detector_sources"] & b["detector_sources"]) for a in left for b in right)
    def metrics(correct, coherent_success):
        count = sum(correct)
        return {"per_b_correct": _ratio(count, 2), "event_any_correct": _ratio(int(count > 0), 1),
                "event_both_correct": _ratio(int(count == 2), 1),
                "coherent_event_both_correct": _ratio(int(coherent_success), 1),
                "known_truth_trials": known, "unknown_truth_trials": 2 - known,
                "known_failed_trials": known - count, "is_lower_bound_due_to_unknown_truth": known != 2}
    top1 = {"candidate_group_count": len(groups[0])}
    if units:
        top1["b_reconstruction"] = metrics(successes[0], coherent(groups[0]))
    pool = {}
    for k in ks:
        item = {"requested_k": k, "evaluated_hypothesis_count": min(k, len(groups)),
                "oracle_uses_truth": True, "oracle_is_deployable_performance": False,
                "unique_candidate_source_group_count": len({group["sources"] for candidate in groups[:k] for group in candidate})}
        if units:
            item["b_reconstruction"] = metrics([any(candidate[index] for candidate in successes[:k]) for index in range(2)], any(coherent(candidate) for candidate in groups[:k]))
        pool[str(k)] = item
    unit_rows = [{"truth_root_position": unit["root"], "truth_membership_known": unit["sources"] is not None,
                  "truth_source_count": len(unit["sources"]) if unit["sources"] is not None else None,
                  "channel": unit["channel"], "top1_correct": successes[0][index],
                  "first_correct_rank": next((rank for rank, candidate in enumerate(successes, 1) if candidate[index]), None),
                  "pool_at_k_correct": {str(k): any(candidate[index] for candidate in successes[:k]) for k in ks}}
                 for index, unit in enumerate(units)]
    continuum = None
    if sample_kind == "continuum":
        components = [root for root in _root_positions(truth) if truth.children(root)]
        source_sets = [truth.source_set(root) if _component_structurally_valid(truth, root) else None for root in components]
        def component_metric(k):
            return _ratio(sum(source is not None and any(group["sources"] == source for candidate in groups[:k] for group in candidate) for source in source_sets), len(source_sets))
        continuum = {"type": source_category, "q_inclusive_reconstruction": {
            "available": False, "value": None, "numerator": None, "denominator": None,
            "reason": "schema_has_no_quark_roots_or_parton_FSP_membership"},
            "retained_component_count": len(source_sets), "unknown_retained_component_count": sum(source is None for source in source_sets),
            "is_lower_bound_due_to_unknown_truth": any(source is None for source in source_sets),
            "retained_component_top1_correct": component_metric(1),
            "retained_component_pool_at_k_correct": {str(k): component_metric(k) for k in ks},
            "retained_component_scope": "explicit_top_level_truth_composites_not_quark_hemispheres"}
    return {"name": "graFEI-inspired FSP candidate assignment", "sample_kind": sample_kind,
            "source_category": source_category, "event_count": 1, "tag_trial_count": 2,
            "candidate_group_policy": _INCLUSIVE_POLICY,
            "correctness": "exact_FSP_source_set_equality_no_missing_extra_or_cross_side_FSPs_ignores_all_PID_and_internal_decay_topology",
            "single_input_FSPs_count_as_reconstructed_groups": False,
            "truth_used_for_inference": False, "is_delivered_partition_accuracy": False,
            "is_grafei_perfectEvent": False, "top1": top1, "pool_at_k": pool,
            "b_units": unit_rows, "continuum": continuum}


def _inclusive_summary(rows):
    bb = [row for row in rows if row["sample_kind"] == "bbbar"]
    keys = sorted({key for row in rows for key in row["pool_at_k"]}, key=int)
    def metrics(items):
        items = list(items)
        result = {name: _sum_ratios(item[name] for item in items) for name in ("per_b_correct", "event_any_correct", "event_both_correct", "coherent_event_both_correct")}
        for name in ("known_truth_trials", "unknown_truth_trials", "known_failed_trials"):
            result[name] = sum(item[name] for item in items)
        result["is_lower_bound_due_to_unknown_truth"] = result["unknown_truth_trials"] > 0
        return result
    return {"event_count": len(rows), "tag_trial_count": 2 * len(rows), "bbbar_event_count": len(bb), "bbbar_trial_count": 2 * len(bb),
            "top1": {"candidate_group_count": sum(row["top1"]["candidate_group_count"] for row in rows),
                     "b_reconstruction": metrics(row["top1"]["b_reconstruction"] for row in bb)},
            "pool_at_k": {key: {"b_reconstruction": metrics(row["pool_at_k"][key]["b_reconstruction"] for row in bb),
                                "unique_candidate_source_group_count": sum(row["pool_at_k"][key]["unique_candidate_source_group_count"] for row in rows)} for key in keys}}


def _summarize_inclusive(rows):
    categories, channels, continuum_types = defaultdict(list), {}, defaultdict(list)
    for row in rows:
        categories[row["source_category"]].append(row)
        if row["continuum"] is not None:
            continuum_types[row["source_category"]].append(row["continuum"])
        for unit in row["b_units"]:
            channel = unit["channel"]
            key = json.dumps([channel["full_truth_channel_id"], channel["reconstructable_channel_id"], channel["stored_channel_id"], channel["retained_signed_channel_id"], channel["signature_scope"]])
            if key not in channels:
                channels[key] = {**channel, "evaluated_b_trials": 0, "unavailable_count": 0,
                                 "top1_correct_count": 0, "pool_at_k_correct_count": {k: 0 for k in row["pool_at_k"]}}
                channels[key].pop("b_side")
            entry = channels[key]
            entry["evaluated_b_trials"] += 1
            entry["unavailable_count"] += int(not unit["truth_membership_known"])
            entry["top1_correct_count"] += int(unit["top1_correct"])
            for k, value in unit["pool_at_k_correct"].items():
                entry["pool_at_k_correct_count"][k] += int(value)
    for channel in channels.values():
        denominator = channel["evaluated_b_trials"]
        channel["top1_efficiency"] = _ratio(channel["top1_correct_count"], denominator)
        channel["pool_at_k_efficiency"] = {k: _ratio(count, denominator) for k, count in channel["pool_at_k_correct_count"].items()}
        channel["covered_top1"] = channel["top1_correct_count"] > 0
        channel["covered_pool_at_k"] = {k: count > 0 for k, count in channel["pool_at_k_correct_count"].items()}
        channel["is_lower_bound_due_to_unknown_truth"] = channel["unavailable_count"] > 0
        channel["known_failed_top1_count"] = denominator - channel["unavailable_count"] - channel["top1_correct_count"]
    types = {}
    for category, records in sorted(continuum_types.items()):
        keys = records[0]["retained_component_pool_at_k_correct"]
        entry = {"event_count": len(records), "tag_trial_count": 2 * len(records),
                 "q_inclusive_reconstruction": records[0]["q_inclusive_reconstruction"],
                 "retained_component_count": sum(record["retained_component_count"] for record in records),
                 "unknown_retained_component_count": sum(record["unknown_retained_component_count"] for record in records),
                 "retained_component_top1_correct": _sum_ratios(record["retained_component_top1_correct"] for record in records),
                 "retained_component_pool_at_k_correct": {k: _sum_ratios(record["retained_component_pool_at_k_correct"][k] for record in records) for k in keys}}
        entry["covered_retained_component_top1"] = entry["retained_component_top1_correct"]["numerator"] > 0
        entry["covered_retained_component_pool_at_k"] = {k: value["numerator"] > 0 for k, value in entry["retained_component_pool_at_k_correct"].items()}
        entry["is_lower_bound_due_to_unknown_truth"] = entry["unknown_retained_component_count"] > 0
        types[category] = entry
    result = {"name": "graFEI-inspired FSP candidate assignment", "candidate_group_policy": _INCLUSIVE_POLICY,
              "correctness": "exact_FSP_source_set_equality_no_missing_extra_or_cross_side_FSPs_ignores_all_PID_and_internal_decay_topology",
              "single_input_FSPs_count_as_reconstructed_groups": False,
              "truth_used_for_inference": False, "is_grafei_perfectEvent": False, "is_delivered_partition_accuracy": False,
              "summary": _inclusive_summary(rows), "by_source_category": {category: _inclusive_summary(records) for category, records in sorted(categories.items())},
              "b_channel_coverage": sorted(channels.values(), key=lambda channel: (channel["full_truth_channel_id"] or 0, channel["retained_signed_channel_id"] or 0)),
              "continuum_type_coverage": types}
    if sum(channel["evaluated_b_trials"] for channel in channels.values()) != result["summary"]["bbbar_trial_count"]:
        raise ValueError("inclusive channel coverage does not account for every B trial")
    return result


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
