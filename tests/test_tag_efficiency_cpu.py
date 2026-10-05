"""Two physical-B trials, exact retained matching, and beam accounting."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from hypertagging.evaluation.tag_efficiency import (
    evaluate_tag_efficiency_event, summarize_tag_efficiency_events,
    validate_tag_efficiency_event, validate_tag_efficiency_report,
)
from tests.test_full_decay_metrics_cpu import _full_tree, _clone, _missing


def _evaluate(truth, candidates, **kwargs):
    return evaluate_tag_efficiency_event(truth, candidates, source_category="mixed", **kwargs)


def test_two_successes_count_two_trials_not_two_events():
    truth = _full_tree()
    record = _evaluate(truth, [truth])
    metrics = record["top1"]["b_reconstruction"]
    assert metrics["per_b_correct"] == {"numerator": 2., "denominator": 2., "value": 1.}
    assert metrics["event_any_correct"]["denominator"] == 1
    assert metrics["event_both_correct"]["numerator"] == 1
    assert record["physical_fei_comparison_ready"] is False
    assert len(record["b_units"]) == 2
    json.dumps(record, allow_nan=False)


def test_one_missing_B_keeps_fixed_denominator_and_event_efficiency():
    truth = _full_tree()
    record = _evaluate(truth, [_missing(truth, {5})])
    metrics = record["top1"]["b_reconstruction"]
    assert metrics["per_b_correct"]["value"] == .5
    assert metrics["event_any_correct"]["value"] == 1
    assert metrics["event_both_correct"]["value"] == 0
    assert metrics["known_failed_trials"] == 1


def test_beam_distinct_Bs_in_different_hypotheses_not_coherent_pair():
    truth = _full_tree()
    left = _missing(truth, {5})
    right = _clone(truth)
    right["node_mask"][0, [4, 8]] = False
    right["daughter_adjacency"][0, 10].zero_()
    right["daughter_adjacency"][0, 10, [6, 9]] = True
    record = _evaluate(truth, [left, right, right], oracle_ks=[1, 2, 4])
    assert record["top1"]["b_reconstruction"]["per_b_correct"]["value"] == .5
    pool = record["pool_at_k"]["2"]["b_reconstruction"]
    assert pool["per_b_correct"]["value"] == 1
    assert pool["event_both_correct"]["value"] == 1
    assert pool["coherent_event_both_correct"]["value"] == 0
    assert record["pool_at_k"]["4"]["evaluated_hypothesis_count"] == 3
    assert record["pool_at_k"]["4"]["b_reconstruction"]["per_b_correct"]["numerator"] == 2
    assert record["pool_at_k"]["4"]["unique_selected_b_candidate_count"] == record["pool_at_k"]["2"]["unique_selected_b_candidate_count"]


@pytest.mark.parametrize("node", [0, 6, 8])
def test_wrong_leaf_mother_or_root_pid_fails_exact_tag(node):
    truth = _full_tree()
    predicted = _clone(truth)
    predicted["current_pid_tokens"][0, node] = 22 if node == 8 else 10
    assert _evaluate(truth, [predicted])["top1"]["b_reconstruction"]["per_b_correct"]["value"] == .5


def test_missing_truth_pid_is_unknown_not_measured_failure():
    truth = _full_tree()
    truth["truth_pid_available"][0, 0] = False
    metrics = _evaluate(truth, [truth])["top1"]["b_reconstruction"]
    assert metrics["unknown_truth_trials"] == 1
    assert metrics["known_failed_trials"] == 0
    assert metrics["per_b_correct"]["denominator"] == 2
    assert metrics["is_lower_bound_due_to_unknown_truth"]


def test_missing_B_roots_preserves_two_unknown_trials():
    truth = _full_tree()
    truth["truth_pid_labels"][0, [8, 9]] = 4
    record = _evaluate(truth, [truth])
    assert record["top1"]["b_reconstruction"]["unknown_truth_trials"] == 2
    assert record["top1"]["b_reconstruction"]["known_failed_trials"] == 0
    assert record["top1"]["b_reconstruction"]["per_b_correct"]["denominator"] == 2


def test_training_policy_does_not_remove_B_trials():
    truth = _full_tree()
    truth["valid_reconstruction_target"] = torch.zeros_like(truth["node_mask"])
    truth["recursive_reconstructable_complete"] = torch.zeros_like(truth["node_mask"])
    result = _evaluate(truth, [truth])
    assert result["top1"]["b_reconstruction"]["per_b_correct"]["numerator"] == 2


def test_channel_provenance_uses_b_side_not_canonical_source_order():
    truth = _full_tree()
    truth["b_side"] = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, -1]])
    event = SimpleNamespace(b1_full_truth_channel_id=111, b2_full_truth_channel_id=222,
                            b1_reconstructable_channel_id=11, b2_reconstructable_channel_id=22)
    record = _evaluate(truth, [_missing(truth, {5})], event=event)
    assert [unit["channel"]["full_truth_channel_id"] for unit in record["b_units"]] == [222, 111]
    summary = summarize_tag_efficiency_events([record, record])
    channels = {item["full_truth_channel_id"]: item for item in summary["b_channel_coverage"]}
    assert channels[222]["evaluated_b_trials"] == 2
    assert channels[222]["top1_correct_count"] == 2
    assert channels[111]["top1_correct_count"] == 0
    assert not channels[111]["covered_top1"]
    assert "anti-B0" in channels[111]["retained_decay_description"]


def test_unknown_channels_and_truth_coverage_reconcile_all_B_trials():
    truth = _full_tree()
    truth["truth_pid_labels"][0, [8, 9]] = 4
    record = _evaluate(truth, [truth])
    summary = summarize_tag_efficiency_events([record])
    assert len(record["b_units"]) == 2
    channels = summary["b_channel_coverage"]
    assert sum(channel["evaluated_b_trials"] for channel in channels) == 2
    assert sum(channel["unavailable_count"] for channel in channels) == 2
    assert channels[0]["is_lower_bound_due_to_unknown_truth"]
    assert channels[0]["known_failed_top1_count"] == 0


def test_unknown_pid_preserves_available_channel_provenance_and_denominator():
    truth = _full_tree()
    truth["b_side"] = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, -1]])
    truth["truth_pid_available"][0, 0] = False
    event = SimpleNamespace(b1_full_truth_channel_id=42)
    summary = summarize_tag_efficiency_events([_evaluate(truth, [truth], event=event)])
    channel = next(row for row in summary["b_channel_coverage"] if row["full_truth_channel_id"] == 42)
    assert channel["evaluated_b_trials"] == channel["unavailable_count"] == 1
    assert channel["retained_signed_pid_signature"] is None
    assert channel["stored_channel_ids_available"]


def test_channel_groups_preserve_distinct_stored_channel_provenance():
    truth = _full_tree()
    truth["b_side"] = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, -1]])
    records = [_evaluate(truth, [truth], event=SimpleNamespace(b1_channel_id=channel_id)) for channel_id in (7, 8)]
    summary = summarize_tag_efficiency_events(records)
    assert {channel["stored_channel_id"] for channel in summary["b_channel_coverage"]} == {None, 7, 8}
    assert sum(channel["evaluated_b_trials"] for channel in summary["b_channel_coverage"]) == 4
    assert all(channel["covered_pool_at_k"]["1"] for channel in summary["b_channel_coverage"])


def test_continuum_has_fake_B_acceptance_and_no_invented_q_truth():
    truth = _full_tree()
    record = evaluate_tag_efficiency_event(truth, [truth, truth], source_category="ccbar")
    assert "b_reconstruction" not in record["top1"]
    assert record["tag_trial_count"] == 2
    cont = record["continuum"]
    assert cont["top1_fake_b_slot_acceptance"]["value"] == 1
    assert cont["q_reconstruction"]["denominator"] is None
    assert cont["retained_component_top1_correct"]["denominator"] == 1
    assert record["pool_at_k"]["2"]["two_slot_acceptance"]["value"] is None
    summary = summarize_tag_efficiency_events([record])
    assert summary["continuum_type_coverage"]["ccbar"]["tag_trial_count"] == 2
    assert summary["summary"]["bbbar_trial_count"] == 0


def test_continuum_no_B_rejection_and_zero_component_denominator():
    truth = _full_tree()
    leaf_only = _clone(truth)
    leaf_only["node_mask"][0, 6:] = False
    record = evaluate_tag_efficiency_event(leaf_only, [leaf_only], source_category="uubar")
    assert record["continuum"]["top1_fake_b_event_rejection"]["value"] == 1
    assert record["continuum"]["retained_component_top1_correct"]["value"] is None


def test_summary_micro_counts_and_validation_detects_tampering():
    truth = _full_tree()
    exact = _evaluate(truth, [truth])
    half = _evaluate(truth, [_missing(truth, {5})])
    summary = summarize_tag_efficiency_events([exact, half])
    assert summary["summary"]["top1"]["b_reconstruction"]["per_b_correct"]["value"] == .75
    report = {"events": [{"scopes": {"half": {"tag_efficiency": {"greedy": exact}}}}],
              "tag_efficiency": {"version": "tag-efficiency-study-v1", "replaces_preregistered_metrics": False,
                                 "summaries": {"half/greedy": summarize_tag_efficiency_events([exact])}}}
    assert validate_tag_efficiency_report(report)
    report["tag_efficiency"]["summaries"]["half/greedy"]["summary"]["event_count"] = 2
    with pytest.raises(ValueError, match="summaries"):
        validate_tag_efficiency_report(report)
    bad = copy.deepcopy(exact)
    bad["top1"]["b_reconstruction"]["per_b_correct"]["numerator"] = 3
    with pytest.raises(ValueError):
        validate_tag_efficiency_event(bad)


def test_hypothesis_inputs_unmodified_and_invalid_prefix_rejected():
    truth = _full_tree()
    predicted = _clone(truth)
    original = _clone(predicted)
    _evaluate(truth, [predicted])
    assert all(torch.equal(predicted[key], value) for key, value in original.items())
    with pytest.raises(ValueError, match="positive"):
        _evaluate(truth, [predicted], oracle_ks=[0])


def test_recovery_checks_later_correct_Bs_after_two_early_false_Bs():
    # Four disjoint B candidates in construction order: two false, then two
    # exact. Nominal acceptance has two slots; truth recovery checks all four.
    size = 16
    adjacency = torch.zeros((1, size, size), dtype=torch.bool)
    for root, leaves in {12: (6, 7, 8), 13: (9, 10, 11),
                         14: (0, 1, 2), 15: (3, 4, 5)}.items():
        adjacency[0, root, list(leaves)] = True
    pid = torch.tensor([[9] * 12 + [21, 38, 21, 38]])
    predicted = {"node_mask": torch.ones((1, size), dtype=torch.bool),
                 "daughter_adjacency": adjacency,
                 "current_pid_tokens": pid.clone(), "truth_pid_labels": pid.clone(),
                 "pid_labels": pid.clone(), "truth_pid_available": torch.ones((1, size), dtype=torch.bool),
                 "level_ids": torch.tensor([[0] * 12 + [1] * 4]),
                 "node_ids": torch.arange(size).reshape(1, size)}
    truth = _clone(predicted)
    truth["node_mask"][0, [12, 13]] = False
    record = _evaluate(truth, [predicted])
    top1 = record["top1"]
    assert top1["selected_b_candidate_count"] == 2
    assert top1["accepted_b_candidate_count"] == 4
    assert top1["b_reconstruction"]["per_b_correct"]["numerator"] == 2
    pool = record["pool_at_k"]["1"]
    assert pool["b_reconstruction"]["per_b_correct"]["value"] == 1
    assert pool["b_reconstruction"]["coherent_event_both_correct"]["value"] == 1
    assert pool["unique_accepted_b_candidate_count"] == 4
    summary = summarize_tag_efficiency_events([record])
    assert summary["summary"]["top1"]["accepted_b_candidate_count"] == 4


@pytest.mark.parametrize("category", ["", "unknown"])
def test_unknown_category_with_one_truth_B_is_incomplete_two_trial_BB(category):
    truth = _full_tree()
    truth["truth_pid_labels"][0, 9] = 4
    record = evaluate_tag_efficiency_event(truth, [truth], source_category=category)
    assert record["sample_kind"] == "bbbar"
    metrics = record["top1"]["b_reconstruction"]
    assert metrics["per_b_correct"]["denominator"] == 2
    assert metrics["per_b_correct"]["numerator"] == 1
    assert metrics["unknown_truth_trials"] == 1
    assert metrics["known_failed_trials"] == 0
    summary = summarize_tag_efficiency_events([record])
    assert sum(channel["evaluated_b_trials"] for channel in summary["b_channel_coverage"]) == 2
    assert sum(channel["unavailable_count"] for channel in summary["b_channel_coverage"]) == 1
