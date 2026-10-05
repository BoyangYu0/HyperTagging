"""Independent tests of inclusive FSP grouping, without PID/tree correctness."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from hypertagging.evaluation.tag_efficiency import (
    evaluate_tag_efficiency_event,
    summarize_tag_efficiency_events,
)
from tests.test_full_decay_metrics_cpu import _clone, _full_tree


def _evaluate(truth, candidates, **kwargs):
    return evaluate_tag_efficiency_event(
        truth, candidates, source_category="mixed", **kwargs
    )


def _flatten_groups(batch, groups):
    """Keep detector leaves, replacing all predicted mothers by explicit groups."""
    output = _clone(batch)
    output["node_mask"][0, 6:] = False
    output["daughter_adjacency"].zero_()
    output["recursive_leaf_source_mask"][0, 6:] = False
    for mother, daughters in groups.items():
        output["node_mask"][0, mother] = True
        output["daughter_adjacency"][0, mother, daughters] = True
        output["recursive_leaf_source_mask"][0, mother] = (
            output["recursive_leaf_source_mask"][0, daughters].any(dim=0)
        )
    return output


def test_inclusive_grouping_ignores_all_predicted_pid_and_intermediate_topology():
    truth = _full_tree()
    predicted = _flatten_groups(truth, {8: [0, 1, 4], 9: [2, 3, 5]})
    # Include wrong B-root and FSP labels: an already-built FSP group is the
    # inclusive object, so no predicted PID may gate its correctness.
    predicted["current_pid_tokens"].fill_(0)
    record = _evaluate(truth, [predicted])
    result = record["inclusive_fsp_grouping"]["top1"]["b_reconstruction"]
    assert result["per_b_correct"] == {
        "numerator": 2.0, "denominator": 2.0, "value": 1.0
    }
    assert result["known_truth_trials"] == 2


def test_merged_whole_event_does_not_count_as_two_inclusive_tags():
    truth = _full_tree()
    merged = _flatten_groups(truth, {10: list(range(6))})
    record = _evaluate(truth, [merged])
    assert record["inclusive_fsp_grouping"]["top1"]["b_reconstruction"]["per_b_correct"]["numerator"] == 0
    assert record["inclusive_fsp_grouping"]["pool_at_k"]["1"]["b_reconstruction"]["per_b_correct"]["denominator"] == 2


def test_contaminated_groups_are_not_repaired_by_truth_selected_leaf_subsets():
    truth = _full_tree()
    contaminated = _flatten_groups(truth, {8: [0, 1, 2, 4], 9: [3, 5]})
    result = _evaluate(truth, [contaminated])["inclusive_fsp_grouping"]["top1"]["b_reconstruction"]
    assert result["per_b_correct"]["numerator"] == 0
    assert result["known_failed_trials"] == 2


def test_inclusive_beam_counts_distinct_truth_groups_not_duplicate_hypotheses():
    truth = _full_tree()
    left_only = _flatten_groups(truth, {8: [0, 1, 4], 9: [2, 3]})
    right_only = _flatten_groups(truth, {8: [0, 1], 9: [2, 3, 5]})
    # Wrong PID does not invalidate an explicitly reconstructed grouping.
    for predicted in (left_only, right_only):
        predicted["current_pid_tokens"].fill_(10)
    record = _evaluate(
        truth, [left_only, right_only, right_only], oracle_ks=[1, 2, 3]
    )
    assert record["inclusive_fsp_grouping"]["top1"]["b_reconstruction"]["per_b_correct"]["value"] == 0.5
    for key in ("2", "3"):
        metrics = record["inclusive_fsp_grouping"]["pool_at_k"][key]["b_reconstruction"]
        assert metrics["per_b_correct"]["value"] == 1.0
        assert metrics["event_both_correct"]["value"] == 1.0
        assert metrics["coherent_event_both_correct"]["value"] == 0.0
    summary = summarize_tag_efficiency_events([record, record])
    counts = summary["inclusive_fsp_grouping"]["summary"]["pool_at_k"]["3"]["b_reconstruction"]
    assert counts["per_b_correct"]["numerator"] == 4
    assert counts["per_b_correct"]["denominator"] == 4
    assert counts["coherent_event_both_correct"]["numerator"] == 0


def test_truth_pid_unavailability_does_not_hide_known_group_or_channel_provenance():
    truth = _full_tree()
    truth["b_side"] = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, -1]])
    truth["truth_pid_available"].fill_(False)
    event = SimpleNamespace(
        b1_full_truth_channel_id=101, b2_full_truth_channel_id=202
    )
    record = _evaluate(truth, [truth], event=event)
    metrics = record["inclusive_fsp_grouping"]["top1"]["b_reconstruction"]
    assert metrics["known_truth_trials"] == 2
    assert metrics["unknown_truth_trials"] == 0
    assert metrics["per_b_correct"]["numerator"] == 2
    channel_by_root = {
        unit["truth_root_position"]: unit["channel"]["full_truth_channel_id"]
        for unit in record["inclusive_fsp_grouping"]["b_units"]
    }
    assert channel_by_root == {8: 202, 9: 101}
    summary = summarize_tag_efficiency_events([record])
    coverage = {row["full_truth_channel_id"]: row for row in summary["inclusive_fsp_grouping"]["b_channel_coverage"]}
    assert set(coverage) == {101, 202}
    for row in coverage.values():
        assert row["evaluated_b_trials"] == row["top1_correct_count"] == 1
        assert row["unavailable_count"] == 0
        assert row["retained_decay_description"] is None
        assert row["stored_channel_ids_available"]


def test_inclusive_continuum_component_recovery_does_not_invent_quark_targets():
    truth = _full_tree()
    predicted = _flatten_groups(truth, {10: list(range(6))})
    predicted["current_pid_tokens"].fill_(0)
    record = evaluate_tag_efficiency_event(
        truth, [predicted], source_category="ccbar"
    )
    continuum = record["inclusive_fsp_grouping"]["continuum"]
    assert continuum["retained_component_top1_correct"]["numerator"] == 1
    assert continuum["retained_component_top1_correct"]["denominator"] == 1
    assert continuum["q_inclusive_reconstruction"]["available"] is False
    assert continuum["q_inclusive_reconstruction"]["value"] is None
    summary = summarize_tag_efficiency_events([record])["inclusive_fsp_grouping"]
    assert summary["summary"]["bbbar_trial_count"] == 0
    assert summary["continuum_type_coverage"]["ccbar"]["q_inclusive_reconstruction"]["available"] is False
