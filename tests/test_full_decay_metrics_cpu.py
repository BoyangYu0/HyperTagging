import json
import math

import pytest
import torch

from hypertagging.evaluation.full_decay_metrics import (
    _TreeView,
    _assign_two_by_sources,
    evaluate_full_decay,
    evaluate_half_decays,
    source_keyed_lcag,
    summarize_decay_evaluations,
)


def _full_tree() -> dict[str, torch.Tensor]:
    #       10(Upsilon)
    #       /         \
    #    8(B0)       9(anti-B0)
    #    /  \          /  \
    #   6    4        7    5
    #  / \           / \
    # 0   1         2   3
    count = 11
    adjacency = torch.zeros((1, count, count), dtype=torch.bool)
    for mother, daughters in {
        6: (0, 1),
        7: (2, 3),
        8: (6, 4),
        9: (7, 5),
        10: (8, 9),
    }.items():
        adjacency[0, mother, list(daughters)] = True
    sources = torch.zeros((1, count, 6), dtype=torch.bool)
    for leaf in range(6):
        sources[0, leaf, leaf] = True
    sources[0, 6, [0, 1]] = True
    sources[0, 7, [2, 3]] = True
    sources[0, 8, [0, 1, 4]] = True
    sources[0, 9, [2, 3, 5]] = True
    sources[0, 10] = True
    pid = torch.tensor([[8, 26, 9, 25, 2, 3, 4, 6, 21, 38, 1]])
    p4 = torch.zeros((1, count, 4), dtype=torch.float32)
    p4[0, :6, 3] = torch.arange(1, 7, dtype=torch.float32)
    for mother in (6, 7, 8, 9, 10):
        p4[0, mother] = p4[0, adjacency[0, mother]].sum(dim=0)
    levels = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3]])
    return {
        "node_mask": torch.ones((1, count), dtype=torch.bool),
        "daughter_adjacency": adjacency,
        "recursive_leaf_source_mask": sources,
        "pid_labels": pid.clone(),
        "pid_target_labels": pid.clone(),
        "truth_pid_labels": pid.clone(),
        "truth_pid_available": torch.ones((1, count), dtype=torch.bool),
        "current_pid_tokens": pid.clone(),
        "level_ids": levels,
        "p4": p4,
        "node_ids": torch.arange(count).reshape(1, count),
        "reco_ids": torch.tensor(
            [[100, 101, 102, 103, 104, 105, -1, -1, -1, -1, -1]]
        ),
        "source_node_ids": torch.arange(count).reshape(1, count),
    }


def _clone(batch):
    return {key: value.clone() for key, value in batch.items()}


def _permute(batch, permutation):
    result = _clone(batch)
    permutation = torch.tensor(permutation)
    count = len(permutation)
    for key, value in list(result.items()):
        if key == "daughter_adjacency":
            result[key] = value[:, permutation][:, :, permutation]
        elif value.ndim >= 2 and value.shape[1] == count:
            result[key] = value[:, permutation]
    result["node_ids"] = torch.tensor(
        [[900 + 17 * index for index in range(count)]], dtype=torch.long
    )
    return result


def _missing(batch, leaves):
    result = _clone(batch)
    if leaves == {5}:
        result["node_mask"][0, [5, 9]] = False
        result["daughter_adjacency"][0, 10].zero_()
        result["daughter_adjacency"][0, 10, [8, 7]] = True
        result["recursive_leaf_source_mask"][0, 10].zero_()
        result["recursive_leaf_source_mask"][0, 10, [0, 1, 2, 3, 4]] = True
    elif leaves == {4, 5}:
        result["node_mask"][0, [4, 5, 8, 9]] = False
        result["daughter_adjacency"][0, 10].zero_()
        result["daughter_adjacency"][0, 10, [6, 7]] = True
        result["recursive_leaf_source_mask"][0, 10].zero_()
        result["recursive_leaf_source_mask"][0, 10, [0, 1, 2, 3]] = True
    else:
        raise AssertionError("unsupported fixture")
    return result


def test_lcag_is_source_keyed_order_id_and_pid_invariant():
    truth = _full_tree()
    predicted = _permute(truth, [10, 3, 8, 0, 6, 5, 1, 9, 2, 7, 4])
    # Detector-conflict provenance may have a different raw source-column
    # order; LCAG membership is rebuilt from unique FSP keys and adjacency.
    predicted["recursive_leaf_source_mask"] = predicted[
        "recursive_leaf_source_mask"
    ][:, :, [4, 1, 5, 0, 3, 2]]
    # The FSP projection itself uses dense runtime IDs; this evaluation-only
    # leaf-order metadata is the bridge back to stable original reco IDs.
    predicted["reco_ids"] = torch.arange(11).reshape(1, 11)
    predicted["source_node_ids"] = torch.arange(11).reshape(1, 11)
    predicted["evaluation_leaf_source_keys"] = torch.tensor(
        [[103, 100, 105, 101, 102, 104]]
    )
    predicted["current_pid_tokens"].fill_(37)

    assert source_keyed_lcag(predicted) == source_keyed_lcag(truth)
    metrics = evaluate_full_decay(predicted, truth)
    assert metrics.perfectLCAG is True
    assert metrics.lcag_pair_accuracy.numerator == 15
    assert metrics.lcag_pair_accuracy.denominator == 15
    assert metrics.leaf_pid_accuracy.value == 0
    assert metrics.root_pid_accuracy.value == 0
    json.dumps(metrics.as_dict())


def test_strict_missing_one_leaf_distinguishes_one_from_two():
    truth = _full_tree()
    one = evaluate_full_decay(_missing(truth, {5}), truth)
    two = evaluate_full_decay(_missing(truth, {4, 5}), truth)

    assert one.perfectLCAG is False
    assert one.strict_missing_one_leaf.value == 1
    assert one.leave_one_out_lcag.value == 1
    assert two.strict_missing_one_leaf.value == 0
    assert two.leave_one_out_lcag.value == 0


def test_b_halves_are_unordered_and_count_as_exactly_two_units():
    truth = _full_tree()
    # Swapping every tensor position (including the two B positions) must not
    # turn source alignment into an ordered-side comparison.
    predicted = _permute(truth, [5, 2, 9, 7, 3, 10, 0, 8, 4, 1, 6])
    result = evaluate_half_decays(predicted, truth, source_category="signal")

    assert result.available
    assert len(result.rows) == 2
    assert all(row.perfectLCAG is True for row in result.rows)
    assert result.both_halves_perfect_lcag.value == 1
    summary = summarize_decay_evaluations([result])
    assert summary["unit_count"] == 2
    assert summary["perfect_lcag"]["denominator"] == 2


def test_continuum_top_level_components_are_multiplicity_units():
    truth = _full_tree()
    truth["node_mask"][0, 10] = False
    for name in (
        "pid_labels",
        "pid_target_labels",
        "truth_pid_labels",
        "current_pid_tokens",
    ):
        truth[name][0, 8] = 4
        truth[name][0, 9] = 6

    result = evaluate_half_decays(
        _clone(truth), truth, source_category="ccbar"
    )

    assert result.available
    assert result.unit_semantics == "continuum_components"
    assert len(result.rows) == 2
    assert all(row.scope == "continuum_component" for row in result.rows)
    assert all(row.perfectLCAG is True for row in result.rows)
    assert result.predicted_component_root_count == 2
    assert result.both_halves_perfect_lcag.denominator == 0


def test_continuum_without_explicit_composite_roots_is_unavailable():
    truth = _full_tree()
    truth["node_mask"][0, 6:] = False

    result = evaluate_half_decays(
        _clone(truth), truth, source_category="continuum"
    )

    assert not result.available
    assert result.rows == ()
    assert result.unit_semantics == "continuum_components"
    assert result.unavailable_reason == "no_explicit_truth_continuum_component_roots"


def test_continuum_counts_nonoverlapping_hallucinated_component_root():
    truth = _full_tree()
    truth["node_mask"][0, 8:] = False
    predicted = _clone(truth)
    predicted["node_mask"][0, 8] = True
    predicted["daughter_adjacency"][0, 8].zero_()
    predicted["daughter_adjacency"][0, 8, [4, 5]] = True
    predicted["recursive_leaf_source_mask"][0, 8].zero_()
    predicted["recursive_leaf_source_mask"][0, 8, [4, 5]] = True
    predicted["level_ids"][0, 8] = 1

    result = evaluate_half_decays(
        predicted, truth, source_category="ccbar"
    )

    assert len(result.rows) == 2
    assert result.predicted_component_root_count == 3
    assert result.assigned_predicted_component_count == 2
    assert result.unassigned_predicted_component_root_count == 1


def test_full_metric_does_not_invent_a_continuum_root():
    truth = _full_tree()
    truth["pid_labels"][0, 10] = 5
    truth["pid_target_labels"][0, 10] = 5
    truth["truth_pid_labels"][0, 10] = 5

    result = evaluate_full_decay(truth, truth)

    assert not result.available
    assert result.unavailable_reason == "truth_root_not_found"
    assert result.perfect_lcag.denominator == 0


def test_momentum_errors_use_source_alignment_and_have_denominators():
    truth = _full_tree()
    truth["p4"][0, 0, 0] = 2.0
    for mother in (6, 8, 10):
        truth["p4"][0, mother] = truth["p4"][
            0, truth["daughter_adjacency"][0, mother]
        ].sum(dim=0)
    predicted = _clone(truth)
    predicted["p4"][0, 10, 0] += 1.0
    metrics = evaluate_full_decay(predicted, truth)

    assert metrics.kinematics.alignment_coverage.value == 1
    # Five composite mothers are aligned; only the root was changed.
    assert metrics.kinematics.p3.numerator == pytest.approx(1.0)
    assert metrics.kinematics.p3.denominator == 5
    assert metrics.kinematics.p3.value == pytest.approx(0.2)
    assert metrics.kinematics.p4_l1.value == pytest.approx(1 / 20)
    assert metrics.kinematics.energy.value == 0
    assert metrics.kinematics.px_bias.value == pytest.approx(0.2)
    assert metrics.kinematics.px_mae.value == pytest.approx(0.2)
    assert math.sqrt(metrics.kinematics.px_squared_error.value) == pytest.approx(
        math.sqrt(1 / 5)
    )
    assert metrics.kinematics.relative_p3.numerator == pytest.approx(0.5)
    assert metrics.kinematics.relative_p3.denominator == 3
    expected_mass_sum = math.sqrt(21.0**2 - 2.0**2) - math.sqrt(
        21.0**2 - 3.0**2
    )
    assert metrics.kinematics.mass.numerator == pytest.approx(expected_mass_sum)


def test_missing_mothers_are_coverage_failures_not_pid_classification_trials():
    truth = _full_tree()
    metrics = evaluate_full_decay(_missing(truth, {5}), truth)

    assert metrics.mother_pid_accuracy.numerator == 3
    assert metrics.mother_pid_accuracy.denominator == 3
    assert metrics.mother_pid_coverage.numerator == 3
    assert metrics.mother_pid_coverage.denominator == 5
    assert metrics.kinematics.alignment_coverage.numerator == 3
    assert metrics.kinematics.alignment_coverage.denominator == 5
    # Errors themselves are conditioned only on the three aligned mothers.
    assert metrics.kinematics.p3.denominator == 3


def test_mother_pid_and_p4_alignment_requires_matching_local_topology():
    truth = _full_tree()
    predicted = _clone(truth)
    predicted["daughter_adjacency"][0, 6].zero_()
    predicted["daughter_adjacency"][0, 6, [0, 4]] = True
    predicted["daughter_adjacency"][0, 8].zero_()
    predicted["daughter_adjacency"][0, 8, [6, 1]] = True
    for mother in (6, 8, 10):
        predicted["p4"][0, mother] = predicted["p4"][
            0, predicted["daughter_adjacency"][0, mother]
        ].sum(dim=0)

    metrics = evaluate_full_decay(predicted, truth)

    assert metrics.perfectLCAG is False
    # Right inner/right B/root align.  The left inner and left B have either a
    # different clade or a different immediate child partition.
    assert metrics.matched_mother_count == 3
    assert metrics.mother_pid_accuracy.denominator == 3
    assert metrics.mother_pid_coverage.numerator == 3
    assert metrics.mother_pid_coverage.denominator == 5
    assert metrics.kinematics.alignment_coverage.numerator == 3


def test_half_assignment_preserves_an_exact_match_before_overlap_sum():
    keys = tuple(range(1, 9))

    def view(source_sets):
        sources = torch.zeros((len(source_sets), len(keys)), dtype=torch.bool)
        for row, values in enumerate(source_sets):
            sources[row, [value - 1 for value in values]] = True
        return _TreeView(
            active=torch.ones(len(source_sets), dtype=torch.bool),
            adjacency=torch.zeros(
                (len(source_sets), len(source_sets)), dtype=torch.bool
            ),
            sources=sources,
            source_keys=keys,
            pid=torch.zeros(len(source_sets), dtype=torch.long),
            truth_pid_available=None,
            p4=None,
        )

    truth = view(({1, 2, 3, 4}, {5, 6, 7, 8}))
    predicted = view(
        (
            {1, 2, 3, 4},
            {1, 2, 3, 5},
            {4, 6, 7, 8},
        )
    )

    assert _assign_two_by_sources(predicted, (0, 1, 2), truth, (0, 1)) == (
        0,
        None,
    )


def test_leaf_pid_uses_runtime_prediction_without_changing_topology():
    truth = _full_tree()
    predicted = _clone(truth)
    predicted["current_pid_tokens"][0, 2] = 17
    metrics = evaluate_full_decay(predicted, truth)

    assert metrics.perfectLCAG is True
    assert metrics.leaf_pid_accuracy.numerator == 5
    assert metrics.leaf_pid_accuracy.denominator == 6
    assert metrics.mother_pid_accuracy.value == 1


def test_unavailable_truth_leaf_pid_is_excluded_not_filled_from_input():
    truth = _full_tree()
    truth["truth_pid_available"][0, 0] = False

    metrics = evaluate_full_decay(_clone(truth), truth)

    assert metrics.leaf_pid_accuracy.numerator == 5
    assert metrics.leaf_pid_accuracy.denominator == 5
    assert metrics.as_dict()["leaf_pid_unavailable_count"] == 1


def test_perfect_lcag_requires_a_single_parent_tree():
    truth = _full_tree()
    predicted = _clone(truth)
    # Leaf zero is now a daughter of both node 6 and node 8.  Its recursive
    # clades and pairwise generations can still look unchanged, but this is a
    # DAG rather than a valid decay tree.
    predicted["daughter_adjacency"][0, 8, 0] = True

    metrics = evaluate_full_decay(predicted, truth)

    assert metrics.lcag_pair_accuracy.value == 1
    assert metrics.structurally_valid.value == 0
    assert metrics.perfectLCAG is False


def test_perfect_lcag_rejects_recursive_detector_source_reuse():
    truth = _full_tree()
    predicted = _clone(truth)
    predicted["recursive_leaf_source_mask"][0, 1, 0] = True

    metrics = evaluate_full_decay(predicted, truth)

    assert metrics.lcag_pair_accuracy.value == 1
    assert metrics.structurally_valid.value == 0
    assert metrics.perfectLCAG is False


def test_ineligible_unary_is_unrepresentable_for_checkpoint_direct_target():
    predicted = _full_tree()
    truth = _clone(predicted)
    count = truth["node_mask"].shape[1]
    truth["node_mask"] = torch.cat(
        (truth["node_mask"], torch.ones((1, 1), dtype=torch.bool)), dim=1
    )
    expanded = torch.zeros((1, count + 1, count + 1), dtype=torch.bool)
    expanded[:, :count, :count] = truth["daughter_adjacency"]
    expanded[0, 10, 9] = False
    expanded[0, 10, 11] = True
    expanded[0, 11, 9] = True
    truth["daughter_adjacency"] = expanded
    truth["recursive_leaf_source_mask"] = torch.cat(
        (
            truth["recursive_leaf_source_mask"],
            truth["recursive_leaf_source_mask"][:, 9:10],
        ),
        dim=1,
    )
    for name in (
        "pid_labels",
        "pid_target_labels",
        "truth_pid_labels",
        "current_pid_tokens",
    ):
        truth[name] = torch.cat(
            (truth[name], torch.tensor([[4]], dtype=torch.long)), dim=1
        )
    truth["truth_pid_available"] = torch.cat(
        (
            truth["truth_pid_available"],
            torch.ones((1, 1), dtype=torch.bool),
        ),
        dim=1,
    )
    truth["p4"] = torch.cat((truth["p4"], truth["p4"][:, 9:10]), dim=1)
    truth["level_ids"][0, 10] = 4
    truth["level_ids"] = torch.cat(
        (truth["level_ids"], torch.tensor([[3]], dtype=torch.long)), dim=1
    )
    truth["node_ids"] = torch.cat(
        (truth["node_ids"], torch.tensor([[11]], dtype=torch.long)), dim=1
    )
    truth["reco_ids"] = torch.cat(
        (truth["reco_ids"], torch.tensor([[-1]], dtype=torch.long)), dim=1
    )
    truth["source_node_ids"] = torch.cat(
        (truth["source_node_ids"], torch.tensor([[11]], dtype=torch.long)), dim=1
    )
    truth["valid_reconstruction_target"] = torch.ones(
        (1, count + 1), dtype=torch.bool
    )
    truth["valid_reconstruction_target"][0, 11] = False
    truth["recursive_reconstructable_complete"] = torch.ones(
        (1, count + 1), dtype=torch.bool
    )

    primary = evaluate_full_decay(
        predicted, truth, target_policy="complete_only"
    )
    diagnostic = evaluate_full_decay(
        predicted,
        truth,
        target_policy="complete_only",
        truth_topology_mode="contracted_diagnostic",
    )

    assert primary.available
    assert primary.target_representable.value == 0
    assert primary.target_unrepresentable_reasons == (
        "ineligible_direct_intermediate",
    )
    assert primary.perfectLCAG is False
    assert primary.truth_mother_count == 5
    assert diagnostic.available
    assert diagnostic.target_representable.value == 1
    assert diagnostic.perfectLCAG is True
    assert diagnostic.truth_mother_count == 5
    assert diagnostic.truth_topology_mode == "contracted_diagnostic"


def test_half_units_keep_multiplicity_when_one_b_target_is_ineligible():
    truth = _full_tree()
    truth["valid_reconstruction_target"] = torch.ones_like(
        truth["node_mask"]
    )
    truth["valid_reconstruction_target"][0, 8] = False
    truth["recursive_reconstructable_complete"] = torch.ones_like(
        truth["node_mask"]
    )

    result = evaluate_half_decays(
        _clone(truth), truth, source_category="signal"
    )

    assert len(result.rows) == 2
    assert not result.available
    assert sum(int(row.available) for row in result.rows) == 1
    assert result.both_halves_perfect_lcag.denominator == 0


def test_recursive_detector_source_conflict_changes_eligibility_not_topology_keys():
    truth = _full_tree()
    truth["recursive_leaf_source_mask"][0, 0, 1] = True

    full = evaluate_full_decay(_clone(truth), truth)
    halves = evaluate_half_decays(
        _clone(truth), truth, source_category="signal"
    )

    assert full.available
    assert full.target_representable.value == 0
    assert full.target_unrepresentable_reasons == (
        "recursive_detector_source_conflict",
    )
    assert full.perfectLCAG is False
    assert len(halves.rows) == 2
    assert all(row.available for row in halves.rows)
    assert sum(
        int(row.target_representable.value == 0) for row in halves.rows
    ) == 1
    assert halves.both_halves_perfect_lcag.value == 0


def test_single_leaf_trivial_tree_has_no_perfect_lcag_trial():
    adjacency = torch.zeros((1, 2, 2), dtype=torch.bool)
    adjacency[0, 1, 0] = True
    sources = torch.ones((1, 2, 1), dtype=torch.bool)
    pid = torch.tensor([[2, 1]], dtype=torch.long)
    batch = {
        "node_mask": torch.ones((1, 2), dtype=torch.bool),
        "daughter_adjacency": adjacency,
        "recursive_leaf_source_mask": sources,
        "pid_labels": pid,
        "pid_target_labels": pid,
        "truth_pid_labels": pid,
        "truth_pid_available": torch.ones((1, 2), dtype=torch.bool),
        "current_pid_tokens": pid,
        "level_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "p4": torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2]),
        "node_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "reco_ids": torch.tensor([[42, -1]], dtype=torch.long),
        "source_node_ids": torch.tensor([[0, 1]], dtype=torch.long),
    }

    result = evaluate_full_decay(
        batch,
        batch,
        target_policy="diagnostic_all",
        minimum_daughters=1,
    )

    assert result.available
    assert result.perfectLCAG is None
    assert result.perfect_lcag.denominator == 0
