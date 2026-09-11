import torch

from test_full_decay_metrics_cpu import _full_tree, _clone, _missing
from hypertagging.evaluation.full_decay_metrics import evaluate_retained_decays, evaluate_half_decays
from hypertagging.evaluation.beam_decay_metrics import evaluate_ranked_decay_candidates
from hypertagging.evaluation.retained_tree_checks import RetainedTreeChecks


def test_ineligible_roots_are_checked_and_failed_not_excluded():
    truth = _full_tree()
    truth['valid_reconstruction_target'] = torch.ones_like(truth['node_mask'])
    truth['recursive_reconstructable_complete'] = torch.ones_like(truth['node_mask'])
    truth['valid_reconstruction_target'][0, 8] = False
    legacy = evaluate_half_decays(truth, truth)
    assert sum(row.available for row in legacy.rows) == 1
    retained = evaluate_retained_decays(truth, truth, scope='half')
    assert len(retained.rows) == 2 and all(row.available for row in retained.rows)
    assert sum(row.perfect_lcag.denominator for row in retained.rows) == 2
    excluded = next(row for row in retained.rows if row.truth_root_position == 8)
    assert excluded.source_recall.value == 1
    assert excluded.lcag_pair_accuracy.value == 1
    assert excluded.target_representable.value == 0
    assert excluded.perfectLCAG is False
    assert excluded.truth_mother_count == 2
    assert 'root_outside_training_target_policy' in excluded.target_unrepresentable_reasons


def test_missing_initial_root_checks_explicit_forest_without_inventing_root():
    truth = _full_tree()
    truth['node_mask'][0, 10] = False
    retained = evaluate_retained_decays(truth, truth)
    assert retained.unit_semantics == 'retained_full_forest'
    assert {row.truth_root_position for row in retained.rows} == {8, 9}
    assert retained.coherent_retained_forest.value == 1
    assert all(row.perfectLCAG for row in retained.rows)
    failed = evaluate_retained_decays(_missing(truth, {5}), truth)
    assert failed.coherent_retained_forest.value == 0
    beam = evaluate_ranked_decay_candidates([failed, retained], scores=[0., -1.], oracle_ks=[1, 2])
    assert beam.oracle_at_k[1]['coherent_event_perfect_lcag']['value'] == 0
    assert beam.oracle_at_k[2]['coherent_event_perfect_lcag']['value'] == 1


def test_singletons_are_checked_without_trivial_lcag_success():
    truth = _full_tree()
    truth['node_mask'][0, 6:] = False
    retained = evaluate_retained_decays(truth, truth)
    assert len(retained.rows) == 6
    assert all(row.source_recall.value == 1 for row in retained.rows)
    assert all(row.perfect_lcag.denominator == 0 for row in retained.rows)
    assert retained.coherent_retained_forest.value == 1
    failed = evaluate_retained_decays(_missing(truth, {5}), truth)
    assert failed.coherent_retained_forest.value == 0
    beam = evaluate_ranked_decay_candidates([failed, retained], scores=[0., -1.])
    assert beam.oracle_at_k[2]['coherent_event_perfect_lcag']['denominator'] == 1
    assert beam.oracle_at_k[2]['coherent_event_perfect_lcag']['value'] == 1
    assert beam.oracle_at_k[2]['perfect_lcag']['denominator'] == 0


def test_empty_truth_stays_unavailable_and_no_b_partition_is_invented():
    truth = _full_tree()
    empty = _clone(truth)
    empty['node_mask'].zero_()
    result = evaluate_retained_decays(truth, empty)
    assert not result.available
    assert result.unavailable_reason == 'no_retained_truth_roots'
    continuum = evaluate_retained_decays(truth, truth, scope='half', source_category='ccbar')
    assert continuum.unit_semantics == 'retained_explicit_components_no_b_partition'
    assert continuum.both_halves_perfect_lcag.denominator == 0


def test_unmatched_extra_component_prevents_coherent_forest_success():
    truth = _full_tree()
    truth['node_mask'][0, 5] = False
    predicted = _clone(truth)
    predicted['node_mask'][0, 5] = True
    predicted['daughter_adjacency'][0, :, 5] = False
    result = evaluate_retained_decays(predicted, truth)
    assert result.unassigned_predicted_component_root_count == 1
    assert result.coherent_retained_forest.value == 0


def test_accumulator_preserves_counts_categories_shapes_and_beam_candidates():
    truth = _full_tree()
    collector = RetainedTreeChecks(target_policy='complete_only', minimum_daughters=2)
    before = _clone(truth)
    exact = collector.evaluate(truth, truth, scope='full', source_category='signal')
    wrong = collector.evaluate(_missing(truth, {5}), truth, scope='full', source_category='signal')
    for row in [exact, wrong]:
        collector.add('full/greedy', row, 'signal')
    collector.add_beam('full/beam', [wrong, exact], [0., -1.], [1, 2])
    report = collector.as_dict()
    summary = report['summaries']['full/greedy']
    assert summary['source_recall']['numerator'] == 11
    assert summary['source_recall']['denominator'] == 12
    assert summary['coherent_retained_forest']['value'] == .5
    assert report['beam']['full/beam']['candidate_count'] == 2
    assert len(report['by_target_shape']['full/greedy']) == 1
    assert report['by_source_category']['full/greedy']['signal'] == summary
    assert all(torch.equal(truth[key], before[key]) for key in truth)


def test_report_validation_rejects_skipped_full_depth_and_proposal_candidates():
    import copy
    import pytest
    from hypertagging.evaluation.retained_tree_checks import validate_retained_tree_report
    truth = _full_tree()
    collector = RetainedTreeChecks(target_policy='complete_only', minimum_daughters=2)
    metrics = {}
    scopes = {}
    for scope in ('full', 'half'):
        result = collector.evaluate(truth, truth, scope=scope, source_category='signal')
        collector.add(f'{scope}/greedy', result, 'signal')
        metrics[scope] = result.as_dict()
        scopes[scope] = {'retained_tree_metrics': result.as_dict(),
            'beam': {'candidate_count': 1},
            'retained_tree_beam': {'candidate_count': 1, 'candidates': [{'metrics': result.as_dict()}]}}
    report = {'retained_tree_checks': collector.as_dict(), 'events': [{'scopes': scopes}],
        'beam_search': {'events': [{'candidate_count': 1, 'candidates': [{'retained_tree_metrics_by_scope': metrics}]}]}}
    assert validate_retained_tree_report(report)
    skipped = copy.deepcopy(report)
    skipped['events'][0]['scopes']['full']['retained_tree_beam']['candidates'].clear()
    with pytest.raises(ValueError, match='not all checked'):
        validate_retained_tree_report(skipped)
    skipped = copy.deepcopy(report)
    del skipped['beam_search']['events'][0]['candidates'][0]['retained_tree_metrics_by_scope']['half']
    with pytest.raises(ValueError, match='scope was skipped'):
        validate_retained_tree_report(skipped)
    skipped = copy.deepcopy(report)
    unit = skipped['events'][0]['scopes']['half']['retained_tree_metrics']
    unit['available'] = False
    unit['unavailable_reason'] = 'no_retained_truth_roots'
    with pytest.raises(ValueError, match='excluded'):
        validate_retained_tree_report(skipped)
