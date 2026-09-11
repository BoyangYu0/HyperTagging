"""Additive, post-inference retained-tree coverage for greedy and beam output."""
from collections import defaultdict

from hypertagging.evaluation.full_decay_metrics import (
    evaluate_retained_decays, summarize_decay_evaluations,
)
from hypertagging.evaluation.beam_decay_metrics import (
    evaluate_ranked_decay_candidates, summarize_beam_decay_evaluations,
)


class RetainedTreeChecks:
    """Collect all scored units with explicit population and search labels."""

    def __init__(self, *, target_policy, minimum_daughters):
        self.target_policy = target_policy
        self.minimum_daughters = minimum_daughters
        self.rows = defaultdict(list)
        self.categories = defaultdict(lambda: defaultdict(list))
        self.shapes = defaultdict(lambda: defaultdict(list))
        self.beams = defaultdict(list)

    def evaluate(self, predicted, truth, *, scope, source_category):
        return evaluate_retained_decays(
            predicted, truth, scope=scope, source_category=source_category,
            target_policy=self.target_policy, minimum_daughters=self.minimum_daughters,
        )

    def add(self, label, evaluation, category):
        self.rows[label].append(evaluation)
        self.categories[label][category].append(evaluation)
        for unit in evaluation.rows:
            if unit.available:
                shape = f"fsp_count={len(unit.truth_sources)};retained_depth={unit.truth_retained_depth}"
                self.shapes[label][shape].append(unit)

    def add_beam(self, label, candidates, scores, oracle_ks):
        result = evaluate_ranked_decay_candidates(candidates, scores=scores, oracle_ks=oracle_ks)
        self.beams[label].append(result)
        return result

    def as_dict(self):
        return {
            "version": "retained-direct-tree-checks-v1",
            "population": "all_explicit_retained_roots_including_outside_training_policy",
            "truth_topology_mode": "checkpoint_direct",
            "training_target_policy": self.target_policy,
            "replaces_preregistered_metrics": False,
            "invented_truth_roots": False,
            "truth_used_for_inference": False,
            "summaries": {label: summarize_decay_evaluations(rows) for label, rows in sorted(self.rows.items())},
            "by_source_category": {label: {category: summarize_decay_evaluations(rows) for category, rows in sorted(groups.items())} for label, groups in sorted(self.categories.items())},
            "by_target_shape": {label: {shape: summarize_decay_evaluations(rows) for shape, rows in sorted(groups.items())} for label, groups in sorted(self.shapes.items())},
            "beam": {label: summarize_beam_decay_evaluations(rows) for label, rows in sorted(self.beams.items())},
        }


def validate_retained_tree_report(report):
    """Reject incomplete retained scoring, including unscored beam candidates."""
    checks = report.get('retained_tree_checks', {})
    if (checks.get('version') != 'retained-direct-tree-checks-v1'
            or checks.get('truth_used_for_inference') is not False
            or checks.get('replaces_preregistered_metrics') is not False):
        raise ValueError('Missing or invalid retained-tree evaluation contract')
    counts = defaultdict(lambda: [0, 0])

    def check_units(value):
        if value['available']:
            if not value['rows'] or not all(row['available'] for row in value['rows']):
                raise ValueError('Retained tree units were skipped')
            if value['unit_count'] != len(value['rows']):
                raise ValueError('Retained tree unit count mismatch')
        elif (value['unavailable_reason'] != 'no_retained_truth_roots'
              or value['rows'] or value['unit_count'] != 0):
            raise ValueError('Retained trees excluded for a training-policy reason')

    for event in report['events']:
        for scope, record in event['scopes'].items():
            result = record['retained_tree_metrics']
            check_units(result)
            counts[scope][0] += result['unit_count']
            counts[scope][1] += sum(row['available'] for row in result['rows'])
            if 'beam' in record:
                retained = record['retained_tree_beam']
                if (retained['candidate_count'] != record['beam']['candidate_count']
                        or len(retained['candidates']) != retained['candidate_count']):
                    raise ValueError('Full-depth beam candidates were not all checked')
                for candidate in retained['candidates']:
                    check_units(candidate['metrics'])
    for scope, (units, available) in counts.items():
        summary = checks['summaries'][f'{scope}/greedy']
        if summary['unit_count'] != units or summary['available_unit_count'] != available:
            raise ValueError('Retained summary does not cover every event unit')
    for event in report.get('beam_search', {}).get('events', []):
        if event['candidate_count'] != len(event['candidates']):
            raise ValueError('Diagnostic beam candidate count mismatch')
        for candidate in event['candidates']:
            metrics = candidate['retained_tree_metrics_by_scope']
            if set(metrics) != set(counts):
                raise ValueError('Diagnostic beam scope was skipped')
            for value in metrics.values():
                check_units(value)
    return True
