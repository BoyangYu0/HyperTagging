"""Versioned Phase70 additive export from authenticated, prediction-identical CPU runs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from hypertagging.evaluation.tag_efficiency import (  # noqa: E402
    summarize_tag_efficiency_events, validate_tag_efficiency_report,
)
from scripts.export_phase72_native_metrics import leaves  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratios(value, prefix=()):
    if not isinstance(value, dict):
        return
    if {'numerator', 'denominator', 'value'} <= value.keys():
        if value['numerator'] is not None and value['denominator'] is not None:
            yield '.'.join(prefix), (value['numerator'], value['denominator'])
        return
    for key, item in value.items():
        yield from ratios(item, prefix + (key,))


def collision_intervals(left, right, seed=20261007, repeats=10000):
    """One shared resample index per collision, including both nominal B trials."""
    assert len(left) == len(right)
    n = len(left)
    if not n:
        return {}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(repeats, n))
    a, b = [list(map(lambda row: dict(ratios(row)), rows)) for rows in (left, right)]
    paths = sorted(set().union(*(row.keys() for row in a + b)))
    result = {}
    for path in paths:
        values, points, counts = [], [], []
        for rows in (a, b):
            data = np.array([row.get(path, (0, 0)) for row in rows], dtype=float)
            sums = data[indices].sum(axis=1)
            valid = sums[:, 1] > 0
            estimates = np.full(repeats, np.nan)
            estimates[valid] = sums[valid, 0] / sums[valid, 1]
            total = data.sum(axis=0)
            points.append(float(total[0] / total[1]) if total[1] else None)
            counts.append(total.tolist())
            values.append(estimates)
        def interval(x):
            x = x[np.isfinite(x)]
            return {'percentile_95': np.quantile(x, [.025, .975]).tolist() if len(x) else None,
                    'valid_resamples': len(x), 'degenerate': bool(len(x) and x.min() == x.max())}
        result[path] = {'set_overlap_off': {'numerator': counts[0][0], 'denominator': counts[0][1], 'value': points[0], **interval(values[0])},
                        'set_overlap_on': {'numerator': counts[1][0], 'denominator': counts[1][1], 'value': points[1], **interval(values[1])},
                        'paired_overlap_on_minus_off': {'value': points[1] - points[0] if None not in points else None, **interval(values[1] - values[0])}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir
    receipts = json.loads((evidence / 'authenticated-evaluation-receipts.json').read_text())
    expected_views = {'independent_complete_target_direct', 'independent_depth_direct',
                      'independent_tree_validity_direct', 'primary_complete_target_beam_direct',
                      'primary_complete_target_contracted_diagnostic', 'primary_complete_target_direct',
                      'primary_complete_target_repeat2_direct'}
    assert len(receipts) == 14
    assert {(r['arm'], r['view']) for r in receipts} == {
        (arm, view) for arm in ('set_overlap_off', 'set_overlap_on') for view in expected_views}

    public = {'version': 'phase72-additive-tag-efficiencies-v1',
              'evaluator_revision': '7f6b28bd62c28656fd2f6600492cbfbb2225d23b',
              'exporter_sha256': sha(Path(__file__)),
              'native_revision': '38124a9e05ca63cd5ac5469e0687952eec2b14dc',
              'preregistered_gates_replaced': False, 'sealed_test_accessed': False,
              'prediction_tensors_unchanged': True, 'original_scientific_counts_and_categories_unchanged': True, 'native_additive_scientific_metrics_exactly_identical': True, 'archived_floating_point_roundoff_present': True,
              'views': {}, 'input_bindings': [], 'candidate_rank_summaries': {},
              'uncertainty': {'unit': 'collision', 'paired_direction': 'overlap_on_minus_off',
                              'resamples': 10000, 'seed': 20261007, 'numpy_version': np.__version__,
                              'interpretation': 'exploratory_percentile_intervals_not_simultaneous_or_confirmatory; degenerate_zero_intervals_do_not_prove_absence; unknown_truth_remains_in_denominator', 'views': {}}}
    grouped = defaultdict(dict)
    scalar_count = 0
    tag_scalar_count = 0
    event_scope_count = 0
    candidate_count = 0
    with gzip.open(evidence / 'additive-all-native-scalars.jsonl.gz', 'xt') as output:
        for receipt in receipts:
            assert receipt['original_scientific_counts_and_categories_unchanged']
            assert receipt['native_additive_prediction_tensors_exactly_identical']
            assert receipt['native_additive_scientific_metrics_exactly_identical']
            assert all(any(word in diff['path'] for word in ('p4', 'mass', 'energy', 'momentum', '.model_only_ranking_scores.'))
                       for diff in receipt['archived_floating_point_differences'])
            arm, view = receipt['arm'], receipt['view']
            run = receipt['evaluations']['additive']
            path = Path(run['report'])
            assert sha(path) == run['report_sha256']
            report = json.loads(path.read_text())
            provenance = report['evaluator_code_provenance']
            assert provenance['git_head'] == public['evaluator_revision']
            assert provenance['provenance_complete'] and provenance['index_matches_worktree']
            assert provenance['worktree_dirty'] is False and provenance['untracked_path_count'] == 0
            assert report['device'] == 'cpu' and report['torch_num_threads'] == 1
            assert report['torch_deterministic_algorithms_enabled'] is True
            assert report['checkpoint_pair']['compatible'] is True
            validate_tag_efficiency_report(report)
            tag_scalar_count += sum(1 for _ in leaves(report['tag_efficiency']))
            tag_scalar_count += sum(sum(1 for _ in leaves(scope['tag_efficiency']))
                                    for event in report['events'] for scope in event['scopes'].values())
            public['views'].setdefault(view, {})[arm] = report['tag_efficiency']
            grouped[view][arm] = report
            command = run['command']
            bindings = {}
            for flag in ('--pretraining-checkpoint', '--reconstruction-checkpoint', '--data', '--dataset-index', '--event-uid-manifest'):
                bindings[flag[2:].replace('-', '_') + '_sha256'] = sha(Path(command[command.index(flag) + 1]))
            public['input_bindings'].append({'arm': arm, 'view': view, **bindings,
                                             'native_report_sha256': receipt['native_report_sha256'],
                                             'archived_floating_point_difference_count': len(receipt['archived_floating_point_differences']),
                                             'maximum_archived_absolute_difference': receipt['maximum_archived_absolute_difference'],
                                             'additive_report_sha256': sha(path),
                                             'prediction_tensor_sequence_sha256': run['predictions_sha256']})
            for row in leaves(report):
                output.write(json.dumps({'arm': arm, 'view': view, **row}, separators=(',', ':')) + '\n')
                scalar_count += 1
            event_scope_count += sum(len(event['scopes']) for event in report['events'])
            candidates = json.loads(Path(run['predictions']).with_suffix('.candidate-tags.json').read_text())
            ranks = defaultdict(list)
            for item in candidates:
                assert len(item['candidate_indices']) == len(item['records'])
                for rank, record in enumerate(item['records'], 1):
                    ranks[f"{item['scope']}/normalized_joint_rank_{rank}"].append(record)
                    candidate_count += 1
                    for row in leaves(record):
                        tag_scalar_count += 1
                        output.write(json.dumps({'arm': arm, 'view': view, 'event_uid': item['event_uid'],
                                                 'scope': item['scope'], 'rank': rank, **row}, separators=(',', ':')) + '\n')
                        scalar_count += 1
            if candidates:
                public['candidate_rank_summaries'][arm] = {key: summarize_tag_efficiency_events(rows) for key, rows in sorted(ranks.items())}
    for view, arms in grouped.items():
        left, right = [arms[arm]['events'] for arm in ('set_overlap_off', 'set_overlap_on')]
        assert [row['event_uid'] for row in left] == [row['event_uid'] for row in right]
        uncertainty = {}
        for scope in ('full', 'half'):
            for method in left[0]['scopes'][scope]['tag_efficiency']:
                rows = [[event['scopes'][scope]['tag_efficiency'][method] for event in events] for events in (left, right)]
                uncertainty[f'{scope}/{method}'] = collision_intervals(*rows)
        public['uncertainty']['views'][view] = uncertainty
    public['counts'] = {'registered_reports': len(receipts), 'event_scope_records': event_scope_count,
                        'retained_candidate_scope_records': candidate_count, 'additive_native_scalar_records': scalar_count,
                        'tagging_scalar_records_including_all_summaries_and_candidates': tag_scalar_count}
    assert event_scope_count == 2480
    assert candidate_count == sum(2*len(event['candidates']) for reports in grouped.values() for report in reports.values() for event in report.get('beam_search',{}).get('events',[]))
    public['archived_floating_point_roundoff_present'] = any(r['archived_floating_point_differences'] for r in receipts)
    public['counts']['public_numeric_records'] = 1 + sum(1 for _ in leaves(public))
    data = (json.dumps(public, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(data, mtime=0))
    (evidence / 'export-counts.json').write_text(json.dumps(public['counts'], indent=2) + '\n')
    print(json.dumps({'bytes': len(data), **public['counts']}))


if __name__ == '__main__':
    main()
