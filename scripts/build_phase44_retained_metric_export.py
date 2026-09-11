#!/usr/bin/env python3
"""Verify additive Phase44 tree coverage against preserved native reports."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.build_reconstruction_phase41_closeout import numeric_rows
from hypertagging.evaluation.retained_tree_checks import validate_retained_tree_report

ARMS = ('late_adaptation_control', 'early_adaptation')
POINTS = ('source_recall', 'source_precision', 'lcag_pair_accuracy', 'mother_pid_coverage',
          'mother_pid_accuracy', 'perfect_lcag', 'coherent_retained_forest', 'target_representable', 'root_pid_accuracy')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_projection(value):
    if isinstance(value, dict):
        return {key: legacy_projection(item) for key, item in value.items() if not key.startswith('retained_tree_')}
    if isinstance(value, list):
        return [legacy_projection(item) for item in value]
    return value



def metric_names(value):
    """Normalize fixed scope/ranker separators to the public metric vocabulary."""
    if isinstance(value, dict):
        names = [str(key).replace('/', '.') for key in value]
        if len(set(names)) != len(names):
            raise ValueError('Metric names collide after separator normalization')
        return {name: metric_names(item) for name, item in zip(names, value.values(), strict=True)}
    if isinstance(value, list):
        return [metric_names(item) for item in value]
    return value



def structure_counts(report, scope):
    rows = [row for event in report['events'] for row in event['scopes'][scope]['retained_tree_metrics']['rows']]
    return {
        'isolated_leaf_units': sum(not row['truth_mother_count'] and len(row['truth_sources']) == 1 for row in rows),
        'single_source_composite_units': sum(bool(row['truth_mother_count']) and len(row['truth_sources']) == 1 for row in rows),
        'nontrivial_topology_units': sum(len(row['truth_sources']) >= 2 for row in rows),
        'source_empty_units': sum(not row['truth_sources'] for row in rows),
        'events_without_flagged_target_incompatibility': sum(bool(event['scopes'][scope]['retained_tree_metrics']['rows']) and all(row['target_representable'] for row in event['scopes'][scope]['retained_tree_metrics']['rows']) for event in report['events']),
        'representable_nontrivial_units': sum(int(row['target_representable']) for row in rows if len(row['truth_sources']) >= 2),
    }


def report_paths(retained, arm):
    paths = list((retained / arm).glob('*.json'))
    if arm == 'early_adaptation':
        paths.extend((retained / 'early_adaptation_retries').glob('*.json'))
    return sorted(paths)


def tree_metric_rows(retained):
    for arm in ARMS:
        for path in report_paths(retained, arm):
            report = json.loads(path.read_text())
            if 'summaries' not in report:
                continue
            def emit(result, event_index, search, candidate_index, scope):
                for row in numeric_rows(result):
                    yield dict(arm=arm, view=path.stem, event_index=event_index,
                               search=search, candidate_index=candidate_index, scope=scope, **row)
            for index, event in enumerate(report['events']):
                for scope, record in event['scopes'].items():
                    yield from emit(record['retained_tree_metrics'], index, 'greedy', -1, scope)
                    for candidate_index, candidate in enumerate(record.get('retained_tree_beam', {}).get('candidates', [])):
                        yield from emit(candidate['metrics'], index, 'full_depth_beam', candidate_index, scope)
            event_indices = {event['event_uid']: index for index, event in enumerate(report['events'])}
            for event in report['beam_search']['events']:
                for candidate in event['candidates']:
                    for scope, result in candidate['retained_tree_metrics_by_scope'].items():
                        yield from emit(result, event_indices[event['event_uid']], 'proposal_beam', candidate['candidate_index'], scope)


def build(legacy, retained):
    provenance = json.loads((retained / 'provenance.json').read_text())
    output = {'version': 'phase44-retained-tree-export-v1', 'status': 'COMPLETE',
              'original_job_status': 'FAILED', 'recovery_classification': provenance['classification'],
              'evaluator_revision': provenance['evaluator_revision'], 'sealed_test_accessed': False,
              'legacy_metrics_unchanged': True, 'all_returned_beam_candidates_checked': True,
              'strict_event_count': 100, 'beam_event_count': 20,
              'source_hashes': [digest(retained / 'provenance.json')], 'arms': {}, 'metric_rows': []}
    for arm in ARMS:
        reports = {}
        for path in report_paths(retained, arm):
            report = json.loads(path.read_text())
            if 'summaries' not in report:
                continue
            validate_retained_tree_report(report)
            code = report['evaluator_code_provenance']
            assert code['git_head'] == output['evaluator_revision'] and code['provenance_complete'] and not code['worktree_dirty']
            assert report['device'] == 'cpu' and report['torch_num_threads'] == 1 and report['torch_deterministic_algorithms_enabled']
            assert report['evaluation_role'] == 'offline_model_evaluation'
            assert report['context']['evaluation_split'] == 'validation'
            assert set(report['context']['evaluated_uid_split_assignments'].values()) == {'validation'}
            assert not report['context']['evaluation_uid_train_overlap']

            old_path = legacy / arm / 'full-decay-reports' / path.name
            old = json.loads(old_path.read_text())
            for family in ('summaries', 'summaries_by_source_category', 'summaries_by_target_shape', 'events', 'beam_search'):
                assert legacy_projection(report[family]) == old[family], (arm, path.name, family, 'legacy result changed')
            for key in ('checkpoint_pair',):
                assert report[key] == old[key]
            assert [e['event_uid'] for e in report['events']] == [e['event_uid'] for e in old['events']]
            assert path.stem not in reports, 'Duplicate completed retained evaluation'
            reports[path.stem] = report
            checks = report['retained_tree_checks']
            output['metric_rows'].extend({'arm': arm, 'view': path.stem, **row} for row in numeric_rows(metric_names(checks)))
            output['source_hashes'].extend([digest(path), digest(old_path)])
        assert len(reports) == 7, (arm, 'Expected all seven checkpoint/search views')
        primary = reports['primary_complete_target_direct']
        repeat = reports['primary_complete_target_repeat2_direct']
        assert primary['retained_tree_checks'] == repeat['retained_tree_checks']
        assert primary['events'] == repeat['events']
        def points(summary):
            return {**{key: summary[key] for key in POINTS},
                    **{key: summary[key] for key in ('unit_count', 'available_unit_count', 'unavailable_unit_count', 'truth_mother_count', 'matched_mother_count', 'unit_semantics_counts')}}
        beam = reports['primary_complete_target_beam_direct']['retained_tree_checks']
        output['arms'][arm] = {
            'primary_repeat_identical': True,
            'primary_structure': {scope: structure_counts(primary, scope) for scope in ('full', 'half')},
            'primary': {scope: points(primary['retained_tree_checks']['summaries'][f'{scope}/greedy']) for scope in ('full', 'half')},
            'beam': {scope: {label.removeprefix(f'{scope}/proposal_beam/'): points(summary) for label, summary in beam['summaries'].items() if label.startswith(f'{scope}/proposal_beam/')} for scope in ('full', 'half')},
            'beam_candidate_count': beam['beam']['full/proposal_beam_normalized_joint']['candidate_count'],
        }
    assert output['arms'][ARMS[0]]['primary_structure'] == output['arms'][ARMS[1]]['primary_structure']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--retained-root', type=Path, required=True)
    parser.add_argument('--details-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.legacy_root, args.retained_root)
    args.details_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.details_dir / 'phase44-retained-all-metrics.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['arm', 'view', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(payload['metric_rows'])
    tree_export = args.details_dir / 'phase44-retained-tree-metrics.csv.gz'
    tree_count = 0
    with gzip.open(tree_export, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['arm', 'view', 'event_index', 'search', 'candidate_index', 'scope', 'metric', 'value'])
        writer.writeheader()
        for row in tree_metric_rows(args.retained_root):
            writer.writerow(row)
            tree_count += 1
    with gzip.open(tree_export, 'rt', newline='') as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == tree_count
    payload['tree_metric_scalar_rows'] = tree_count
    payload['tree_metric_export_sha256'] = digest(tree_export)
    payload['source_hashes'].append(digest(tree_export))
    payload['detailed_metric_count'] = len(payload['metric_rows'])
    (args.details_dir / 'phase44-retained-all-metrics.json').write_text(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    payload['metric_rows'] = [row for row in payload['metric_rows'] if not row['metric'].startswith(('by_source_category.', 'by_target_shape.'))]
    payload['aggregate_metric_count'] = len(payload['metric_rows'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    registry = [[row['arm'], row['view'], row['metric']] for row in payload['metric_rows']]
    assert len(registry) == len(set(map(tuple, registry)))
    (ROOT / 'docs/_ext/phase44_retained_metric_registry.json').write_text(json.dumps(registry, separators=(',', ':')) + '\n')
    print(json.dumps({'aggregate_rows': len(registry), 'detailed_rows': payload['detailed_metric_count'], 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
