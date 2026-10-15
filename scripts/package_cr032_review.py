"""Deterministic aggregate-only export of HIGH-bound CR032 and CR009 evidence.

No datasets, checkpoints, scheduler submission, new model calls or bootstrap.
Every input is authenticated before projection. Explicit root/field selection
keeps private unit proofs, event/source identities and process metadata private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '6e1b5bd4a34a9a70174baf1099a01847ba7c0a1f'
LIMIT = 5_000_000
EXPECTED = {
    'cr009-final-report.md': 'e28fccd5d4d4f0912cf786c8dddda7397015d4fe4fd98e5dcabe5c02ae4b08a3',
    'results-summary.json': '00542f7337842d3165b6c6dbc6afd5186c386c47655f8252336869af3e12c711',
    'unit-proof-table.json': '4c34f6d693d766219a41cc27fed9635ec7a4211a90604e91b4f1d4fe3a57e6ad',
    'event-proof-table.json': 'ffe19db4295f307f46ad66fd42fb1ad00542128b1302b054a6c804f5ef451037',
    'complete-metric-tables.json': '4013727c9706aa44d068de878bdeb23e57e9e6efa739d59f8008fcd0ff10fabf',
    'paired-comparison.json': 'c7634b8ab82ff68b5ea0d63b3305300f0acedfe49769c400fc5425453c8413bb',
    'terminal-accounting.json': '658cd470095bb9246c31dc0996b1e70f9ce1398c069eea97abf96a1b83f97987',
}
METRIC_ROOTS = ('availability', 'bootstrap_calls', 'comparison', 'denominator_audit',
                'exposure', 'interpretation', 'issues', 'scientific_gate', 'status', 'strata')
FIT_FIELDS = ('completed_steps', 'configured_schedule_horizon', 'cpu_seconds',
              'fixed_stop_steps', 'metrics', 'primary_slots', 'status', 'wall_seconds')
ACCOUNT_FIELDS = ('actual_cpu_seconds', 'arm', 'assessment_events_completed',
                  'completed_optimizer_steps', 'counts', 'elapsed_seconds', 'maxrss_kib',
                  'scheduler_steps_applied', 'status')
ALLOCATION_FIELDS = ('elapsed', 'passed', 'peaks', 'scientific_consumer_started', 'source_hashes_preserved')
SCHEDULER_FIELDS = ('AllocCPUS', 'Elapsed', 'ExitCode', 'MaxRSS', 'ReqMem', 'State', 'TotalCPU')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'docs/_ext'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def select(value, fields):
    return {key: value[key] for key in fields}


def stats(value):
    result = Counter()
    def walk(v):
        if isinstance(v, dict):
            result['objects'] += 1
            result['object_fields'] += len(v)
            for child in v.values(): walk(child)
        elif isinstance(v, list):
            result['arrays'] += 1
            result['array_items'] += len(v)
            for child in v: walk(child)
        else:
            result['scalars'] += 1
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                result['numeric_scalars'] += 1
    walk(value)
    return dict(result)


def field_schema(value):
    """Explicit public leaf paths/types, with array positions represented by []."""
    result = {}
    def walk(v, path):
        if isinstance(v, dict):
            if not v: result.setdefault(path, set()).add('empty_object')
            for key, child in v.items(): walk(child, path+'.'+key)
        elif isinstance(v, list):
            if not v: result.setdefault(path, set()).add('empty_array')
            for child in v: walk(child, path+'[]')
        else:
            result.setdefault(path, set()).add(type(v).__name__)
    walk(value, '$')
    return {key: sorted(types) for key, types in sorted(result.items())}


def fit(value):
    return select(value, FIT_FIELDS)


def accounting(value):
    # Failed attempts may contain only status/error and partial counters.
    result = {key: value[key] for key in ACCOUNT_FIELDS if key in value}
    if value.get('fit_result') is not None:
        result['fit_result'] = fit(value['fit_result'])
    return result


def project(inputs):
    summary = inputs['results-summary.json']
    public_summary = select(summary, ('counts', 'limits', 'scientific_gate', 'strata', 'study_id', 'summary'))
    # Detach before removing the evaluator's per-event side ordinal. Channel IDs
    # and signed decay descriptions are scientific aggregate dimensions.
    public_summary = json.loads(canonical(public_summary))
    for row in public_summary['strata']['channel']:
        row['channel'].pop('b_side', None)
    resources = summary['resources']
    public_summary['resources'] = {
        'allocation': select(resources['allocation'], ALLOCATION_FIELDS),
        'consumer': select(resources['consumer'], ('counts', 'cpu_seconds', 'elapsed_seconds', 'maxrss_kib', 'status')),
        'allocation_exits': [row['actual_exit'] for row in resources['allocation']['records']],
    }
    units = inputs['unit-proof-table.json']
    events = inputs['event-proof-table.json']
    if len(units) != 64 or len(events) != 32:
        raise ValueError('Frozen proof population differs')
    public_summary['charge_obstruction_histogram'] = dict(sorted(Counter(
        str(unit['fixed_integer_charge']) for unit in units if unit['certificates']['charge']).items()))
    public_summary['proof_population'] = {'units': len(units), 'events': len(events)}
    complete = inputs['complete-metric-tables.json']
    metrics = select(complete, METRIC_ROOTS)
    # All metric trees above are copied byte-value-exact. This scoped resource
    # projection excludes parameter/RNG structures and UID-keyed panel tables.
    metrics['resources'] = {}
    for arm, value in complete['resources'].items():
        metrics['resources'][arm] = {
            'execution_accounting': accounting(value['execution-accounting.json']),
            'fit_result': fit(value['fit-result.json']),
            'panel_access': select(value['panel-access.json'], ('complete_only_capacity_passed', 'counts', 'replacements')),
            'training_observer': select(value['training-observer-summary.json'], ('active', 'counts', 'scope')),
        }
    terminal = inputs['terminal-accounting.json']
    public_terminal = select(terminal, ('all_jobs_terminal', 'artifacts_verified', 'complete_scientific_execution',
        'cumulative_counts', 'first_failed_attempt_retained', 'source_binding_verified'))
    public_terminal['attempts'] = {}
    for arm, value in terminal['jobs'].items():
        public_terminal['attempts'][arm] = {
            'accounting': accounting(value['accounting']),
            'allocation': select(value['allocation'], tuple(key for key in ALLOCATION_FIELDS if key != 'scientific_consumer_started')),
            'scheduler': select(value['scheduler'], SCHEDULER_FIELDS),
            'actual_exit': value['child_exit']['actual_exit'],
            'source_files_verified': value['source_files_verified'],
            'source_sha': value['source_sha'],
        }
    report = inputs['cr009-final-report.md']
    if 'totals2,565 model/5,130 encoder/20,520 event presentations' not in report:
        raise ValueError('Bound telemetry correction does not reconcile')
    public_terminal['pid_subset_telemetry_correction'] = {
        'status': 'corrected_in_bound_final_report; raw subset values retained above',
        'source_report_sha256': EXPECTED['cr009-final-report.md'],
        'cause': 'Shared no_grad wrapper code object classified assessment calls again inside the training PID diagnostic subset.',
        'corrected_training_subset': {'pid_mode_diagnostic_model_calls': 2565,
                                     'pid_mode_diagnostic_encoder_calls': 5130,
                                     'pid_mode_diagnostic_event_presentations': 20520},
        'unaffected': 'Overall training and assessment counts, resource guards, optimizer behavior and scientific outcomes.',
        'new_model_calls': 0,
    }
    metrics['resources_annotation'] = 'PID diagnostic subset fields preserve raw telemetry. Use cr009-resources.json pid_subset_telemetry_correction for corrected totals from the bound final report.'
    return {'cr032-summary': public_summary, 'cr009-metrics': metrics,
            'cr009-paired': inputs['paired-comparison.json'], 'cr009-resources': public_terminal}


def authenticate(plan):
    if plan['scientific_source_cutoff'] != SOURCE_SHA:
        raise ValueError('Scientific cutoff differs')
    inputs, provenance = {}, []
    for name, expected in EXPECTED.items():
        matches = [entry for entry in plan['immutable_evidence']
                   if (entry['sha256'] == expected if name == 'cr009-final-report.md'
                       else Path(entry['path']).name == name)]
        if len(matches) != 1 or matches[0]['sha256'] != expected:
            raise ValueError('Missing or unapproved evidence binding: '+name)
        entry = matches[0]
        raw = Path(entry['path']).read_bytes()
        if len(raw) != entry['bytes'] or digest(raw) != expected or len(raw) > LIMIT:
            raise ValueError('Changed or oversize evidence: '+name)
        inputs[name] = raw.decode() if name.endswith('.md') else json.loads(raw)
        provenance.append({'source': name, 'bytes': len(raw), 'sha256': expected,
                           'original_cardinality': stats(inputs[name])})
    return inputs, provenance


def package(plan, output):
    inputs, provenance = authenticate(plan)
    documents = project(inputs)
    codec = module('wiki_phase72_compact')
    # encode() applies the existing full privacy projection before producing any
    # compressed public artifact. Strings are not silently redacted or rounded.
    inventory = {
        'schema': 'cr032-public-field-inventory-v1', 'source_bindings': provenance,
        'policy': 'Exact saved aggregate values; no new scientific estimates. Private proof rows, event/source identities, model and RNG metadata, process identifiers, paths and detailed monitor journal remain private.',
        'selected_metric_roots': list(METRIC_ROOTS),
        'excluded_source_subtrees': {
            'results-summary.json': ['resources.allocation.readiness_events', 'resources.allocation.scratch', 'resources.allocation.records operational fields except actual_exit', 'resources.consumer.completed', 'resources.consumer.cursor', 'strata.channel[].channel.b_side'],
            'unit-proof-table.json': ['all individual rows; only charge obstruction histogram and row count exported'],
            'event-proof-table.json': ['all individual rows; only row count exported'],
            'complete-metric-tables.json': ['retained_target_inventory', 'resources.*.initialization-binding.json', 'resources.*.replay-contract.json', 'resources.*.panel-access.json.summaries', 'resources.*.panel-access.json.uids', 'resources.*.training-observer-summary.json.bindings', 'resources.*.fit-result.json.checkpoints'],
            'terminal-accounting.json': ['scheduler_command', 'jobs.*.job', 'jobs.*.scheduler.JobID', 'jobs.*.e1', 'jobs.*.child_exit operational fields except actual_exit'],
        },
        'public_documents': {key: {'cardinality': stats(value), 'field_schema': field_schema(value),
                                 'sha256': digest(canonical(value)), 'bytes': len(canonical(value))}
                             for key, value in documents.items()},
    }
    documents['inventory'] = inventory
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, value in documents.items():
        raw = canonical(value)
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        envelope = codec.encode(raw, packed, LIMIT)
        path = output/(name+'.json.gz')
        path.write_bytes(packed)
        records.append({'file': path.name, 'download': ('cr032-inventory' if name == 'inventory' else name)+'.json',
                        'compressed_bytes': len(packed), 'compressed_sha256': digest(packed),
                        'decoded_bytes': len(raw), 'decoded_sha256': digest(raw),
                        'envelope_bytes': len(envelope), 'envelope_sha256': digest(envelope)})
    binding = {'schema': 'cr032-public-binding-v1', 'source_sha': SOURCE_SHA,
        'context_source_sha': '6ca8ef769c5137e1c5347d879372dbf0fe7c6472', 'files': records,
        'public_summary': documents['cr032-summary']['summary'],
        'no_new_science': True, 'complete_post_study_evaluation': False}
    (output/'binding.json').write_bytes(canonical(binding))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binding = package(json.loads(args.plan.read_text()), args.output)
    print(json.dumps({'files': len(binding['files']), 'envelope_bytes': sum(row['envelope_bytes'] for row in binding['files'])}))


if __name__ == '__main__':
    main()
