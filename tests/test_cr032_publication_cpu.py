"""Publication evidence fidelity and bounded privacy checks; no scientific data."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


EXPORT = load('scripts/package_cr032_review.py', 'cr032_export_test')
WIKI = load('docs/_ext/wiki_cr032.py', 'cr032_wiki_test')
SOURCE = ROOT / WIKI.SOURCE


def documents():
    return {path.name[:-8]: json.loads(gzip.decompress(path.read_bytes()))
            for path in SOURCE.glob('*.json.gz')}


def test_scientific_denominators_and_limits(tmp_path):
    value = WIKI.generate(ROOT, tmp_path)
    summary = value['summary']
    assert summary['summary']['event_any_upper'] == 31
    assert summary['summary']['event_both_upper'] == 19
    assert summary['limits']['whole_events'] == 32
    assert summary['strata']['category']['charged']['accepted_impossible'] == 8
    assert summary['strata']['category']['mixed']['accepted_impossible'] == 6
    assert sum(summary['charge_obstruction_histogram'].values()) == 14
    for axis in ('category', 'intrinsic_height', 'source_count'):
        assert sum(row['eligible'] for row in summary['strata'][axis].values()) == 64
    assert sum(row['eligible'] for row in summary['strata']['channel']) == 64
    for name in ('training_updates', 'model_calls', 'new_rollouts', 'bootstrap_draws'):
        assert summary['counts'][name] == 0
    assert summary['resources']['allocation_exits'] == [0, 0]
    reports = value['context']['reports']
    assert [(reports[key]['raw_correct'], reports[key]['accepted_correct']) for key in
            ('mixed4376', 'generic_b4166', 'generic_b3416')] == [(1, 1), (5, 3), (4, 2)]
    assert all(row['scientific_gate'] == 'failed' for row in reports.values())
    assert value['context']['seed_uncertainty'].startswith('unavailable')


def test_inventory_schema_and_roundtrip(tmp_path):
    value = WIKI.generate(ROOT, tmp_path)
    docs = documents()
    inventory = docs['inventory']
    assert len(inventory['source_bindings']) == 7
    for name, record in inventory['public_documents'].items():
        assert record['field_schema'] == EXPORT.field_schema(docs[name])
        assert record['cardinality'] == EXPORT.stats(docs[name])
        assert record['sha256'] == EXPORT.digest(EXPORT.canonical(docs[name]))
    for record in value['metric_downloads']:
        if record['filename'].endswith('.txt'):
            continue
        path = tmp_path / record['filename']
        data = WIKI._codec().decode(json.loads(path.read_bytes()), WIKI.LIMIT)
        original_name = 'inventory' if path.name == 'cr032-inventory.json' else path.stem
        assert json.loads(data) == docs[original_name]
    # Duplicate bound paired table agrees in every scalar, including full-precision CIs.
    assert docs['cr009-metrics']['comparison'] == docs['cr009-paired']
    assert sum(row['bytes'] for row in value['metric_downloads']) < 1_000_000


def test_private_identity_and_operational_fields_absent():
    forbidden = {'event_uid', 'event_uids', 'uid', 'uids', 'b_side', 'sources', 'resources_source_ids',
                 'node_id', 'daughter_node_ids', 'JobID', 'job', 'job_id', 'pid', 'argv', 'scratch',
                 'source_file', 'checkpoints', 'checkpoint_path', 'hostname', 'username', 'readiness_events'}
    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for value in documents().values():
        walk(value)
    # Failed attempts remain visible, with actual status and exit.
    resources = documents()['cr009-resources']
    assert resources['first_failed_attempt_retained']
    assert resources['attempts']['mixed']['actual_exit'] == 1
    assert resources['attempts']['mixed']['scheduler']['State'] == 'FAILED'


def test_tampered_source_and_path_rejected(tmp_path):
    source = tmp_path / WIKI.SOURCE
    source.mkdir(parents=True)
    binding = json.loads((SOURCE / 'binding.json').read_bytes())
    for path in SOURCE.glob('*'):
        (source / path.name).write_bytes(path.read_bytes())
    output = tmp_path / 'out'
    output.mkdir()
    first = source / binding['files'][0]['file']
    first.write_bytes(first.read_bytes()+b'x')
    with pytest.raises(ValueError, match='binding differs'):
        WIKI.generate(tmp_path, output)
    first.write_bytes((SOURCE / first.name).read_bytes())
    binding['files'][0]['file'] = '../escape.json.gz'
    (source / 'binding.json').write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='filename'):
        WIKI.generate(tmp_path, output)


def test_oversize_and_private_payload_rejected():
    codec = WIKI._codec()
    data = EXPORT.canonical({'event_uid': '/home/private/user'})
    with pytest.raises(ValueError, match='Private'):
        codec.encode(data, gzip.compress(data), WIKI.LIMIT)
    with pytest.raises(ValueError, match='byte limit'):
        codec.encode(b' '*(WIKI.LIMIT+1), b'', WIKI.LIMIT)


def test_unknown_scientific_cutoff_rejected():
    with pytest.raises(ValueError, match='cutoff'):
        EXPORT.authenticate({'scientific_source_cutoff': 'unapproved'})


def test_corrected_pid_telemetry_preserves_raw_evidence():
    resources = documents()['cr009-resources']
    assert resources['cumulative_counts']['pid_mode_diagnostic_model_calls'] == 4293
    assert resources['cumulative_counts']['pid_mode_diagnostic_encoder_calls'] == 8586
    assert resources['cumulative_counts']['pid_mode_diagnostic_event_presentations'] == 22248
    correction = resources['pid_subset_telemetry_correction']
    assert correction['corrected_training_subset'] == {
        'pid_mode_diagnostic_model_calls': 2565,
        'pid_mode_diagnostic_encoder_calls': 5130,
        'pid_mode_diagnostic_event_presentations': 20520}
    assert correction['source_report_sha256'] == EXPORT.EXPECTED['cr009-final-report.md']


def test_standalone_decoder_matches_bounded_library(tmp_path):
    import subprocess
    import sys
    WIKI.generate(ROOT, tmp_path)
    result = tmp_path / 'restored.json'
    run = subprocess.run([sys.executable, str(tmp_path / 'cr032-decoder.txt'),
                          str(tmp_path / 'cr032-summary.json'), str(result)], capture_output=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(result.read_bytes()) == documents()['cr032-summary']
    rerun = subprocess.run([sys.executable, str(tmp_path / 'cr032-decoder.txt'),
                            str(tmp_path / 'cr032-summary.json'), str(result)], capture_output=True)
    assert rerun.returncode != 0
