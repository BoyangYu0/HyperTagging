"""Lossless additive publication and collision-level uncertainty contracts."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def publisher():
    spec = importlib.util.spec_from_file_location('efficiencies', ROOT / 'docs/_ext/wiki_phase69_efficiencies.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_typed_dag_preserves_types_duplicates_and_unknowns():
    m = publisher()
    value = {'a': [0, False, 0.0, None, '0'], 'b': [0, False, 0.0, None, '0'],
             'channel': {'count': 2, 'correct': 0, 'available': False}}
    packed = m.pack(value)
    assert m.canonical(m.unpack(packed)) == m.canonical(value)
    assert len(packed['nodes']) < 20
    packed['nodes'][-1][1][0][1] = len(packed['nodes'])
    with pytest.raises(ValueError, match='reference'):
        m.unpack(packed)


def test_collision_bootstrap_does_not_treat_two_b_trials_as_independent():
    from scripts.export_phase69_efficiencies import collision_intervals
    rows = [{'b': {'numerator': n, 'denominator': 2, 'value': n / 2}} for n in (0, 2)]
    result = collision_intervals(rows, rows, repeats=1000)['b']
    assert result['refined_encoder']['percentile_95'] == [0.0, 1.0]
    assert result['paired_pre_minus_refined']['percentile_95'] == [0.0, 0.0]
    assert result['paired_pre_minus_refined']['degenerate'] is True
    missing = collision_intervals([{'x': {'numerator': 0, 'denominator': 0, 'value': None}}],
                                  [{'x': {'numerator': 0, 'denominator': 0, 'value': None}}], repeats=10)['x']
    assert missing['refined_encoder']['value'] is None
    assert missing['refined_encoder']['percentile_95'] is None


def test_complete_additive_bundle_keeps_old_bundle_and_every_registered_view(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / 'docs/_ext'))
    import wiki_phase69
    m = publisher()
    summary = json.loads((ROOT / 'artifacts/codex/phase69_publication_20261005/summary.json').read_text())
    wiki_phase69.generate(ROOT, tmp_path, summary)
    native = json.loads((tmp_path / 'phase69-all-aggregate-metrics.json').read_text())
    result = m.generate(ROOT, tmp_path, native)
    bundle = json.loads((tmp_path / m.BUNDLE).read_text())
    assert bundle['native_aggregate_bundle'] == native
    decoded = m.unpack(bundle['additive_efficiencies'])
    assert decoded['counts']['registered_reports'] == 14
    assert decoded['counts']['retained_candidate_scope_records'] == 262
    assert len(decoded['views']) == 7
    assert all(set(arms) == {'refined_encoder', 'pre_refinement_encoder'} for arms in decoded['views'].values())
    assert result['native_bundle_preserved'] is True
    assert len(m.canonical(decoded)) == result['decoded_efficiencies']['bytes']


@pytest.mark.parametrize('mutation', ['binding', 'coverage', 'event_identifier'])
def test_efficiency_publisher_rejects_corruption_incomplete_or_private_exports(tmp_path, mutation):
    import gzip
    import hashlib
    m = publisher()
    source = ROOT / m.SOURCE
    value = json.loads(gzip.decompress(source.read_bytes()))
    binding = json.loads(source.with_name('binding.json').read_text())
    if mutation == 'coverage':
        value['views'].pop(next(iter(value['views'])))
    if mutation == 'event_identifier':
        value['extra'] = {'event_uid': 'not-public'}
    data = m.canonical(value)
    if mutation != 'binding':
        binding.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    else:
        binding['sha256'] = '0' * 64
    target = tmp_path / m.SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(data, mtime=0))
    target.with_name('binding.json').write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        m.generate(tmp_path, tmp_path, {})
