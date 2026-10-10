"""Keep the retrospective diagnostic scope and downloadable evidence authentic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('phase86_publication', ROOT / 'docs/_ext/wiki_phase86.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_frozen_scope_and_lossless_selected_aggregates(tmp_path):
    result = M.generate(ROOT, tmp_path)
    raw = (tmp_path / 'phase86-cleanup-summary.json').read_bytes()
    source = (ROOT / M.SOURCE / 'summary.json').read_bytes()
    value = json.loads(raw)
    original = json.loads(source)
    assert set(value) == {'schema', 'source_summary_sha256', 'cutoff', 'phase86', 'cleanup', 'cr001'}
    assert value['source_summary_sha256'] == hashlib.sha256(source).hexdigest()
    assert all(value[key] == original[key] for key in ('phase86', 'cleanup', 'cr001', 'cutoff'))
    assert raw != source
    assert result['source_set_gates_passed']
    assert not result['recursive_topology_pid_reachability_verified']
    assert result['branches']['generation']['necessary_height_deep_fit'] == 4
    assert result['branches']['repair']['necessary_height_deep_fit'] == 4
    assert value['cleanup']['accounting']['home_logical_bytes_reclaimed_net'] == 941618410
    assert value['cleanup']['accounting']['home_allocated_bytes_reclaimed_net'] == 254145536
    assert value['cr001']['final_suite_unexecuted'] == 2797
    assert not value['cr001']['recursive96_executed_at_cutoff']
    for lines in (M.render(result), (ROOT / 'docs/wiki/phase86.rst').read_text().splitlines()):
        for i, line in enumerate(lines):
            if i and line and set(line) <= set('=-~'):
                assert len(line) >= len(lines[i - 1])


@pytest.mark.parametrize('mutation', ['bytes', 'scope', 'recursive_claim', 'denominator'])
def test_reject_corruption_or_false_scope(tmp_path, mutation):
    source = tmp_path / M.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / M.SOURCE, source)
    p = source / 'summary.json'
    if mutation == 'bytes':
        p.write_bytes(p.read_bytes() + b' ')
    else:
        value = json.loads(p.read_text())
        key = {'scope': 'training_updates', 'recursive_claim': 'recursive_topology_pid_reachability_verified', 'denominator': 'eligible_mothers'}[mutation]
        value['phase86'][key] = 1
        raw = json.dumps(value).encode()
        p.write_bytes(raw)
        b = json.loads((source / 'binding.json').read_text())
        b.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (source / 'binding.json').write_text(json.dumps(b))
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(ValueError):
        M.generate(tmp_path, output)
