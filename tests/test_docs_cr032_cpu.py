"""Validate the new review's rendered prose and archived source identifiers."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_review_renders_without_warnings_or_private_tokens(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'conf.py').write_text("project='fixture'\nmaster_doc='index'\n")
    (source / 'index.rst').write_bytes((ROOT / 'docs/wiki/cr032.rst').read_bytes())
    for target in ('phase84_85', 'phase86', '_generated/status/downloads'):
        page = source / (target + '.rst')
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(':orphan:\n\nFixture\n=======\n')
    output = tmp_path / 'text'
    subprocess.run([sys.executable, '-m', 'sphinx', '-W', '-b', 'text',
                    str(source), str(output)], check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location('cr032_prose_privacy', ROOT / 'docs/_ext/wiki_privacy.py')
    privacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(privacy)
    rendered = (output / 'index.txt').read_text()
    assert not privacy.violations(rendered)
    assert '50/64' in rendered and '61/64' in rendered
    assert 'not proved achievable' in rendered
    assert '2,565' in rendered and '4,293' in rendered


def test_generated_review_heading_underlines_are_complete():
    spec = importlib.util.spec_from_file_location('cr032_render', ROOT / 'docs/_ext/wiki_cr032.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = module.render({})
    for index, line in enumerate(lines):
        if index and line and set(line) <= {'=', '-'}:
            assert len(line) >= len(lines[index - 1])
