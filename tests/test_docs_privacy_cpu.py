"""Representative publication regressions; integration/workflow checks live in test_docs_wiki_cpu."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
import wiki
import wiki_privacy as privacy


def _search_index():
    return {
        "docnames": ["api/module"], "filenames": ["api/module.rst"],
        "titles": ["Public module reference"], "terms": {"username": [0], "job_id": 0},
        "titleterms": {"api_key": [0]}, "objects": {}, "objnames": {}, "objtypes": {},
        "envversion": {"sphinx": 1}, "alltitles": {}, "indexentries": {},
    }

def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path

def _validation_site(root, payload):
    for name in ("objects.inv", "searchindex.js", ".nojekyll"):
        _write(root, name, "")
    _write(root, "index.html", payload)
    return root

def test_sphinx_source_gate_rejects_authored_raw_markup_before_rendering(tmp_path):
    _write(
        tmp_path,
        "index.rst",
        ".. raw:: html\n\n   <p>/ho<span>me/PRIVATE_USER/</span>private.txt</p>\n",
    )
    app = SimpleNamespace(
        srcdir=str(tmp_path),
        config=SimpleNamespace(wiki_content_path=""),
        extensions={},
    )
    with pytest.raises(ValueError, match="publication policy"):
        wiki.on_builder_inited(app)
    assert not hasattr(app, "_wiki_authored_sources_validated")

@pytest.mark.parametrize("value", [
    "/home/PRIVATE_USER/private.json", "C:\\Users\\PRIVATE_USER\\private.txt",
    "hostname=login123.example.org", "username=PRIVATE_USER",
    "GPU-1234abcd-1234-abcd-1234-123456789abc", "job_id=982731",
    "authority/private_authorization.json", "checkpoint/private.ckpt",
    "api_key=OPAQUE_PRIVATE_TOKEN", "ghp_1234567890abcdefghijklmnop",
    "%252Fhome%252FPRIVATE_USER%252Fprivate.txt",
    "&#47;home&#47;PRIVATE_USER&#47;private.txt",
    r"\u002fhome\u002fPRIVATE_USER\u002fprivate.txt",
    r"\x2fhome\x2fPRIVATE_USER\x2fprivate.txt",
])
def test_redactor_and_scanner_detect_private_and_encoded_values(value):
    assert privacy.violations(value)
    redacted = privacy.redact(value)
    assert "PRIVATE_USER" not in redacted
    assert not privacy.violations(redacted)

def test_relative_sphinx_links_are_not_absolute_paths(tmp_path):
    for value in ("../../index.html", "../../../api/module.html#name", "./status/index.html", "../_static/wiki.css"):
        assert privacy.violations(value) == []
        assert privacy.redact(value) == value
    for value in ("/home/PRIVATE_USER/private", "file:///home/PRIVATE_USER/private", r"\\PRIVATE_HOST\PRIVATE_SHARE\private"):
        assert privacy.violations(value)
    _write(tmp_path, "index.html", '<a href="../../index.html">Parent</a><a href="../_static/wiki.css">Local style</a>')
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"

def test_search_index_words_are_typed_document_references_not_private_fields(tmp_path):
    _write(tmp_path, "searchindex.js", "Search.setIndex(" + json.dumps(_search_index()) + ")")
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"

def test_search_index_numeric_references_do_not_invent_private_identifiers(tmp_path, monkeypatch):
    index = _search_index()
    index["docnames"] = [f"api/module_{i}" for i in range(8)]
    index["filenames"] = [f"api/module_{i}.rst" for i in range(8)]
    index["titles"] = [f"Public module {i}" for i in range(8)]
    index["terms"] = {"module": list(range(8))}
    monkeypatch.setattr(privacy, "sensitive_literals", lambda root: {"01234567"})
    site = tmp_path / "site"
    _write(site, "searchindex.js", "Search.setIndex(" + json.dumps(index) + ")")
    assert privacy.validate_artifact(site, root=tmp_path)["privacy"] == "PASS"
    index["titles"][0] = "Recorded identifier 01234567"
    _write(site, "searchindex.js", "Search.setIndex(" + json.dumps(index) + ")")
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(site, root=tmp_path)

def test_signature_defaults_are_elided_recursively_without_losing_dotted_names():
    value = "Example._run(self, value=Path('OPAQUE_PRIVATE_TOKEN'), *, config={'api_key': 'OPAQUE_NESTED_TOKEN'}) -> str"
    result = privacy.safe_signature(value)
    assert result == "Example._run(self, value=..., *, config=...) -> str"
    assert "OPAQUE" not in result
    assert privacy.safe_signature("broken(x=Path('OPAQUE_PRIVATE_TOKEN')") == "broken(...)"

@pytest.mark.parametrize("relative,payload", [
    ("index.html", '<p>/ho<span>me/PRIVATE_USER/</span>private.txt</p>'),
    ("index.html", '<span data-path="&#47;home&#47;PRIVATE_USER&#47;private.txt">safe</span>'),
    ("index.html", '<!-- /home/PRIVATE_USER/private.txt -->'),
    ("data.json", '{"unknown":{"nested":[{"apiKey":{"value":"OPAQUE_PRIVATE_TOKEN"}}]}}'),
    ("data.json", '{"path":"\\u002fhome\\u002fPRIVATE_USER\\u002fprivate.txt"}'),
    ("report.txt", r'\x2fhome\x2fPRIVATE_USER\x2fprivate.txt'),
    ("searchindex.js", 'Search.setIndex({"titles":["\\u002fhome\\u002fPRIVATE_USER\\u002fprivate.txt"]})'),
    ("_static/custom.js", 'const path = "\\u002funusual\\u002fPRIVATE_USER\\u002fprivate.txt";'),
    ("_static/custom.css", '.x { background: url("/unusual/PRIVATE_USER/private.txt"); }'),
])
def test_artifact_scans_encoded_html_json_text_and_search_index(tmp_path, relative, payload):
    _write(tmp_path, relative, payload)
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)

@pytest.mark.parametrize("directory", ["sources", "evidence", "_sources", "_modules"])
def test_artifact_rejects_forbidden_directories_even_when_empty(tmp_path, directory):
    (tmp_path / directory).mkdir()
    with pytest.raises(ValueError, match="forbidden-source-directory"):
        privacy.validate_artifact(tmp_path)

@pytest.mark.parametrize("relative", ["download.py", "download.yaml", "download.csv", "download.zip", "download.parquet"])
def test_artifact_rejects_raw_or_unclassified_downloads(tmp_path, relative):
    _write(tmp_path, relative, "RAW_SOURCE_BODY_CANARY")
    with pytest.raises(ValueError, match="forbidden-raw-download"):
        privacy.validate_artifact(tmp_path)

def test_allowed_ast_projection_and_static_markup_do_not_trigger_raw_body_rule(tmp_path):
    source = tmp_path / "source"
    _write(source, "scripts/public.py", 'def public(value):\n    """Documented callable."""\n    return "RAW_BODY_CANARY"\n')
    output = tmp_path / "artifact"
    _write(output, "api.txt", "public(value)\nDocumented callable.\n")
    _write(output, "status.rst", '.. raw:: html\n\n   <section><h2>Status</h2><p>NO_GO</p></section>\n')
    _write(output, "_static/custom.css", '/* https://developer.mozilla.org/en-US/docs/Web/API */\n')
    _write(output, "manifest.json", '{"job_id": null, "credentials": "[redacted]", "status": "NO_GO"}')
    assert privacy.validate_artifact(output, source)["privacy"] == "PASS"


@pytest.mark.parametrize('placement', ['typed_references', 'literal_title', 'literal_numeric_version', 'split_object_strings', 'generic_json'])
def test_private_numeric_literal_is_not_fabricated_from_search_doc_indexes(tmp_path, placement):
    root = tmp_path / 'repository'
    root.mkdir()
    _write(root, 'private.json', json.dumps({'job_id': '1203'}))
    output = tmp_path / 'site'
    output.mkdir()
    index = _search_index()
    index['docnames'] = [f'page{i}' for i in range(4)]
    index['filenames'] = [f'page{i}.rst' for i in range(4)]
    index['titles'] = ['Public page'] * 4
    index['terms'] = {'public': [1, 2, 0, 3]}
    if placement == 'literal_title': index['titles'][0] = '1203'
    if placement == 'literal_numeric_version': index['envversion'] = {'sphinx': 1203}
    if placement == 'split_object_strings':
        index['objnames'] = {'0': ['12', '03', 'Public']}
    if placement == 'generic_json':
        _write(output, 'data.json', json.dumps({'values': [1, 2, 0, 3]}))
    else:
        _write(output, 'searchindex.js', 'Search.setIndex(' + json.dumps(index) + ')')
    if placement == 'typed_references':
        assert privacy.validate_artifact(output, root)['privacy'] == 'PASS'
    else:
        with pytest.raises(ValueError, match='private-source-literal'):
            privacy.validate_artifact(output, root)
