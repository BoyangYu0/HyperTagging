"""Adversarial publication-boundary checks without scientific imports."""
from __future__ import annotations

import ast
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
import wiki
import wiki_privacy as privacy

_validation_spec = importlib.util.spec_from_file_location(
    "privacy_test_validation", ROOT / "scripts" / "validate_docs.py",
)
validation = importlib.util.module_from_spec(_validation_spec)
_validation_spec.loader.exec_module(validation)


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


def test_public_docs_links_remain_readable_without_masking_private_url_material():
    allowed = (
        "https://software.belle2.org/development/sphinx/framework/doc/atend-doctools.html",
        "https://github.com/actions/checkout", "https://docs.python.org/3/library/ast.html",
        "https://developer.mozilla.org/en-US/docs/Web/API", "http://www.w3.org/1999/xhtml",
        "https://pypi.org/project/Sphinx/", "https://github.com/org/project/issues",
        "https://docs.python.org/3/data/model.html",
    )
    for value in allowed:
        assert privacy.redact(value) == value
        assert privacy.violations(value) == []
    for value in (
        "https://github.com.evil.invalid/private", "https://github.com@private.invalid/",
        "https://PRIVATE_USER:OPAQUE_PRIVATE_TOKEN@github.com/actions/checkout",
        "https://github.com/actions/checkout?api_key=OPAQUE_PRIVATE_TOKEN",
        "https://github.com/actions/checkout?token=OPAQUE_PRIVATE_TOKEN",
        "https://github.com/actions/OPAQUE_PRIVATE_TOKEN?username=PRIVATE_USER",
        "https://github.com/home/PRIVATE_USER/private",
        "https://docs.python.org/tmp/PRIVATE_USER/private",
        "https://login123.example.org/private",
    ):
        assert privacy.violations(value)
        assert privacy.redact(value) == privacy.REDACTED


def test_relative_sphinx_links_are_not_absolute_paths(tmp_path):
    for value in ("../../index.html", "../../../api/module.html#name", "./status/index.html", "../_static/wiki.css"):
        assert privacy.violations(value) == []
        assert privacy.redact(value) == value
    for value in ("/home/PRIVATE_USER/private", "file:///home/PRIVATE_USER/private", r"\\PRIVATE_HOST\PRIVATE_SHARE\private"):
        assert privacy.violations(value)
    _write(tmp_path, "index.html", '<a href="../../index.html">Parent</a><a href="../_static/wiki.css">Local style</a>')
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def _search_index():
    return {
        "docnames": ["api/module"], "filenames": ["api/module.rst"],
        "titles": ["Public module reference"], "terms": {"username": [0], "job_id": 0},
        "titleterms": {"api_key": [0]}, "objects": {}, "objnames": {}, "objtypes": {},
        "envversion": {"sphinx": 1}, "alltitles": {}, "indexentries": {},
    }


def test_search_index_words_are_typed_document_references_not_private_fields(tmp_path):
    _write(tmp_path, "searchindex.js", "Search.setIndex(" + json.dumps(_search_index()) + ")")
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


@pytest.mark.parametrize("parts", [
    ["AKIA", "ABCDEFGHIJKLMNOP", "public"],
    ["user@", "example.com", "public"],
    ["PRIVATE_", "noise", "CANARY"],
])
def test_search_index_reconstructs_related_typed_string_fields(
    tmp_path, parts,
):
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write(
        source, "private.json",
        json.dumps({"api_key": "PRIVATE_CANARY"}),
    )
    value = _search_index()
    value["objnames"] = {"0": parts}
    _write(
        output, "searchindex.js",
        "Search.setIndex(" + json.dumps(value) + ")",
    )
    with pytest.raises(ValueError):
        privacy.validate_artifact(output, source)


def test_json_publication_rejects_duplicate_keys_that_discard_private_bytes(tmp_path):
    _write(tmp_path, "data.json", '{"path":"/home/PRIVATE_USER/private","path":"safe"}')
    with pytest.raises(ValueError, match="unreadable-publication") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_json_publication_rejects_encoded_duplicate_key_spellings(tmp_path):
    _write(tmp_path, "data.json", '{"path":"/home/PRIVATE_USER/private","\\u0070ath":"safe"}')
    with pytest.raises(ValueError, match="unreadable-publication") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_search_index_rejects_duplicate_keys_that_discard_private_bytes(tmp_path):
    encoded = json.dumps(_search_index(), separators=(",", ":"))
    encoded = encoded.replace(
        '"titles":["Public module reference"]',
        '"titles":["/home/PRIVATE_USER/private"],"titles":["Public module reference"]',
        1,
    )
    _write(tmp_path, "searchindex.js", "Search.setIndex(" + encoded + ")")
    with pytest.raises(ValueError, match="unreadable-publication") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_legacy_sphinx_search_index_object_shape_is_typed():
    value = _search_index()
    value.pop("alltitles")
    value.pop("indexentries")
    value["objects"] = {
        "hypertagging.module": {
            "callable": [0, 1, 1, "callable"],
        }
    }
    assert privacy._valid_search_index(value) is True


@pytest.mark.parametrize("trailing", [
    ', "/home/PRIVATE_USER/private.txt"',
    '/* /home/PRIVATE_USER/private.txt */',
])
def test_legacy_search_index_parser_rejects_ignored_trailing_content(tmp_path, monkeypatch, trailing):
    value = _search_index()
    canonical = json.dumps(value, separators=(",", ":")).replace('"docnames"', "docnames", 1)
    # Model Sphinx 3's data-only parser, which returns after one root object.
    # The production guard must use its encoder to prove full consumption.
    legacy = SimpleNamespace(loads=lambda _encoded: value, dumps=lambda _value: canonical)
    import sphinx.util
    monkeypatch.setattr(sphinx.util, "jsdump", legacy, raising=False)
    _write(tmp_path, "searchindex.js", "Search.setIndex(" + canonical + trailing + ")")
    with pytest.raises(ValueError, match="unreadable-publication") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_only_name_and_digest_authenticated_vendor_bytes_are_exempt(tmp_path, monkeypatch):
    vendor = b'const docs = "file://vendor.example/path";\n'
    digest = hashlib.sha256(vendor).hexdigest()
    monkeypatch.setattr(privacy, "_trusted_vendor_hashes", lambda: {"vendor.js": {digest}})
    path = tmp_path / "_static" / "vendor.js"
    path.parent.mkdir()
    path.write_bytes(vendor)
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"
    path.write_bytes(vendor + b"// modified\n")
    with pytest.raises(ValueError):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("mutation", ["private_string", "nested_secret", "unknown_key", "invalid_index", "private_version"])
def test_search_index_exception_cannot_hide_secrets_or_untyped_records(tmp_path, mutation):
    value = _search_index()
    if mutation == "private_string":
        value["titles"][0] = r"\u002fhome\u002fPRIVATE_USER\u002fprivate.txt"
    elif mutation == "nested_secret":
        value["terms"]["username"] = {"value": "OPAQUE_PRIVATE_TOKEN"}
    elif mutation == "unknown_key":
        value["api_key"] = "OPAQUE_PRIVATE_TOKEN"
    elif mutation == "invalid_index":
        value["terms"]["job_id"] = 982731
    else:
        value["envversion"]["job_id"] = 982731
    _write(tmp_path, "searchindex.js", "Search.setIndex(" + json.dumps(value) + ")")
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


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


@pytest.mark.parametrize("payload", [
    "<?probe /home/PRIVATE_USER/private?>",
    '<!DOCTYPE html PUBLIC "/home/PRIVATE_USER/private">',
    "<![CDATA[/home/PRIVATE_USER/private]]>",
    '<div data-/home/PRIVATE_USER/private="safe"></div>',
])
def test_html_processing_instruction_declaration_and_name_channels_are_scanned(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x OPAQUE_="" PRIVATE_TOKEN',
    '<x a="unterminated',
    "</section",
    "<!-- incomplete comment",
    "<?probe incomplete",
    "<!DOCTYPE html",
    "<![CDATA[incomplete",
])
def test_incomplete_html_markup_fails_closed(tmp_path, payload):
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    with pytest.raises(ValueError, match="unparseable-html"):
        privacy.validate_artifact(privacy_site)
    offline_site = _validation_site(tmp_path / "offline", payload)
    with pytest.raises(ValueError, match="incomplete HTML markup"):
        validation.validate_html(offline_site)


@pytest.mark.parametrize("payload", [
    "plain 2 < 3",
    "literal <",
    "literal << text",
    "literal <3 is text",
    '<!DOCTYPE html PUBLIC "public>identifier"><p title="x > y">safe</p>',
    "<?probe safe?><p>safe</p>",
    "<!-- closed comment --><p>safe</p>",
])
def test_benign_less_than_text_and_complete_generated_markup_pass(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


@pytest.mark.parametrize("payload", [
    '<meta name="api_key" content="OPAQUE_PRIVATE_TOKEN">',
    '<meta http-equiv="authorization" content="OPAQUE_PRIVATE_TOKEN">',
    '<input name="token" value="OPAQUE_PRIVATE_TOKEN">',
    '<input type="password" value="OPAQUE_PRIVATE_TOKEN">',
    '<div data-secret="OPAQUE_PRIVATE_TOKEN"></div>',
])
def test_html_semantic_sensitive_attributes_are_rejected(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-field-value") as caught:
        privacy.validate_artifact(tmp_path)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<api_ key="=OPAQUE_PRIVATE_TOKEN"></api_>',
    '<x api_="" key="=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x api_="" inert="" key="=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x api_="" inert="public" key="=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x api_="" key="=none" inert="" OPAQUE_PRIVATE_TOKEN=""></x>',
    '<x api_="" key="=none" inert="OPAQUE_PRIVATE_TOKEN"></x>',
    '<x a="api_" key="=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x a="api_" b="key=OPAQUE_PRIVATE_TOKEN"></x>',
    '<api_ key="OPAQUE_PRIVATE_TOKEN"></api_>',
    '<x api_="" key="OPAQUE_PRIVATE_TOKEN"></x>',
    '<x a="api_" key="OPAQUE_PRIVATE_TOKEN"></x>',
    '<x api_="" key="none" inert="OPAQUE_PRIVATE_TOKEN"></x>',
    '<x api_="" key:="" OPAQUE_PRIVATE_TOKEN=""></x>',
    '<x api_="" key="=OPAQUE_%50RIVATE_TOKEN"></x>',
])
def test_generic_private_assignment_split_across_raw_tag_channels_is_rejected(
    tmp_path, payload,
):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-field-assignment") as caught:
        privacy.validate_artifact(tmp_path)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x data_="" key="=public"></x>',
    '<x a="public_" key="=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x a="public_" key="public"></x>',
    '<x data_="" key:="" public=""></x>',
])
def test_benign_raw_tag_channel_reconstruction_remains_allowed(
    tmp_path, payload,
):
    _write(tmp_path, "index.html", payload)
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


@pytest.mark.parametrize("tail", [
    "TOKEN", "%54OKEN", "&#84;OKEN", r"\54 OKEN",
])
def test_deny_literal_split_across_value_name_value_channels_is_rejected(
    tmp_path, tail,
):
    source = tmp_path / "source"
    _write(source, "private.json", '{"api_key":"OPAQUE_PRIVATE_TOKEN"}')
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="OPAQUE_" PRIVATE_="" b="' + tail + '"></x>',
    )
    with pytest.raises(ValueError, match="private-source-literal") as caught:
        privacy.validate_artifact(output, source)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


def test_ordered_deny_literal_scan_does_not_require_a_raw_source_body(tmp_path):
    source = tmp_path / "source"
    _write(
        source, "docs/wiki/private.json",
        '{"api_key":"OPAQUE_PRIVATE_TOKEN"}',
    )
    assert privacy._source_bodies(source)[1] == []
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        '<x a="OPAQUE_" PRIVATE_="" b="TOKEN"></x>',
    )
    with pytest.raises(ValueError, match="private-source-literal") as caught:
        privacy.validate_artifact(output, source)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


def test_deny_literal_survives_many_visible_markup_gaps(tmp_path):
    source = tmp_path / "source"
    _write(source, "private.json", '{"api_key":"OPAQUE_PRIVATE_TOKEN"}')
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        "O" + "<x></x>" * 24 + "PAQUE_PRIVATE_TOKEN",
    )
    with pytest.raises(ValueError, match="private-source-literal") as caught:
        privacy.validate_artifact(output, source)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x OPAQUE_=""></x><!--public-->PRIVATE_TOKEN',
    '<x OPAQUE_=""></x><span title="public"></span>PRIVATE_TOKEN',
    '<x OPAQUE_="public"></x>PRIVATE_TOKEN',
    'O' + '<x></x>' * 9 + '<x PAQUE_PRIVATE_TOKEN=""></x>',
])
def test_deny_literal_skips_nonvisible_html_channel_noise(tmp_path, payload):
    source = tmp_path / "source"
    _write(
        source, "docs/wiki/private.json",
        '{"api_key":"OPAQUE_PRIVATE_TOKEN"}',
    )
    output = tmp_path / "artifact"
    _write(output, "index.html", payload)
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


def test_deny_literal_split_across_json_values_is_rejected(tmp_path):
    source = tmp_path / "source"
    _write(source, "private.json", '{"api_key":"OPAQUE_PRIVATE_TOKEN"}')
    output = tmp_path / "artifact"
    _write(output, "data.json", json.dumps(["OPAQUE_", "PRIVATE_TOKEN"]))
    with pytest.raises(ValueError, match="private-source-literal") as caught:
        privacy.validate_artifact(output, source)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x AKIA="" ABCDEFGHIJKLMNOP=""></x>',
    'boy<x title="public"></x>ang.yu',
    '<x AKIA="public"></x>ABCDEFGHIJKLMNOP',
    '<x AKIA=""></x><!--public-->ABCDEFGHIJKLMNOP',
    '<x AKIA="" inert="ABCDEFGHIJKLMNOP"></x>',
    '<x AKIA="public" inert="ABCDEFGHIJKLMNOP"></x>',
    '<x a="AKIA" public="noise" b="ABCDEFGHIJKLMNOP"></x>',
    '<x a="AKIA" inert="" ABCDEFGHIJKLMNOP=""></x>',
    '<AKIA public="noise" inert="ABCDEFGHIJKLMNOP"></AKIA>',
    '<AKIA public="noise">ABCDEFGHIJKLMNOP</AKIA>',
    '<AKIA></AKIA><!--ABCDEFGHIJKLMNOP-->',
    '<x a="AKIA" public="noise">ABCDEFGHIJKLMNOP</x>',
])
def test_private_patterns_split_across_html_projections_are_rejected(
    tmp_path, payload,
):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    ["AKIA", "ABCDEFGHIJKLMNOP"],
    ["boy", "ang.yu"],
])
def test_private_patterns_split_across_json_projections_are_rejected(
    tmp_path, payload,
):
    _write(tmp_path, "data.json", json.dumps(payload))
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload,rule", [
    (["AKIA", "!", "ABCDEFGHIJKLMNOP"], "private-pattern"),
    (["user@", "!", "example.com"], "private-pattern"),
    (["api_", "!", "key=OPAQUE_PRIVATE_TOKEN"], "private-field-assignment"),
])
def test_private_material_crosses_selectable_json_scalars(
    tmp_path, payload, rule,
):
    _write(tmp_path, "data.json", json.dumps(payload))
    with pytest.raises(ValueError, match=rule):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("literal,payload", [
    ("OPAQUE_PRIVATE_TOKEN", ["OPAQUE_", "!", "PRIVATE_TOKEN"]),
    ("Token", ["To", "!", "ken"]),
])
def test_source_material_crosses_selectable_json_scalars(
    tmp_path, literal, payload,
):
    source = tmp_path / "source"
    _write(source, "private.json", json.dumps({"api_key": literal}))
    output = tmp_path / "artifact"
    _write(output, "data.json", json.dumps(payload))
    with pytest.raises(ValueError, match="private-(?:source-literal|pattern)"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("payload", [
    '<x a="private.user@" public="#" b="example.com"></x>',
    '<x a="192.168." public="#" b="1.2"></x>',
    '<x a="private." public="#" b="internal"></x>',
    '<x a="login" public="#" b="123.private"></x>',
    '<x a="/" n1="#" b="home" n2="#" c="/" n3="#" '
    'd="PRIVATE_USER" n4="#" e="/" n5="#" f="private"></x>',
])
def test_broad_private_patterns_cross_selected_value_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<!--noise user@--><x title="!"></x><y title="example.com"></y>',
    '<!--noise 10.--><x title="!"></x><y title="0.0.1"></y>',
    '<!--noise foo.--><x title="!"></x><y title="internal"></y>',
])
def test_broad_private_patterns_start_inside_hidden_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<!--noise model.--><x title="!"></x><y title="ckpt"></y>',
    '<!--noise arti--><x title="!"></x><y title="facts/private"></y>',
    '<x title="noise /"><!--!--><!--home/private-->',
    '<x title="noise/"><!--!--><!--home/private-->',
    '<!--noise/--><!--!--><!--home/private-->',
    '<!--noise C:--><x title="!"></x><y title="\\private\\file"></y>',
    '<!--noise \\--><x title="!"></x><y title="\\host\\share"></y>',
])
def test_private_paths_start_inside_hidden_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


def test_relative_navigation_cannot_exempt_a_private_home_path(tmp_path):
    _write(tmp_path, "index.html", '<a href="../../home/private">unsafe</a>')
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload,target", [
    ('<!--noise model.--><!--!--><a href="ckpt.html">open</a>', "ckpt.html"),
    (
        '<!--noise arti--><!--!--><a href="facts/private.html">open</a>',
        "facts/private.html",
    ),
    (
        '<!--noise model.--><!--!--><!--ckpt.html-->'
        '<a href="ckpt.html">open</a>',
        "ckpt.html",
    ),
    (
        '<!--noise arti--><!--!--><!--facts/private.html-->'
        '<a href="facts/private.html">open</a>',
        "facts/private.html",
    ),
])
def test_safe_navigation_only_suppresses_new_path_starts(tmp_path, payload, target):
    _write(tmp_path, "index.html", payload)
    _write(tmp_path, target, "public")
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<x a="-----BEGIN " public="-" b="PRIVATE KEY-----"></x>',
    '<!-- -----BEGIN --><!-- - --><!-- PRIVATE KEY----- -->',
])
def test_private_key_header_crosses_selected_hidden_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<!-- token=OPAQUE_PRIVATE_TOKEN -->',
    '<!-- tok --><!-- en=OPAQUE_PRIVATE_TOKEN -->',
    '<!-- tok --><!-- public --><!-- en=OPAQUE_PRIVATE_TOKEN -->',
    '<!-- tok --><x title="public"></x><!-- en=OPAQUE_PRIVATE_TOKEN -->',
    '<!-- private_ --><x></x><!-- key=OPAQUE_PRIVATE_TOKEN -->',
    '<?tok?><!en=OPAQUE_PRIVATE_TOKEN>',
])
def test_private_assignments_cross_hidden_parser_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-field-assignment"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<!--noise customto--><x title="!" data-x="ken=OPAQUE_PRIVATE_TOKEN"></x>',
    '<!--noise api_ses--><x title="!" data-x="sion=OPAQUE_PRIVATE_TOKEN"></x>',
    '<!--noise private_--><x title="!" data-x="key=OPAQUE_PRIVATE_TOKEN"></x>',
    '<x title="noise customto" data-x="!"></x><!--ken=OPAQUE_PRIVATE_TOKEN-->',
])
def test_generic_private_assignments_cross_hidden_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-(?:field-assignment|pattern)"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<?user@?><x title="!"></x><y title="example.com"></y>',
    '<?customto?><x title="!"></x><y title="ken=OPAQUE_PRIVATE_TOKEN"></y>',
    '<customto title="!"></customto><y title="ken=OPAQUE_PRIVATE_TOKEN"></y>',
    '<x customto=""></x><y title="!"></y>'
    '<z title="ken=OPAQUE_PRIVATE_TOKEN"></z>',
])
def test_private_values_cross_structural_hidden_channels(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="private-(?:field-assignment|pattern)"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("wrapper", [
    "<template>!</template>",
    "<span hidden>!</span>",
    "<head>!</head>",
    '<span style="display:none">!</span>',
    '<span style="visibility: hidden !important">!</span>',
    "<dialog>!</dialog>",
])
@pytest.mark.parametrize("left,right", [
    ("AKIA", "ABCDEFGHIJKLMNOP"),
    ("user@", "example.com"),
    ("customto", "ken=OPAQUE_PRIVATE_TOKEN"),
])
def test_nonvisible_data_does_not_reset_hidden_reconstruction(
    tmp_path, wrapper, left, right,
):
    _write(
        tmp_path, "index.html",
        f'<x title="{left}"></x>{wrapper}<y title="{right}"></y>',
    )
    with pytest.raises(ValueError, match="private-(?:field-assignment|pattern)"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("wrapper", [
    "<title>!</title>",
    "<datalist>!</datalist>",
    "<details>!</details>",
    r'<span style="display\3a none">!</span>',
    r'<span style="\64 isplay:none">!</span>',
    r'<span style="visibility\3a hidden">!</span>',
])
def test_default_hidden_and_css_escaped_hidden_content_remains_selectable(
    tmp_path, wrapper,
):
    _write(
        tmp_path, "index.html",
        f'<x title="AKIA"></x>{wrapper}'
        '<y title="ABCDEFGHIJKLMNOP"></y>',
    )
    with pytest.raises(ValueError, match="private-pattern"):
        privacy.validate_artifact(tmp_path)


def test_open_details_content_resets_hidden_reconstruction(tmp_path):
    _write(
        tmp_path, "index.html",
        '<x title="AKIA"></x><details open>!</details>'
        '<y title="ABCDEFGHIJKLMNOP"></y>',
    )
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def test_nonvisible_data_does_not_reset_source_literal_reconstruction(tmp_path):
    literal = "ABCDEFGHJKLMNOPQRSTUV"
    source = tmp_path / "source"
    _write(source, "configs/private.json", json.dumps({"api_key": literal}))
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        '<x title="ABCDEFGH"></x><template>!</template>'
        '<y title="JKLMNOPQRSTUV"></y>',
    )
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("opening", ["<template/>", "<span hidden/>", "<head/>"])
def test_nonvoid_self_close_is_rejected_as_browser_ambiguous(tmp_path, opening):
    _write(tmp_path, "index.html", opening + "!<p>public</p>")
    with pytest.raises(ValueError, match="unparseable-html-markup"):
        privacy.validate_artifact(tmp_path)


def test_private_host_suffix_is_case_insensitive_but_python_enum_is_public():
    for value in ("server.LOCAL", "host.INTERNAL", "foo.DESY.DE"):
        assert "private-pattern-13" in privacy.violations(value)
    assert privacy.violations(
        "hypertagging.data.selection_repromotion_publication.LOCAL_CONTROLLER_VERSION"
    ) == []


@pytest.mark.parametrize("value", [
    "hypertagging.data.selection_repromotion_publication.LOCAL_CONTROLLER_VERSION",
    "foo.locality",
    "192.168.1.2x",
    "model.ckptools",
])
def test_ordered_hidden_scanner_preserves_regex_word_boundaries(tmp_path, value):
    _write(tmp_path, "index.html", f'<x title="{value}"></x>')
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


@pytest.mark.parametrize("prefix_width", [7, 8])
def test_ordered_private_literal_does_not_expire_in_nonvisible_markup(
    tmp_path, prefix_width,
):
    literal = "ABCDEFGHJKLMNOPQRSTUV"
    source = tmp_path / "source"
    _write(source, "docs/wiki/private.json", json.dumps({"api_key": literal}))
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        '<x a="' + literal[:prefix_width] + '"></x>'
        + "<!--z-->" * 100
        + '<x ' + literal[prefix_width:] + '=""></x>',
    )
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("payload", [
    "XOPAQUE_PRIVATE_TOKENY", "OPAQUE_PRIVATE_TOKENY",
    "XOPAQUE_PRIVATE_TOKEN",
])
def test_full_deny_literal_is_private_inside_alphanumeric_carrier(
    tmp_path, payload,
):
    source = tmp_path / "source"
    _write(source, "private.json", '{"api_key":"OPAQUE_PRIVATE_TOKEN"}')
    output = tmp_path / "artifact"
    _write(output, "report.txt", payload)
    with pytest.raises(ValueError, match="private-source-literal") as caught:
        privacy.validate_artifact(output, source)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("relative,payload", [
    ("index.html", '<script type="application/json">{"token":"OPAQUE_PRIVATE_TOKEN"}</script>'),
    ("index.html", '<div data-config=\'{"api_key":"OPAQUE_PRIVATE_TOKEN"}\'></div>'),
    ("_static/custom.js", 'const state = {token: "OPAQUE_PRIVATE_TOKEN"};'),
    ("_static/custom.js", 'const secret = "OPAQUE_PRIVATE_TOKEN";'),
    ("_static/custom.js", 'state["token"] = "OPAQUE_PRIVATE_TOKEN";'),
    ("_static/custom.js", 'const state = {["token"]: "OPAQUE_PRIVATE_TOKEN"};'),
    ("_static/custom.js", 'const token /* comment */ = "OPAQUE_PRIVATE_TOKEN";'),
])
def test_sensitive_assignments_in_embedded_json_and_javascript_are_rejected(tmp_path, relative, payload):
    _write(tmp_path, relative, payload)
    with pytest.raises(ValueError, match="private-field-assignment") as caught:
        privacy.validate_artifact(tmp_path)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    "token: OPAQUE_PRIVATE_TOKEN",
    "private_key: OPAQUE_PRIVATE_TOKEN",
    "ssh_key: OPAQUE_PRIVATE_TOKEN",
    "session: OPAQUE_PRIVATE_TOKEN",
    "cookie: OPAQUE_PRIVATE_TOKEN",
    "job_id: 982731",
    "checkpoint_path: private-value",
    "session:token=OPAQUE_PRIVATE_TOKEN",
    "auth:private_key=OPAQUE_PRIVATE_TOKEN",
    "token: str.OPAQUE_PRIVATE_TOKEN",
    "token: str|OPAQUE_PRIVATE_TOKEN",
    "token: str[OPAQUE_PRIVATE_TOKEN]",
    "token: Path)OPAQUE_PRIVATE_TOKEN",
    "token: dict,OPAQUE_PRIVATE_TOKEN",
])
def test_unquoted_sensitive_fields_and_type_prefixes_are_rejected(payload):
    assert "private-field-assignment" in privacy.violations(payload)


@pytest.mark.parametrize("payload", [
    "job_id: int", "checkpoint_path: Path", "token: str",
    "token: torch.Tensor", "authorization: False", "id-token: write",
])
def test_complete_nonprivate_type_annotations_and_empty_values_are_allowed(payload):
    assert privacy.violations(payload) == []


def test_rst_type_option_is_not_a_sensitive_value_but_multiline_assignment_is():
    assert privacy.violations("token\n   :type: int") == []
    assert "private-field-assignment" in privacy.violations(
        "token\n   : OPAQUE_PRIVATE_TOKEN"
    )


@pytest.mark.parametrize("payload", [
    "token: str # OPAQUE_PRIVATE_TOKEN",
    "token: str; OPAQUE_PRIVATE_TOKEN",
    "token: str¶OPAQUE_PRIVATE_TOKEN",
    "token: str→OPAQUE_PRIVATE_TOKEN",
    "token: str\nOPAQUE_PRIVATE_TOKEN",
    "token: str = None; OPAQUE_PRIVATE_TOKEN",
    "token: int, name: str OPAQUE_PRIVATE_TOKEN",
    "token: int) -> str OPAQUE_PRIVATE_TOKEN",
    "token: int = None, name: str OPAQUE_PRIVATE_TOKEN",
    "token: int\nSource: src/x.py:1 OPAQUE_PRIVATE_TOKEN",
])
def test_type_annotation_exemption_consumes_the_complete_value(payload):
    assert "private-field-assignment" in privacy.violations(payload)


@pytest.mark.parametrize("payload", [
    "token: int, *, name: str = ...",
    "checkpoint_path: str | Path, configuration: Mapping[str, Any] = ...",
    "mother_charge_by_token: tuple[tuple[int, float], ...]",
    "token: int¶\nSource: src/package/module.py:1",
    'token: int\n\n   Source: "src/package/module.py:1".',
    "token: str, job_name: str | None=...) -> tuple[list[str], Path]",
])
def test_complete_generated_annotation_contexts_remain_allowed(payload):
    assert privacy.violations(payload) == []


def test_excessive_annotation_line_fails_closed_without_suffix_reparsing():
    payload = ", ".join(f"x{index}_token: int" for index in range(5_000))
    assert "private-field-assignment" in privacy.violations(payload)


def test_repeated_unterminated_private_key_headers_fail_fast():
    payload = "-----BEGIN X PRIVATE KEY-----" * 10_000
    assert "private-pattern-0" in privacy.violations(payload)
    assert privacy.redact(payload) == privacy.REDACTED


@pytest.mark.parametrize("payload", [
    '<meta name="api_key" name="description" content="OPAQUE_PRIVATE_TOKEN">',
    '<input name="password" name="description" value="OPAQUE_PRIVATE_TOKEN">',
    '<div DATA-ID="first" data-id="second"></div>',
])
def test_html_duplicate_attributes_cannot_discard_sensitive_semantics(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="duplicate-html-attribute") as caught:
        privacy.validate_artifact(tmp_path)
    assert "OPAQUE_PRIVATE_TOKEN" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x data-a="/ho"></x><!--me/PRIVATE_--><p>USER/private</p>',
    '<x data-a="/ho" data-b="me/PRIVATE_" data-c="USER/private"></x>',
])
def test_html_private_material_split_across_content_channels_is_rejected(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<script>document.body.textContent=["/","home","/","PRIVATE_USER","/","private"].join("")</script>',
    '<script>document.body.textContent=atob("L2hvbWUvUFJJVkFURV9VU0VSL3ByaXZhdGU=")</script>',
    '<iframe src="data:text/html;base64,L2hvbWUvUFJJVkFURV9VU0VSL3ByaXZhdGU="></iframe>',
    '<div onclick="document.body.textContent=atob(\'L2hvbWU=\')">safe</div>',
    '<style>body::before{content:"/" "home" "/" "PRIVATE_USER" "/private"}</style>',
    '<div style="content:attr(data-private)" data-private="safe"></div>',
    '<img srcset="safe.png 1x, data:image/svg+xml;base64,L2hvbWU= 2x">',
    '<base href="//evil.invalid/">',
    '<meta http-equiv="refresh" content="0;url=data:text/html;base64,L2hvbWU=">',
])
def test_active_html_cannot_synthesize_unscannable_private_material(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<svg><image href="https://docs.python.org/3/_static/py.svg"/></svg>',
    '<svg><use xlink:href="https://docs.python.org/3/icon.svg#mark"/></svg>',
    '<video poster="https://docs.python.org/3/_static/poster.png"></video>',
    '<track src="https://docs.python.org/3/_static/captions.vtt">',
    '<input type="image" src="https://docs.python.org/3/_static/submit.png">',
    '<input formaction="https://docs.python.org/3/search.html">',
    '<button formaction="https://docs.python.org/3/search.html">go</button>',
    '<input type="submit" formmethod="post">',
    '<button type="submit" formmethod="post">go</button>',
    '<button type="submit" formmethod="get">go</button>',
    '<form action="https://docs.python.org/3/search.html"></form>',
    '<object data="https://docs.python.org/3/object.bin"></object>',
    '<body background="https://docs.python.org/x.png"></body>',
    '<table><colgroup background="https://docs.python.org/x.png"><col></colgroup><tr><td>x</td></tr></table>',
    '<table><colgroup><col background="https://docs.python.org/x.png"></colgroup><tr><td>x</td></tr></table>',
    '<table><thead background="https://docs.python.org/x.png"><tr><th>x</th></tr></thead></table>',
    '<table><tbody background="https://docs.python.org/x.png"><tr><td>x</td></tr></tbody></table>',
    '<table><tfoot background="https://docs.python.org/x.png"><tr><td>x</td></tr></tfoot></table>',
    '<table><tr background="https://docs.python.org/x.png"><td>x</td></tr></table>',
    '<a href="#" ping="https://docs.python.org/collect">docs</a>',
    '<link rel="preload" as="image" imagesrcset="https://docs.python.org/x.png 1x">',
    '<link rel="preload" as="image" imagesrcset="%68ttps://docs.python.org/x.png 1x">',
    '<img attributionsrc="https://docs.python.org/report">',
    '<img src="_static/x.png" attributionsrc>',
    '<img src="_static/x.png" attributionsrc="">',
    '<script attributionsrc="//docs.python.org/report"></script>',
    '<a href="#" attributionsrc="https://docs.python.org/report">docs</a>',
    '<area href="#" attributionsrc="https://docs.python.org/report">',
])
def test_runtime_and_submission_attributes_are_rejected_by_both_validators(
    tmp_path, payload,
):
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)

    offline_site = _validation_site(tmp_path / "offline", payload)
    with pytest.raises(ValueError):
        validation.validate_html(offline_site)


def test_backslash_protocol_relative_runtime_asset_is_rejected(tmp_path):
    payload = r'<img src="\\docs.python.org/x.png">'
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)

    offline_site = _validation_site(tmp_path / "offline", payload)
    _write(offline_site, r"\\docs.python.org/x.png", "decoy")
    with pytest.raises(ValueError, match="Remote runtime asset"):
        validation.validate_html(offline_site)


def test_external_documentation_anchor_remains_allowed_by_both_validators(tmp_path):
    payload = '<a href="https://docs.python.org/3/library/ast.html">AST</a>'
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    assert privacy.validate_artifact(privacy_site)["privacy"] == "PASS"
    offline_site = _validation_site(tmp_path / "offline", payload)
    assert validation.validate_html(offline_site)["remote_assets"] == 0


def test_local_legacy_background_asset_remains_allowed(tmp_path):
    payload = '<body background="_static/background.txt">docs</body>'
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    _write(privacy_site, "_static/background.txt", "public")
    assert privacy.validate_artifact(privacy_site)["privacy"] == "PASS"
    offline_site = _validation_site(tmp_path / "offline", payload)
    _write(offline_site, "_static/background.txt", "public")
    assert validation.validate_html(offline_site)["remote_assets"] == 0


@pytest.mark.parametrize("action", ["../search.html", ""])
def test_local_get_search_forms_remain_allowed_by_both_validators(
    tmp_path, action,
):
    payload = '<form class="search" action="' + action + '" method="get"></form>'
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "search.html", "safe")
    _write(privacy_site, "deep/page.html", payload)
    assert privacy.validate_artifact(privacy_site)["privacy"] == "PASS"

    offline_site = _validation_site(tmp_path / "offline", "safe")
    _write(offline_site, "search.html", "safe")
    _write(offline_site, "deep/page.html", payload)
    assert validation.validate_html(offline_site)["remote_assets"] == 0


@pytest.mark.parametrize("payload", [
    '<form action="/search.html" method="get"></form>',
    '<form action="../outside.html" method="get"></form>',
    '<form action="%2e%2e/outside.html" method="get"></form>',
    '<form action="%252e%252e/outside.html" method="get"></form>',
    '<form action="&#46;&#46;/outside.html" method="get"></form>',
    '<form action="%68ttps://docs.python.org/3/search.html" method="get"></form>',
    '<form action="search.html" method="post"></form>',
])
def test_unsafe_search_form_targets_and_methods_are_rejected(tmp_path, payload):
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)
    offline_site = _validation_site(tmp_path / "offline", payload)
    with pytest.raises(ValueError):
        validation.validate_html(offline_site)


def test_backslash_form_traversal_cannot_use_a_local_decoy(tmp_path):
    payload = r'<form action="..\..\search.html" method="get"></form>'
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "deep/page.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)

    offline_site = _validation_site(tmp_path / "offline", "safe")
    _write(offline_site, "deep/page.html", payload)
    _write(offline_site, r"deep/..\..\search.html", "decoy")
    with pytest.raises(ValueError, match="Form action escapes publication"):
        validation.validate_html(offline_site)


def test_scheme_detection_does_not_match_metadata_words(tmp_path):
    _write(tmp_path, "index.html", '<div title="Metadata: public">Safe</div>')
    _write(tmp_path, "_static/safe.css", ":root{--metadata:public}")
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def test_external_css_cannot_synthesize_private_material(tmp_path):
    _write(
        tmp_path, "_static/custom.css",
        'body::before{content:"/" "home" "/" "PRIVATE_USER" "/private"}',
    )
    with pytest.raises(ValueError, match="untrusted-active-content") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '@import "\\68 ttps://github.com/example/theme.css";',
    '@import/**/"https://github.com/example/theme.css";',
    '@import "http:docs.python.org/theme.css";',
    '@import "\\68 ttp:docs.python.org/theme.css";',
    'body{background:url(/**/"https://docs.python.org/theme.css")}',
    'body{background:url(http:docs.python.org/theme.css)}',
    '@counter-style private{system:cyclic;symbols:"/" "home" "/PRIVATE_USER/private"}',
    'li{list-style-type:symbols(cyclic "/" "home" "/PRIVATE_USER/private")}',
    'li{list-style-type:"/home/PRIVATE_USER/private"}',
])
def test_external_css_cannot_load_decoded_or_comment_obfuscated_remote_assets(tmp_path, payload):
    _write(tmp_path, "_static/custom.css", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    r'a{background-image:url(\\\\evil.invalid/x)}',
    r'a{background-image:url(\5c\5c evil.invalid/x)}',
    r'a{background-image:url(\5c/evil.invalid/x)}',
])
def test_css_backslash_url_normalization_cannot_load_remote_assets(
    tmp_path, payload,
):
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "_static/custom.css", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)

    offline_site = _validation_site(tmp_path / "offline", "safe")
    _write(offline_site, "_static/custom.css", payload)
    with pytest.raises(ValueError, match="CSS"):
        validation.validate_html(offline_site)


def test_safe_sphinx_generated_content_punctuation_remains_allowed(tmp_path):
    _write(
        tmp_path, "_static/theme.css",
        "a::before{content: ':'} b::after{content: ''} "
        ".symbols:before{content: ':'} :root{--symbols:safe}",
    )
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def test_css_quotes_cannot_synthesize_a_deny_literal(tmp_path):
    source = tmp_path / "source"
    _write(source, "private.json", '{"api_key":"OPAQUE_PRIVATE_TOKEN"}')
    output = tmp_path / "artifact"
    _write(
        output, "_static/custom.css",
        'q{quotes:"OPAQUE_" "" "PRIVATE_TOKEN" ""}',
    )
    _write(output, "index.html", '<link rel="stylesheet" href="_static/custom.css"><q><q>x</q></q>')
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(output, source)
    with pytest.raises(ValueError, match="CSS"):
        validation.validate_html(output)


@pytest.mark.parametrize("payload,rule", [
    ("/*AKIA*//*ABCDEFGHIJKLMNOP*/", "private-pattern"),
    ("/*user@*//*example.com*/", "private-pattern"),
    ("/*api_*//*key=OPAQUE_PRIVATE_TOKEN*/", "private-field-assignment"),
])
def test_adjacent_css_comments_cannot_split_private_material(
    tmp_path, payload, rule,
):
    _write(tmp_path, "_static/custom.css", payload)
    with pytest.raises(ValueError, match=rule):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<x style="/*AKIA*//*ABCDEFGHIJKLMNOP*/">safe</x>',
    '<x style="/*user@*//*example.com*/">safe</x>',
    '<x style="/*api_*//*key=OPAQUE_PRIVATE_TOKEN*/">safe</x>',
])
def test_inline_css_comments_cannot_hide_split_private_material(tmp_path, payload):
    privacy_site = tmp_path / "privacy"
    _write(privacy_site, "index.html", payload)
    with pytest.raises(ValueError, match="untrusted-active-content"):
        privacy.validate_artifact(privacy_site)

    offline_site = _validation_site(tmp_path / "offline", payload)
    with pytest.raises(ValueError, match="inline CSS"):
        validation.validate_html(offline_site)


def test_svg_is_never_a_publishable_artifact(tmp_path):
    _write(tmp_path, "payload.svg", "<svg><script>document.body.textContent='private'</script></svg>")
    with pytest.raises(ValueError, match="forbidden-raw-download"):
        privacy.validate_artifact(tmp_path)


def test_untrusted_external_javascript_is_rejected_even_without_cleartext_secret(tmp_path):
    _write(tmp_path, "_static/custom.js", 'document.body.textContent = atob("L2hvbWU=")')
    with pytest.raises(ValueError, match="untrusted-active-script"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    r"\2f home\2f PRIVATE_USER\2f private.txt",
    r"\00002fhome\00002fPRIVATE_USER\00002fprivate.txt",
    r"\057home\057PRIVATE_USER\057private.txt",
    r"\/\h\o\m\e\/\P\R\I\V\A\T\E\_\U\S\E\R\/\p\r\i\v\a\t\e",
])
def test_css_and_javascript_octal_escaped_private_material_is_rejected(tmp_path, payload):
    _write(tmp_path, "report.txt", payload)
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


@pytest.mark.parametrize("depth", [9, 40])
def test_nested_encoding_is_decoded_or_fails_closed_without_disclosure(tmp_path, depth):
    payload = "/home/PRIVATE_USER/private"
    for _ in range(depth):
        payload = payload.replace("%", "%25").replace("/", "%2F")
    _write(tmp_path, "report.txt", payload)
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_untrusted_binary_asset_is_rejected_without_parsing_metadata(tmp_path):
    path = tmp_path / "_static" / "probe.png"
    path.parent.mkdir()
    path.write_bytes(b"\x89PNG\r\n\x1a\nprivate=/home/PRIVATE_USER/private")
    with pytest.raises(ValueError, match="untrusted-binary-asset") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_sphinx_inventory_payload_is_decompressed_and_scanned(tmp_path):
    header = (
        b"# Sphinx inventory version 2\n"
        b"# Project: Public\n"
        b"# Version: 1\n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    body = b"/home/PRIVATE_USER/private py:function 1 public.html public\n"
    (tmp_path / "objects.inv").write_bytes(header + zlib.compress(body))
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_malformed_sphinx_inventory_fails_closed(tmp_path):
    (tmp_path / "objects.inv").write_bytes(b"not an inventory /home/PRIVATE_USER/private")
    with pytest.raises(ValueError, match="unreadable-publication") as caught:
        privacy.validate_artifact(tmp_path)
    assert "PRIVATE_USER" not in str(caught.value)


def test_unknown_nested_source_fields_supply_deny_literals(tmp_path, capsys):
    source = tmp_path / "source"
    _write(source, "configs/secret.json", json.dumps({
        "newMetadata": [{"credentials": {"arbitrary": "OPAQUE_NESTED_TOKEN"}}],
        "identity": {"execution": {"userName": "PRIVATE_USER"}},
    }))
    _write(source, "schemas/fields.json", json.dumps({"properties": {
        "source_authorization": {"type": "string"},
        "checkpoint_path": {"type": "string", "description": "Path metadata"},
    }}))
    literals = privacy.sensitive_literals(source)
    assert {"OPAQUE_NESTED_TOKEN", "PRIVATE_USER"}.issubset(literals)
    assert "string" not in literals
    assert "Path metadata" not in literals
    assert capsys.readouterr().out == ""
    output = tmp_path / "artifact"
    _write(output, "data.json", '{"unrecognized": "OPAQUE_NESTED_TOKEN"}')
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("key", [
    "accessKeyId", "access_key_id", "awsAccessKeyId",
])
def test_access_key_identifiers_are_sensitive_fields(tmp_path, key):
    source = tmp_path / "source"
    literal = "OPAQUE_CLOUD_ID_123456"
    _write(source, "private.json", json.dumps({key: literal}))
    assert literal in privacy.sensitive_literals(source)
    assert "private-field-assignment" in privacy.violations(
        f"{key}={literal}",
    )
    output = tmp_path / "artifact"
    _write(output, "report.txt", literal)
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("relative,payload", [
    ("configs/broken.yaml", "password: [PRIVATE_YAML_VALUE\n"),
    ("configs/tagged.yaml", "password: !private PRIVATE_YAML_VALUE\n"),
    ("configs/broken.json", '{"password": "PRIVATE_JSON_VALUE"'),
])
def test_unreadable_structured_privacy_source_fails_closed_without_disclosure(tmp_path, relative, payload):
    source = tmp_path / "source"
    _write(source, relative, payload)
    with pytest.raises(ValueError, match="Unreadable structured privacy source") as caught:
        privacy.sensitive_literals(source)
    assert "PRIVATE_YAML_VALUE" not in str(caught.value)
    assert "PRIVATE_JSON_VALUE" not in str(caught.value)
    assert str(source) not in str(caught.value)


@pytest.mark.parametrize("relative,payload", [
    ("configs/duplicate.json", '{"path":"/home/PRIVATE_USER/private","\\u0070ath":"safe"}'),
    ("configs/duplicate.yaml", 'path: /home/PRIVATE_USER/private\n"\\u0070ath": safe\n'),
    ("configs/duplicate.JSON", '{"path":"/home/PRIVATE_USER/private","\\u0070ath":"safe"}'),
    ("configs/duplicate.YmL", 'path: /home/PRIVATE_USER/private\n"\\u0070ath": safe\n'),
])
def test_duplicate_source_keys_fail_closed_including_encoded_spellings(tmp_path, relative, payload):
    source = tmp_path / "source"
    _write(source, relative, payload)
    with pytest.raises(ValueError, match="Duplicate key in structured privacy source") as caught:
        privacy.sensitive_literals(source)
    assert "PRIVATE_USER" not in str(caught.value)


def test_schema_values_and_yaml_binary_private_values_supply_deny_literals(tmp_path):
    source = tmp_path / "source"
    _write(source, "schemas/fields.json", json.dumps({"properties": {
        "checkpoint_path": {
            "type": "string", "description": "Public description",
            "default": "OPAQUE_SCHEMA_DEFAULT", "enum": ["OPAQUE_SCHEMA_ENUM"],
        },
    }}))
    _write(source, "configs/secret.yaml", "credentials: !!binary T1BBUVVFX1BSSVZBVEVfVE9LRU4=\n")
    literals = privacy.sensitive_literals(source)
    assert {"OPAQUE_SCHEMA_DEFAULT", "OPAQUE_SCHEMA_ENUM", "OPAQUE_PRIVATE_TOKEN"}.issubset(literals)
    assert "Public description" not in literals


def test_nested_schema_containers_preserve_private_defaults(tmp_path):
    source = tmp_path / "source"
    _write(source, "schemas/nested.json", json.dumps({
        "credentials": {
            "type": "object",
            "properties": {
                "password": {"type": "string", "default": "OPAQUE_NESTED_SCHEMA_TOKEN"},
            },
        },
    }))
    assert "OPAQUE_NESTED_SCHEMA_TOKEN" in privacy.sensitive_literals(source)


def test_oversize_structured_privacy_source_fails_closed(tmp_path):
    source = tmp_path / "source"
    _write(
        source, "configs/large.json",
        '{"api_key":"OPAQUE_PRIVATE_TOKEN","padding":"'
        + "Z" * privacy._MAX_SOURCE_TEXT_BYTES + '"}',
    )
    with pytest.raises(ValueError, match="exceeds scan limit"):
        privacy.sensitive_literals(source)


def test_oversize_readable_source_fails_closed_before_markup_split(tmp_path):
    source = tmp_path / "source"
    _write(
        source, "configs/large.txt",
        "Z" * (privacy._MAX_SOURCE_TEXT_BYTES + 1),
    )
    output = tmp_path / "artifact"
    _write(output, "index.html", "safe")
    with pytest.raises(ValueError, match="raw-body scan limit"):
        privacy.validate_artifact(output, source)


def test_utf8_signature_cannot_hide_a_markup_split_source_body(tmp_path):
    body = "OPAQUE_UNCLASSIFIED_SOURCE_BODY_123456789"
    source = tmp_path / "source"
    path = source / "configs/public.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    output = tmp_path / "artifact"
    _write(output, "index.html", "<span>" + body[:20] + "</span>" + body[20:])
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_non_utf8_classified_text_source_fails_closed(tmp_path):
    source = tmp_path / "source"
    path = source / "configs/public.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes("OPAQUE_SOURCE_BODY".encode("utf-16"))
    output = tmp_path / "artifact"
    _write(output, "index.html", "safe")
    with pytest.raises(ValueError, match="Unreadable classified text source"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize(
    "key",
    [
        "private_key", "ssh_key", "token", "session_id", "host",
        "hostname", "host_name", "apikey", "accesstoken", "authtoken",
        "privatekey", "sshkey", "sessionid", "jobid", "schedulerid",
        "authoritypath", "checkpointpath", "gpuuuid",
        "refreshtoken", "deploytoken", "apitoken", "fooprivatekey",
        "foosshkey", "sessioncookie", "motherchargebytoken",
        "clientsecret", "dbpassword", "loginpasswd", "servicecredentials",
        "accountusername", "serverhostname",
        "httpauthorization", "clientauthorization", "authheader",
        "clientauth", "clientbearer", "usersession", "usersessionid",
        "dbhost", "serverhost", "accesskey", "apiaccesskey",
        "privatekeyid", "sshkeypath", "httpAuthorization",
        "proxyAuthorization", "authHeader", "secretkey", "authkey",
        "signingkey", "encryptionkey", "passphrase", "dbpass", "pwd",
    ],
)
def test_exact_sensitive_key_names_are_private(key):
    assert privacy._private_key(key)


@pytest.mark.parametrize("key", [
    "apikey", "accesstoken", "authtoken", "privatekey", "sshkey",
    "sessionid", "jobid", "schedulerid", "authoritypath",
    "checkpointpath", "gpuuuid",
    "refreshtoken", "deploytoken", "apitoken", "fooprivatekey",
    "foosshkey", "sessioncookie", "motherchargebytoken",
    "clientsecret", "dbpassword", "loginpasswd", "servicecredentials",
    "accountusername", "serverhostname",
    "httpauthorization", "clientauthorization", "authheader", "clientauth",
    "clientbearer", "usersession", "usersessionid", "dbhost", "serverhost",
    "accesskey", "apiaccesskey", "privatekeyid", "sshkeypath",
    "secretkey", "authkey", "signingkey", "encryptionkey", "passphrase",
    "dbpass", "pwd",
])
def test_collapsed_sensitive_source_keys_supply_deny_literals(tmp_path, key):
    source = tmp_path / "source"
    _write(source, "private.json", json.dumps({key: "OPAQUE_PRIVATE_TOKEN"}))
    assert "OPAQUE_PRIVATE_TOKEN" in privacy.sensitive_literals(source)
    output = tmp_path / "artifact"
    _write(output, "report.txt", "OPAQUE_PRIVATE_TOKEN")
    with pytest.raises(ValueError, match="private-source-literal"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("key", [
    "refreshtoken", "deploytoken", "apitoken", "fooprivatekey",
    "foosshkey", "sessioncookie", "motherchargebytoken",
    "clientsecret", "dbpassword", "loginpasswd", "servicecredentials",
    "accountusername", "serverhostname",
    "httpauthorization", "clientauthorization", "authheader", "clientauth",
    "clientbearer", "usersession", "usersessionid", "dbhost", "serverhost",
    "accesskey", "apiaccesskey", "privatekeyid", "sshkeypath",
    "secretkey", "authkey", "signingkey", "encryptionkey", "passphrase",
    "dbpass", "pwd",
])
def test_collapsed_sensitive_assignment_suffixes_are_private(key):
    assert "private-field-assignment" in privacy.violations(
        key + "=OPAQUE_PRIVATE_TOKEN"
    )


def test_host_metric_key_does_not_supply_a_private_literal(tmp_path):
    assert not privacy._private_key("host_peak_memory")
    source = tmp_path / "source"
    _write(source, "status.json", '{"host_peak_memory":"MEGABYTES"}')
    assert "MEGABYTES" not in privacy.sensitive_literals(source)


def test_artifact_scans_injected_filenames_and_hides_them_in_errors(tmp_path):
    _write(tmp_path, "GPU-1234abcd-1234-abcd-1234-123456789abc.txt", "Safe contents")
    with pytest.raises(ValueError) as caught:
        privacy.validate_artifact(tmp_path)
    assert "GPU-1234abcd" not in str(caught.value)


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


@pytest.mark.parametrize("source_name", [
    "scripts/public.py", "reports/public.html", "reports/metrics.csv",
    "environment/runtime.lock", "configs/runtime.cfg",
])
@pytest.mark.parametrize("render_html", [False, True])
def test_raw_source_body_cannot_hide_in_innocent_text_or_html_download(tmp_path, source_name, render_html):
    source = tmp_path / "source"
    body = 'def public(value):\n    """Documented callable."""\n    result = "RAW_SOURCE_BODY_CANARY_UNRELATED_TO_SECRETS"\n    return result + value\n'
    _write(source, source_name, body)
    output = tmp_path / "artifact"
    if render_html:
        text = '<html><body><pre>' + ''.join('<span>' + html.escape(line) + '</span>\n' for line in body.splitlines()) + '</pre></body></html>'
    else:
        text = body
    _write(output, "innocent.html" if render_html else "innocent.txt", text)
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, source)
    assert "RAW_SOURCE_BODY_CANARY" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    '<x data-a="{first}"></x><!--{second}-->',
    '<x data-a="{first}" data-b="{second}"></x>',
])
def test_raw_source_body_split_across_html_channels_is_rejected(tmp_path, payload):
    source = tmp_path / "source"
    body = "RAW_SOURCE_BODY_CANARY_" * 4
    _write(source, "configs/public.cfg", body)
    midpoint = len(body) // 2
    output = tmp_path / "artifact"
    _write(output, "index.html", payload.format(first=body[:midpoint], second=body[midpoint:]))
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, source)
    assert "RAW_SOURCE_BODY_CANARY" not in str(caught.value)


def test_raw_source_body_in_trailing_unclosed_html_comment_is_rejected(tmp_path):
    source = tmp_path / "source"
    body = "PROPRIETARY_IMPLEMENTATION_CANARY_" * 3
    _write(source, "configs/public.cfg", body)
    midpoint = len(body) // 2
    output = tmp_path / "artifact"
    _write(output, "index.html", body[:midpoint] + "<!--" + body[midpoint:])
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("length", [27, 63])
def test_complete_short_source_body_is_rejected_inside_a_larger_artifact(tmp_path, length):
    source = tmp_path / "source"
    body = "Q" * length
    _write(source, "configs/short.cfg", body)
    output = tmp_path / "artifact"
    _write(output, "report.txt", "safe-prefix\n" + body + "\nsafe-suffix")
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_raw_source_body_split_across_html_tag_and_attribute_names_is_rejected(tmp_path):
    source = tmp_path / "source"
    body = "opaquename"
    _write(source, "configs/public.cfg", body)
    output = tmp_path / "artifact"
    _write(output, "index.html", "<opaque name></opaque>")
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("body,payload", [
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<OPAQUE_ PRIVATE_TOKEN="">x</OPAQUE_>'),
    ("OpAqUe_PrIvAtE_ToKeN", '<x OpAqUe_="" PrIvAtE_ToKeN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x></OPAQUE_></PRIVATE_TOKEN>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a=foo/bar OPAQUE_="" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a/ OPAQUE_="" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x/ OPAQUE_="" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a / OPAQUE_="" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="PRIVATE_TOKEN">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a="OPAQUE_" PRIVATE_TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="" a="PRIVATE_TOKEN">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="PRIVATE_" TOKEN="">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="">PRIVATE_TOKEN</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<OPAQUE_>PRIVATE_TOKEN</OPAQUE_>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_=""><!--PRIVATE_TOKEN--></x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a="OPAQUE_"></x><p PRIVATE_TOKEN=""></p>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_=""></x><?q PRIVATE_TOKEN?>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_=""></x><![if PRIVATE_TOKEN]>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x XXOPAQUE_="">PRIVATE_TOKEN</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<XXOPAQUE_>PRIVATE_TOKEN</XXOPAQUE_>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x XXOPAQUE_=""></x><!--PRIVATE_TOKEN-->'),
    ("OPAQUE_PRIVATE_TOKEN", '<x a="XXOPAQUE_"></x><p PRIVATE_TOKEN=""></p>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_=""></x><p XXPRIVATE_TOKEN=""></p>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_=""></x><p XXPRIVATE_TOKENYY=""></p>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="\\50 RIVATE_TOKEN">x</x>'),
    ("OPAQUE_PRIVATE_TOKEN", '<x OPAQUE_="\\120RIVATE_TOKEN">x</x>'),
    ("OPAQUE PRIVATE_TOKEN", '<OPAQUE>\\20PRIVATE_TOKEN</OPAQUE>'),
    ("OPAQUE PRIVATE_TOKEN", '<OPAQUE>\\20<PRIVATE_TOKEN></PRIVATE_TOKEN>'),
    ("OPAQUE PRIVATE_TOKEN", '<x a="OPAQUE\\20"></x><PRIVATE_TOKEN></PRIVATE_TOKEN>'),
])
def test_raw_source_matching_preserves_tag_and_attribute_name_case(tmp_path, body, payload):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", body)
    output = tmp_path / "artifact"
    _write(output, "index.html", payload)
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("body,tokens", [
    ("opaquename", ["opaque", "XXnameYY"]),
    ("ABCDEFGHIJKL", ["ABCDEFGH", "xxIJKLyy"]),
    ("ABCDEFGHIJKLMNO", ["ABCDEFGH", "I", "JK", "LM", "xxNOyy"]),
])
def test_ordered_raw_matcher_handles_short_carriers_and_noisy_tail(body, tokens):
    assert privacy._ordered_token_stream_contains([body], tokens)


@pytest.mark.parametrize("first_width", range(1, 6))
def test_ordered_raw_matcher_bootstraps_one_to_five_character_carriers(first_width):
    body = "abcdefghijk"
    assert privacy._ordered_token_stream_contains(
        [body],
        [
            (body[:first_width], False, True),
            (body[first_width:8], True, True),
            (body[8:], False, False),
        ],
    )


@pytest.mark.parametrize("fragment_width", range(1, 6))
def test_ordered_raw_matcher_bootstraps_fragments_inside_noisy_carriers(fragment_width):
    body = "abcdefghijk"
    assert privacy._ordered_token_stream_contains(
        [body],
        [
            ("XX" + body[:fragment_width] + "YY", True, True),
            (body[fragment_width:], False, False),
        ],
    )


def test_ordered_raw_matcher_keeps_shorter_embedded_branch_start():
    assert privacy._ordered_token_stream_contains(
        ["abX12345", "abc12345"],
        [
            ("ZZabcYY", True, True),
            ("X12345", False, False),
        ],
    )


def test_ordered_raw_matcher_keeps_shorter_embedded_branch_progress():
    assert privacy._ordered_token_stream_contains(
        ["abX12345", "abc12345"],
        [
            ("a", False, True),
            ("ZZbcYY", True, True),
            ("X12345", False, False),
        ],
    )


@pytest.mark.parametrize("tokens", [
    [
        ("a", False, True),
        ("XXbcdefghijYY", True, True),
        ("klmnop", False, False),
    ],
    [
        ("XXabcdefghijYY", True, True),
        ("klmnop", False, False),
    ],
])
def test_ordered_raw_matcher_keeps_maximal_run_after_width_anchor(tokens):
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"], tokens,
    )


def test_ordered_raw_matcher_advances_established_state_inside_noisy_carrier():
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"],
        [
            ("abcdefgh", False, True),
            ("XXijkYY", True, True),
            ("lmnop", False, False),
        ],
    )


@pytest.mark.parametrize("tokens", [
    [
        ("XXabcdefghijYY", True, True),
        ("jklmnop", False, False),
    ],
    [
        ("abcdefgh", False, True),
        ("XXijkYY", True, True),
        ("jklmnop", False, False),
    ],
    [
        ("a", False, True),
        ("XXbcdefghijYY", True, True),
        ("jklmnop", False, False),
    ],
])
def test_ordered_raw_matcher_retains_each_noisy_run_endpoint(tokens):
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"], tokens,
    )


@pytest.mark.parametrize("body,tokens", [
    (
        "abcdefgh ijklmnop",
        [
            ("abcdefgh ", False, False),
            (" ", False, True),
            ("ijklmnop", False, False),
        ],
    ),
    (
        "# abcdefgh",
        [
            ("# ", False, False),
            (" ", False, True),
            ("abcdefgh", False, False),
        ],
    ),
])
def test_ordered_raw_matcher_treats_adjacent_canonical_space_as_idempotent(
    body, tokens,
):
    assert privacy._ordered_token_stream_contains([body], tokens)


def test_ordered_raw_matcher_retains_space_state_for_css_escape_alternative():
    assert privacy._ordered_token_stream_contains(
        ["abcdefgh ijklmnop"],
        [
            ("abcdefgh ", False, False),
            (r"\20 ", False, True),
            (r"\20 ", False, True),
            ("ijklmnop", False, False),
        ],
    )


@pytest.mark.parametrize("continuation", ["\\\n", "%5C%0A", "&#92;&#10;"])
def test_ordered_raw_matcher_retains_state_for_empty_css_continuation(
    continuation,
):
    assert "" in privacy._canonical_token_variants(continuation)
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"],
        [
            ("abcdefgh", False, False),
            (continuation, False, True),
            ("ijklmnop", False, False),
        ],
    )


@pytest.mark.parametrize("continuation", ["\\\n", "%5C%0A", "&#92;&#10;"])
def test_ordered_raw_matcher_retains_bootstrap_for_empty_css_continuation(
    continuation,
):
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"],
        [
            ("abc", False, True),
            (continuation, False, True),
            ("defghijklmnop", False, False),
        ],
    )


def test_empty_css_continuation_retains_bootstrap_across_nonvisible_markup():
    assert privacy._ordered_token_stream_contains(
        ["abcdefgh"],
        [
            ("a", False, True),
            ("\\\n", False, True),
            *([("ZZ", True, True)] * 15),
            ("bcdefgh", False, False),
        ],
    )


def test_ordered_raw_matcher_keeps_decoding_alternatives_frozen(monkeypatch):
    canonical_variants = privacy._canonical_token_variants

    def mixed_variants(value):
        if value == "MIXED":
            return (" ", " i", "jkl")
        return canonical_variants(value)

    monkeypatch.setattr(privacy, "_canonical_token_variants", mixed_variants)
    assert not privacy._ordered_token_stream_contains(
        ["abcdefgh ijkl"],
        [
            ("abcdefgh ", False, False),
            ("MIXED", False, False),
        ],
    )


def test_ordered_raw_matcher_keeps_empty_decoding_alternative_frozen(monkeypatch):
    canonical_variants = privacy._canonical_token_variants

    def mixed_variants(value):
        if value == "MIXED_EMPTY":
            return ("", "ijkl", "mnop")
        return canonical_variants(value)

    monkeypatch.setattr(privacy, "_canonical_token_variants", mixed_variants)
    assert not privacy._ordered_token_stream_contains(
        ["abcdefghijklmnop"],
        [
            ("abcdefgh", False, False),
            ("MIXED_EMPTY", False, False),
        ],
    )


@pytest.mark.parametrize("body,tokens", [
    (
        "abcdefgh ijklmnop",
        [
            ("abcdefgh ", False, False),
            (" ijklmnop", False, False),
        ],
    ),
    (
        "# abcdefgh",
        [
            ("# ", False, False),
            (" abcdefgh", False, False),
        ],
    ),
])
def test_ordered_raw_matcher_skips_one_redundant_strict_leading_space(
    body, tokens,
):
    assert privacy._ordered_token_stream_contains([body], tokens)


def test_ordered_raw_matcher_does_not_retain_nonspace_state_across_space():
    assert not privacy._ordered_token_stream_contains(
        ["abcdefghijkl"],
        [
            ("abcd", False, False),
            (" ", False, True),
            ("efghijkl", False, False),
        ],
    )


def test_ordered_raw_matcher_bootstrap_crosses_multiple_structural_carriers():
    assert privacy._ordered_token_stream_contains(
        ["abcdefghijk"],
        [
            ("a", False, True),
            ("b", True, True),
            ("cde", True, True),
            ("fghijk", False, False),
        ],
    )


def test_ordered_raw_matcher_bootstrap_obeys_only_semantic_reset():
    body = "abcdefgh"
    wrappers = [("zz", True, True)] * 15
    assert privacy._ordered_token_stream_contains(
        [body], [("a", False, True), *wrappers, ("bcdefgh", False, False)],
    )
    assert privacy._ordered_token_stream_contains(
        [body],
        [("a", False, True), *wrappers, ("zz", True, True),
         ("bcdefgh", False, False)],
    )
    assert not privacy._ordered_token_stream_contains(
        [body],
        [("a", False, True), ("ordinary prose", False, False),
         ("bcdefgh", False, False)],
    )


def test_ordered_raw_matcher_bootstrap_detects_short_terminal_and_renews_start():
    assert privacy._ordered_token_stream_contains(
        ["abcde"], [("a", False, True), ("bcde", True, True)],
    )
    assert privacy._ordered_token_stream_contains(
        ["abcdefgh"],
        [
            ("a", False, True),
            *([("wrapper", True, True)] * 9),
            ("a", True, True),
            *([("wrapper", True, True)] * 7),
            ("bcdefgh", False, False),
        ],
    )


@pytest.mark.parametrize("name_carrier", ["Python", "PythonYY", "XXPythonYY"])
def test_actual_gitignore_body_cannot_cross_short_attribute_and_name_channels(
    tmp_path, name_carrier,
):
    body = privacy._canonical_body((ROOT / ".gitignore").read_text(encoding="utf-8"))
    prefix = "# Python"
    assert body.startswith(prefix)
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="# " ' + name_carrier + '="">'
        + html.escape(body[len(prefix):]) + "</x>",
    )
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, ROOT)
    assert prefix not in str(caught.value)


def test_actual_agents_body_cannot_hide_prefix_in_noisy_attribute_name(tmp_path):
    body = privacy._canonical_body((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    prefix = "# HyperTagging"
    continuation = "# HyperTaggin"
    assert body.startswith(prefix)
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="# " XXHyperTaggingYY="">'
        + html.escape(body[len(continuation):]) + "</x>",
    )
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, ROOT)
    assert prefix not in str(caught.value)


def test_actual_agents_body_survives_adjacent_canonical_whitespace(tmp_path):
    body = privacy._canonical_body((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    prefix = "# HyperTagging "
    assert body.startswith(prefix)
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="# " XXHyperTaggingYY="" b=" " c=" ">'
        + html.escape(body[len(prefix):]) + "</x>",
    )
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, ROOT)
    assert prefix not in str(caught.value)


def test_actual_agents_body_survives_css_escaped_whitespace(tmp_path):
    body = privacy._canonical_body((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    prefix = "# HyperTagging "
    assert body.startswith(prefix)
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="# " XXHyperTaggingYY="" b="\\20 " c="\\20 ">'
        + html.escape(body[len(prefix):]) + "</x>",
    )
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, ROOT)
    assert prefix not in str(caught.value)


@pytest.mark.parametrize("continuation", ["\\\n", "%5C%0A", "&#92;&#10;"])
def test_actual_agents_body_survives_empty_css_continuation(
    tmp_path, continuation,
):
    body = privacy._canonical_body((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    prefix = "# HyperTagging agent gui"
    assert body.startswith(prefix)
    output = tmp_path / "artifact"
    _write(
        output,
        "index.html",
        '<x a="# " XXHyperTaggingYY="" c=" agent gui" b="'
        + continuation + '">' + html.escape(body[len(prefix):]) + "</x>",
    )
    with pytest.raises(ValueError, match="raw-source-body") as caught:
        privacy.validate_artifact(output, ROOT)
    assert prefix not in str(caught.value)


def test_ordered_raw_matcher_deduplicates_repetitive_states():
    body = "A" * 400
    assert privacy._ordered_token_stream_contains(
        [body], ["A" * 6, *(["A"] * 394)],
    )


def test_ordered_raw_matcher_operation_limit_fails_closed(monkeypatch):
    monkeypatch.setattr(privacy, "_MAX_MATCH_OPERATIONS", 10)
    with pytest.raises(ValueError, match="operation limit"):
        privacy._ordered_token_stream_contains(["A" * 100], ["A"] * 100)


def test_ordered_raw_matcher_charges_substring_byte_work(monkeypatch):
    monkeypatch.setattr(privacy, "_MAX_MATCH_OPERATIONS", 1_000)
    with pytest.raises(ValueError, match="operation limit"):
        privacy._ordered_token_stream_contains(
            ["ABCDEFG"], ["X" * 200_000 + "ABCDEFG"],
        )


def test_publication_wide_match_budget_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "OPAQUE_PRIVATE_TOKEN")
    output = tmp_path / "artifact"
    _write(output, "index.html", "<p>ordinary public documentation</p>")
    monkeypatch.setattr(privacy, "_MAX_PUBLICATION_MATCH_OPERATIONS", 1)
    with pytest.raises(ValueError, match="operation limit"):
        privacy.validate_artifact(output, source)


def test_large_literal_set_uses_a_bounded_linear_matcher():
    literals = [f"OPAQUE_PRIVATE_{index:05d}_TOKEN" for index in range(10_000)]
    build_budget = [privacy._MAX_PUBLICATION_MATCH_OPERATIONS]
    matcher = privacy._literal_automaton(literals, build_budget)
    assert build_budget[0] < privacy._MAX_PUBLICATION_MATCH_OPERATIONS
    with pytest.raises(ValueError, match="operation limit"):
        privacy._literal_automaton_contains(
            matcher, "ordinary publication text", [1], [1],
        )


def test_nested_prefix_literal_outputs_remain_constant_size():
    literals = ["A" * size for size in range(4, 2_004)]
    budget = [privacy._MAX_PUBLICATION_MATCH_OPERATIONS]
    _transitions, _failures, terminal = privacy._literal_automaton(
        literals, budget,
    )
    assert len(terminal) == len(literals[-1]) + 1
    assert all(type(item) is bool for item in terminal)


@pytest.mark.parametrize("body_length", [24, 96])
def test_many_raw_source_bodies_charge_the_publication_budget(
    tmp_path, monkeypatch, body_length,
):
    bodies = [
        ("S" * (body_length - 8)) + f"{index:08d}"
        for index in range(10_000)
    ]
    monkeypatch.setattr(privacy, "sensitive_literals", lambda _root: set())
    monkeypatch.setattr(
        privacy, "_source_bodies", lambda _root: (set(), bodies),
    )
    monkeypatch.setattr(privacy, "_MAX_PUBLICATION_MATCH_OPERATIONS", 1)
    output = tmp_path / "artifact"
    _write(output, "report.txt", "ordinary publication text " * 8)
    with pytest.raises(ValueError, match="operation limit"):
        privacy.validate_artifact(output, tmp_path / "source")


def test_publication_wide_match_budget_is_finite_and_proportional():
    assert (
        privacy._MAX_PUBLICATION_MATCH_OPERATIONS
        == 10 * privacy._MAX_MATCH_OPERATIONS
    )


def test_ordered_raw_matcher_skips_structure_but_not_unrelated_content():
    body = "abcdefgh"
    assert privacy._ordered_token_stream_contains(
        [body], [("abcdef", False), ("wrapper", True), ("gh", False)],
    )
    assert not privacy._ordered_token_stream_contains(
        [body], ["abcdef", "ordinary unrelated prose", "gh"],
    )


def test_unrelated_visible_text_does_not_form_a_raw_source_body(tmp_path):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "OPAQUE_PRIVATE_TOKEN")
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        '<x OPAQUE_="">ordinary unrelated prose PRIVATE_TOKEN</x>',
    )
    assert privacy.validate_artifact(output, source)["privacy"] == "PASS"


@pytest.mark.parametrize("limit_name", [
    "_MAX_PUBLICATION_ENTRIES",
    "_MAX_PUBLICATION_FILE_BYTES",
    "_MAX_PUBLICATION_BYTES",
])
def test_publication_resource_limits_fail_closed(tmp_path, monkeypatch, limit_name):
    _write(tmp_path, "index.html", "safe")
    _write(tmp_path, "second.txt", "safe")
    monkeypatch.setattr(privacy, limit_name, 1)
    with pytest.raises(ValueError, match="limit|privacy check"):
        privacy.validate_artifact(tmp_path)


@pytest.mark.parametrize("payload", [
    '<x a="foo\\"bar" OPAQUE_="" PRIVATE_TOKEN="">x</x>',
    "<x a='foo\\'bar' OPAQUE_='' PRIVATE_TOKEN=''>x</x>",
])
def test_raw_html_name_lexer_fails_closed_on_parser_disagreement(tmp_path, payload):
    _write(tmp_path, "index.html", payload)
    with pytest.raises(ValueError, match="unparseable-html-names"):
        privacy.validate_artifact(tmp_path)


def test_raw_source_body_split_across_json_values_is_rejected(tmp_path):
    source = tmp_path / "source"
    body = "RAW_SOURCE_BODY_CANARY_" * 4
    _write(source, "configs/public.cfg", body)
    midpoint = len(body) // 2
    output = tmp_path / "artifact"
    _write(output, "data.json", json.dumps({"left": body[:midpoint], "right": body[midpoint:]}))
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_raw_source_body_encoded_across_html_name_and_value_is_rejected(tmp_path):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "Token")
    output = tmp_path / "artifact"
    _write(output, "index.html", '<x a="%5" 4oken=""></x>')
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("payload", [
    '<x a="%5" inert="!" 4oken=""></x>',
    '<x a="%4" inert="!" 1KIAABCDEFGHIJKLMNOP=""></x>',
    '<x a="user%" inert="!" b="40example.com"></x>',
    '<x a="api_key%" inert="!" b="3DOPAQUE_PRIVATE_TOKEN"></x>',
    '<!--api_key%--><x title="!"></x><!--3DOPAQUE_PRIVATE_TOKEN-->',
    '<x a="&#x5" inert="!" 4;oken=""></x>',
    r'<x a="\u00" inert="!" 54oken=""></x>',
    r'<x a="\x" inert="!" 54oken=""></x>',
    r'<x a="\5" inert="!" b="4 oken"></x>',
    '<x a="user&#x4" inert="!" b="0;example.com"></x>',
    r'<x a="user\" inert="!" b="x40" inert2="!" c="example.com"></x>',
    '<x a="user&" inert="!" b="commat;example.com"></x>',
    '<x a="api_key&" inert="!" b="#61;OPAQUE_PRIVATE_TOKEN"></x>',
])
def test_incomplete_encoded_fragments_fail_closed(tmp_path, payload):
    source = tmp_path / "source"
    _write(source, "configs/private.json", json.dumps({"api_key": "Token"}))
    output = tmp_path / "artifact"
    _write(output, "index.html", payload)
    with pytest.raises(ValueError, match="incomplete-encoded-fragment"):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("css", [False, True])
@pytest.mark.parametrize("left,right,literal", [
    ("\\", "54 oken", "Token"),
    ("&", "#233;OMEGA", "éOMEGA"),
    ("&", "#xE9;OMEGA", "éOMEGA"),
    ("&eac", "ute;OMEGA", "éOMEGA"),
    ("%C", "3%A9OMEGA", "éOMEGA"),
    ("%C3", "%A9OMEGA", "éOMEGA"),
])
def test_comment_bridged_unicode_and_css_encodings_fail_closed(
    tmp_path, css, left, right, literal,
):
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write(source, "private.json", json.dumps({"api_key": literal}))
    wrap = (lambda value: f"/*{value}*/") if css else (
        lambda value: f"<!--{value}-->"
    )
    name = "_static/custom.css" if css else "index.html"
    _write(output, name, wrap(left) + wrap("!") + wrap(right))
    with pytest.raises(ValueError):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("as_json", [False, True])
def test_javascript_surrogate_pair_cannot_hide_a_source_literal(
    tmp_path, as_json,
):
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write(source, "private.json", json.dumps({"api_key": "🔐OMEGA"}))
    value = r"\uD83D\uDD10OMEGA"
    _write(
        output,
        "data.json" if as_json else "report.txt",
        json.dumps({"value": value}) if as_json else value,
    )
    with pytest.raises(ValueError):
        privacy.validate_artifact(output, source)


@pytest.mark.parametrize("payload", [
    '<table style="width: 100%"><tr><td>safe</td></tr></table>',
    '<p title="R&D">safe</p>',
    '<!--R&D--><p>safe</p>',
])
def test_literal_percentages_and_ampersands_are_not_incomplete_encodings(
    tmp_path, payload,
):
    _write(tmp_path, "index.html", payload)
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def test_benign_completed_percent_escape_is_inspected_not_blanket_rejected(tmp_path):
    _write(
        tmp_path, "index.html",
        '<x title="rate %" inert="!" data-v="4 items"></x>',
    )
    assert privacy.validate_artifact(tmp_path)["privacy"] == "PASS"


def test_named_entity_split_cannot_hide_an_email(tmp_path):
    _write(
        tmp_path, "index.html",
        '<x a="user&com" inert="!" b="mat;example.com"></x>',
    )
    with pytest.raises(ValueError, match="incomplete-encoded-fragment"):
        privacy.validate_artifact(tmp_path)


def test_named_entity_split_cannot_hide_normalized_source_body(tmp_path):
    source = tmp_path / "source"
    _write(source, "private.json", json.dumps({"api_key": "To ken"}))
    output = tmp_path / "artifact"
    _write(
        output, "index.html",
        '<x a="To&Ta" inert="!" b="b;ken"></x>',
    )
    with pytest.raises(ValueError, match="incomplete-encoded-fragment"):
        privacy.validate_artifact(output, source)


def test_encoded_fragment_reconstruction_is_bounded():
    tokens = [("&#", False, True, True)] + [
        ("1", False, True, True)
    ] * 4_000
    publication_budget = [privacy._MAX_MATCH_OPERATIONS]
    file_budget = [privacy._MAX_MATCH_OPERATIONS]
    assert not privacy._has_useful_cross_fragment_encoding(
        tokens, publication_budget, file_budget,
    )


def test_raw_source_body_survives_many_visible_markup_gaps(tmp_path):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "AB")
    output = tmp_path / "artifact"
    _write(output, "index.html", "A" + "<x></x>" * 24 + "B")
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_raw_source_body_encoded_across_json_key_and_value_is_rejected(tmp_path):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "Token")
    output = tmp_path / "artifact"
    _write(output, "data.json", json.dumps({"%5": "4oken"}))
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_complete_structured_source_body_nested_in_json_is_rejected(tmp_path):
    source = tmp_path / "source"
    body = '{"public":"' + "RAW_SOURCE_BODY_CANARY_" * 4 + '"}'
    _write(source, "configs/public.cfg", body)
    output = tmp_path / "artifact"
    _write(output, "data.json", '{"wrapper":' + body + "}")
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_whitespace_normalized_source_body_split_across_json_values_is_rejected(
    tmp_path,
):
    source = tmp_path / "source"
    _write(source, "configs/public.cfg", "LEFT RIGHT")
    output = tmp_path / "artifact"
    _write(output, "data.json", json.dumps(["LEFT", "RIGHT"]))
    with pytest.raises(ValueError, match="raw-source-body"):
        privacy.validate_artifact(output, source)


def test_allowed_ast_projection_and_static_markup_do_not_trigger_raw_body_rule(tmp_path):
    source = tmp_path / "source"
    _write(source, "scripts/public.py", 'def public(value):\n    """Documented callable."""\n    return "RAW_BODY_CANARY"\n')
    output = tmp_path / "artifact"
    _write(output, "api.txt", "public(value)\nDocumented callable.\n")
    _write(output, "status.rst", '.. raw:: html\n\n   <section><h2>Status</h2><p>NO_GO</p></section>\n')
    _write(output, "_static/custom.css", '/* https://developer.mozilla.org/en-US/docs/Web/API */\n')
    _write(output, "manifest.json", '{"job_id": null, "credentials": "[redacted]", "status": "NO_GO"}')
    assert privacy.validate_artifact(output, source)["privacy"] == "PASS"


def test_security_validator_contains_no_asserts_and_works_with_python_optimization(tmp_path):
    tree = ast.parse(Path(privacy.__file__).read_text())
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    _write(tmp_path, "private.txt", "/home/PRIVATE_USER/private.txt")
    program = "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from wiki_privacy import validate_artifact\ntry:\n validate_artifact(Path(sys.argv[2]))\nexcept ValueError:\n sys.exit(0)\nsys.exit(1)\n"
    result = subprocess.run([sys.executable, "-O", "-c", program, str(ROOT / "docs/_ext"), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "PRIVATE_USER" not in result.stdout + result.stderr
