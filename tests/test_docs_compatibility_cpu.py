"""Bounded basf2/Sphinx compatibility checks requiring no scientific imports."""
from __future__ import annotations

import builtins
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXTENSIONS = {"sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode",
                       "sphinx.ext.mathjax", "sphinx.ext.autosectionlabel", "sphinx.ext.intersphinx"}
OPTIONAL_IMPORTS = {"basf2", "ROOT", "torch", "numpy", "pandas", "awkward", "pyarrow", "uproot",
                    "onnx", "onnxruntime", "scipy", "matplotlib", "networkx", "hypertagging"}


@pytest.mark.parametrize("configuration", ["docs/conf.py", "docs/wiki/conf.py"])
def test_shared_configuration_retains_basf2_settings_without_scientific_imports(configuration, monkeypatch):
    original = builtins.__import__
    def restricted(name, *args, **kwargs):
        if name.split(".")[0] in OPTIONAL_IMPORTS:
            raise AssertionError("Documentation configuration attempted a scientific import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", restricted)
    configuration_values = runpy.run_path(str(ROOT / configuration))
    assert REQUIRED_EXTENSIONS <= set(configuration_values["extensions"])
    assert OPTIONAL_IMPORTS - {"hypertagging"} <= set(configuration_values["autodoc_mock_imports"])
    assert configuration_values["autosectionlabel_prefix_document"] is True
    assert configuration_values["default_role"] == "any"
    assert configuration_values["numfig"] is True
    assert configuration_values["napoleon_google_docstring"] is True
    assert configuration_values["napoleon_numpy_docstring"] is True
    assert configuration_values["html_copy_source"] is False
    assert configuration_values["html_show_sourcelink"] is False
    assert configuration_values["viewcode_follow_imported_members"] is False
    assert configuration_values["intersphinx_mapping"] == {}
    assert configuration_values["html_math_renderer"] == "offline-text"
    assert configuration_values["mathjax_path"] == ""


def _children(path):
    in_tree = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(".. toctree::"):
            in_tree = True
            continue
        if not in_tree or not line.strip():
            continue
        if not line.startswith(" "):
            in_tree = False
            continue
        value = line.strip()
        if not value.startswith(":"):
            yield (path.parent / (value + ".rst")).resolve()


def test_package_discovery_entry_reaches_every_authored_guide_once():
    entry = ROOT / "doc" / "index-hypertagging.rst"
    assert entry in list((ROOT / "doc").glob("index*.rst"))
    assert list(_children(entry)) == [ROOT / "docs" / "index.rst"]
    assert list(_children(ROOT / "docs" / "index.rst")) == [ROOT / "docs" / "wiki" / "index.rst"]
    reached, incoming = set(), Counter()
    def walk(path):
        assert path.is_relative_to(ROOT)
        assert path not in reached, "An authored page is cyclic or belongs to multiple toctrees"
        reached.add(path)
        for child in _children(path):
            incoming[child] += 1
            if "_generated" not in child.parts:
                assert child.is_file(), "An authored toctree entry is missing"
                walk(child)
    walk(entry)
    assert set((ROOT / "docs" / "wiki").glob("*.rst")) <= reached
    assert all(count == 1 for count in incoming.values())
    guide = (ROOT / "docs" / "wiki" / "basf2.rst").read_text()
    assert "https://software.belle2.org/development/sphinx/framework/doc/atend-doctools.html" in guide


class _Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img", "iframe", "source"} and attrs.get("src"):
            self.resources.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.resources.append(attrs.get("href", ""))


@pytest.mark.parametrize("builder", ["html", "text"])
def test_warning_fatal_math_and_static_api_build_offline_without_importing_modules(tmp_path, builder):
    pytest.importorskip("sphinx")
    source = tmp_path / "source"
    source.mkdir()
    # Reuse the actual extensions, configuration and viewcode guard. Stub only
    # expensive full-repository generation so this remains a bounded fixture.
    configuration = (
        "from pathlib import Path\nimport runpy\n"
        f"_shared = runpy.run_path({str(ROOT / 'docs' / 'conf.py')!r})\n"
        "globals().update({key: value for key, value in _shared.items() if not key.startswith('_')})\n"
        "import wiki\nwiki.generate = lambda *_args: {}\n"
        "html_static_path = []\nhtml_extra_path = []\nhtml_css_files = []\n"
        "wiki_content_path = ''\n"
    )
    (source / "conf.py").write_text(configuration, encoding="utf-8")
    (source / "index.rst").write_text(
        "Offline compatibility\n=====================\n\n"
        "Inline conservation: :math:`p = \\sum_i p_i`.\n\n"
        ".. math::\n   :label: closure\n\n   p_{mother} = \\sum_i p_i\n\n"
        "Equation :eq:`closure` remains labelled.\n\n"
        ".. py:module:: hypertagging.never_import_fixture\n\n"
        ".. py:function:: decode(value)\n\n   Static signature, no runtime import.\n",
        encoding="utf-8",
    )
    runner = (
        "import builtins, socket, sys\n"
        "from sphinx.cmd.build import build_main\n"
        f"blocked = {sorted(OPTIONAL_IMPORTS)!r}\n"
        "original = builtins.__import__\n"
        "def restricted(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in blocked: raise RuntimeError('Scientific import attempted')\n"
        "    return original(name, *args, **kwargs)\n"
        "def no_network(*args, **kwargs): raise RuntimeError('Network access attempted')\n"
        "builtins.__import__ = restricted\n"
        "socket.create_connection = no_network\n"
        "socket.socket.connect = no_network\n"
        "socket.socket.connect_ex = no_network\n"
        "raise SystemExit(build_main(sys.argv[1:]))\n"
    )
    output = tmp_path / builder
    result = subprocess.run(
        [sys.executable, "-c", runner, "-W", "--keep-going", "-E", "-a", "-b", builder,
         "-d", str(tmp_path / "doctrees"), str(source), str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(path.name in {"_sources", "_modules"} for path in output.rglob("*"))
    assert not any(path.suffix in {".py", ".rst"} for path in output.rglob("*"))
    if builder == "html":
        document = (output / "index.html").read_text()
        assert "p_{mother}" in document and "equation-closure" in document
        parser = _Resources()
        parser.feed(document)
        assert parser.resources
        assert all(not resource.startswith(("http:", "https:", "//")) for resource in parser.resources)
        assert not any("mathjax" in resource.lower() for resource in parser.resources)
    else:
        document = (output / "index.txt").read_text()
        assert "p_{mother}" in document and "Static signature" in document
