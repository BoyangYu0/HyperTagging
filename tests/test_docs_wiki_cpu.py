"""Documentation integration checks without importing scientific dependencies."""
from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
from wiki_repository import generate_repository, repository_files

_spec = importlib.util.spec_from_file_location("wiki_validation", ROOT / "scripts" / "validate_docs.py")
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def test_whole_repository_counts_snapshots_without_publishing_their_bodies(tmp_path):
    for rel in ("src/hypertagging/__init__.py", "tests/test_contract.py",
                "reconstruction/snapshots/frozen/runtime/src/hypertagging/example.py"):
        path = tmp_path / "source" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('def public(value: int) -> int:\n    """Documented function."""\n    return value\n')
    root = tmp_path / "source"
    output = tmp_path / "generated"
    manifest = generate_repository(root, output)
    assert manifest["python_modules"] == 3
    assert len(manifest["files"]) == 3
    assert sorted(path.name for path in output.iterdir()) == ["index.rst", "manifest.json"]
    published = "\n".join(path.read_text() for path in output.iterdir())
    for raw in ("def public", "Documented function", "test_contract.py", "frozen/runtime"):
        assert raw not in published


def test_source_inventory_does_not_follow_external_symlinks(tmp_path):
    root = tmp_path / "source"
    (root / "src").mkdir(parents=True)
    external = tmp_path / "external.py"
    external.write_text("raise RuntimeError('must never execute')\n")
    (root / "src" / "escape.py").symlink_to(external)
    assert repository_files(root) == []


def test_rst_sources_are_hashed_without_a_source_view_or_download(tmp_path):
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.rst").write_text("Guide\n=====\n")
    output = tmp_path / "generated"
    generate_repository(root, output)
    assert sorted(path.name for path in output.iterdir()) == ["index.rst", "manifest.json"]
    assert "guide.rst" not in (output / "manifest.json").read_text()


def _site(tmp_path):
    for name in ("objects.inv", "searchindex.js", ".nojekyll"):
        (tmp_path / name).write_text("")
    return tmp_path


def test_local_html_validator_accepts_project_subpath_links(tmp_path):
    site = _site(tmp_path)
    (site / "deep").mkdir()
    (site / "index.html").write_text('<a href="deep/page.html#target">page</a>')
    (site / "deep" / "page.html").write_text('<h1 id="target">API</h1><a href="../index.html">home</a>')
    assert validation.validate_html(site)["local_links_checked"] == 2


def test_local_html_validator_allows_benign_metadata_attribute(tmp_path):
    site = _site(tmp_path)
    (site / "index.html").write_text('<div title="Metadata: public">Safe</div>')
    assert validation.validate_html(site)["remote_assets"] == 0


@pytest.mark.parametrize("bad", [
    '<script src="https://cdn.invalid/runtime.js"></script>',
    '<link rel="stylesheet" href="//cdn.invalid/theme.css">',
    '<a href="/repository/page.html">absolute</a>',
    '<a href="index.html#missing">anchor</a>',
    '<a href="../outside.html">escape</a>',
    '<script>document.body.textContent=["/","home"].join("")</script>',
    '<iframe src="data:text/html;base64,L2hvbWU="></iframe>',
    '<div onclick="document.body.textContent=\'private\'">safe</div>',
    '<style>body::before{content:"/" "home"}</style>',
    '<img srcset="safe.png 1x, data:image/svg+xml;base64,L2hvbWU= 2x">',
    '<base href="//evil.invalid/">',
    '<meta http-equiv="refresh" content="0;url=data:text/html;base64,L2hvbWU=">',
])
def test_local_html_validator_rejects_missing_or_nonlocal_assets(tmp_path, bad):
    site = _site(tmp_path)
    (site / "index.html").write_text(bad)
    with pytest.raises(ValueError):
        validation.validate_html(site)


def test_local_html_validator_rejects_active_css_and_svg(tmp_path):
    site = _site(tmp_path)
    (site / "index.html").write_text('<link rel="stylesheet" href="x.css">')
    (site / "x.css").write_text('body::before{content:"/" "home"}')
    (site / "payload.svg").write_text("<svg><script>alert(1)</script></svg>")
    with pytest.raises(ValueError):
        validation.validate_html(site)


def test_local_html_validator_fails_closed_on_raw_name_disagreement(tmp_path):
    site = _site(tmp_path)
    (site / "index.html").write_text(
        '<x a="foo\\"bar" OPAQUE_="" PRIVATE_TOKEN="">x</x>'
    )
    with pytest.raises(ValueError, match="Unparseable HTML name channel"):
        validation.validate_html(site)


def test_pages_workflow_privileges_and_untrusted_events():
    assert validation.validate_workflow()["permission_boundary"] == "PASS"


def test_security_validators_have_no_assert_statements():
    for path in (ROOT / "scripts/validate_docs.py", ROOT / "docs/_ext/wiki_privacy.py"):
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(path.read_text())))


@pytest.mark.parametrize("before,after", [
    ("contents: read", "contents: write"),
    ("persist-credentials: false", "persist-credentials: true"),
    ("-e -o pipefail", "-e"),
    ("github.ref == 'refs/heads/master'", "github.ref == 'refs/heads/other'"),
    ("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7"),
])
def test_mutated_workflow_fails_even_with_python_optimization(tmp_path, before, after):
    target = tmp_path / ".github/workflows/docs.yml"
    target.parent.mkdir(parents=True)
    original = (ROOT / ".github/workflows/docs.yml").read_text()
    assert before in original
    target.write_text(original.replace(before, after))
    program = (
        "import importlib.util, pathlib, sys; "
        "spec=importlib.util.spec_from_file_location('validation',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.ROOT=pathlib.Path(sys.argv[2]); module.validate_workflow()"
    )
    result = subprocess.run([sys.executable, "-O", "-c", program,
                             str(ROOT / "scripts/validate_docs.py"), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_docs_configuration_does_not_import_hypertagging_or_fetch_inventory():
    import runpy
    before = {name for name in sys.modules if name.startswith("hypertagging")}
    conf = runpy.run_path(str(ROOT / "docs" / "wiki" / "conf.py"))
    assert conf["intersphinx_mapping"] == {}
    assert "basf2" in conf["autodoc_mock_imports"] and "torch" in conf["autodoc_mock_imports"]
    assert {name for name in sys.modules if name.startswith("hypertagging")} == before
