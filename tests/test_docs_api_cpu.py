"""Import-free API generation checks; no model/data dependencies are required."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
SPEC = importlib.util.spec_from_file_location("wiki_api_for_test", ROOT / "docs/_ext/wiki_api.py")
assert SPEC and SPEC.loader
WIKI_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WIKI_API)


def _fixture(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repository"
    for name, content in files.items():
        path = root / "src/hypertagging" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_api_never_imports_project_and_preserves_docstrings(tmp_path):
    root = _fixture(tmp_path, {
        "__init__.py": "raise RuntimeError('must not execute')\n",
        "danger.py": '''"""Module docs with :unknown:role:`text`."""
import package_that_must_not_exist
raise RuntimeError("must not execute")
def public(x: "MissingType", /, *, y: int = 3) -> "MissingType":
    """Useful docs.

    .. include:: /etc/passwd
    """
    return x
''',
    })
    before = set(sys.modules)
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    assert not any(name.startswith("hypertagging") for name in set(sys.modules) - before)
    text = (tmp_path / "api/hypertagging.danger.rst").read_text()
    assert ".. py:function:: public(x: 'MissingType', /, *, y: int=...) -> 'MissingType'" in text
    assert "      .. include:: [redacted]" in text
    assert "/etc/passwd" not in text
    assert "      Useful docs." in text
    assert inventory["module_count"] == 2


def test_aliases_relative_imports_lazy_exports_and_cycles(tmp_path):
    root = _fixture(tmp_path, {
        "__init__.py": '''from .api import alias as public_alias
__all__ = ["public_alias"]
''',
        "api.py": '''from .impl import actual
alias = actual
__all__ = ("alias",)
''',
        "impl.py": '''def actual(value=1):
    """Implementation documentation."""
    return value
''',
        "lazy/__init__.py": '''_EXPORT_MODULE = {
    **{name: "hypertagging.impl" for name in ("actual",)}
}
__all__ = sorted(_EXPORT_MODULE)
def __getattr__(name):
    raise RuntimeError("must not execute")
''',
        "cycle.py": '''from .cycle import looping
__all__ = ["looping"]
''',
    })
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    symbols = {item["name"]: item for item in inventory["symbols"]}
    for alias in ("hypertagging.public_alias", "hypertagging.api.alias", "hypertagging.lazy.actual"):
        assert symbols[alias]["canonical"] == "hypertagging.impl.actual"
        assert symbols[alias]["documented"]
    assert inventory["dynamic_exports"] == []
    assert inventory["unresolved_exports"] == [{"name": "hypertagging.cycle.looping", "target": "hypertagging.cycle.looping"}]


def test_members_properties_async_and_explicit_private_exports(tmp_path):
    root = _fixture(tmp_path, {"__init__.py": '''__all__ = ["_explicit", "EXPORTED_CONSTANT"]
EXPORTED_CONSTANT = 3
def _explicit():
    """Explicitly public despite underscore."""
class Example:
    """Example docs."""
    field: int = 1
    def __init__(self, value: int = 1):
        """Constructor docs."""
        self.value = value
    @property
    def value(self) -> int:
        """Getter docs."""
        return 1
    @value.setter
    def value(self, value):
        pass
    @classmethod
    def make(cls, x=1):
        """Factory docs."""
        return cls(x)
    async def run(self, /, *values, enabled=True, **options):
        """Async docs."""
        return values
    def _private(self):
        pass
    def __len__(self):
        """Special method docs."""
        return 1
def outside_all():
    """Still accessible as a public local function."""
'''} )
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    symbols = {item["name"]: item for item in inventory["symbols"]}
    assert symbols["hypertagging.Example._private"]["surface"] == "internal_unstable"
    assert not symbols["hypertagging.Example._private"]["documented"]
    assert "hypertagging._explicit" in symbols
    assert symbols["hypertagging._explicit"]["surface"] == "internal_unstable"
    assert "hypertagging.outside_all" in symbols
    accessors = [item for item in inventory["symbols"] if item["name"] == "hypertagging.Example.value"]
    assert len(accessors) == 2
    assert accessors[0]["kind"] == "property" and accessors[0]["documented"]
    assert accessors[0]["indexed"] and not accessors[1]["indexed"]
    assert len({item["definition_id"] for item in accessors}) == 2
    assert symbols["hypertagging.Example.__len__"]["documented"]
    text = (tmp_path / "api/hypertagging.rst").read_text()
    assert ".. py:class:: Example(value: int=...)" in text
    assert ".. py:method:: Example.run(*values, enabled=..., **options)" in text
    assert "   :async:" in text
    assert "   :classmethod:" in text


def test_generation_is_deterministic_and_keeps_unrelated_files(tmp_path):
    root = _fixture(tmp_path, {"__init__.py": '"""Package docs."""\n'})
    output = tmp_path / "api"
    output.mkdir()
    unrelated = output / "user-notes.txt"
    unrelated.write_text("keep me")
    first = WIKI_API.generate_api(root, output)
    contents = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    modified = {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    second = WIKI_API.generate_api(root, output)
    assert first == second
    assert contents == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert modified == {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    assert unrelated.read_text() == "keep me"
    assert json.loads((output / "inventory.json").read_text()) == first
    other = tmp_path / "different/depth/api"
    assert WIKI_API.generate_api(root, other) == first
    assert {name: data for name, data in contents.items() if name != Path("user-notes.txt")} == {
        path.relative_to(other): path.read_bytes() for path in other.rglob("*") if path.is_file()
    }


def test_module_facade_collision_keeps_canonical_function_indexed(tmp_path):
    root = _fixture(tmp_path, {
        "__init__.py": 'from .run import run\n__all__ = ["run"]\n',
        "run.py": 'def run():\n    """Run something."""\n',
    })
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    symbols = {item["name"]: item for item in inventory["symbols"]}
    assert not symbols["hypertagging.run"]["indexed"]
    assert symbols["hypertagging.run.run"]["indexed"]
    assert "   :noindex:" in (tmp_path / "api/hypertagging.rst").read_text()


def test_literal_export_updates_and_unhandled_dynamic_exports(tmp_path):
    root = _fixture(tmp_path, {
        "__init__.py": '''__all__ = ["A"]
__all__ += ["B"]
__all__.append("C")
__all__.extend(("D",))
A = B = C = D = 1
''',
        "dynamic.py": '__all__ = ["A"]\n__all__ += external_code()\nA = 1\n',
    })
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    modules = {item["name"]: item for item in inventory["modules"]}
    assert modules["hypertagging"]["exports"] == ["A", "B", "C", "D"]
    assert modules["hypertagging.dynamic"]["exports"] is None
    assert inventory["dynamic_exports"] == [{"module": "hypertagging.dynamic", "sha256": modules["hypertagging.dynamic"]["sha256"]}]
    assert "external_code()" not in (tmp_path / "api/hypertagging.dynamic.rst").read_text()


def test_sphinx_builds_signatures_aliases_and_literal_docs_without_imports(tmp_path):
    if importlib.util.find_spec("sphinx") is None:
        pytest.skip("Sphinx is available in the documentation-only environment")
    root = _fixture(tmp_path, {
        "__init__.py": 'from .run import run, Example\n',
        "run.py": '''import a_dependency_that_does_not_exist
raise RuntimeError("Never import source")
def run(value: "MissingType", /, *, enabled=True) -> "MissingType":
    """Run with unknown external types.

    .. include:: /no-such-secret-file
    :nonexistent:role:`literal text`
    """
class Example:
    """Class docs."""
    @property
    def value(self) -> int:
        """Value docs."""
    @classmethod
    def make(cls, x: int = 1):
        """Factory docs."""
    @staticmethod
    async def process(self, /, *args, enabled=True, **kwargs):
        """Static async docs."""
def consume(value: Example) -> Example:
    """Canonical annotation references remain unambiguous across reexports."""
''',
    })
    source = tmp_path / "api"
    WIKI_API.generate_api(root, source)
    assert "consume(value: hypertagging.run.Example) -> hypertagging.run.Example" in (
        source / "hypertagging.run.rst"
    ).read_text()
    (source / "conf.py").write_text(
        "project = 'API fixture'\nmaster_doc = 'index'\nhtml_theme = 'basic'\n"
        "html_copy_source = False\nhtml_show_sourcelink = False\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-E", "-W", "--keep-going", "-q", str(source), str(tmp_path / "html")],
        text=True, capture_output=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = (tmp_path / "html/hypertagging.run.html").read_text()
    assert "no-such-secret-file" not in rendered
    assert "literal text" in rendered
    assert 'id="hypertagging.run.run"' in rendered
    assert 'id="hypertagging.run.Example.value"' in rendered
    # Some Sphinx versions create the empty directory despite copy_source=False.
    assert not any(path.is_file() for path in (tmp_path / "html/_sources").rglob("*"))


def _independent_definitions(tree, module):
    """Walk lexical scopes independently of the generator, retaining locations."""
    definitions = set()
    def walk(node, prefix):
        if isinstance(node, ast.ClassDef):
            name = prefix + "." + node.name
            definitions.add((name, node.lineno))
            for child in node.body:
                walk(child, name)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + "." + node.name
            definitions.add((name, node.lineno))
            for child in node.body:
                walk(child, name)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, prefix)
    walk(tree, module)
    return definitions


def test_repository_covers_every_module_and_every_definition(tmp_path):
    inventory = WIKI_API.generate_api(ROOT, tmp_path / "api")
    paths = sorted((ROOT / "src/hypertagging").rglob("*.py"))
    modules = {item["source"]: item for item in inventory["modules"]}
    definitions = {(item["source"], item["canonical"], item["line"])
                   for item in inventory["symbols"] if item["definition"] and not item["alias"]}
    assert set(modules) == {path.relative_to(ROOT).as_posix() for path in paths}
    for path in paths:
        module = modules[path.relative_to(ROOT).as_posix()]["name"]
        tree = ast.parse(path.read_bytes())
        actual = {(name, line) for source, name, line in definitions if source == path.relative_to(ROOT).as_posix()}
        assert actual == _independent_definitions(tree, module), path.relative_to(ROOT)
    assert len(definitions) == inventory["definition_count"]
    assert inventory["dynamic_exports"] == []
    assert inventory["unresolved_exports"] == []


def test_independent_validator_detects_missing_nested_function(tmp_path, monkeypatch):
    root = _fixture(tmp_path, {"__init__.py": '''class Public:
    """Public class."""
    def documented(self):
        """Documented method."""
    def _undocumented(self):
        pass
def outer():
    def _nested():
        pass
    return _nested
'''} )
    generated = tmp_path / "generated"
    inventory = WIKI_API.generate_api(root, generated / "api")
    from wiki_repository import generate_repository
    from wiki_catalog import generate_catalog
    generate_repository(root, generated / "repository")
    generate_catalog(root, generated / "catalog")
    spec = importlib.util.spec_from_file_location("coverage_validation_for_test", ROOT / "scripts/validate_docs.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    monkeypatch.setattr(validator, "ROOT", root)
    assert validator.validate_coverage(generated)["api_symbols"] == 5
    missing = "hypertagging.outer._nested"
    inventory["symbols"] = [item for item in inventory["symbols"] if item["name"] != missing]
    inventory["modules"][0]["symbols"].remove(missing)
    inventory["symbol_count"] -= 1
    (generated / "api/inventory.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Missing .*API|Missing .*definitions"):
        validator.validate_coverage(generated)


def test_api_publishes_only_redacted_projection_and_hashes(tmp_path):
    raw = '''"""Use password=fixture-password and host login123.private.example.
    /home/fixture-user/private.py GPU-12345678-abcd job_id=987654
    authority/private.json model-secret.ckpt
    """
__all__ = ["EXPORTED_CONFIG"]
EXPORTED_CONFIG = {"raw_source_marker": "must-never-publish", "secret": "unlabeled-config-secret"}
def entry(secret="unlabeled-default-secret", options={"token": "nested-default-secret"}):
    """Module API description."""
    sensitive_internal_body = "no-source-body"
    return sensitive_internal_body
'''
    root = _fixture(tmp_path, {"__init__.py": raw})
    output = tmp_path / "api"
    inventory = WIKI_API.generate_api(root, output)
    all_text = "\n".join(path.read_text() for path in output.rglob("*") if path.is_file())
    for private in (
        "fixture-password", "login123.private.example", "/home/fixture-user", "GPU-12345678-abcd",
        "987654", "authority/private.json", "model-secret.ckpt", "raw_source_marker",
        "must-never-publish", "unlabeled-config-secret", "unlabeled-default-secret",
        "nested-default-secret", "sensitive_internal_body", "no-source-body",
    ):
        assert private not in all_text, private
    assert not any(path.name in {"sources", "evidence", "_sources"} for path in output.rglob("*"))
    assert not any(path.suffix == ".py" for path in output.rglob("*"))
    assert "source_download" not in all_text
    assert inventory["modules"][0]["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "src/hypertagging/__init__.py" in all_text
    assert "Module API description." in all_text


def test_overloads_conditional_methods_dunders_and_function_local_classes(tmp_path):
    source = '''from typing import overload
@overload
def _choose(value: int) -> int: ...
@overload
def _choose(value: str) -> str: ...
def _choose(value):
    return value
if available:
    def conditional(): pass
else:
    def conditional(): pass
class _Private:
    if available:
        def _hidden(self): pass
    else:
        def _hidden(self): pass
    def __len__(self): return 0
    def factory(self):
        class Local:
            def _local(self): pass
        return Local
def outer():
    def ordinary_nested(token="nested-default-secret"):
        """Nested docs."""
        nested_body_secret = "must-not-publish"
        async def deeper(): pass
        return deeper
    class Collector:
        def __init__(self): pass
        def event(self):
            def helper(): pass
            return helper
    return Collector
'''
    root = _fixture(tmp_path, {"__init__.py": source})
    inventory = WIKI_API.generate_api(root, tmp_path / "api")
    actual = {(item["canonical"], item["line"]) for item in inventory["symbols"] if item["definition"] and not item["alias"]}
    assert actual == _independent_definitions(ast.parse(source), "hypertagging")
    symbols = {item["canonical"]: item for item in inventory["symbols"]}
    nested = symbols["hypertagging.outer.ordinary_nested"]
    assert nested["kind"] == "function"
    assert nested["surface"] == "internal_unstable" and not nested["indexed"]
    assert symbols["hypertagging.outer.ordinary_nested.deeper"]["kind"] == "function"
    assert symbols["hypertagging.outer.Collector.event.helper"]["kind"] == "function"
    assert symbols["hypertagging.outer.Collector.event"]["kind"] == "method"
    assert not symbols["hypertagging.outer.Collector"]["indexed"]
    choose = [item for item in inventory["symbols"] if item["name"] == "hypertagging._choose"]
    assert len(choose) == 3 and sum(item["indexed"] for item in choose) == 1
    assert len({item["anchor"] for item in inventory["symbols"]}) == len(inventory["symbols"])
    assert all(item["surface"] == "internal_unstable" for item in choose)
    text = (tmp_path / "api/hypertagging.rst").read_text()
    assert "Public / stable definitions" in text and "Internal / unstable definitions" in text
    assert ".. py:function:: outer.ordinary_nested(token=...)" in text
    assert "Nested docs." in text
    assert "nested-default-secret" not in text
    assert "nested_body_secret" not in text and "must-not-publish" not in text
