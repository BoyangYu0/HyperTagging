"""Static documentation checks; no project, training or scheduler imports."""

from __future__ import annotations

import ast
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
SPEC = importlib.util.spec_from_file_location("wiki_catalog", ROOT / "docs/_ext/wiki_catalog.py")
assert SPEC and SPEC.loader
CATALOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CATALOG)


def _files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _snapshot_scripts(snapshot: Path) -> Path:
    """Freeze inputs while other contributors may be editing the checkout."""
    for name in ("scripts", "examples"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"} and "__pycache__" not in path.parts:
                copied = snapshot / path.relative_to(ROOT)
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes(path.read_bytes())
    return snapshot


def _independent_definitions(source: bytes) -> set[tuple]:
    """Collect lexical definition identities independently of the catalog code."""
    definitions = set()

    def visit(node, prefix="", scope="module", function_local=False):
        if isinstance(node, ast.ClassDef):
            name = prefix + "." + node.name if prefix else node.name
            internal = function_local or any(part.startswith("_") for part in name.split("."))
            definitions.add((
                name, node.lineno, node.end_lineno, "class",
                "internal/unstable" if internal else "public",
            ))
            for child in node.body:
                visit(child, name, "class", function_local)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + "." + node.name if prefix else node.name
            internal = function_local or any(part.startswith("_") for part in name.split("."))
            definitions.add((
                name, node.lineno, node.end_lineno,
                "method" if scope == "class" else "function",
                "internal/unstable" if internal else "public",
            ))
            for child in node.body:
                visit(child, name, "function", True)
            return
        if not isinstance(node, ast.Lambda):
            for child in ast.iter_child_nodes(node):
                visit(child, prefix, scope, function_local)

    visit(ast.parse(source))
    return definitions


def test_catalog_covers_every_script_without_publishing_sources(tmp_path):
    snapshot = _snapshot_scripts(tmp_path / "source-snapshot")
    output = tmp_path / "catalogue"
    manifest = CATALOG.generate_catalog(snapshot, output)
    expected = {
        path.relative_to(snapshot).as_posix()
        for name in ("scripts", "examples")
        for path in (snapshot / name).rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"} and "__pycache__" not in path.parts
    }
    assert {entry["path"] for entry in manifest["entries"]} == expected
    assert manifest["total"] == len(expected)
    assert sum(manifest["counts"].values()) == len(expected)
    assert manifest["schema_version"] == 2
    expected_definition_count = 0
    for entry in manifest["entries"]:
        source = (snapshot / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(source).hexdigest()
        page = output / (entry["docname"] + ".rst")
        assert page.is_file()
        assert entry["sha256"] in page.read_text()
        assert entry["prerequisites"] and entry["side_effects"]
        assert entry["line"] == 1 and entry["end_line"] >= 1
        assert all("declaration" not in argument for argument in entry.get("arguments", []))
        if entry["path"].endswith(".py"):
            expected_definitions = _independent_definitions(source)
            actual_definitions = {
                (item["name"], item["line"], item["end_line"], item["kind"], item["surface"])
                for item in entry["definitions"]
            }
            assert actual_definitions == expected_definitions, entry["path"]
            assert len(actual_definitions) == len(entry["definitions"])
            expected_definition_count += len(expected_definitions)
        else:
            assert not entry.get("definitions")
    assert manifest["definition_count"] == expected_definition_count
    assert not any(path.name in {"sources", "evidence", "_sources"} for path in output.rglob("*"))
    assert all(path.suffix in {".rst", ".json"} for path in output.rglob("*") if path.is_file())
    assert "Download the exact source" not in "\n".join(path.read_text() for path in output.rglob("*.rst"))
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_catalog_is_deterministic_without_repository_location_or_timestamps(tmp_path):
    # Freeze inputs so an unrelated editor adding a script during this test does
    # not turn a source change into a false nondeterminism report.
    snapshot = _snapshot_scripts(tmp_path / "source-snapshot")
    first = tmp_path / "first"
    second = tmp_path / "second"
    CATALOG.generate_catalog(snapshot, first)
    CATALOG.generate_catalog(snapshot, second)
    assert _files(first) == _files(second)
    before = {path: path.stat().st_mtime_ns for path in first.rglob("*") if path.is_file()}
    CATALOG.generate_catalog(snapshot, first)
    assert {path: path.stat().st_mtime_ns for path in before} == before


def test_catalog_exposes_real_cpu_commands_and_never_advertises_fake_helper_help(tmp_path):
    entries = {entry["path"]: entry for entry in CATALOG.generate_catalog(ROOT, tmp_path / "site")["entries"]}
    for name in ("toy_mc", "grafei", "gpt_like"):
        entry = entries[f"examples/{name}_minimal/run_example.py"]
        assert entry["fixture"]
        assert entry["command"].endswith(f"python examples/{name}_minimal/run_example.py")
        assert 'CUDA_VISIBLE_DEVICES=""' in entry["command"]
    for name in ("train_hyperbolic_pretrain", "train_level_reconstruction"):
        assert entries[f"scripts/{name}.py"]["command"].endswith("--dry-run --tiny --device cpu --max-steps 2 --batch-size 2")
    for entry in entries.values():
        if entry["command"] and entry["command"].endswith("--help"):
            assert entry["arguments"]
            assert entry["path"].endswith(".py")
        if entry["kind"] in {"Slurm batch entry point", "Slurm execution wrapper", "imported helper module"}:
            assert entry["command"] is None
    assert entries["scripts/slurm/submit_phase3_batch_efficiency_calibration.py"]["command"] is None
    assert "Even --dry-run" in entries["scripts/condor/submit_mdst_production_10m.sh"]["side_effects"]
    assert entries["scripts/activate_env.sh"]["command"].startswith("source ")


def test_inventory_never_executes_source_and_discovers_new_cli_declarations(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    sentinel = tmp_path / "must-not-exist"
    source = f'''"""A new command with a literal source description."""
import basf2
import torch
from pathlib import Path
raise RuntimeError("Source must never be imported")
Path({str(sentinel)!r}).write_text("executed")
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--count", type=int, default=3)
    parser.parse_args()
if __name__ == "__main__":
    main()
'''
    (scripts / "new.py").write_text(source)
    (scripts / "new.sh").write_text("#!/bin/bash\n# Shell sibling.\necho not-run\n")
    (scripts / "helper.py").write_text('"""Helper; text mentioning argparse is not a parser."""\ndef f():\n    return 1\n')
    result = CATALOG.generate_catalog(repo, tmp_path / "site")
    entries = {entry["path"]: entry for entry in result["entries"]}
    assert not sentinel.exists()
    assert [item["names"] for item in entries["scripts/new.py"]["arguments"]] == [["--output"], ["--count"]]
    assert entries["scripts/helper.py"]["command"] is None
    assert entries["scripts/new.py"]["command"] is None
    assert entries["scripts/new.sh"]["command"] is None
    assert entries["scripts/new.py"]["docname"] != entries["scripts/new.sh"]["docname"]
    assert entries["scripts/new.py"]["definitions"][0]["signature"] == "main()"
    assert "executed" not in json.dumps(result)
    assert str(sentinel) not in json.dumps(result)
    original_hash = entries["scripts/new.py"]["sha256"]
    (scripts / "new.py").write_text(source + "\n# source update\n")
    updated = CATALOG.generate_catalog(repo, tmp_path / "site")
    assert next(entry for entry in updated["entries"] if entry["path"] == "scripts/new.py")["sha256"] != original_hash


def test_catalog_rejects_sources_that_escape_repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    external = tmp_path / "private.py"
    external.write_text("secret = 1\n")
    (repo / "scripts/linked.py").symlink_to(external)
    with pytest.raises(ValueError, match="escapes the repository"):
        CATALOG.generate_catalog(repo, tmp_path / "site")


def test_catalog_rejects_generation_into_source_directories(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        CATALOG.generate_catalog(repo, repo / "scripts/generated")


def test_catalog_publishes_only_redacted_ast_projection(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "private_fixture.py").write_text('''"""Example command.

Username: confidential_user; hostname=login123.example.org
Input /home/confidential_user/project/private.json
GPU-1234abcd-1234-abcd-1234-123456789abc
job_id=982731 checkpoint_path=checkpoints/private.ckpt
api_key=ghp_1234567890abcdefghijklmnop
"""
import argparse
from pathlib import Path
# RAW_COMMENT_BODY_MUST_NOT_PUBLISH
RAW_SECRET = "RAW_ASSIGNMENT_BODY_MUST_NOT_PUBLISH"
def _helper(value="OPAQUE_DEFAULT", nested=Path("OPAQUE_NESTED_DEFAULT")):
    """Internal helper with password=DOCSTRING_PASSWORD."""
    return "RAW_FUNCTION_BODY_MUST_NOT_PUBLISH"
class Example:
    def _method(self, *, token="OPAQUE_METHOD_DEFAULT"):
        """Method documentation."""
        return RAW_SECRET
def main():
    parser = argparse.ArgumentParser(description="RAW_PARSER_DESCRIPTION")
    parser.add_argument("--output", help="RAW_ARGUMENT_HELP", default="RAW_ARGUMENT_DEFAULT")
    Path("OPAQUE_CALL_PATH").write_text(RAW_SECRET)
if __name__ == "__main__":
    main()
''')
    (scripts / "private_fixture.sh").write_text('#!/bin/bash\n# RAW_SHELL_COMMENT\nprintf RAW_SHELL_BODY\n')
    output = tmp_path / "site"
    manifest = CATALOG.generate_catalog(repo, output)
    published = "\n".join(path.read_text() for path in output.rglob("*") if path.is_file())
    for forbidden in (
        "confidential_user", "login123.example.org", "GPU-1234abcd", "982731",
        "private.ckpt", "ghp_1234567890", "DOCSTRING_PASSWORD",
        "RAW_COMMENT_BODY", "RAW_ASSIGNMENT_BODY", "RAW_FUNCTION_BODY",
        "RAW_PARSER_DESCRIPTION", "RAW_ARGUMENT_HELP", "RAW_ARGUMENT_DEFAULT",
        "RAW_SHELL_COMMENT", "RAW_SHELL_BODY", "OPAQUE_DEFAULT",
        "OPAQUE_NESTED_DEFAULT", "OPAQUE_METHOD_DEFAULT", "OPAQUE_CALL_PATH",
        "/home/", ":literalinclude:", ":download:`Download the exact source",
    ):
        assert forbidden not in published
    assert "[redacted]" in published
    entry = next(item for item in manifest["entries"] if item["path"].endswith(".py"))
    definitions = {item["name"]: item for item in entry["definitions"]}
    assert set(definitions) == {"_helper", "Example", "Example._method", "main"}
    assert definitions["_helper"]["signature"] == "_helper(value=..., nested=...)"
    assert definitions["Example._method"]["surface"] == "internal/unstable"
    assert definitions["Example._method"]["docstring"] == "Method documentation."
    assert entry["io_evidence"] == [{"line": 23, "operation": "write_text"}]


def test_catalog_qualifies_and_classifies_every_nested_definition(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "nested.py").write_text('''def outer():
    def nested(value: "PRIVATE_ANNOTATION", option="PRIVATE_DEFAULT"):
        """Nested documentation."""
        body_value = "PRIVATE_BODY"
        return body_value
    async def async_nested():
        pass
    class Inner:
        def method(self, option="PRIVATE_METHOD_DEFAULT"):
            def callback():
                pass
            return callback
    return Inner
class Container:
    def method(self):
        pass
    class Nested:
        def member(self):
            pass
''')
    manifest = CATALOG.generate_catalog(repo, tmp_path / "site")
    entry = manifest["entries"][0]
    definitions = {item["name"]: item for item in entry["definitions"]}
    assert set(definitions) == {
        "outer", "outer.nested", "outer.async_nested", "outer.Inner",
        "outer.Inner.method", "outer.Inner.method.callback", "Container",
        "Container.method", "Container.Nested", "Container.Nested.member",
    }
    assert definitions["outer.nested"]["kind"] == "function"
    assert definitions["outer.Inner.method"]["kind"] == "method"
    assert definitions["outer.Inner.method.callback"]["kind"] == "function"
    assert all(
        definitions[name]["surface"] == "internal/unstable"
        for name in definitions if name.startswith("outer.")
    )
    assert definitions["Container.Nested.member"]["surface"] == "public"
    assert definitions["outer.nested"]["signature"] == (
        "outer.nested(value: '[redacted]', option=...)"
    )
    published = json.dumps(manifest) + "\n" + "\n".join(
        path.read_text() for path in (tmp_path / "site").rglob("*.rst")
    )
    for private in (
        "PRIVATE_ANNOTATION", "PRIVATE_DEFAULT", "PRIVATE_BODY",
        "PRIVATE_METHOD_DEFAULT", "body_value",
    ):
        assert private not in published


def test_independent_validator_rejects_missing_nested_catalog_definition(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src/hypertagging").mkdir(parents=True)
    (repo / "src/hypertagging/__init__.py").write_text('"""Fixture package."""\n')
    (repo / "scripts").mkdir()
    (repo / "scripts/example.py").write_text(
        "def outer():\n    def nested():\n        pass\n    return nested\n"
    )
    generated = tmp_path / "generated"
    from wiki_api import generate_api
    from wiki_repository import generate_repository
    generate_api(repo, generated / "api")
    generate_repository(repo, generated / "repository")
    manifest = CATALOG.generate_catalog(repo, generated / "catalog")
    validator_spec = importlib.util.spec_from_file_location(
        "catalog_coverage_validation_for_test", ROOT / "scripts/validate_docs.py"
    )
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    monkeypatch.setattr(validator, "ROOT", repo)
    assert validator.validate_coverage(generated)["catalog_definitions"] == 2
    entry = manifest["entries"][0]
    entry["definitions"] = [
        item for item in entry["definitions"] if item["name"] != "outer.nested"
    ]
    manifest["definition_count"] -= 1
    (generated / "catalog/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing catalog definitions"):
        validator.validate_coverage(generated)


def test_catalog_docstrings_cannot_activate_rst_directives(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/inert.py").write_text('''"""Ordinary description.

.. include:: PRIVATE_FILE

.. raw:: html

   <script>RAW_SCRIPT_MARKER</script>
"""
def main():
    """.. include:: PRIVATE_FILE"""
''')
    output = tmp_path / "site"
    CATALOG.generate_catalog(repo, output)
    page = (output / "entries/scripts/inert.py.rst").read_text()
    assert "\n.. include::" not in page
    assert "\n.. raw::" not in page
    assert "   .. include:: PRIVATE_FILE" in page
