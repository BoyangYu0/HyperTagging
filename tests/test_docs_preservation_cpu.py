"""Documentation generation must preserve unowned files and filesystem aliases."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
import wiki

_SPEC = importlib.util.spec_from_file_location("build_docs", ROOT / "scripts" / "build_docs.py")
assert _SPEC and _SPEC.loader
build_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_docs)


@pytest.fixture
def generated(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    output = tmp_path / "generated"

    def generate(_root, destination):
        destination.mkdir(parents=True)
        (destination / "index.rst").write_text("Generated page\n", encoding="utf-8")
        return {"source": "fixture"}

    for name in ("generate_api", "generate_catalog", "generate_repository", "generate_status"):
        monkeypatch.setattr(wiki, name, generate)
    return root, output


def test_generation_updates_owned_files_and_preserves_unknown_files(generated):
    root, output = generated
    first = wiki.generate(root, output)
    sentinel = output / "personal-note.txt"
    sentinel.write_text("User-owned content", encoding="utf-8")
    assert wiki.generate(root, output) == first
    assert sentinel.read_text() == "User-owned content"


def test_manually_modified_generated_file_blocks_all_output_mutation(generated):
    root, output = generated
    wiki.generate(root, output)
    (output / "status" / "index.rst").write_text("Manual changes", encoding="utf-8")
    before = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="manual changes"):
        wiki.generate(root, output)
    after = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_new_target_parent_symlink_cannot_escape_owned_directory(generated, tmp_path):
    root, output = generated
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (output / ".wiki-generated.json").write_text("{}", encoding="utf-8")
    (output / "api").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe|symlink"):
        wiki.generate(root, output)
    assert list(external.iterdir()) == []
    assert sorted(path.name for path in output.iterdir()) == [".wiki-generated.json", "api"]


@pytest.mark.parametrize("alias_kind", ["output", "ancestor", "marker_symlink", "marker_hardlink"])
def test_output_and_marker_aliases_are_rejected(generated, tmp_path, alias_kind):
    root, output = generated
    external = tmp_path / "external"
    external.mkdir()
    if alias_kind == "output":
        output.symlink_to(external, target_is_directory=True)
    elif alias_kind == "ancestor":
        output.symlink_to(external, target_is_directory=True)
        output = output / "child"
    else:
        output.mkdir()
        target = external / "manifest.json"
        target.write_text("{}", encoding="utf-8")
        if alias_kind == "marker_symlink":
            (output / ".wiki-generated.json").symlink_to(target)
        else:
            os.link(target, output / ".wiki-generated.json")
    with pytest.raises(ValueError, match="symlink|singly linked"):
        wiki.generate(root, output)
    assert not (external / "child").exists()
    assert not (external / "api").exists()


def test_owned_file_hardlink_is_rejected_without_touching_alias(generated, tmp_path):
    root, output = generated
    wiki.generate(root, output)
    external = tmp_path / "external.txt"
    os.link(output / "api" / "index.rst", external)
    before = external.read_bytes()
    with pytest.raises(ValueError, match="singly linked"):
        wiki.generate(root, output)
    assert external.read_bytes() == before


def test_unowned_collision_or_parent_file_fails_before_writes(generated):
    root, output = generated
    output.mkdir()
    (output / ".wiki-generated.json").write_text("{}", encoding="utf-8")
    (output / "status").write_text("User-owned obstruction", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        wiki.generate(root, output)
    assert not (output / "api").exists()
    assert (output / "status").read_text() == "User-owned obstruction"


@pytest.mark.parametrize("name", ["../external", "/absolute", "api/../index.rst", "api//index.rst"])
def test_ownership_manifest_cannot_name_noncanonical_paths(generated, name):
    root, output = generated
    output.mkdir()
    (output / ".wiki-generated.json").write_text(json.dumps({name: hashlib.sha256(b"").hexdigest()}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        wiki.generate(root, output)
    assert not (output / "api").exists()


def test_build_destination_rejects_aliases_sources_and_nonempty_paths(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)
    for destination in (alias, alias / "child", root, root / ".git" / "empty", root / "src" / "empty"):
        with pytest.raises(ValueError):
            build_docs._output_path(root, destination)
    (external / "sentinel").write_text("Preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="existing files"):
        build_docs._output_path(root, external)
    assert (external / "sentinel").read_text() == "Preserve me"
    assert build_docs._output_path(root, root / "artifacts" / "docs-new") == root / "artifacts" / "docs-new"
    assert build_docs._output_path(root, root / "docs" / "_build" / "new") == root / "docs" / "_build" / "new"


@pytest.fixture
def staging_repository(tmp_path, monkeypatch):
    """Exercise staging preflight without running a whole documentation build."""
    root = tmp_path / "repository"
    files = {
        "docs/wiki/index.rst": "Wiki\n====\n",
        "docs/wiki/conf.py": "project = 'Fixture'\n",
        "docs/wiki/_static/wiki.css": ".wiki { color: black; }\n",
        "docs/wiki/.nojekyll": "",
        "docs/index.rst": "Standalone\n==========\n",
        "docs/conf.py": "project = 'Fixture'\n",
        "doc/index-hypertagging.rst": "Package\n=======\n",
    }
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(build_docs, "__file__", str(root / "scripts" / "build_docs.py"))
    calls = []
    monkeypatch.setattr(build_docs.subprocess, "run", lambda *args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0))
    return root, tmp_path / "build", calls


@pytest.mark.parametrize("entry,layout", [
    ("docs/index.rst", "standalone"),
    ("doc/index-hypertagging.rst", "basf2"),
    ("docs/conf.py", "standalone"),
    ("docs/wiki/conf.py", "standalone"),
])
def test_entry_source_alias_is_rejected_before_any_staging(staging_repository, tmp_path, entry, layout):
    root, output, calls = staging_repository
    private = tmp_path / "private.rst"
    private.write_text("UNRELATED-SOURCE-BODY\n", encoding="utf-8")
    (root / entry).unlink()
    (root / entry).symlink_to(private)
    with pytest.raises(SystemExit) as failure:
        build_docs.main(["--output", str(output), "--layout", layout])
    assert failure.value.code == 2
    assert not output.exists()
    assert not calls
    assert private.read_text() == "UNRELATED-SOURCE-BODY\n"


@pytest.mark.parametrize("name", ["private-config.json", "raw-report.txt", "subdirectory/raw-repository.rst"])
def test_static_staging_never_copies_unrecognized_source_bodies(staging_repository, name):
    root, output, _calls = staging_repository
    injected = root / "docs" / "wiki" / "_static" / name
    injected.parent.mkdir(parents=True, exist_ok=True)
    body = "UNRELATED-RAW-CONFIGURATION-BODY"
    injected.write_text(body, encoding="utf-8")
    # Either rejecting unexpected assets before mutation or omitting them is a
    # safe policy. Recursively copying arbitrary _static content is not.
    try:
        result = build_docs.main(["--output", str(output)])
    except SystemExit as failure:
        assert failure.code == 2
        assert not output.exists()
    else:
        assert result == 0
        assert not any(body in path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    assert injected.read_text() == body


def test_staged_generator_failure_preserves_previous_outputs(generated, monkeypatch):
    root, output = generated
    wiki.generate(root, output)
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    def fail_after_partial_generation(_root, destination):
        destination.mkdir(parents=True)
        (destination / "partial.rst").write_text("Partial staged output")
        raise ValueError("Synthetic staged generation failure")
    monkeypatch.setattr(wiki, "generate_status", fail_after_partial_generation)
    with pytest.raises(ValueError, match="staged generation failure"):
        wiki.generate(root, output)
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_failed_staged_privacy_validation_preserves_previous_outputs(generated, monkeypatch):
    root, output = generated
    wiki.generate(root, output)
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    def reject(_stage, _root, **_kwargs):
        raise ValueError("Synthetic privacy rejection")
    monkeypatch.setattr(wiki, "validate_artifact", reject)
    with pytest.raises(ValueError, match="privacy rejection"):
        wiki.generate(root, output)
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_unowned_sphinx_source_is_preserved_but_cannot_join_generated_site(generated):
    root, output = generated
    wiki.generate(root, output)
    unowned = output / "unowned-orphan.rst"
    # An orphan flag suppresses toctree warnings. A plain body need not contain
    # any recognizable secret, so publication must be blocked by ownership.
    unowned.write_text(":orphan:\n\nUser source\n===========\n\nUNOWNED-RAW-SOURCE-BODY\n", encoding="utf-8")
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="unowned|Unowned"):
        wiki.generate(root, output)
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_nojekyll_marker_cannot_publish_an_arbitrary_body(staging_repository):
    root, output, calls = staging_repository
    marker = root / "docs" / "wiki" / ".nojekyll"
    marker.write_text("UNRELATED-MARKER-PAYLOAD", encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        build_docs.main(["--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()
    assert not calls
    assert marker.read_text() == "UNRELATED-MARKER-PAYLOAD"
