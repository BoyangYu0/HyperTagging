"""Privacy and whole-tree coverage of the compact repository inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
from wiki_repository import generate_repository, repository_files


def _put(root: Path, relative: str, text: str = "private body\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_compact_inventory_covers_whole_git_without_disclosing_paths_or_bodies(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--quiet")
    files = {
        "src/hypertagging/example.py": 'def f():\n    """PRIVATE_DOCSTRING"""\n    return "PRIVATE_FUNCTION_BODY"\n',
        "tests/test_contract.py": 'raise RuntimeError("MUST_NOT_IMPORT")\n',
        "reconstruction/snapshots/private_user/example.py": 'invalid_python_but_inventory_must_hash_it !\n',
        "docs/audits/private_audit.md": "hostname=login123.example.org\njob_id=982731\n",
        "configs/private_config.yaml": "api_key: PRIVATE_CONFIG_SECRET\n",
        "reports/private_user_report.json": '{"username": "private_user"}\n',
        "private_user_directory/PRIVATE_FILENAME.txt": "PRIVATE_CONTENT\n",
        "README.md": "PRIVATE_README_BODY\n",
        ".gitignore": "ignored-private.txt\n",
    }
    for relative, payload in files.items():
        _put(root, relative, payload)
    _git(root, "add", "README.md")
    # Tracked working-tree modifications must be hashed as they exist now.
    files["README.md"] += "CURRENT_UNCOMMITTED_BODY\n"
    (root / "README.md").write_text(files["README.md"])
    _put(root, "ignored-private.txt", "IGNORED_SECRET\n")
    _put(root, "src/__pycache__/ignored.py", "CACHE_SECRET\n")
    output = tmp_path / "inventory"
    manifest = generate_repository(root, output)
    expected_ids = {hashlib.sha256(relative.encode("utf-8")).hexdigest(): relative for relative in files}
    assert {entry["path_id"] for entry in manifest["files"]} == set(expected_ids)
    assert manifest["total_files"] == len(files)
    assert manifest["python_modules"] == 3
    assert manifest["groups"]["other"] == 1
    assert sum(manifest["groups"].values()) == len(files)
    assert sum(manifest["kinds"].values()) == len(files)
    assert sorted(path.name for path in output.iterdir()) == ["index.rst", "manifest.json"]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for entry in manifest["files"]:
        assert set(entry) == {"path_id", "group", "kind", "bytes", "sha256", "content_status"}
        payload = files[expected_ids[entry["path_id"]]].encode("utf-8")
        assert entry["sha256"] == hashlib.sha256(
            b"hypertagging-docs-public-content-v1\0" + payload,
        ).hexdigest()
        assert entry["bytes"] == len(payload)
        assert entry["content_status"] == "hashed_text"
    published = "\n".join(path.read_text() for path in output.iterdir())
    for value in (*files, "PRIVATE_", "CURRENT_UNCOMMITTED_BODY", "private_user", "login123", "982731", str(root)):
        assert value not in published
    assert "source_pages" not in manifest
    assert "documented_public_definitions" not in published


def test_binary_model_data_inventory_uses_stat_only_without_reading_payloads(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--quiet")
    suffixes = (".pt", ".pth", ".ckpt", ".parquet", ".root", ".npz", ".onnx", ".bin", ".png", ".unknown")
    inputs = {_put(root, "private_artifacts/model" + suffix) for suffix in suffixes}
    output = tmp_path / "inventory"
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path in inputs:
            raise RuntimeError("data/model/binary payload must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = generate_repository(root, output)
    assert manifest["stat_only_files"] == len(inputs)
    assert manifest["fingerprinted_files"] == 0
    assert all(entry["sha256"] is None for entry in manifest["files"])
    assert all(entry["content_status"] == "stat_only_not_read" for entry in manifest["files"])
    assert all(entry["group"] == "other" for entry in manifest["files"])


def test_repository_inventory_is_relocatable_and_deterministic(tmp_path):
    first = tmp_path / "first-checkout"
    second = tmp_path / "second-checkout"
    for root in (first, second):
        _put(root, "docs/audit.md", "same private body\n")
        _put(root, "src/hypertagging/example.py", '"""same documentation"""\n')
    outputs = [tmp_path / "first-output", tmp_path / "second-output"]
    first_manifest = generate_repository(first, outputs[0])
    assert generate_repository(second, outputs[1]) == first_manifest
    first_files = {path.name: path.read_bytes() for path in outputs[0].iterdir()}
    assert {path.name: path.read_bytes() for path in outputs[1].iterdir()} == first_files
    before = {path: path.stat().st_mtime_ns for path in outputs[0].iterdir()}
    generate_repository(first, outputs[0])
    assert {path: path.stat().st_mtime_ns for path in before} == before


def test_repository_discovery_does_not_follow_external_symlinks(tmp_path):
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    external = _put(tmp_path / "external", "private.py", "PRIVATE_SOURCE\n")
    (root / "src/escape.py").symlink_to(external)
    (root / "src/escape-directory").symlink_to(external.parent, target_is_directory=True)
    assert repository_files(root) == []


def test_repository_inventory_refuses_old_mirrors_and_preserves_them(tmp_path):
    root = tmp_path / "checkout"
    _put(root, "README.md")
    output = tmp_path / "old-output"
    old = _put(output, "sources/private.txt", "USER_OWNED_SOURCE\n")
    with pytest.raises(ValueError, match="source mirrors"):
        generate_repository(root, output)
    assert old.read_text() == "USER_OWNED_SOURCE\n"


def test_repository_inventory_rejects_output_aliases(tmp_path):
    root = tmp_path / "checkout"
    _put(root, "README.md")
    with pytest.raises(ValueError, match="inputs"):
        generate_repository(root, root)
    external = tmp_path / "external-output"
    external.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        generate_repository(root, alias)
    assert list(external.iterdir()) == []
