"""Portable registry parity and archive integrity/overwrite boundaries."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
from wiki_status import load_registry

spec = importlib.util.spec_from_file_location("artifact_storage", ROOT / "scripts/manage_repository_artifacts.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_every_packed_registry_preserves_original_bytes_and_values():
    manifest = json.loads((ROOT / "configs/repository_artifacts.json").read_text())
    for entry in manifest["entries"]:
        if "packed_path" in entry:
            payload = storage.read_payload(entry, ROOT, None)
            assert load_registry(ROOT / entry["path"]) == json.loads(payload)


def test_conflicting_regenerated_registry_fails_closed(tmp_path):
    path = tmp_path / "registry.json"
    path.with_suffix(".json.gz").write_bytes(gzip.compress(b'{"metric":1}'))
    path.write_text('{"metric":2}')
    with pytest.raises(ValueError, match="Conflicting"):
        load_registry(path)


def test_archive_restoration_preserves_existing_files_and_rejects_escape(tmp_path):
    payload = b"original scientific evidence\n"
    record = {"path": "reports/evidence.json", "bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    storage.restore(record, payload, tmp_path)
    storage.restore(record, payload, tmp_path)
    (tmp_path / record["path"]).write_bytes(b"user edit")
    with pytest.raises(ValueError, match="integrity"):
        storage.restore(record, payload, tmp_path)
    assert (tmp_path / record["path"]).read_bytes() == b"user edit"
    with pytest.raises(ValueError, match="escapes"):
        storage.beneath(tmp_path, "../outside")
    with pytest.raises(ValueError, match="integrity"):
        storage.checked(b"corrupt", record)


def test_private_manifest_cli_roundtrip_and_preflight(tmp_path):
    import subprocess

    packed_root = tmp_path / "archive"
    packed_root.mkdir()
    payload = b"completed study log\n"
    packed = gzip.compress(payload, mtime=0)
    (packed_root / "log.gz").write_bytes(packed)
    entry = {"path": "historical/run.log", "bytes": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest(), "mode": 0o640,
             "packed_path": "log.gz", "packed_bytes": len(packed),
             "packed_sha256": hashlib.sha256(packed).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [entry]}))
    output = tmp_path / "restored"
    command = [sys.executable, str(ROOT / "scripts/manage_repository_artifacts.py"),
               "restore", "--manifest", str(manifest), "--packed-root", str(packed_root),
               "--output-dir", str(output)]
    import os
    env = dict(os.environ)
    env.pop("HYPERTAGGING_DATA_ROOT", None)
    restored = subprocess.run(command, check=True, env=env, capture_output=True)
    assert json.loads(restored.stdout)["source"] == "packed_artifacts"
    assert (output / entry["path"]).read_bytes() == payload
    assert (output / entry["path"]).stat().st_mode & 0o777 == 0o640
    bad = dict(entry, path="other.log", sha256="0" * 64)
    manifest.write_text(json.dumps({"entries": [entry, bad]}))
    second = tmp_path / "preflight"
    command[-1] = str(second)
    result = subprocess.run(command, env=env, capture_output=True)
    assert result.returncode != 0
    assert not second.exists()
    manifest.write_text(json.dumps({"entries": [dict(entry, packed_path="../outside.gz")]}))
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
