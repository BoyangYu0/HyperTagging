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
