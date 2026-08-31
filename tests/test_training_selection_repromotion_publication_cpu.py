from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

import hypertagging.data.selection_repromotion as repromotion
import hypertagging.data.selection_repromotion_publication as publication
from hypertagging.data.selection_repromotion import (
    build_repromotion_package,
    load_repromotion_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "configs/training_selection/repromotion/train_035k_16key_repromotion.contract.json"
)
NOW = datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(contract: dict) -> dict:
    return {
        "repository_head": "1" * 40,
        "implementation_tag": "fixture-publication-helper-tag",
        "implementation_tag_object": "2" * 40,
        "implementation_tag_type": "tag",
        "tracked_worktree_clean": True,
        "base_code_commit": contract["base_code"]["commit"],
        "base_code_tag": contract["base_code"]["tag"],
        "base_tag_is_ancestor": True,
    }


def _context() -> publication.CommandContext:
    return publication.CommandContext(
        cwd=str(ROOT),
        argv=(
            str(Path(sys.executable).resolve()),
            str(ROOT / "scripts/publish_training_selection_repromotion_once.py"),
            "--authorization",
            "/tmp/exact-publication-authorization.json",
        ),
        environment={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
        },
    )


def _tool(name: str, path: Path) -> dict:
    canonical = path.resolve(strict=True)
    return {
        "name": name,
        "path": str(canonical),
        "canonical_path": str(canonical),
        "sha256": _sha(canonical),
    }


def _write_authorization(path: Path, payload: dict) -> None:
    payload[publication.AUTHORIZATION_HASH_FIELD] = publication._canonical_hash(
        payload, publication.AUTHORIZATION_HASH_FIELD
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _authorization(tmp_path: Path) -> tuple[Path, dict, publication.CommandContext]:
    contract = load_repromotion_contract(CONTRACT_PATH)
    output_root = tmp_path / "immutable-package"
    context = _context()
    provenance = _provenance(contract)
    package = build_repromotion_package(
        contract,
        ROOT,
        implementation_provenance=provenance,
        output_namespace=str(output_root),
    )
    paths = publication._namespace_paths(output_root)
    payload = {
        "schema_version": publication.AUTHORIZATION_VERSION,
        "authorization_id": "fixture-one-use-publication",
        "created_at_utc": "2026-08-31T10:00:00Z",
        "expires_at_utc": "2026-08-31T11:00:00Z",
        "one_use": True,
        "retry_authorized": False,
        "repository_root": str(ROOT),
        "implementation": {
            "commit": provenance["repository_head"],
            "tree": "3" * 40,
            "tag": provenance["implementation_tag"],
            "tag_object": provenance["implementation_tag_object"],
        },
        "contract": {
            "path": str(CONTRACT_PATH),
            "file_sha256": _sha(CONTRACT_PATH),
            "internal_hash": contract["contract_sha256"],
        },
        "inputs": contract["inputs"],
        "namespace": {
            **{key: str(value) for key, value in paths.items()},
            "all_must_be_absent": True,
        },
        "expected_outputs": publication._expected_outputs(contract, output_root),
        "expected_package_receipt_sha256": hashlib.sha256(
            package.receipt_bytes
        ).hexdigest(),
        "command": {
            "cwd": context.cwd,
            "argv": list(context.argv),
            "environment": context.environment,
        },
        "tools": [
            _tool("env", Path("/usr/bin/env")),
            _tool("git", Path("/usr/bin/git")),
            _tool("python", Path(sys.executable)),
        ],
        "authorization": {
            **publication.AUTHORIZATION_FLAGS,
            "metadata_package_publication_authorized": True,
        },
    }
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path, payload)
    return authorization_path, payload, context


def _publish(tmp_path: Path, monkeypatch):
    authorization_path, payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    paths = publication.publish_authorized_once(
        authorization_path,
        repository_root=ROOT,
        contract_path=CONTRACT_PATH,
        command_context=context,
        now=NOW,
    )
    return paths, payload


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "false",
        "expired",
        "command",
        "tool",
        "tool_path",
        "environment",
    ),
)
def test_authorization_schema_and_context_fail_before_metadata_or_lock(
    tmp_path, monkeypatch, mutation
):
    authorization_path, payload, context = _authorization(tmp_path)
    authorization_path.chmod(0o644)
    if mutation == "missing":
        payload.pop("one_use")
    elif mutation == "extra":
        payload["unexpected"] = False
    elif mutation == "false":
        payload["authorization"]["metadata_package_publication_authorized"] = False
    elif mutation == "expired":
        payload["expires_at_utc"] = "2026-08-31T10:01:00Z"
    elif mutation == "command":
        payload["command"]["cwd"] = "/wrong"
    elif mutation == "tool":
        payload["tools"][0]["sha256"] = "0" * 64
    elif mutation == "tool_path":
        payload["tools"][1] = _tool("git", Path("/usr/bin/env"))
    else:
        context.environment["PATH"] = "/tmp"
    _write_authorization(authorization_path, payload)
    monkeypatch.setattr(
        publication,
        "build_repromotion_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata opened after invalid authorization")
        ),
    )
    with pytest.raises(ValueError):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert not any(
        name.endswith("authorization.lock.json") for name in os.listdir(tmp_path)
    )


def test_expected_receipt_mismatch_fails_before_lock(tmp_path, monkeypatch):
    authorization_path, payload, context = _authorization(tmp_path)
    authorization_path.chmod(0o644)
    payload["expected_package_receipt_sha256"] = "0" * 64
    _write_authorization(authorization_path, payload)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    with pytest.raises(ValueError, match="expected package receipt"):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert not any(
        name.endswith("authorization.lock.json") for name in os.listdir(tmp_path)
    )


def test_descriptor_owner_mismatch_fails_before_metadata_or_lock(tmp_path, monkeypatch):
    authorization_path, _payload, context = _authorization(tmp_path)
    owner_uid = os.geteuid()
    monkeypatch.setattr(publication.os, "geteuid", lambda: owner_uid + 1)
    monkeypatch.setattr(
        publication,
        "build_repromotion_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata opened after owner mismatch")
        ),
    )
    with pytest.raises(ValueError, match="owned unique"):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert not any(
        name.endswith("authorization.lock.json") for name in os.listdir(tmp_path)
    )


@pytest.mark.parametrize("collision", ("regular", "symlink"))
def test_lock_collision_is_fail_closed_without_publication(
    tmp_path, monkeypatch, collision
):
    authorization_path, payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    lock = Path(payload["namespace"]["authorization_lock"])
    if collision == "regular":
        lock.write_text("stale")
    else:
        lock.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert not Path(payload["namespace"]["output_root"]).exists()


def test_failure_after_lock_is_permanent_no_retry(tmp_path, monkeypatch):
    authorization_path, payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    monkeypatch.setattr(
        publication,
        "publish_repromotion_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert Path(payload["namespace"]["authorization_lock"]).is_file()
    assert not Path(payload["namespace"]["accepted_receipt"]).exists()
    with pytest.raises(FileExistsError):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )


def test_success_writes_accepted_receipt_last_and_keeps_all_authority_inert(
    tmp_path, monkeypatch
):
    authorization_path, payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    order: list[str] = []
    original = publication._exclusive_write

    def observed(path, content):
        if path.name.endswith("accepted.json"):
            assert Path(
                payload["expected_outputs"]["package_receipt"]["path"]
            ).is_file()
        order.append(path.name)
        return original(path, content)

    monkeypatch.setattr(publication, "_exclusive_write", observed)
    paths = publication.publish_authorized_once(
        authorization_path,
        repository_root=ROOT,
        contract_path=CONTRACT_PATH,
        command_context=context,
        now=NOW,
    )
    assert order == [
        Path(payload["namespace"]["authorization_lock"]).name,
        Path(payload["namespace"]["accepted_receipt"]).name,
    ]
    accepted = json.loads(paths["accepted_receipt"].read_text())
    assert accepted[publication.ACCEPTED_HASH_FIELD] == publication._canonical_hash(
        accepted, publication.ACCEPTED_HASH_FIELD
    )
    assert accepted["gates"] == {
        "consumer_pin_authorized": False,
        "live_pointer_activation_authorized": False,
        "metadata_package_publication_accepted": True,
        "raw_or_source_payload_accessed": False,
        "science_authorized": False,
        "slurm_actions_performed": False,
        "submission_authorized": False,
    }
    for key in ("authorization_lock", "accepted_receipt"):
        metadata = paths[key].stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1
    assert stat.S_IMODE(paths["output_root"].stat().st_mode) == 0o555


def test_accepted_receipt_failure_strands_package_without_retry(tmp_path, monkeypatch):
    authorization_path, payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    original = publication._exclusive_write

    def fail_accepted(path, content):
        if path.name.endswith("accepted.json"):
            raise OSError("injected accepted-receipt failure")
        return original(path, content)

    monkeypatch.setattr(publication, "_exclusive_write", fail_accepted)
    with pytest.raises(OSError, match="accepted-receipt"):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )
    assert Path(payload["expected_outputs"]["package_receipt"]["path"]).is_file()
    assert not Path(payload["namespace"]["accepted_receipt"]).exists()
    with pytest.raises(FileExistsError):
        publication.publish_authorized_once(
            authorization_path,
            repository_root=ROOT,
            contract_path=CONTRACT_PATH,
            command_context=context,
            now=NOW,
        )


def test_wrapper_build_opens_only_four_authenticated_metadata_documents(
    tmp_path, monkeypatch
):
    authorization_path, _payload, context = _authorization(tmp_path)
    contract = load_repromotion_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        publication,
        "_validate_implementation",
        lambda *_args, **_kwargs: _provenance(contract),
    )
    opened: list[str] = []
    original = repromotion._read_bound

    def observed(root, name, spec):
        opened.append(name)
        return original(root, name, spec)

    monkeypatch.setattr(repromotion, "_read_bound", observed)
    publication.publish_authorized_once(
        authorization_path,
        repository_root=ROOT,
        contract_path=CONTRACT_PATH,
        command_context=context,
        now=NOW,
    )
    assert opened == ["inventory", "roles", "legacy_selection", "legacy_index"]
