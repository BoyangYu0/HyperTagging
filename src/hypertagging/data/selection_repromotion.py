"""Fail-closed, metadata-only 15-to-16-key selection repromotion."""

from __future__ import annotations

from copy import deepcopy
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Mapping

from hypertagging.data.dataset_index import (
    _index_hash,
    _selection_fingerprint_from_strings,
    _validate_dataset_index_metadata_payload,
)
from hypertagging.data.training_selection import (
    HASH_FIELD,
    INVENTORY_VERSION,
    ROLE_MANIFEST_VERSION,
    SELECTION_ENTRY_KEYS,
    SELECTION_MANIFEST_VERSION,
    _validate_hashed_manifest_payload,
    _validate_training_selection_index_metadata_pure,
    canonical_manifest_hash,
    validate_training_selection_metadata,
)

CONTRACT_VERSION = "hypertagging-training-selection-repromotion-contract-v1"
RECEIPT_VERSION = "hypertagging-training-selection-repromotion-receipt-v1"
CLAIM_VERSION = "hypertagging-training-selection-repromotion-claim-v1"
CONTRACT_HASH_FIELD = "contract_sha256"
RECEIPT_HASH_FIELD = "receipt_sha256"
OUTPUT_RECEIPT_NAME = "repromotion.receipt.json"
LEGACY_ENTRY_KEYS = SELECTION_ENTRY_KEYS - {"campaign_config_digest"}
MAX_METADATA_BYTES = 8 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_INPUT_VERSIONS = {
    "inventory": ("manifest_hash", INVENTORY_VERSION),
    "roles": ("manifest_hash", ROLE_MANIFEST_VERSION),
    "legacy_selection": ("manifest_hash", SELECTION_MANIFEST_VERSION),
    "legacy_index": ("index_hash", None),
}
_AUTHORIZATION_KEYS = {
    "execution_authorized",
    "live_pointer_publish_authorized",
    "raw_or_source_payload_access_authorized",
    "science_authorized",
    "slurm_actions_authorized",
    "submission_authorized",
}


@dataclass(frozen=True)
class _BoundDocument:
    path: str
    payload: dict[str, Any]
    file_sha256: str
    mode: int
    nlink: int


@dataclass(frozen=True)
class RepromotionPackage:
    selection_payload: dict[str, Any]
    index_payload: dict[str, Any]
    receipt_payload: dict[str, Any]
    selection_bytes: bytes
    index_bytes: bytes
    receipt_bytes: bytes
    selection_name: str
    index_name: str


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and set(value) <= _HEX


def _canonical_hash(payload: Mapping[str, Any], excluded: str) -> str:
    document = {key: value for key, value in payload.items() if key != excluded}
    data = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _relative_json(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("repromotion input path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".json"
    ):
        raise ValueError("repromotion input must be canonical relative JSON")
    return value


def load_repromotion_contract(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid repromotion contract JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("repromotion contract must be an object")
    if payload.get(CONTRACT_HASH_FIELD) != _canonical_hash(
        payload, CONTRACT_HASH_FIELD
    ):
        raise ValueError("repromotion contract hash mismatch")
    _validate_contract(payload)
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        set(contract)
        != {
            "schema_version",
            "package_id",
            "base_code",
            "inputs",
            "target",
            "output_policy",
            "authorization",
            CONTRACT_HASH_FIELD,
        }
        or contract.get("schema_version") != CONTRACT_VERSION
    ):
        raise ValueError("repromotion contract schema is invalid")
    base = contract.get("base_code")
    if (
        not isinstance(base, Mapping)
        or not _is_hex(base.get("commit"), 40)
        or not _is_hex(base.get("tree"), 40)
        or not _is_hex(base.get("tag_object"), 40)
        or base.get("tag")
        != "ht-full-decay-data-loading-hardening-code-only-20260831-v1"
        or base.get("clean_tagged_descendant_required") is not True
    ):
        raise ValueError("repromotion base-code provenance is invalid")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(_INPUT_VERSIONS):
        raise ValueError("repromotion inputs are invalid")
    for name, (internal_field, _version) in _INPUT_VERSIONS.items():
        spec = inputs[name]
        if (
            not isinstance(spec, Mapping)
            or set(spec)
            != {"path", "file_sha256", "internal_hash_field", "internal_hash"}
            or spec.get("internal_hash_field") != internal_field
            or not _is_hex(spec.get("file_sha256"), 64)
            or not _is_hex(spec.get("internal_hash"), 64)
        ):
            raise ValueError(f"repromotion {name} binding is invalid")
        _relative_json(spec.get("path"))
    target = contract.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("selection_manifest_version") != SELECTION_MANIFEST_VERSION
        or target.get("selection_entry_keys") != sorted(SELECTION_ENTRY_KEYS)
        or target.get("selection_order")
        != ["split", "category", "task_id", "inventory_entry_hash"]
        or target.get("index_included_splits") != ["train", "validation"]
        or target.get("required_splits") != ["train", "validation"]
        or target.get("metadata_only_excluded_splits") != ["test"]
    ):
        raise ValueError("repromotion target is invalid")
    for field in (
        "selection_order_sha256",
        "roles_hash",
        "selection_manifest_hash",
        "selection_file_sha256",
        "selection_fingerprint",
        "dataset_index_hash",
        "dataset_index_file_sha256",
    ):
        if not _is_hex(target.get(field), 64):
            raise ValueError(f"repromotion target {field} is invalid")
    for field in ("selection_file_bytes", "dataset_index_file_bytes"):
        if not isinstance(target.get(field), int) or target[field] <= 0:
            raise ValueError(f"repromotion target {field} is invalid")
    if contract.get("output_policy") != {
        "claim": "sibling_dot_claim",
        "exact_root_must_be_absent": True,
        "claim_must_be_absent": True,
        "create_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
        "cas_filenames": True,
        "hidden_staging_files": True,
        "promotion_primitive": "renameat2_RENAME_NOREPLACE",
        "receipt_written_last": True,
        "live_pointer_updates_allowed": False,
        "overwrite_allowed": False,
        "retry_namespace_allowed": False,
    }:
        raise ValueError("repromotion output policy is invalid")
    authorization = contract.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or set(authorization) != _AUTHORIZATION_KEYS
        or any(value is not False for value in authorization.values())
    ):
        raise ValueError("repromotion authorization must remain false")


def _read_bound(root: Path, name: str, spec: Mapping[str, Any]) -> _BoundDocument:
    source = root / PurePosixPath(_relative_json(spec["path"]))
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{name} is not a unique regular metadata file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_METADATA_BYTES:
                raise ValueError("repromotion metadata exceeds size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec["file_sha256"]:
        raise ValueError(f"{name} metadata file hash mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} metadata JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get(spec["internal_hash_field"]) != spec["internal_hash"]
    ):
        raise ValueError(f"{name} metadata internal hash mismatch")
    return _BoundDocument(
        str(spec["path"]),
        payload,
        digest,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
    )


def _authenticate_inputs(
    contract: Mapping[str, Any], root: Path
) -> dict[str, _BoundDocument]:
    documents = {
        name: _read_bound(root, name, contract["inputs"][name])
        for name in _INPUT_VERSIONS
    }
    for name, (_field, version) in _INPUT_VERSIONS.items():
        if version is not None:
            _validate_hashed_manifest_payload(
                documents[name].payload,
                source=root / documents[name].path,
                expected_version=version,
            )
    _validate_dataset_index_metadata_payload(documents["legacy_index"].payload)
    return documents


def _unique(entries: object, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} entries are invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} entry is invalid")
        identity = entry.get("inventory_entry_hash")
        if not _is_hex(identity, 64) or identity in result:
            raise ValueError(f"{label} identity is invalid or duplicate")
        result[str(identity)] = entry
    return result


def _order_hash(entries: list[Mapping[str, Any]]) -> str:
    order = [
        [
            entry["split"],
            entry["category"],
            entry["task_id"],
            entry["inventory_entry_hash"],
        ]
        for entry in entries
    ]
    return hashlib.sha256(
        json.dumps(order, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _promote_selection(
    inventory: Mapping[str, Any],
    roles: Mapping[str, Any],
    legacy: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        legacy.get("inventory_hash") != inventory.get(HASH_FIELD)
        or roles.get("inventory_hash") != inventory.get(HASH_FIELD)
        or legacy.get("roles_hash") != roles.get(HASH_FIELD)
        or target.get("roles_hash") != roles.get(HASH_FIELD)
    ):
        raise ValueError("inventory and role bindings disagree")
    inventory_by_hash = _unique(inventory.get("entries"), "inventory")
    role_by_hash = _unique(roles.get("entries"), "role")
    legacy_entries = legacy.get("entries")
    if not isinstance(legacy_entries, list):
        raise ValueError("legacy selection entries are invalid")
    expected_roles = {
        "test": "test",
        "train": "training_pool",
        "validation": "validation",
    }
    output: list[dict[str, Any]] = []
    for position, entry in enumerate(legacy_entries):
        if not isinstance(entry, Mapping) or set(entry) != LEGACY_ENTRY_KEYS:
            raise ValueError(f"legacy entry {position} is not exact 15-key schema")
        identity = str(entry["inventory_entry_hash"])
        inventory_entry = inventory_by_hash.get(identity)
        role_entry = role_by_hash.get(identity)
        if inventory_entry is None or role_entry is None:
            raise ValueError("selection identity is absent from metadata")
        for field in LEGACY_ENTRY_KEYS - {"split"}:
            if inventory_entry.get(field) != entry.get(field):
                raise ValueError(f"legacy projection mismatch for {field}")
        if (
            role_entry.get("role") != expected_roles.get(str(entry["split"]))
            or role_entry.get("source_file") != entry.get("source_file")
            or role_entry.get("task_id") != entry.get("task_id")
        ):
            raise ValueError("selection role isolation mismatch")
        digest = inventory_entry.get("campaign_config_digest")
        if not _is_hex(digest, 64):
            raise ValueError("authenticated campaign digest is invalid")
        promoted = dict(entry)
        promoted["campaign_config_digest"] = digest
        output.append(promoted)
    selection = deepcopy(dict(legacy))
    selection["entries"] = output
    selection[HASH_FIELD] = canonical_manifest_hash(selection)
    validate_training_selection_metadata(selection)
    if (
        len(output) != target["selection_entry_count"]
        or _order_hash(output) != target["selection_order_sha256"]
    ):
        raise ValueError("target entry count or order mismatch")
    for field in (
        "split_counts",
        "split_shard_counts",
        "category_split_shard_counts",
        "training_category_shard_quotas",
        "roles_hash",
    ):
        if selection.get(field) != target[field]:
            raise ValueError(f"target {field} mismatch")
    if selection[HASH_FIELD] != target["selection_manifest_hash"]:
        raise ValueError("target selection hash mismatch")
    return selection


def _promote_index(
    legacy_index: Mapping[str, Any],
    legacy_selection: Mapping[str, Any],
    selection: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    index = deepcopy(dict(legacy_index))
    binding = index.get("selection_contract")
    if (
        not isinstance(binding, dict)
        or binding.get("selection_manifest_hash") != legacy_selection.get(HASH_FIELD)
        or binding.get("included_splits") != target["index_included_splits"]
    ):
        raise ValueError("legacy index selection binding is invalid")
    binding["selection_manifest_hash"] = selection[HASH_FIELD]
    binding["fingerprint"] = _selection_fingerprint_from_strings(
        index["paths"],
        mode=binding["mode"],
        max_events=binding["max_events"],
        selection_manifest_hash=selection[HASH_FIELD],
    )
    if binding["fingerprint"] != target["selection_fingerprint"]:
        raise ValueError("target fingerprint mismatch")
    index["index_hash"] = _index_hash(index)
    _validate_dataset_index_metadata_payload(index)
    _validate_training_selection_index_metadata_pure(
        selection,
        index,
        include_splits=target["index_included_splits"],
        required_splits=target["required_splits"],
    )
    if index["index_hash"] != target["dataset_index_hash"]:
        raise ValueError("target dataset-index hash mismatch")
    identity = index.get("event_identity_validation", {})
    if (
        index.get("event_count") != target["index_event_count"]
        or len(index.get("paths", ())) != target["index_path_count"]
        or len(index.get("shards", ())) != target["index_shard_count"]
        or identity.get("status") != target["event_identity_status"]
        or identity.get("sealed_test_opened") is not target["sealed_test_opened"]
    ):
        raise ValueError("target index count or sealed-role contract mismatch")
    return index


def _validate_provenance(
    value: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        set(value)
        != {
            "repository_head",
            "implementation_tag",
            "implementation_tag_object",
            "implementation_tag_type",
            "tracked_worktree_clean",
            "base_code_commit",
            "base_code_tag",
            "base_tag_is_ancestor",
        }
        or not _is_hex(value.get("repository_head"), 40)
        or not _is_hex(value.get("implementation_tag_object"), 40)
        or value.get("implementation_tag_type") != "tag"
        or not isinstance(value.get("implementation_tag"), str)
        or not value["implementation_tag"]
        or value.get("tracked_worktree_clean") is not True
        or value.get("base_tag_is_ancestor") is not True
        or value.get("base_code_commit") != contract["base_code"]["commit"]
        or value.get("base_code_tag") != contract["base_code"]["tag"]
    ):
        raise ValueError("clean tagged descendant provenance is required")
    return dict(value)


def build_repromotion_package(
    contract: Mapping[str, Any],
    repository_root: str | Path,
    *,
    implementation_provenance: Mapping[str, Any],
    output_namespace: str,
) -> RepromotionPackage:
    _validate_contract(contract)
    provenance = _validate_provenance(implementation_provenance, contract)
    documents = _authenticate_inputs(contract, Path(repository_root))
    target = contract["target"]
    selection = _promote_selection(
        documents["inventory"].payload,
        documents["roles"].payload,
        documents["legacy_selection"].payload,
        target,
    )
    index = _promote_index(
        documents["legacy_index"].payload,
        documents["legacy_selection"].payload,
        selection,
        target,
    )
    selection_bytes = _json_bytes(selection)
    index_bytes = _json_bytes(index)
    if (
        len(selection_bytes) != target["selection_file_bytes"]
        or hashlib.sha256(selection_bytes).hexdigest()
        != target["selection_file_sha256"]
    ):
        raise ValueError("target selection file hash mismatch")
    if (
        len(index_bytes) != target["dataset_index_file_bytes"]
        or hashlib.sha256(index_bytes).hexdigest()
        != target["dataset_index_file_sha256"]
    ):
        raise ValueError("target dataset-index file hash mismatch")
    selection_name = f"selection.{target['selection_file_sha256']}.json"
    index_name = f"index.{target['dataset_index_file_sha256']}.json"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "package_id": contract["package_id"],
        "contract_sha256": contract[CONTRACT_HASH_FIELD],
        "output_namespace": output_namespace,
        "implementation_provenance": provenance,
        "inputs": {
            name: {
                "path": document.path,
                "file_sha256": document.file_sha256,
                "internal_hash": document.payload[
                    contract["inputs"][name]["internal_hash_field"]
                ],
                "mode": oct(document.mode),
                "nlink": document.nlink,
            }
            for name, document in documents.items()
        },
        "outputs": {
            selection_name: {
                "file_sha256": target["selection_file_sha256"],
                "manifest_hash": selection[HASH_FIELD],
                "mode": "0o444",
                "size_bytes": target["selection_file_bytes"],
            },
            index_name: {
                "file_sha256": target["dataset_index_file_sha256"],
                "index_hash": index["index_hash"],
                "mode": "0o444",
                "size_bytes": target["dataset_index_file_bytes"],
            },
        },
        "gates": {
            "metadata_only": True,
            "raw_or_source_payloads_opened": False,
            "excluded_test_role_resolved_or_opened": False,
            "live_pointers_updated": False,
            "execution_authorized": False,
            "science_authorized": False,
            "submission_authorized": False,
            "slurm_actions_performed": False,
        },
    }
    receipt[RECEIPT_HASH_FIELD] = _canonical_hash(receipt, RECEIPT_HASH_FIELD)
    return RepromotionPackage(
        selection,
        index,
        receipt,
        selection_bytes,
        index_bytes,
        _json_bytes(receipt),
        selection_name,
        index_name,
    )


def _exclusive_write(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o444, dir_fd=directory_fd)
    try:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written <= 0:
                raise OSError("short repromotion-package write")
            position += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _verify_artifact(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_size != len(payload)
        ):
            raise ValueError("repromotion artifact metadata mismatch")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.digest() != hashlib.sha256(payload).digest():
            raise ValueError("repromotion artifact content mismatch")
    finally:
        os.close(descriptor)


def _rename_noreplace(directory_fd: int, source: str, target: str) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is required") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _stage_and_promote(directory_fd: int, name: str, payload: bytes) -> None:
    staging_name = f".{name}.staging"
    _exclusive_write(directory_fd, staging_name, payload)
    _verify_artifact(directory_fd, staging_name, payload)
    _rename_noreplace(directory_fd, staging_name, name)
    _verify_artifact(directory_fd, name, payload)


def publish_repromotion_package(
    package: RepromotionPackage, output_root: str | Path
) -> dict[str, Path]:
    root = Path(output_root)
    if not root.is_absolute() or root.name in {"", ".", ".."}:
        raise ValueError("output root must be absolute")
    if package.receipt_payload.get("output_namespace") != str(root):
        raise ValueError("receipt namespace mismatch")
    parent = root.parent
    claim_name = f".{root.name}.claim.json"
    if root.exists() or (parent / claim_name).exists():
        raise FileExistsError("output root or claim already exists")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, flags)
    try:
        claim = _json_bytes(
            {
                "schema_version": CLAIM_VERSION,
                "package_id": package.receipt_payload["package_id"],
                "contract_sha256": package.receipt_payload["contract_sha256"],
                "receipt_sha256": package.receipt_payload[RECEIPT_HASH_FIELD],
                "output_namespace": str(root),
            }
        )
        _exclusive_write(parent_fd, claim_name, claim)
        os.fsync(parent_fd)
        os.mkdir(root.name, mode=0o755, dir_fd=parent_fd)
        output_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            _stage_and_promote(
                output_fd, package.selection_name, package.selection_bytes
            )
            _stage_and_promote(output_fd, package.index_name, package.index_bytes)
            _exclusive_write(output_fd, OUTPUT_RECEIPT_NAME, package.receipt_bytes)
            os.fsync(output_fd)
            os.fchmod(output_fd, 0o555)
        finally:
            os.close(output_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "claim": parent / claim_name,
        "root": root,
        "selection": root / package.selection_name,
        "index": root / package.index_name,
        "receipt": root / OUTPUT_RECEIPT_NAME,
    }


def collect_git_provenance(
    repository_root: str | Path, *, implementation_tag: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repository_root)

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    head = run("rev-parse", "HEAD").stdout.strip()
    tag_commit = run("rev-parse", f"{implementation_tag}^{{commit}}").stdout.strip()
    base = contract["base_code"]
    if (
        run("rev-parse", base["tag"]).stdout.strip() != base["tag_object"]
        or run("cat-file", "-t", base["tag"]).stdout.strip() != "tag"
        or run("rev-parse", f"{base['tag']}^{{commit}}").stdout.strip()
        != base["commit"]
        or run("rev-parse", f"{base['commit']}^{{tree}}").stdout.strip() != base["tree"]
    ):
        raise ValueError("base code tag provenance mismatch")
    provenance = {
        "repository_head": head,
        "implementation_tag": implementation_tag,
        "implementation_tag_object": run(
            "rev-parse", implementation_tag
        ).stdout.strip(),
        "implementation_tag_type": run(
            "cat-file", "-t", implementation_tag
        ).stdout.strip(),
        "tracked_worktree_clean": not run(
            "status", "--porcelain", "--untracked-files=no"
        ).stdout.strip(),
        "base_code_commit": contract["base_code"]["commit"],
        "base_code_tag": contract["base_code"]["tag"],
        "base_tag_is_ancestor": run(
            "merge-base",
            "--is-ancestor",
            contract["base_code"]["commit"],
            head,
            check=False,
        ).returncode
        == 0,
    }
    if tag_commit != head:
        raise ValueError("implementation tag does not resolve to HEAD")
    return _validate_provenance(provenance, contract)
