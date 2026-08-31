"""One-use authorization boundary for inert selection-repromotion publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

from hypertagging.data.selection_repromotion import (
    CONTRACT_HASH_FIELD,
    OUTPUT_RECEIPT_NAME,
    RECEIPT_HASH_FIELD,
    build_repromotion_package,
    collect_git_provenance,
    load_repromotion_contract,
    publish_repromotion_package,
)

AUTHORIZATION_VERSION = "hypertagging-training-selection-publication-authorization-v2"
LOCK_VERSION = "hypertagging-training-selection-publication-lock-v1"
ACCEPTED_VERSION = "hypertagging-training-selection-publication-accepted-v1"
LOCAL_CONTROLLER_VERSION = "hypertagging-external-local-controller-authorization-v1"
TRUSTED_EXECUTION_UID = 12184
AUTHORITY_PARENT = Path(
    "/home/b/Boyang.Yu/.hypertagging-authority/"
    "training-selection-repromotion-publication-v2"
)
AUTHORIZATION_HASH_FIELD = "authorization_sha256"
LOCK_HASH_FIELD = "lock_sha256"
ACCEPTED_HASH_FIELD = "accepted_sha256"

AUTHORIZATION_FLAGS = {
    "consumer_pin_authorized": False,
    "live_pointer_activation_authorized": False,
    "metadata_package_publication_authorized": False,
    "raw_or_source_payload_access_authorized": False,
    "science_authorized": False,
    "slurm_actions_authorized": False,
    "submission_authorized": False,
}
_AUTHORIZED_FLAGS = {
    **AUTHORIZATION_FLAGS,
    "metadata_package_publication_authorized": True,
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "authorization_id",
    "created_at_utc",
    "expires_at_utc",
    "one_use",
    "retry_authorized",
    "trusted_execution",
    "external_local_controller_authorization",
    "repository_root",
    "implementation",
    "contract",
    "inputs",
    "namespace",
    "expected_outputs",
    "expected_package_receipt_sha256",
    "command",
    "tools",
    "authorization",
    AUTHORIZATION_HASH_FIELD,
}
_HEX = frozenset("0123456789abcdef")
_LOCAL_CONTROLLER_KEYS = {
    "schema_version",
    "controller_kind",
    "controller_host_id",
    "controller_principal",
    "authorization_event_id",
    "authorized_at_utc",
    "authorized_request_sha256",
    "decision",
}


@dataclass(frozen=True)
class CommandContext:
    cwd: str
    argv: tuple[str, ...]
    environment: dict[str, str]


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and set(value) <= _HEX


def _canonical_hash(payload: Mapping[str, Any], excluded: str) -> str:
    document = {key: value for key, value in payload.items() if key != excluded}
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"bound file is not a unique regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _trusted_execution(authority_parent: Path) -> dict[str, Any]:
    return {
        "uid": TRUSTED_EXECUTION_UID,
        "authority_parent": str(authority_parent),
        "authority_parent_mode": "0o700",
        "authorization_file_mode": "0o444",
        "authorization_file_must_be_direct_child": True,
    }


def _authority_parent_descriptor(authority_parent: Path) -> tuple[int, os.stat_result]:
    if not authority_parent.is_absolute():
        raise ValueError("authority parent must be absolute")
    try:
        canonical_parent = authority_parent.resolve(strict=True)
    except OSError as error:
        raise ValueError("authority parent is unavailable") from error
    if str(canonical_parent) != str(authority_parent):
        raise ValueError("authority parent must be an exact canonical path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(authority_parent, flags)
    except OSError as error:
        raise ValueError("authority parent is unavailable or follows a symlink") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise ValueError("authority parent must be owned mode-0700 directory")
    try:
        named = os.stat(authority_parent, follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        raise
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        os.close(descriptor)
        raise ValueError("authority parent identity mismatch")
    return descriptor, metadata


def _read_authorization(
    path: Path, *, authority_parent: Path
) -> tuple[dict[str, Any], str]:
    if (
        not path.is_absolute()
        or path.parent != authority_parent
        or path.name in {"", ".", ".."}
    ):
        raise ValueError("authorization must be a direct authority-parent child")
    parent_descriptor, parent_metadata = _authority_parent_descriptor(authority_parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_uid != os.geteuid()
        ):
            raise ValueError(
                "authorization must be an owned unique mode-0444 regular file"
            )
        raw = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 1024 * 1024:
                raise ValueError("authorization is too large")
        final_metadata = os.fstat(descriptor)
        final_parent_metadata = os.fstat(parent_descriptor)
        named_authorization = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        named_parent = os.stat(authority_parent, follow_symlinks=False)
        authorization_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_nlink,
            metadata.st_size,
        )
        if (
            not stat.S_ISREG(final_metadata.st_mode)
            or not stat.S_ISREG(named_authorization.st_mode)
            or final_metadata.st_uid != os.geteuid()
            or named_authorization.st_uid != os.geteuid()
            or stat.S_IMODE(final_metadata.st_mode) != 0o444
            or stat.S_IMODE(named_authorization.st_mode) != 0o444
            or final_metadata.st_nlink != 1
            or named_authorization.st_nlink != 1
            or authorization_identity
            != (
                final_metadata.st_dev,
                final_metadata.st_ino,
                final_metadata.st_uid,
                stat.S_IMODE(final_metadata.st_mode),
                final_metadata.st_nlink,
                final_metadata.st_size,
            )
            or authorization_identity
            != (
                named_authorization.st_dev,
                named_authorization.st_ino,
                named_authorization.st_uid,
                stat.S_IMODE(named_authorization.st_mode),
                named_authorization.st_nlink,
                named_authorization.st_size,
            )
        ):
            raise ValueError("authorization metadata changed while reading")
        parent_identity = (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
            parent_metadata.st_uid,
            stat.S_IMODE(parent_metadata.st_mode),
        )
        if (
            not stat.S_ISDIR(final_parent_metadata.st_mode)
            or not stat.S_ISDIR(named_parent.st_mode)
            or final_parent_metadata.st_uid != os.geteuid()
            or named_parent.st_uid != os.geteuid()
            or stat.S_IMODE(final_parent_metadata.st_mode) != 0o700
            or stat.S_IMODE(named_parent.st_mode) != 0o700
            or parent_identity
            != (
                final_parent_metadata.st_dev,
                final_parent_metadata.st_ino,
                final_parent_metadata.st_uid,
                stat.S_IMODE(final_parent_metadata.st_mode),
            )
            or parent_identity
            != (
                named_parent.st_dev,
                named_parent.st_ino,
                named_parent.st_uid,
                stat.S_IMODE(named_parent.st_mode),
            )
        ):
            raise ValueError("authority parent metadata changed while reading")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("authorization JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("authorization must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("authorization timestamps must use UTC Z form")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("authorization timestamp must be UTC")
    return parsed


def _validate_authorization(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    context: CommandContext,
    authority_parent: Path,
) -> None:
    if (
        set(payload) != _TOP_LEVEL_KEYS
        or payload.get("schema_version") != AUTHORIZATION_VERSION
    ):
        raise ValueError("publication authorization schema is invalid")
    if payload.get(AUTHORIZATION_HASH_FIELD) != _canonical_hash(
        payload, AUTHORIZATION_HASH_FIELD
    ):
        raise ValueError("publication authorization hash mismatch")
    if (
        not isinstance(payload.get("authorization_id"), str)
        or not payload["authorization_id"]
        or payload.get("one_use") is not True
        or payload.get("retry_authorized") is not False
        or payload.get("authorization") != _AUTHORIZED_FLAGS
        or payload.get("trusted_execution") != _trusted_execution(authority_parent)
    ):
        raise ValueError("publication authorization capability is invalid")
    created = _parse_utc(payload.get("created_at_utc"))
    expires = _parse_utc(payload.get("expires_at_utc"))
    if now.tzinfo is None:
        raise ValueError("authorization comparison time must be aware")
    now = now.astimezone(timezone.utc)
    if (
        not created <= now <= expires
        or not created < expires
        or expires - created > timedelta(hours=1)
    ):
        raise ValueError("publication authorization is not currently valid")
    controller = payload.get("external_local_controller_authorization")
    if not isinstance(controller, Mapping) or set(controller) != _LOCAL_CONTROLLER_KEYS:
        raise ValueError("external local-controller authorization is invalid")
    controller_time = _parse_utc(controller.get("authorized_at_utc"))
    if (
        controller.get("schema_version") != LOCAL_CONTROLLER_VERSION
        or controller.get("controller_kind") != "codex-desktop-local-controller"
        or controller.get("decision")
        != "authorize-one-metadata-package-publication"
        or not all(
            isinstance(controller.get(key), str) and bool(controller[key])
            for key in (
                "controller_host_id",
                "controller_principal",
                "authorization_event_id",
            )
        )
        or not _is_hex(controller.get("authorized_request_sha256"), 64)
        or not controller_time <= created
        or created - controller_time > timedelta(minutes=15)
    ):
        raise ValueError("external local-controller authorization facts mismatch")
    command = payload.get("command")
    if command != {
        "cwd": context.cwd,
        "argv": list(context.argv),
        "environment": context.environment,
    }:
        raise ValueError("publication command context mismatch")


def _validate_tool_bindings(tools: object, context: CommandContext) -> None:
    if not isinstance(tools, list) or not tools:
        raise ValueError("publication tool bindings are invalid")
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping) or set(tool) != {
            "name",
            "path",
            "canonical_path",
            "sha256",
        }:
            raise ValueError("publication tool binding is invalid")
        name = tool.get("name")
        path = Path(str(tool.get("path")))
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not path.is_absolute()
            or str(path.resolve(strict=True)) != tool.get("canonical_path")
            or _sha256_file(path) != tool.get("sha256")
        ):
            raise ValueError("publication tool binding mismatch")
        names.add(name)
    if names != {"env", "git", "python"}:
        raise ValueError("publication tool closure must be exact")
    by_name = {str(tool["name"]): tool for tool in tools}
    if (
        by_name["env"]["path"] != "/usr/bin/env"
        or by_name["git"]["path"] != "/usr/bin/git"
        or context.environment.get("PATH") != "/usr/bin:/bin"
        or not context.argv
        or str(Path(context.argv[0]).resolve(strict=True))
        != by_name["python"]["canonical_path"]
    ):
        raise ValueError("publication executable closure mismatch")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _namespace_paths(root: Path) -> dict[str, Path]:
    parent = root.parent
    return {
        "output_root": root,
        "claim": parent / f".{root.name}.claim.json",
        "authorization_lock": parent / f".{root.name}.authorization.lock.json",
        "accepted_receipt": parent / f".{root.name}.accepted.json",
    }


def _validate_static_bindings(
    authorization: Mapping[str, Any], repository_root: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    if (
        not repository_root.is_absolute()
        or str(repository_root) != authorization.get("repository_root")
        or not contract_path.is_absolute()
    ):
        raise ValueError("publication repository or contract path mismatch")
    contract_spec = authorization.get("contract")
    if not isinstance(contract_spec, Mapping) or set(contract_spec) != {
        "path",
        "file_sha256",
        "internal_hash",
    }:
        raise ValueError("publication contract binding is invalid")
    if str(contract_path) != contract_spec.get("path") or _sha256_file(
        contract_path
    ) != contract_spec.get("file_sha256"):
        raise ValueError("publication contract file binding mismatch")
    contract = load_repromotion_contract(contract_path)
    if contract[CONTRACT_HASH_FIELD] != contract_spec.get("internal_hash"):
        raise ValueError("publication contract internal hash mismatch")
    if authorization.get("inputs") != contract["inputs"]:
        raise ValueError("publication input bindings mismatch")
    namespace = authorization.get("namespace")
    if not isinstance(namespace, Mapping) or set(namespace) != {
        "output_root",
        "claim",
        "authorization_lock",
        "accepted_receipt",
        "all_must_be_absent",
    }:
        raise ValueError("publication namespace binding is invalid")
    root = Path(str(namespace.get("output_root")))
    if not root.is_absolute() or namespace.get("all_must_be_absent") is not True:
        raise ValueError("publication namespace must be fresh and absolute")
    paths = _namespace_paths(root)
    if any(str(paths[key]) != namespace.get(key) for key in paths):
        raise ValueError("publication namespace path mismatch")
    return contract, paths


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", *args],
        cwd=repository_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _validate_implementation(
    authorization: Mapping[str, Any], repository_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    implementation = authorization.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "commit",
        "tree",
        "tag",
        "tag_object",
    }:
        raise ValueError("publication implementation binding is invalid")
    tag = str(implementation.get("tag"))
    provenance = collect_git_provenance(
        repository_root, implementation_tag=tag, contract=contract
    )
    if (
        provenance["repository_head"] != implementation.get("commit")
        or provenance["implementation_tag_object"] != implementation.get("tag_object")
        or _git(repository_root, "rev-parse", "HEAD^{tree}")
        != implementation.get("tree")
    ):
        raise ValueError("publication implementation provenance mismatch")
    return provenance


def _expected_outputs(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    target = contract["target"]
    selection_name = f"selection.{target['selection_file_sha256']}.json"
    index_name = f"index.{target['dataset_index_file_sha256']}.json"
    return {
        "root_mode": "0o555",
        "selection": {
            "path": str(root / selection_name),
            "file_sha256": target["selection_file_sha256"],
            "internal_hash": target["selection_manifest_hash"],
            "size_bytes": target["selection_file_bytes"],
            "mode": "0o444",
            "nlink": 1,
        },
        "index": {
            "path": str(root / index_name),
            "file_sha256": target["dataset_index_file_sha256"],
            "internal_hash": target["dataset_index_hash"],
            "size_bytes": target["dataset_index_file_bytes"],
            "mode": "0o444",
            "nlink": 1,
        },
        "package_receipt": {
            "path": str(root / OUTPUT_RECEIPT_NAME),
            "mode": "0o444",
            "nlink": 1,
        },
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written <= 0:
                raise OSError("short publication-boundary write")
            position += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _verify_output(path: Path, expected: Mapping[str, Any]) -> None:
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != expected["nlink"]
        or oct(stat.S_IMODE(metadata.st_mode)) != expected["mode"]
        or ("size_bytes" in expected and metadata.st_size != expected["size_bytes"])
        or ("file_sha256" in expected and _sha256_file(path) != expected["file_sha256"])
    ):
        raise ValueError("published artifact verification failed")


def publish_authorized_once(
    authorization_path: str | Path,
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    command_context: CommandContext,
    authority_parent: str | Path = AUTHORITY_PARENT,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Consume one exact capability and publish an inert package once."""

    auth_path = Path(authorization_path)
    if os.geteuid() != TRUSTED_EXECUTION_UID:
        raise ValueError("publication trusted execution uid mismatch")
    authority_root = Path(authority_parent)
    authorization, authorization_file_sha256 = _read_authorization(
        auth_path, authority_parent=authority_root
    )
    _validate_authorization(
        authorization,
        now=now or datetime.now(timezone.utc),
        context=command_context,
        authority_parent=authority_root,
    )
    _validate_tool_bindings(authorization["tools"], command_context)
    root = Path(repository_root)
    contract, paths = _validate_static_bindings(
        authorization, root, Path(contract_path)
    )
    if command_context.environment != {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{root / 'src'}:{root}",
        "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
    }:
        raise ValueError("publication environment closure mismatch")
    provenance = _validate_implementation(authorization, root, contract)
    expected_outputs = _expected_outputs(contract, paths["output_root"])
    if authorization.get("expected_outputs") != expected_outputs:
        raise ValueError("publication output bindings mismatch")
    package = build_repromotion_package(
        contract,
        root,
        implementation_provenance=provenance,
        output_namespace=str(paths["output_root"]),
    )
    package_receipt_sha256 = hashlib.sha256(package.receipt_bytes).hexdigest()
    if package_receipt_sha256 != authorization.get("expected_package_receipt_sha256"):
        raise ValueError("expected package receipt hash mismatch")
    if any(_lexists(path) for path in paths.values()):
        raise FileExistsError("publication namespace or one-use state already exists")
    lock: dict[str, Any] = {
        "schema_version": LOCK_VERSION,
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_sha256": authorization[AUTHORIZATION_HASH_FIELD],
        "output_root": str(paths["output_root"]),
        "retry_authorized": False,
    }
    lock[LOCK_HASH_FIELD] = _canonical_hash(lock, LOCK_HASH_FIELD)
    lock_bytes = _json_bytes(lock)
    _exclusive_write(paths["authorization_lock"], lock_bytes)
    published = publish_repromotion_package(package, paths["output_root"])
    for key in ("selection", "index"):
        _verify_output(published[key], expected_outputs[key])
    _verify_output(published["receipt"], expected_outputs["package_receipt"])
    if _sha256_file(published["receipt"]) != package_receipt_sha256:
        raise ValueError("published package receipt hash mismatch")
    if stat.S_IMODE(paths["output_root"].stat().st_mode) != 0o555:
        raise ValueError("published package root mode mismatch")
    accepted: dict[str, Any] = {
        "schema_version": ACCEPTED_VERSION,
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_sha256": authorization[AUTHORIZATION_HASH_FIELD],
        "lock": {
            "path": str(paths["authorization_lock"]),
            "file_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "internal_hash": lock[LOCK_HASH_FIELD],
        },
        "package": {
            "output_root": str(paths["output_root"]),
            "package_receipt_path": str(published["receipt"]),
            "package_receipt_file_sha256": package_receipt_sha256,
            "package_receipt_internal_hash": package.receipt_payload[
                RECEIPT_HASH_FIELD
            ],
            "outputs": expected_outputs,
        },
        "gates": {
            "consumer_pin_authorized": False,
            "live_pointer_activation_authorized": False,
            "metadata_package_publication_accepted": True,
            "raw_or_source_payload_accessed": False,
            "science_authorized": False,
            "slurm_actions_performed": False,
            "submission_authorized": False,
        },
    }
    accepted[ACCEPTED_HASH_FIELD] = _canonical_hash(accepted, ACCEPTED_HASH_FIELD)
    _exclusive_write(paths["accepted_receipt"], _json_bytes(accepted))
    return {**published, **paths, "accepted_receipt": paths["accepted_receipt"]}


__all__ = [
    "ACCEPTED_HASH_FIELD",
    "ACCEPTED_VERSION",
    "AUTHORITY_PARENT",
    "AUTHORIZATION_FLAGS",
    "AUTHORIZATION_HASH_FIELD",
    "AUTHORIZATION_VERSION",
    "CommandContext",
    "LOCAL_CONTROLLER_VERSION",
    "TRUSTED_EXECUTION_UID",
    "publish_authorized_once",
]
