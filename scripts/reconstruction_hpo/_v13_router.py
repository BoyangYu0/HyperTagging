"""Private v13 production router and unforgeable capability issuer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any, Iterable, Mapping

from . import _v13_core as _core

_NODE_SCHEMAS = (
    "TerminalNode.v5", "PairNode.v5", "LocatorNode.v5", "PilotNode.v5",
    "LadderNode.v5", "PointerNode.v5", "HpoNode.v5", "FinalNode.v5",
)
_ISSUER_SECRET = object()
_LIVE: dict[str, tuple[int, int, str]] = {}
_MAX_AGE_NS = 300_000_000_000


def _keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    wanted = tuple(expected)
    if not isinstance(value, Mapping) or len(value) != len(wanted) or set(value) != set(wanted):
        raise _core.VerificationError(f"{label} keys differ from v13 contract")


@dataclass(frozen=True, slots=True)
class _VerifiedChain:
    _nonce: str
    _pid: int
    _issued_ns: int
    _chain_digest: str
    _issuer: object


def _stable_bytes(ref: Mapping[str, Any], roots: Iterable[str]) -> bytes:
    _keys(ref, ("path", "byte_length", "sha256", "media_type", "schema", "version"), "ArtifactRef.v5")
    path = Path(ref["path"])
    if not path.is_absolute() or path != Path(os.path.normpath(str(path))):
        raise _core.VerificationError("ArtifactRef path is not normalized absolute")
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    if not resolved_roots or not any(path.is_relative_to(root) for root in resolved_roots):
        raise _core.VerificationError("ArtifactRef path is outside trusted roots")
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise _core.VerificationError("ArtifactRef target is not regular/nonsymlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise _core.VerificationError("ArtifactRef identity changed before read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
        raise _core.VerificationError("ArtifactRef identity changed during read")
    raw = b"".join(chunks)
    if len(raw) != ref["byte_length"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise _core.VerificationError("ArtifactRef length/hash mismatch")
    return raw


def _digest_without(value: Mapping[str, Any], omitted: str = "digest") -> str:
    return hashlib.sha256(_core.jcs({k: v for k, v in value.items() if k != omitted})).hexdigest()


def _false_authorization(value: Any) -> None:
    if not isinstance(value, Mapping) or not value or any(v is not False for v in value.values()):
        raise _core.VerificationError("node authorization is not exact false")


def _walk_refs(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if tuple(value.keys()) == ("path", "byte_length", "sha256", "media_type", "schema", "version"):
            yield value
            return
        for child in value.values():
            yield from _walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_refs(child)


def _reopen_leaf(ref: Mapping[str, Any], roots: Iterable[str], spec: Mapping[str, Any]) -> None:
    raw = _stable_bytes(ref, roots)
    media = str(ref["media_type"])
    if media.endswith("json") or "json" in media:
        value = _core.parse_json_bytes(raw)
        if not isinstance(value, Mapping):
            raise _core.VerificationError("JSON evidence leaf is not an object")
        schema_name = ref["schema"]
        if schema_name in spec["schemas"]:
            schema = spec["schemas"][schema_name]
            keys = schema.get("exact_keys")
            if keys is not None:
                _keys(value, keys, schema_name)
            if "authorization" in value:
                _false_authorization(value["authorization"])
            if "digest" in value and value["digest"] != _digest_without(value):
                raise _core.VerificationError(f"{schema_name} digest mismatch")


def verify_nodes(repo: str, refs: tuple[Mapping[str, Any], ...], trusted_roots: Iterable[str]) -> _VerifiedChain:
    if len(refs) != 8:
        raise TypeError("exactly eight node ArtifactRefs are required")
    spec = _core.load_spec(repo)
    previous: str | None = None
    node_digests: list[str] = []
    for index, (ref, schema_name) in enumerate(zip(refs, _NODE_SCHEMAS)):
        raw = _stable_bytes(ref, trusted_roots)
        node = _core.parse_json_bytes(raw)
        if not isinstance(node, Mapping):
            raise _core.VerificationError("node is not an object")
        schema = spec["schemas"][schema_name]
        _keys(node, schema["exact_keys"], schema_name)
        for key, expected in schema.get("required", {}).items():
            if node.get(key) != expected:
                raise _core.VerificationError(f"{schema_name}.{key} mismatch")
        _false_authorization(node["authorization"])
        digest = _digest_without(node)
        if node["digest"] != digest:
            raise _core.VerificationError(f"{schema_name} digest mismatch")
        if index and node["previous_digest"] != previous:
            raise _core.VerificationError(f"{schema_name} previous_digest mismatch")
        for leaf_ref in _walk_refs(node):
            if leaf_ref is ref:
                continue
            _reopen_leaf(leaf_ref, trusted_roots, spec)
        previous = digest
        node_digests.append(digest)
    chain_digest = hashlib.sha256("".join(node_digests).encode("ascii")).hexdigest()
    nonce = secrets.token_hex(32)
    issued = time.monotonic_ns()
    pid = os.getpid()
    _LIVE[nonce] = (pid, issued + _MAX_AGE_NS, chain_digest)
    return _VerifiedChain(nonce, pid, issued, chain_digest, _ISSUER_SECRET)


def decide(capability: object) -> str:
    if type(capability) is not _VerifiedChain or capability._issuer is not _ISSUER_SECRET:
        raise _core.VerificationError("decision requires a live issued VerifiedChain")
    live = _LIVE.pop(capability._nonce, None)
    now = time.monotonic_ns()
    if live is None or capability._pid != os.getpid() or live[0] != os.getpid() or now > live[1] or live[2] != capability._chain_digest:
        raise _core.VerificationError("VerifiedChain is stale, consumed, forged, or cross-process")
    return "E_FEASIBILITY_SHAPE_AUTHORITY"


__all__: list[str] = []
