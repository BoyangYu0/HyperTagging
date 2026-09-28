#!/usr/bin/env python3
"""Verify or restore explicitly archived repository outputs; never submit jobs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def beneath(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes root: {relative}")
    return path


def checked(payload: bytes, record: dict) -> bytes:
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Artifact integrity mismatch: {record['path']}")
    return payload


def read_payload(record: dict, repository: Path, data_root: Path | None) -> bytes:
    if data_root is not None:
        # An explicitly requested data archive must be present and verified.
        return checked(beneath(data_root, record["data_path"]).read_bytes(), record)
    if "packed_path" not in record:
        raise ValueError(f"External data root required: {record['path']}")
    packed = beneath(repository, record["packed_path"]).read_bytes()
    if len(packed) != record["packed_bytes"] or hashlib.sha256(packed).hexdigest() != record["packed_sha256"]:
        raise ValueError(f"Packed artifact integrity mismatch: {record['path']}")
    return checked(gzip.decompress(packed), record)


def restore(record: dict, payload: bytes, output: Path) -> None:
    destination = beneath(output, record["path"])
    if destination.exists():
        checked(destination.read_bytes(), record)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete bytes atomically, without replacing an existing file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(checked(payload, record))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "restore"))
    parser.add_argument("--data-root", type=Path, default=os.environ.get("HYPERTAGGING_DATA_ROOT"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--path", action="append", help="Exact manifest path; repeat to select outputs")
    args = parser.parse_args()
    if args.action == "restore" and args.output_dir is None:
        parser.error("restore requires an explicit --output-dir")
    manifest = json.loads((ROOT / "configs/repository_artifacts.json").read_text())
    entries = manifest["entries"]
    if args.path:
        unknown = set(args.path) - {entry["path"] for entry in entries}
        if unknown:
            parser.error(f"Unknown artifact paths: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["path"] in args.path]
    elif args.data_root is None:
        entries = [entry for entry in entries if "packed_path" in entry]
    # Validate the entire requested set before writing any destination.
    payloads = [(entry, read_payload(entry, ROOT, args.data_root)) for entry in entries]
    if args.action == "restore":
        for entry, payload in payloads:
            restore(entry, payload, args.output_dir)
    print(json.dumps({"action": args.action, "verified_files": len(payloads),
                      "verified_bytes": sum(len(payload) for _, payload in payloads),
                      "source": "external_archive" if args.data_root else "portable_registries"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
