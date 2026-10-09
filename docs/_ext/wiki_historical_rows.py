"""Move exact historical rows to bounded lossless downloads, never discard them."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase42_44_delivery_20261009"
PHASES = ("phase42", "phase43", "phase44")
FILES = {f"{p}-metric-rows.json" for p in PHASES} | {
    "phase42-44-row-integrity.json",
    "phase42-44-row-decoder.txt",
}


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def compact(root, output, manifest):
    spec = importlib.util.spec_from_file_location(
        "historical_compact", Path(__file__).with_name("wiki_phase72_compact.py")
    )
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    binding = json.loads((root / SOURCE / "binding.json").read_text())
    if binding["version"] != "phase42-44-lossless-row-binding-v1" or [
        r["phase"] for r in binding["files"]
    ] != list(PHASES):
        raise ValueError("Wrong historical row archive")
    receipts = []
    for record in binding["files"]:
        phase = record["phase"]
        path = root / SOURCE / record["file"]
        if (
            path.name != f"{phase}-metric-rows.json.gz"
            or path.is_symlink()
            or path.stat().st_size > 5_000_000
        ):
            raise ValueError("Unsafe historical archive")
        packed = path.read_bytes()
        if hashlib.sha256(packed).hexdigest() != record["compressed_sha256"]:
            raise ValueError("Historical compressed hash mismatch")
        with gzip.open(path, "rb") as stream:
            data = stream.read(5_000_001)
        if (
            len(data) > 5_000_000
            or len(data) != record["decoded_bytes"]
            or hashlib.sha256(data).hexdigest() != record["decoded_sha256"]
        ):
            raise ValueError("Historical decoded hash mismatch")
        current = manifest["reconstruction"][phase]["metric_rows"]
        if len(current) != record["rows"] or canonical(current) != data:
            raise ValueError("Historical rows changed; archive update required")
        encoded = codec.encode(data, packed, 5_000_000)
        filename = f"{phase}-metric-rows.json"
        (output / filename).write_bytes(encoded)
        receipt = {
            "phase": phase,
            "filename": filename,
            "rows": len(current),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "decoded_sha256": record["decoded_sha256"],
            "decoded_bytes": len(data),
        }
        receipts.append(receipt)
        target = manifest["reconstruction"][phase]
        target.pop("metric_rows")
        target["metric_rows_download"] = receipt
    (output / "phase42-44-row-integrity.json").write_bytes(
        canonical(
            {
                "version": "phase42-44-lossless-row-integrity-v1",
                "files": receipts,
                "all_original_rows_preserved": True,
                "publication_limits_unchanged": True,
            }
        )
    )
    decoder = """# Python 3 standard-library decoder; reconstructs every historical metric row.
import base64,hashlib,json,pathlib,zlib
for phase in ("phase42","phase43","phase44"):
    e=json.loads(pathlib.Path(phase+"-metric-rows.json").read_text())
    assert set(e)=={"encoding","decoded_bytes","decoded_sha256","data"}
    assert e["encoding"]=="bounded-gzip-base32-json-v1"
    n=e["decoded_bytes"]
    assert isinstance(n,int) and 0<=n<=5000000 and len(e["data"])<=10485760
    d=zlib.decompressobj(31)
    b=d.decompress(base64.b32decode(e["data"],casefold=False),n+1)
    assert len(b)==n and d.eof and not d.unconsumed_tail and not d.unused_data
    assert hashlib.sha256(b).hexdigest()==e["decoded_sha256"]
    pathlib.Path(phase+"-metric-rows-decoded.json").write_bytes(b)
"""
    (output / "phase42-44-row-decoder.txt").write_text(decoder)
    return receipts
