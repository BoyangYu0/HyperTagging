"""Move exact historical rows to bounded lossless downloads, never discard them."""

import gzip
import io
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase42_44_delivery_20261009"
PHASES = ("phase42", "phase43", "phase44")
FILES = {f"{p}-metric-rows.json" for p in PHASES} | {
    "phase42-44-row-integrity.json",
    "phase42-44-row-decoder.txt",
    "phase41-61-status-records.json",
    "phase41-61-status-decoder.txt",
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


def compact_status_records(output, manifest):
    """Archive complete historical status records after rendering the full page.

    Existing metric downloads and row archives remain byte-preserved. Each
    pointer identifies a complete original record in one bounded envelope.
    """
    spec = importlib.util.spec_from_file_location(
        "historical_status_compact", Path(__file__).with_name("wiki_phase72_compact.py")
    )
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    reconstruction = manifest["reconstruction"]
    records = {f"phase{number}": reconstruction[f"phase{number}"]
               for number in range(41, 62) if f"phase{number}" in reconstruction}
    if not records:
        return
    data = canonical(records)
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=9, mtime=0) as zipped:
        zipped.write(data)
    encoded = codec.encode(data, stream.getvalue(), 5_000_000)
    if codec.decode(json.loads(encoded), 5_000_000) != data:
        raise ValueError("Historical status delivery mismatch")
    filename = "phase41-61-status-records.json"
    output.mkdir(parents=True, exist_ok=True)
    (output / filename).write_bytes(encoded)
    for phase, record in records.items():
        reconstruction[phase] = {
            "status": record.get("status"),
            "full_record_download": {
                "filename": filename,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
                "decoded_bytes": len(data),
                "decoded_sha256": hashlib.sha256(data).hexdigest(),
                "decoded_key_path": [phase],
                "record_sha256": hashlib.sha256(canonical(record)).hexdigest(),
                "decoder": "phase41-61-status-decoder.txt",
                "full_record_in_download": True,
            },
        }
    (output / "phase41-61-status-decoder.txt").write_text(STATUS_DECODER)


STATUS_DECODER = r'''# Python 3 standard library: restore the complete historical status records.
# Usage: python phase41-61-status-decoder.txt
# Run beside status.json and phase41-61-status-records.json.
import base64, hashlib, json, pathlib, zlib
p = pathlib.Path("phase41-61-status-records.json")
assert not p.is_symlink() and p.stat().st_size <= 10485760
raw = p.read_bytes()
e = json.loads(raw)
assert set(e) == {"encoding", "decoded_bytes", "decoded_sha256", "data"}
assert e["encoding"] == "bounded-gzip-base32-json-v1"
n = e["decoded_bytes"]
assert type(n) is int and 0 <= n <= 5000000
assert isinstance(e["data"], str) and len(e["data"]) <= 10485760
d = zlib.decompressobj(31)
b = d.decompress(base64.b32decode(e["data"], casefold=False), n + 1)
assert len(b) == n and d.eof and not d.unconsumed_tail and not d.unused_data
assert hashlib.sha256(b).hexdigest() == e["decoded_sha256"]
records = json.loads(b)
status = json.loads(pathlib.Path("status.json").read_bytes())
for phase, record in records.items():
    assert phase in {"phase" + str(i) for i in range(41, 62)}
    ref = status["reconstruction"][phase]["full_record_download"]
    assert ref["filename"] == p.name and ref["decoded_key_path"] == [phase]
    assert ref["sha256"] == hashlib.sha256(raw).hexdigest() and ref["bytes"] == len(raw)
    assert ref["decoded_sha256"] == e["decoded_sha256"] and ref["decoded_bytes"] == n
    canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    assert hashlib.sha256(canonical).hexdigest() == ref["record_sha256"]
    status["reconstruction"][phase] = record
pathlib.Path("phase41-61-status-records-decoded.json").write_bytes(b)
pathlib.Path("status-with-historical-records.json").write_text(
    json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
'''
