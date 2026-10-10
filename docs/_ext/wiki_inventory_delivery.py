"""Lossless bounded transport for the current generated API coverage inventory."""
import gzip
import io
import json
from pathlib import Path

from wiki_phase72_compact import canonical, decode, encode

LIMIT = 5_000_000


def write_inventory_delivery(output: Path, inventory: dict) -> dict:
    """Keep every inventory field and verify the decoded canonical bytes."""
    data = canonical(inventory)
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=9, mtime=0) as zipped:
        zipped.write(data)
    encoded = encode(data, stream.getvalue(), LIMIT)
    if decode(json.loads(encoded), LIMIT) != data:
        raise ValueError('API inventory lossless delivery mismatch')
    (output / 'inventory-download.json').write_bytes(encoded)
    (output / 'inventory-decoder.txt').write_text(DECODER)
    return {'decoded_bytes': len(data), 'encoded_bytes': len(encoded)}


DECODER = '''# Python 3 standard library: reconstruct the complete API coverage inventory.
import base64, hashlib, json, pathlib, zlib
p = pathlib.Path("inventory-download.json")
assert not p.is_symlink() and p.stat().st_size <= 10485760
e = json.loads(p.read_bytes())
assert set(e) == {"encoding", "decoded_bytes", "decoded_sha256", "data"}
assert e["encoding"] == "bounded-gzip-base32-json-v1"
assert type(e["decoded_bytes"]) is int and 0 <= e["decoded_bytes"] <= 5000000
assert isinstance(e["data"], str) and len(e["data"]) <= 10485760
d = zlib.decompressobj(31)
raw = d.decompress(base64.b32decode(e["data"], casefold=False), e["decoded_bytes"] + 1)
assert d.eof and not d.unused_data and not d.unconsumed_tail
assert len(raw) == e["decoded_bytes"]
assert hashlib.sha256(raw).hexdigest() == e["decoded_sha256"]
json.loads(raw)
pathlib.Path("inventory-decoded.json").write_bytes(raw)
'''
