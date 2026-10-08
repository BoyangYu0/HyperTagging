"""Bounded textual envelopes of authenticated, privacy-checked JSON aggregates.

Reuse immutable gzip bytes instead of recompressing during documentation builds,
so Python/zlib versions cannot change the published files. No raw binary download
or publication limit exception is introduced.
"""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

MAX_BYTES = 10 * 1024 * 1024
ENCODING = 'bounded-gzip-base64-json-v1'


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def decode(envelope, limit=MAX_BYTES):
    if set(envelope)!={'encoding','decoded_bytes','decoded_sha256','data'} or envelope['encoding']!=ENCODING:
        raise ValueError('Unknown compact envelope')
    size=envelope['decoded_bytes']
    if type(size) is not int or not 0<=size<=limit or not isinstance(envelope['data'],str) or len(envelope['data'])>MAX_BYTES:
        raise ValueError('Compact byte limit')
    compressed=base64.b64decode(envelope['data'],validate=True)
    decoder=zlib.decompressobj(31)
    data=decoder.decompress(compressed,size+1)
    if len(data)!=size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('Compact expansion or stream mismatch')
    if hashlib.sha256(data).hexdigest()!=envelope['decoded_sha256']:
        raise ValueError('Compact decoded digest mismatch')
    return data


def encode(data, compressed, limit=MAX_BYTES):
    if len(data)>limit or len(compressed)>MAX_BYTES:
        raise ValueError('Compact byte limit')
    spec=importlib.util.spec_from_file_location('compact_privacy',Path(__file__).with_name('wiki_privacy.py'))
    privacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(privacy)
    value=privacy._strict_json(data.decode())
    if privacy._contains_private_fields(value) or privacy.redact(data.decode())!=data.decode():
        raise ValueError('Private decoded aggregate')
    envelope={'encoding':ENCODING,'decoded_bytes':len(data),'decoded_sha256':hashlib.sha256(data).hexdigest(),
              'data':base64.b64encode(compressed).decode('ascii')}
    if decode(envelope,limit)!=data:
        raise ValueError('Compact roundtrip mismatch')
    result=canonical(envelope)
    if len(result)>MAX_BYTES:raise ValueError('Compact byte limit')
    return result
