"""Lossless envelopes retain decoded privacy checks and fixed resource bounds."""
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compact',ROOT/'docs/_ext/wiki_phase72_compact.py')
compact=importlib.util.module_from_spec(spec);spec.loader.exec_module(compact)
standalone=runpy.run_path(str(ROOT/'docs/wiki/phase72_scalar_decoder.txt'))['read_compact']


def test_compact_is_lossless_and_reuses_authenticated_bytes():
    raw=compact.canonical({'count':8000,'missing':None,'exact':False,'value':0.0})
    compressed=gzip.compress(raw,mtime=0)
    envelope=compact.encode(raw,compressed)
    assert compact.decode(json.loads(envelope))==raw==standalone(envelope)
    assert base64.b32decode(json.loads(envelope)['data'])==compressed
    with pytest.raises(ValueError,match='mismatch'):
        compact.encode(raw,gzip.compress(b'{}',mtime=0))


def test_compression_cannot_bypass_decoded_privacy():
    raw=compact.canonical({'note':'/home/private/person/checkpoint.pt'})
    with pytest.raises(ValueError,match='Private decoded'):
        compact.encode(raw,gzip.compress(raw,mtime=0))


@pytest.mark.parametrize('mutation',['declared_limit','hidden_expansion','trailing_stream','digest'])
def test_bounded_standalone_rejects_malformed_envelopes(mutation):
    raw=b'"'+b'x'*4096+b'"';compressed=gzip.compress(raw,mtime=0)
    value={'encoding':compact.ENCODING,'decoded_bytes':len(raw),'decoded_sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b32encode(compressed).decode()}
    if mutation=='declared_limit':value['decoded_bytes']=11*1024*1024
    elif mutation=='hidden_expansion':value['decoded_bytes']=4
    elif mutation=='trailing_stream':value['data']=base64.b32encode(compressed+gzip.compress(b'',mtime=0)).decode()
    else:value['decoded_sha256']='0'*64
    with pytest.raises(ValueError):compact.decode(value)
    with pytest.raises(ValueError):standalone(compact.canonical(value))


def test_compact_payload_passes_unmodified_publication_privacy(tmp_path):
    raw=compact.canonical({'values':list(range(4096)),'missing':None})
    envelope=compact.encode(raw,gzip.compress(raw,mtime=0))
    assert '/' not in json.loads(envelope)['data']
    (tmp_path/'aggregate.json').write_bytes(envelope)
    spec=importlib.util.spec_from_file_location('privacy_fixture',ROOT/'docs/_ext/wiki_privacy.py')
    privacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(privacy)
    assert privacy.validate_artifact(tmp_path,generated_projection=True)['privacy']=='PASS'
