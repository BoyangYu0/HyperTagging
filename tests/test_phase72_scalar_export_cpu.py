"""Lossless scalar export preserves missing/null and bool/integer distinctions."""
import hashlib
import runpy
from pathlib import Path
import pytest
from scripts.package_phase72_histories import pack, unpack, canonical
DECODER=runpy.run_path(str(Path(__file__).parents[1]/'docs/wiki/phase72_scalar_decoder.txt'))

def test_rle_roundtrip_with_sparse_null_and_typed_values():
    rows=[{'record':i,'metric':'loss','value':v} for i,v in enumerate([0,0,True,1,None,0.0])]
    rows += [{'record':i,'metric':'validation','value':None} for i in (2,5)]
    expected=sorted(rows,key=lambda r:(r['record'],r['metric']))
    # Native checkpoint traversal is not the public alphabetical track order.
    packed=pack(list(reversed(rows)))
    assert canonical(unpack(packed))==canonical(expected)
    value={'version':'phase72-training-scalar-rle-v1','scalar_count':len(rows),'columns':packed,'decoded_scalar_sha256':hashlib.sha256(canonical(expected)).hexdigest()}
    assert canonical(DECODER['decode'](value))==canonical(expected)
    value['columns'][0]['value_runs'][0][0]=10**10
    with pytest.raises(ValueError,match='bounds'):DECODER['decode'](value)
