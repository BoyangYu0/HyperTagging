import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, '.')
from scripts.reconstruction_hpo._v13_core import (
    VerificationError, jcs, load_spec, validate_all_schema_objects,
    validate_authorization, validate_usage_denials, validate_spec_bytes,
)

ROOT = Path(__file__).resolve().parents[1]

def auth():
    return {k: False for k in ('submission_authorized','execution_authorized','scheduler_authorized','payload_access_authorized','scientific_execution_authorized','root_final_go')}

def denials():
    return {k: False for k in ('sealed_test_used','stress_used','restricted_raw_used','restricted_source_used','train_loss_used')}

def test_normative_registry_and_oracle():
    spec = load_spec(ROOT)
    assert len(spec['schemas']) == 38
    assert len(spec['test_oracle']) == 194
    assert all(value is False for value in spec['authorization'].values())

def test_duplicate_json_keys_rejected():
    with pytest.raises(VerificationError):
        validate_spec_bytes(b'{"a":1,"a":2}')

def test_authorization_and_usage_are_exact_false():
    value = {'authorization': auth(), 'usage_denials': denials()}
    validate_authorization(value)
    validate_usage_denials(value)
    bad = dict(value, authorization=dict(auth(), root_final_go=True))
    with pytest.raises(VerificationError):
        validate_authorization(bad)

def test_all_38_schema_key_maps_are_consumed():
    spec = load_spec(ROOT)
    objects = {}
    common = spec['schemas']['CommonReceipt.v5']['exact_keys']
    for name, schema in spec['schemas'].items():
        keys = schema.get('exact_keys') or schema.get('common_plus_exact_keys')
        if schema.get('common_plus_exact_keys'):
            keys = list(common) + list(keys)
        objects[name] = {k: ({x: False for x in ('submission_authorized','execution_authorized','scheduler_authorized','payload_access_authorized','scientific_execution_authorized','root_final_go')} if k == 'authorization' else ({x: False for x in ('sealed_test_used','stress_used','restricted_raw_used','restricted_source_used','train_loss_used')} if k == 'usage_denials' else None)) for k in keys}
    # This synthetic object intentionally lacks authorization/denials values;
    # exact key routing itself must still cover every schema.
    validate_all_schema_objects(spec, objects)

def test_jcs_is_deterministic_and_rejects_float():
    assert jcs({'b': 2, 'a': [True, 'x']}) == b'{"a":[true,"x"],"b":2}'
    with pytest.raises(VerificationError):
        jcs({'x': 1.0})
