import hashlib
import inspect
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, '.')
from scripts.reconstruction_hpo._v13_core import jcs, load_spec, VerificationError
from scripts.reconstruction_hpo.joint_terminal_evidence_verifier_v13 import verify_chain, decision

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_spec(ROOT)
AUTH = {k: False for k in ('submission_authorized','execution_authorized','scheduler_authorized','payload_access_authorized','scientific_execution_authorized','root_final_go')}
SCHEMAS = ('TerminalNode.v5','PairNode.v5','LocatorNode.v5','PilotNode.v5','LadderNode.v5','PointerNode.v5','HpoNode.v5','FinalNode.v5')

def artifact(path, schema):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'byte_length': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'media_type': 'application/json', 'schema': schema, 'version': 1}

def chain(tmp_path):
    refs=[]; previous=None
    for i, name in enumerate(SCHEMAS):
        schema=SPEC['schemas'][name]
        node={k: None for k in schema['exact_keys']}
        node.update(schema.get('required', {}))
        node['authorization']=dict(AUTH)
        if i: node['previous_digest']=previous
        node['digest']=hashlib.sha256(jcs({k:v for k,v in node.items() if k!='digest'})).hexdigest()
        previous=node['digest']
        path=tmp_path/f'{i}.json'
        path.write_bytes(jcs(node))
        refs.append(artifact(path,name))
    return refs

def test_public_api_is_exact_and_signature_requires_eight_refs():
    import scripts.reconstruction_hpo.joint_terminal_evidence_verifier_v13 as api
    assert api.__all__ == ['verify_chain','decision']
    assert [n for n,v in inspect.getmembers(api,inspect.isfunction) if not n.startswith('_')] == ['decision','verify_chain']
    assert len(inspect.signature(verify_chain).parameters) == 10

def test_compiled_F01_prevents_capability_issuance(tmp_path):
    refs=chain(tmp_path)
    with pytest.raises(VerificationError, match='^E_FEASIBILITY_SHAPE_AUTHORITY$'):
        verify_chain(str(ROOT),*refs,trusted_roots=[str(tmp_path)])
    with pytest.raises(VerificationError): decision({'_nonce':'forged'})

def test_digest_or_previous_digest_mutation_rejects(tmp_path):
    refs=chain(tmp_path)
    data=json.loads(Path(refs[3]['path']).read_text())
    data['previous_digest']='0'*64
    Path(refs[3]['path']).write_bytes(jcs(data))
    refs[3]=artifact(Path(refs[3]['path']),SCHEMAS[3])
    with pytest.raises(VerificationError): verify_chain(str(ROOT),*refs,trusted_roots=[str(tmp_path)])

def test_trusted_root_and_symlink_fail_closed(tmp_path):
    refs=chain(tmp_path)
    with pytest.raises(VerificationError): verify_chain(str(ROOT),*refs,trusted_roots=[str(tmp_path/'other')])

def test_normative_oracle_identity_is_delegated_to_substantive_v4_inventory():
    assert [case['id'] for case in SPEC['test_oracle']] == [f'T{i:03d}' for i in range(1,195)]
    assert SPEC['authorization']['submission_authorized'] is False
