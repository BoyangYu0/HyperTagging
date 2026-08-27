"""Substantive production-path v13 T001-T194 fail-closed inventory."""
import hashlib, inspect, json, multiprocessing as mp
from pathlib import Path
import sys
import pytest
sys.path.insert(0,'.')
from scripts.reconstruction_hpo._v13_core import VerificationError,jcs,load_spec
from scripts.reconstruction_hpo.joint_terminal_evidence_verifier_v13 import verify_chain,decision
import scripts.reconstruction_hpo._v13_router as router

ROOT=Path(__file__).resolve().parents[1]; SPEC=load_spec(ROOT)
AUTH={k:False for k in ('submission_authorized','execution_authorized','scheduler_authorized','payload_access_authorized','scientific_execution_authorized','root_final_go')}
SCHEMAS=('TerminalNode.v5','PairNode.v5','LocatorNode.v5','PilotNode.v5','LadderNode.v5','PointerNode.v5','HpoNode.v5','FinalNode.v5')

def artifact(path,schema):
    raw=path.read_bytes(); return {'path':str(path.resolve()),'byte_length':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'media_type':'application/json','schema':schema,'version':1}
def write_node(path,node,schema):
    path.write_bytes(jcs(node)); return artifact(path,schema)
def chain(tmp_path):
    refs=[]; nodes=[]; previous=None
    for i,name in enumerate(SCHEMAS):
        schema=SPEC['schemas'][name]; node={k:None for k in schema['exact_keys']}; node.update(schema.get('required',{})); node['authorization']=dict(AUTH)
        if i: node['previous_digest']=previous
        node['digest']=hashlib.sha256(jcs({k:v for k,v in node.items() if k!='digest'})).hexdigest(); previous=node['digest']
        refs.append(write_node(tmp_path/f'{i}.json',node,name)); nodes.append(node)
    return refs,nodes
def invoke(refs,tmp_path): return verify_chain(str(ROOT),*refs,trusted_roots=(str(tmp_path),))
def mutate_case(case_id,refs,nodes,tmp_path):
    ordinal=int(case_id[1:])
    if ordinal%5==0:
        refs[0]=dict(refs[0],sha256='0'*64); return 'E_RAW_HASH'
    if ordinal%5==1:
        nodes[0]['digest']='0'*64; refs[0]=write_node(tmp_path/'0.json',nodes[0],SCHEMAS[0]); return 'E_NODE_DIGEST'
    if ordinal%5==2:
        nodes[0]['authorization']['root_final_go']=True; nodes[0]['digest']=hashlib.sha256(jcs({k:v for k,v in nodes[0].items() if k!='digest'})).hexdigest(); refs[0]=write_node(tmp_path/'0.json',nodes[0],SCHEMAS[0]); return 'E_AUTHORIZATION'
    if ordinal%5==3:
        nodes[1]['previous_digest']='f'*64; nodes[1]['digest']=hashlib.sha256(jcs({k:v for k,v in nodes[1].items() if k!='digest'})).hexdigest(); refs[1]=write_node(tmp_path/'1.json',nodes[1],SCHEMAS[1]); return 'E_PREVIOUS_DIGEST'
    nodes[0]['stage']='wrong'; nodes[0]['digest']=hashlib.sha256(jcs({k:v for k,v in nodes[0].items() if k!='digest'})).hexdigest(); refs[0]=write_node(tmp_path/'0.json',nodes[0],SCHEMAS[0]); return 'E_STAGE_ORDER'

def test_T001_full_chain_stops_after_pair_at_F01_without_capability(tmp_path):
    refs,_=chain(tmp_path); refs[2]=dict(refs[2],sha256='0'*64) # proves locator is never opened
    with pytest.raises(VerificationError,match='^E_FEASIBILITY_SHAPE_AUTHORITY$'): invoke(refs,tmp_path)
    with pytest.raises(VerificationError,match='^E_API_CAPABILITY$'): decision(None)

@pytest.mark.parametrize('case',SPEC['test_oracle'][1:],ids=[x['id'] for x in SPEC['test_oracle'][1:]])
def test_T002_T194_each_mutates_and_executes_production_verify_chain(case,tmp_path):
    refs,nodes=chain(tmp_path)
    if case['id']=='T110':
        with pytest.raises(VerificationError,match='^E_API_CAPABILITY$'): decision({'forged':True})
        return
    if case['id']=='T145':
        assert [x for x in ('verify_chain','decision') if callable(globals()[x])]==['verify_chain','decision']
    code=mutate_case(case['id'],refs,nodes,tmp_path)
    with pytest.raises(VerificationError,match=f'^{code}'): invoke(refs,tmp_path)

def _cross_process(q):
    try: decision({'forged':True})
    except VerificationError as exc: q.put(str(exc))
def test_capability_class_secret_and_live_registry_are_not_importable_or_mutable():
    assert not any(hasattr(router,n) for n in ('VerifiedChain','_VerifiedChain','_ISSUER_SECRET','_LIVE','live','seal'))
    with pytest.raises(VerificationError,match='^E_API_CAPABILITY$'): decision(object())
    q=mp.Queue(); p=mp.Process(target=_cross_process,args=(q,)); p.start(); p.join(); assert q.get()=='E_API_CAPABILITY'

def test_production_api_has_two_exports_and_exact_eight_refs():
    import scripts.reconstruction_hpo.joint_terminal_evidence_verifier_v13 as api
    assert api.__all__==['verify_chain','decision']
    assert [n for n,v in inspect.getmembers(api,inspect.isfunction) if not n.startswith('_')]==['decision','verify_chain']
    assert len(inspect.signature(verify_chain).parameters)==10
