"""Source-role metadata checks require no private payload or sealed-test access."""
from pathlib import Path
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]

def load(name):return json.loads((ROOT/name).read_text())

def test_expansion_preserves_every_prior_source_role_and_train_payload():
    old='configs/training_selection/phase60_validation_expansion_20260925/'
    new='configs/training_selection/phase70_validation_expansion_20261006/'
    before=load(old+'roles.json');after=load(new+'roles.json')
    old_roles={e['source_file']:e['role'] for e in before['entries']}
    new_roles={e['source_file']:e['role'] for e in after['entries']}
    assert all(new_roles[k]==v for k,v in old_roles.items())
    fresh=set(new_roles)-set(old_roles)
    assert len(fresh)==12 and all(new_roles[k]=='validation' for k in fresh)
    a,b=load(old+'train_070k.json'),load(new+'train_070k.json')
    def training(m):return {e['source_file']:e['parquet_sha256_reference'] for e in m['entries'] if e['split']=='train'}
    assert training(a)==training(b)
    assert b['split_counts']=={'train':70000,'validation':160000,'test':0}
    assert not b['selection_includes_test']
    assert Counter(e['category'] for e in b['entries'] if e['source_file'] in fresh)==dict.fromkeys(('charged','mixed','ccbar','uubar','ddbar','ssbar'),2)
