"""Independent policy admission and final-coverage regressions for next campaign."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts import run_reconstruction_phase72 as runner
from scripts.validate_phase72_final_coverage import validate_final_coverage

CATEGORIES=('charged','mixed','ccbar','uubar','ddbar','ssbar')


def plan():
    return {'by_category':{cat:[f'{cat}-{i}' for i in range(2000)] for cat in CATEGORIES},
            'sealed_test_accessed':False,'status':'RESERVED_NOT_PROCESSED',
            'required_distinct_processed_per_category':2000,'scopes':['full','half'],
            'checkpoint_selection_uses_primary_cohort':False}


def admission(tmp_path,monkeypatch,value):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    p=tmp_path/'plan.json';p.write_text(json.dumps(value))
    return {'policy_evaluation_plan':{'path':'plan.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}


def test_reservation_is_not_processed_final_evaluation(tmp_path,monkeypatch):
    p=plan();uids={u for v in p['by_category'].values() for u in v}
    c=admission(tmp_path,monkeypatch,p)
    got=runner.validate_policy_plan(c,uids,{'historical'},{'selection'},set(sorted(uids)[:100]))
    assert got['reserved']==12000 and got['actual_processed']==0 and not got['final_evaluation_compliant']


@pytest.mark.parametrize('case',['short_category','duplicate','history','selection','sealed','missing_scope','false_complete'])
def test_plan_rejects_ineligible_or_mislabelled_coverage(tmp_path,monkeypatch,case):
    p=plan();uids={u for v in p['by_category'].values() for u in v};history={'historical'};selection={'selection'}
    if case=='short_category':p['by_category']['ssbar'].pop()
    if case=='duplicate':p['by_category']['ssbar'][0]=p['by_category']['charged'][0]
    if case=='history':history.add('charged-0')
    if case=='selection':selection.add('mixed-0')
    if case=='sealed':p['sealed_test_accessed']=True
    if case=='missing_scope':p['scopes']=['full']
    if case=='false_complete':p['status']='COMPLETE'
    c=admission(tmp_path,monkeypatch,p)
    with pytest.raises(RuntimeError):runner.validate_policy_plan(c,uids,history,selection,set(sorted(uids)[:100]))


def test_final_closeout_requires_processed_same_cohort_every_arm_scope():
    p=plan();rows=[{'event_uid':u,'source_category':cat,'attempted':True,'processed':True,'truth_unavailable':True} for cat,us in p['by_category'].items() for u in us]
    views={a:{s:copy.deepcopy(rows) for s in ('full','half')} for a in runner.ARM_ROLES}
    assert validate_final_coverage(p,views)['set_overlap_off']['full']['complete']
    views['set_overlap_on']['half'][0]['processed']=False
    with pytest.raises(ValueError,match='Incomplete category'):validate_final_coverage(p,views)


def test_seed_and_single_factor_guards():
    c={'seed':20261008,'balanced_level_replay_contract':{'seed':20261008}}
    runner.validate_seed_contract(c,20261008)
    with pytest.raises(RuntimeError):runner.validate_seed_contract(c,20261006)
    old=json.loads((Path(__file__).resolve().parents[1]/'configs/reconstruction/ht_reconstruction_phase70_20261005.json').read_text())
    for arm,role,weight in zip(old["arms"],runner.ARM_ROLES,(0.0,1.0)):
        arm["role"]=role
        arm["overrides"].update(type_conditioned_daughter_relation_bias=True,auxiliary_teacher_weight=0.5,pointer_set_overlap_weight=weight)
    runner.validate_context_contrast(old)
    old['arms'][1]['overrides']['pointer_positive_weight']=99
    with pytest.raises(RuntimeError,match='second scientific factor'):runner.validate_context_contrast(old)
