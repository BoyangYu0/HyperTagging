"""Objective balance factor isolation and explicit selection reuse/fresh strict guards."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase57 import ARM_ROLES, validate_closeout_basis, validate_seed_contract, validation_exclusions
from scripts.run_phase57_pretraining import validate_pretraining_contract, refinement_config
ROOT=Path(__file__).resolve().parents[1]


def load(stem):
    return json.loads((ROOT/'configs/reconstruction'/stem).read_text())


def prereg():return load('ht_reconstruction_phase57_20260922.json')


def contract(p,arm):
    pre=copy.deepcopy(p['pretraining_refinement'])
    pre['parent_ranking_weight']=2.0
    pre['config']['leaf_pid_phase_weights']=arm['pretraining_leaf_pid_phase_weights']
    return {'arm_role':arm['role'],'pretraining_refinement':pre}


def test_phase57_only_late_pid_balance_differs():
    p=prereg();assert tuple(a['role'] for a in p['arms'])==ARM_ROLES
    assert set(p['capacity_admission']['reports_by_arm'])==set(ARM_ROLES)
    assert all(v['production_training_allowed'] and v['query_overflow_count']==v['cardinality_overflow_count']==0 for v in p['capacity_admission']['reports_by_arm'].values())
    a,b=[contract(p,arm) for arm in p['arms']]
    for c in (a,b):validate_pretraining_contract(c,p)
    assert a['pretraining_refinement']['config'].pop('leaf_pid_phase_weights')==[1,1,.4,.4]
    assert b['pretraining_refinement']['config'].pop('leaf_pid_phase_weights')==[1,1,.2,.2]
    assert a['pretraining_refinement']==b['pretraining_refinement']
    assert p['arms'][0]['overrides']==p['arms'][1]['overrides']
    old=load('ht_reconstruction_phase56_20260921.json')
    for k in ('post_training_gates','evaluation_contract','data_binding'):assert p[k]==old[k]
    assert p['common_training_contract']['max_steps']==4376
    assert p['data_binding']['train_events']==70000
    assert p['validation_budget']['remaining_untouched_after_phase57']==291
    assert {a['checkpoint_step'] for a in p['arms']}=={81096}
    assert len({a['checkpoint_sha256'] for a in p['arms']})==1
    validate_closeout_basis(p)
    for key in ('phase56_closeout_basis','phase56_retained_metric_basis','validation_cohort'):
        binding=p[key];assert hashlib.sha256((ROOT/binding['path']).read_bytes()).hexdigest()==binding['sha256']


def test_phase57_selection_reuse_is_exact_and_strict_is_fresh(tmp_path):
    p=prereg();c=load('ht_reconstruction_phase57_validation_cohort_20260922.json')
    old=load('ht_reconstruction_phase56_validation_cohort_20260921.json')
    excluded=set(validation_exclusions(c))
    assert c['checkpoint_selection_event_uids']==old['checkpoint_selection_event_uids']
    assert len(excluded)==48609
    assert len(set(c['event_uids']))==100 and not set(c['event_uids']) & (excluded|set(c['checkpoint_selection_event_uids']))
    assert set(old['event_uids'])<=excluded
    validate_seed_contract(p['common_training_contract'],c['seed'])
    for phase in range(36,56):
        past=json.loads((ROOT/c['source_bindings'][f'phase{phase}_cohort']['path']).read_text())
        assert set(past['checkpoint_selection_event_uids'])|set(past['event_uids'])<=excluded
    for arm in p['arms']:
        cfg=refinement_config(contract(p,arm),{'selection_manifest':'selection','dataset_index':'index','checkpoint':'checkpoint'},c,tmp_path)
        assert cfg.validation_event_uids==tuple(c['checkpoint_selection_event_uids'])
        assert not set(cfg.validation_event_uids)&set(c['event_uids']) and cfg.resume is None


@pytest.mark.parametrize('change',['selection_order','old_strict','count'])
def test_phase57_rejects_unauthorized_cohort_reuse(change):
    c=load('ht_reconstruction_phase57_validation_cohort_20260922.json')
    if change=='selection_order':c['checkpoint_selection_event_uids'].reverse()
    if change=='old_strict':c['event_uids'][0]=c['checkpoint_selection_event_uids'][0]
    if change=='count':c['permitted_selection_reuse_count']=999
    with pytest.raises(RuntimeError):validation_exclusions(c)


@pytest.mark.parametrize('key,value',[('objective_dominance_ratio',21.0),('pilot_objective_violation_action','warn'),('max_steps',4376),('seed',20260923),('leaf_pid_phase_weights',[1,1,.1,.1])])
def test_phase57_rejects_guard_or_factor_drift(key,value):
    p=prereg();c=contract(p,p['arms'][0]);c['pretraining_refinement']['config'][key]=value
    with pytest.raises(RuntimeError):validate_pretraining_contract(c,p)


def test_phase57_rejects_stale_replay_seed():
    p=prereg();c=copy.deepcopy(p['common_training_contract']);c['balanced_level_replay_contract']['seed']=20260923
    with pytest.raises(RuntimeError):validate_seed_contract(c,20260924)
