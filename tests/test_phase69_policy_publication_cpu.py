"""Small synthetic publication-contract fixtures; no scientific-performance claim."""
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('policy_publication',ROOT/'docs/_ext/wiki_phase69_policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    categories=('charged','mixed','ccbar','uubar','ddbar','ssbar')
    count={'requested':2000,'unique_attempted':2000,'processed':2000,'failed':0,'truth_unavailable':3}
    tag={'summary':{'bbbar_trial_count':8000,'event_count':12000},
         'by_source_category':{c:{'event_count':2000} for c in categories},
         'b_channel_coverage':[{'evaluated_b_trials':8000}]}
    arm={'beam_diagnostic':{'coverage':{'status':'DIAGNOSTIC_NOT_QUOTA_COMPLIANT','processed_by_category':dict.fromkeys(categories,10)}},
         'greedy':{'summaries':{s:{'inference':{'event_count':12000}} for s in ('full','half')},
                   'summaries_by_source_category':{c:{s:{'inference':{'event_count':2000}} for s in ('full','half')} for c in categories},
                   'tag_efficiency':{'summaries':{s+'/greedy':{**copy.deepcopy(tag),'inclusive_fsp_grouping':copy.deepcopy(tag)} for s in ('full','half')}}}}
    return {'version':'phase69-policy-reevaluation-v1-20261006','status':'COMPLETE_GREEDY_WITH_DIAGNOSTIC_BEAM',
            'historical_reports_replaced':False,'sealed_test_accessed':False,'physical_fei_comparison_ready':False,
            'arms':{a:copy.deepcopy(arm) for a in ('refined_encoder','pre_refinement_encoder')},
            'coverage':{'views':{a:{c:{s:copy.deepcopy(count) for s in ('full','half')} for c in categories} for a in ('refined_encoder','pre_refinement_encoder')}}}


def test_complete_publication_contract_retains_unavailable_truth():
    module.validate(fixture())


def test_publication_rejects_shortage_in_one_arm_scope():
    value=fixture();value['coverage']['views']['refined_encoder']['ssbar']['half']['processed']=1999
    with pytest.raises(ValueError,match='Quota'):module.validate(value)


def test_publication_rejects_missing_channel_and_private_event_data():
    value=fixture();value['arms']['refined_encoder']['greedy']['tag_efficiency']['summaries']['full/greedy']['b_channel_coverage']=[]
    with pytest.raises(ValueError,match='Channel'):module.validate(value)
    value=fixture();value['event_uids']=['private']
    with pytest.raises(ValueError,match='Private event'):module.validate(value)


def test_diagnostic_beam_cannot_inherit_greedy_completeness():
    value=fixture();value['arms']['refined_encoder']['beam_diagnostic']['coverage']['status']='COMPLETE'
    with pytest.raises(ValueError,match='Beam diagnostic'):module.validate(value)


def test_dag_expansion_is_bounded_before_decoding():
    nodes=[['str','x']]
    for i in range(12):nodes.append(['list',[i]*10])
    with pytest.raises(ValueError,match='decoded byte budget'):module.decoded_size(nodes,len(nodes)-1)
    with pytest.raises(ValueError,match='reference'):module.decoded_size([['list',[0]]],0)


def test_dag_size_counts_shared_subtrees_without_expansion(monkeypatch):
    nodes=[['str','x'],['list',[0,0]],['dict',[[0,1]]]]
    expected=len(b'{"x":["x","x"]}\n')
    assert module.decoded_size(nodes,2)==expected
    monkeypatch.setattr(module,'MAX_DECODED_BYTES',expected-1)
    with pytest.raises(ValueError,match='decoded byte budget'):module.decoded_size(nodes,2)
