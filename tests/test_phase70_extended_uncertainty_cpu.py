"""Check collision-coherent pair and continuum numerator/denominator accounting."""
from scripts.export_phase70_extended_uncertainty import event_ratios
from scripts.export_phase70_policy import paired_intervals


def ratio(n,d):return {'numerator':n,'denominator':d}


def test_exact_and_inclusive_continuum_keep_their_own_recovery_and_background():
    tag={'pool_at_k':{'1':{}},'continuum':{'retained_component_top1_correct':ratio(1,3),
         'top1_fake_b_slot_acceptance':ratio(2,2),'top1_fake_b_event_acceptance':ratio(1,1)},
         'inclusive_fsp_grouping':{'pool_at_k':{'1':{}},'continuum':{'retained_component_top1_correct':ratio(2,3)}}}
    event={'source_category':'ccbar','scopes':{'full':{'tag_efficiency':{'greedy':tag}}}}
    values=event_ratios(event)
    assert values['full/ccbar/exact_component_recovery']==[1,3]
    assert values['full/ccbar/inclusive_component_recovery']==[2,3]
    assert values['full/ccbar/top1_fake_b_event_acceptance']==[1,1]
    assert values['full/uubar/exact_component_recovery']==[0,0]
    assert values['full/exact/coherent_pair']==[0,0]


def test_bootstrap_uses_sufficient_statistics_and_preserves_zero_denominator():
    categories={f'{cat}-{i}':cat for cat in ('charged','mixed','ccbar','uubar','ddbar','ssbar') for i in range(2)}
    left={u:{'rate':[1,2 if u.endswith('0') else 8],'unavailable':[0,0]} for u in categories}
    right={u:{'rate':[2,2 if u.endswith('0') else 8],'unavailable':[0,0]} for u in categories}
    result=paired_intervals(left,right,categories,repeats=32,batch=4)
    assert result['metrics']['rate']['type_bias']['value']==.2
    assert result['metrics']['rate']['no_type_bias']['value']==.4
    assert result['metrics']['rate']['disabled_minus_enabled']['value']==.2
    assert result['metrics']['unavailable']['type_bias']['value'] is None
    assert result['metrics']['unavailable']['type_bias']['valid_resamples']==0
