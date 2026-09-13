"""Aggregation populations must not turn failed or trivial units into success."""
from scripts.build_phase47_aggregation_supplement import METRICS,summarize


def row(n,d,**changes):
    r={'available':True,'target_representable':True,'structurally_valid':True,'truth_sources':[1,2],'predicted_sources':[1,2], 'leaf_pid_unavailable_count':0,'truth_leaf_count':2,'truth_mother_count':1,'predicted_mother_count':1}
    for key in METRICS:r.update({key+'_numerator':n,key+'_denominator':d})
    r.update(changes);return r


def test_micro_macro_keep_failures_and_undefined_denominators_separate():
    out=summarize([{'rows':[row(1,1)]},{'rows':[row(0,9),row(0,0)]}])['source_recall']
    assert out['micro']==0.1 and out['micro_denominator']==10
    assert out['unit_macro']==0.5 and out['unit_macro_denominator']==2
    assert out['event_macro']==0.5 and out['event_macro_denominator']==2
    assert out['unit_macro_unavailable']==1


def test_source_and_topology_do_not_imply_all_pid_success():
    r=row(1,1,leaf_pid_accuracy_numerator=1,leaf_pid_accuracy_denominator=2)
    out=summarize([{'rows':[r]}])['exact']['nontrivial']
    assert out['source']['numerator']==out['source_topology']['numerator']==1
    assert out['source_leaf_pid']['numerator']==out['source_topology_all_pid']['numerator']==0
    r['target_representable']=False;r['perfectLCAG_numerator']=0
    out=summarize([{'rows':[r]}])['exact']['nontrivial']
    assert out['source']['numerator']==0 and out['source']['denominator']==1


def test_isolated_leaf_does_not_receive_topology_credit():
    r=row(1,1,truth_sources=[1],predicted_sources=[1],truth_leaf_count=1,truth_mother_count=0,predicted_mother_count=0,perfectLCAG_numerator=0,perfectLCAG_denominator=0)
    out=summarize([{'rows':[r]}])
    assert out['exact']['all_retained']['source']['numerator']==1
    assert out['exact']['all_retained']['source_topology']['numerator']==0
    assert out['perfectLCAG']['unit_macro'] is None
    assert out['exact']['nontrivial']['unit_count']==0
