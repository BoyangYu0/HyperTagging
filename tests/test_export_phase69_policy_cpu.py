from dataclasses import fields
import math
import pytest
from scripts.export_phase69_policy import merge_statistics, paired_intervals
from hypertagging.evaluation.full_decay_metrics import (
    DecayEvaluation, KinematicErrorMetrics, RatioMetric, summarize_decay_evaluations,
)


def unit(n, d, squared):
    kwargs={field.name: RatioMetric(n,d) for field in fields(DecayEvaluation) if field.type=='RatioMetric'}
    kwargs.update(scope='full',unit_index=0,available=True,unavailable_reason=None,
                  truth_root_position=0,predicted_root_position=0,truth_sources=tuple(range(d)),
                  predicted_sources=tuple(range(n)),kinematics=KinematicErrorMetrics(px_squared_error=RatioMetric(squared,d)),
                  leaf_pid_confusion=((2,2,n),))
    return DecayEvaluation(**kwargs)


def test_sufficient_statistics_merge_matches_shared_event_aggregation():
    rows=[unit(1,2,8),unit(9,10,10)]
    chunks=[summarize_decay_evaluations([row]) for row in rows]
    merged=merge_statistics(chunks)
    assert merged==summarize_decay_evaluations(rows)
    assert merged['source_recall']['value']==10/12
    assert merged['px_rmse']['value']==math.sqrt(18/12)
    assert merged['leaf_pid_confusion']==[{'truth_token':2,'predicted_token':2,'count':10}]


def test_merge_rejects_unregistered_list_and_metadata_drift():
    with pytest.raises(ValueError,match='Unknown list'):merge_statistics([{'x':[1]},{'x':[2]}])
    with pytest.raises(ValueError,match='Conflicting'):merge_statistics([{'policy':'a'},{'policy':'b'}])


def test_bootstrap_batch_size_does_not_change_paired_stratified_result():
    left={str(i):{'metric':[i%3,3]} for i in range(12)}
    right={str(i):{'metric':[i%3+1,3]} for i in range(12)}
    categories={str(i):('charged','mixed')[i%2] for i in range(12)}
    a=paired_intervals(left,right,categories,repeats=100,batch=7)
    b=paired_intervals(left,right,categories,repeats=100,batch=100)
    assert a['metrics']==b['metrics']
    assert a['metrics']['metric']['pre_minus_refined']['value']==pytest.approx(1/3)
    with pytest.raises(ValueError,match='identities'):paired_intervals(left,{},categories,repeats=2)
