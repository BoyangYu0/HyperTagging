"""Reserve one matched-daughter set-overlap campaign after complete Phase71 review."""
import copy
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256, uid_set_sha256
from hypertagging.data.capacity import production_capacity_report
from hypertagging.training.model_config import MODEL_PRESETS
ARTIFACTS=Path('/project/agkuhr/users/boyang/data/HyperTagging_artifacts')
REVIEW=ARTIFACTS/'phase71_review_20261007'
PRIVATE=ROOT/'runtime_inputs/reconstruction_phase72_20261007'
CATEGORIES=('charged','mixed','ccbar','uubar','ddbar','ssbar')

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def build():
    import pyarrow.parquet as pq
    queue=subprocess.check_output(['squeue','--noheader','--user',getpass.getuser(),'--format=%j|%k'],text=True)
    if 'phase72-' in queue or PRIVATE.exists():raise RuntimeError('Existing next campaign or reservation')
    decision=load(REVIEW/'phase72-decision.json');aggregate_path=REVIEW/'primary-v1/aggregate.json'
    if decision['selected_campaign']!='matched_daughter_set_overlap_off_vs_on' or decision['review_complete'] is not True or decision['aggregate_sha256']!=sha(aggregate_path):raise RuntimeError('Missing authenticated complete scientific decision')
    import importlib.util
    spec=importlib.util.spec_from_file_location('policy',ROOT/'docs/_ext/wiki_phase71.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.validate(load(aggregate_path))
    prior=ROOT/'configs/reconstruction/ht_reconstruction_phase71_20261006.json';p=load(prior);old=load(ROOT/p['validation_cohort']['path'])
    assert sha(ROOT/p['validation_cohort']['path'])==p['validation_cohort']['sha256']
    hb=old['source_bindings']['previous_validation_universe'];assert sha(ROOT/hb['path'])==hb['sha256']
    used=set(load(ROOT/hb['path'])['event_uids'])
    registry_path=ROOT/'configs/reconstruction/supplementary_validation_reservations.json'
    for r in load(registry_path)['reservations']:
        b=r['cohort_manifest'];f=ARTIFACTS/b['artifact_relative_path'];assert sha(f)==b['sha256'];u=set(load(f)['event_uids']);assert len(u)==r['reserved_count'];used |=u
    assert len(used)==98000
    selection_path=ROOT/p['data_binding']['selection_manifest'];index_path=ROOT/p['data_binding']['dataset_index']
    assert sha(selection_path)==p['data_binding']['selection_manifest_sha256'] and sha(index_path)==p['data_binding']['dataset_index_sha256']
    selection,index=load(selection_path),load(index_path);train=set();categories={}
    assert not selection['selection_includes_test'] and index['normalizer_scope']=='train'
    for e in selection['entries']:
        path=Path(selection['data_root'])/e['path'];assert sha(path)==e['parquet_sha256_reference']
        for row in pq.read_table(path,columns=['event_uid','source_category','source_file'],use_threads=False).to_pylist():
            u=str(row['event_uid']);assert u not in train and u not in categories
            assert row['source_category']==e['category'] and row['source_file']==e['source_file']
            if e['split']=='train':train.add(u)
            elif e['split']=='validation':categories[u]=e['category']
            else:raise RuntimeError('Sealed or unknown role')
    universe=set(categories);assert len(train)==70000 and len(universe)==160000 and not train&universe and used<=universe
    pools={c:ranked({u for u,k in categories.items() if k==c}-used,20261008) for c in CATEGORIES}
    if any(len(x)<2000 for x in pools.values()):raise RuntimeError('Independent category capacity shortage: '+str({c:len(x) for c,x in pools.items()}))
    groups={c:x[:2000] for c,x in pools.items()};primary=[u for c in CATEGORIES for u in groups[c]]
    selected=ranked(universe-used-set(primary),20261008)[:1000];diagnostic=ranked(set(primary),20261008)[:100]
    assert len(set(primary+selected))==13000 and not set(primary+selected)&used
    PRIVATE.mkdir(parents=True)
    write(PRIVATE/'universe.json',{'event_uids':sorted(universe),'event_uids_sha256':uid_set_sha256(universe)})
    write(PRIVATE/'reservation-registry-before.json',load(registry_path))
    write(PRIVATE/'history.json',{'event_uids':sorted(used),'event_uids_sha256':uid_set_sha256(used),'registry_binding':binding(PRIVATE/'reservation-registry-before.json'),'original_registry_sha256':sha(registry_path)})
    assert sha(ROOT/p['policy_evaluation_plan']['path'])==p['policy_evaluation_plan']['sha256']
    plan=copy.deepcopy(load(ROOT/p['policy_evaluation_plan']['path']))
    plan.update(version='phase72-primary-policy-plan-v1',event_uids=primary,by_category=groups,exclusion_uid_set_sha256=uid_set_sha256(used),seed=20261008,available_before_reservation={c:len(x) for c,x in pools.items()})
    write(PRIVATE/'policy-plan.json',plan)
    c=copy.deepcopy(old);c.pop('remaining_untouched_after_phase71',None)
    c.update(manifest_version='hypertagging-reconstruction-phase72-cohort-v1',study_id='phase72-matched-daughter-set-objective-20261007',created_at=datetime.now(timezone.utc).isoformat(),seed=20261008,selection_original_seed=20261008,strict_selection_seed=20261008,remaining_untouched_after_phase72=49000,historical_used_event_uid_count=98000,checkpoint_selection_event_uids=selected,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),evaluation_event_uids=diagnostic,event_uids=diagnostic,evaluation_event_uids_sha256=uid_sequence_sha256(diagnostic),event_uids_sha256=uid_sequence_sha256(diagnostic),validation_exclusion_event_uid_count=159000,validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selected)),overlap_audit={'strict_vs_history':0,'selection_vs_history':0,'strict_vs_selection':0},policy_evaluation_plan=binding(PRIVATE/'policy-plan.json'))
    c.update(dataset_index=p['data_binding']['dataset_index'],dataset_index_sha256=p['data_binding']['dataset_index_sha256'],selection_manifest=p['data_binding']['selection_manifest'],selection_manifest_sha256=p['data_binding']['selection_manifest_sha256'])
    c['source_bindings']={'validation_universe':binding(PRIVATE/'universe.json'),'previous_validation_universe':binding(PRIVATE/'history.json')};write(PRIVATE/'cohort.json',c)
    for name,source in [('parent-closeout.json',REVIEW/'native/closeout.json'),('geometry.json',REVIEW/'native/geometry.json'),('review-decision.json',REVIEW/'phase72-decision.json'),('parent-policy-aggregate.json',aggregate_path)]:write(PRIVATE/name,load(source))
    p['geometry_admission'].update(binding(PRIVATE/'geometry.json'));p['parent_phase71']=binding(prior)
    p.update(study_id=c['study_id'],preregistration_version='hypertagging-reconstruction-phase72-preregistration-v1',created_at=c['created_at'],pilot_classification='MATCHED_DAUGHTER_SET_OBJECTIVE_PILOT',scientific_question='Does adding a matched-daughter soft-Jaccard training objective improve exact and inclusive retained tagging without increasing fake-B acceptance or degrading precision and recursive structure?')
    base=copy.deepcopy(p['arms'][0]);p['arms']=[]
    for role,w in [('set_overlap_off',0.0),('set_overlap_on',1.0)]:
        a=copy.deepcopy(base);a.update(role=role,label=role,hypothesis='Jointly penalizing foreign and missing daughters may improve model-produced grouping; local set overlap does not guarantee recursive correctness.')
        a['overrides'].update(auxiliary_teacher_weight=0.5,pointer_set_overlap_weight=w);p['arms'].append(a)
    p['exposure_expectations']={a['role']:copy.deepcopy(p['exposure_expectations']['aux_teacher_050']) for a in p['arms']}
    p['pretraining_refinement']['mode']='fixed_encoder_matched_daughter_set_objective'
    p['scientific_source_boundary']['limitation']='New local membership-objective mechanism, not another auxiliary teacher dose or a pretraining/data-size comparison. No model promotion.'
    p.pop('phase70_closeout_basis');p.pop('phase70_retained_metric_basis')
    p['phase71_closeout_basis']={**binding(PRIVATE/'parent-closeout.json'),'classification':'completed_pair_one_passes_original_gates','selected_next_factor':'pointer_set_overlap_weight','sealed_test_accessed':False}
    p['phase71_retained_metric_basis']={**binding(PRIVATE/'parent-closeout.json'),'version':'phase71-closeout-v1'}
    p['review_decision']=binding(PRIVATE/'review-decision.json');p['parent_policy_aggregate']=binding(PRIVATE/'parent-policy-aggregate.json')
    common=p['common_training_contract'];common['seed']=20261008;common['balanced_level_replay_contract']['seed']=20261008
    capacity=production_capacity_report(index,global_n_queries=MODEL_PRESETS[common['model_preset']].n_queries,global_max_cardinality=17,n_queries_by_level=dict(common['n_queries_by_level']),max_cardinality_by_level=dict(common['max_cardinality_by_level']),target_policy='complete_only')
    assert capacity['production_training_allowed'] and capacity['query_overflow_count']==capacity['cardinality_overflow_count']==0
    p['capacity_admission']['reports_by_arm']={a['role']:copy.deepcopy(capacity) for a in p['arms']}
    p['shared_capacity_repair']['provenance']='Inherited unchanged from Phase71; no Phase72 capacity intervention.'
    stats=load(ROOT/p['fresh_statistics_binding']['path']);stats['training_normalizer_identical_to_phase71']=True;write(PRIVATE/'statistics.json',stats);p['fresh_statistics_binding']=binding(PRIVATE/'statistics.json')
    p['validation_cohort'].update(binding(PRIVATE/'cohort.json'),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget'].pop('remaining_untouched_after_phase71');p['validation_budget'].update(previously_used=98000,remaining_untouched_after_phase72=49000,actual_primary_processed=0)
    p['policy_evaluation_plan']=binding(PRIVATE/'policy-plan.json')
    p['objective_definition']={'name':'matched_daughter_soft_jaccard','formula':'1 - sum(sigmoid(pointer_logit)*target) / sum(sigmoid(pointer_logit)+target-sigmoid(pointer_logit)*target)','coefficient':1.0,'reduction':'mean over matched mothers within each event-level, then existing level/context normalization','precision':'FP32','population':'existing representable matched mother daughter vectors; no targets introduced','inference_changed':False,'limitations':'Local immediate-daughter surrogate, not end-to-end exact B loss; old and new weighted total losses are not comparable scientific quality.'}
    p['decision_rules'].update(authority='User2026-10-07 authorizes one bounded successor; no further campaign.',historical_basis=decision['rationale'],execution_check='Verify matched shared architecture, overlap coefficient0/1, positive native overlap loss only in candidate and finite gradients; auxiliary teacher fixed0.5.',stop_rule='Review this one mechanism contrast. No automatic sweep, successor, sealed test or promotion.',confirmation='Require joint exact and inclusive per-B improvement with positive paired collision lower bounds, useful gain at least4/8000, no point increase in pooled continuum fake-B acceptance and no source precision/recursive-structure deterioration. Preserve all original diagnostic gates.')
    p['runtime_variation_boundary']['policy']='Only added set-overlap objective differs; common seed20261008, initialization, data, replay budget and fixed inference. No equal-FLOP or post-update trajectory identity claim.'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase72_20261007.json',p)
    write(PRIVATE/'reservation.json',{'event_uids':primary+selected,'primary_policy_events':12000,'checkpoint_selection_events':1000,'by_category':groups,'sealed_test_accessed':False})
    print(json.dumps({'preregistration':str(ROOT/'configs/reconstruction/ht_reconstruction_phase72_20261007.json'),'reservation':binding(PRIVATE/'reservation.json'),'remaining':49000}))
if __name__=='__main__':build()
