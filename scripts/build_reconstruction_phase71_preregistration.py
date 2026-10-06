"""Reserve one bounded auxiliary-teacher objective contrast after complete Phase70 review.

Private collision identities remain in ignored, hash-bound runtime inputs.
The 100-event native suite remains diagnostic; it cannot close the primary study.
"""
import copy
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256, uid_set_sha256
from hypertagging.data.capacity import production_capacity_report
from hypertagging.training.model_config import MODEL_PRESETS

ARTIFACTS = Path('/project/agkuhr/users/boyang/data/HyperTagging_artifacts')
REVIEW = ARTIFACTS/'phase70_review_20261006'
PRIVATE = ROOT/'runtime_inputs/reconstruction_phase71_20261006'
CATEGORIES = ('charged', 'mixed', 'ccbar', 'uubar', 'ddbar', 'ssbar')


def load(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def build():
    import pyarrow.parquet as pq
    queue = subprocess.check_output(['squeue','--noheader','--user',getpass.getuser(),'--format=%j|%k'],text=True)
    if 'phase71' in queue or PRIVATE.exists():
        raise RuntimeError('Existing Phase71 jobs or reservation; refuse duplicate campaign')
    decision = load(REVIEW/'phase71-decision.json')
    if decision['selected_campaign'] != 'auxiliary_teacher_objective_050_vs_100' or decision['review_complete'] is not True:
        raise RuntimeError('Complete evidence synthesis must authorize this specific bounded study')
    aggregate_path = REVIEW/'supplemental-v1/aggregate.json'
    aggregate = load(aggregate_path)
    if decision['aggregate_sha256'] != hashlib.sha256(aggregate_path.read_bytes()).hexdigest():
        raise RuntimeError('Reviewed primary results changed')
    # Publication validation independently enforces actual category coverage, channel accounting and both scopes.
    import importlib.util
    spec=importlib.util.spec_from_file_location('p70_policy',ROOT/'docs/_ext/wiki_phase70.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator);validator.validate(aggregate)
    prior = ROOT/'configs/reconstruction/ht_reconstruction_phase70_20261005.json'
    p = load(prior); old = load(ROOT/p['validation_cohort']['path'])
    history_path = ROOT/old['source_bindings']['previous_validation_universe']['path']
    used = set(load(history_path)['event_uids']) | set(old['event_uids']) | set(old['checkpoint_selection_event_uids'])
    registry_path = ROOT/'configs/reconstruction/supplementary_validation_reservations.json'
    registry = load(registry_path)
    for row in registry['reservations']:
        b = row['cohort_manifest']; path = ARTIFACTS/b['artifact_relative_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != b['sha256']:
            raise RuntimeError('Existing reservation hash changed')
        uids=set(load(path)['event_uids'])
        if len(uids)!=row['reserved_count'] or used & uids:
            raise RuntimeError('Existing reservation identity/overlap mismatch')
        used |= uids
    assert len(used) == 85000
    selection_path=ROOT/'configs/training_selection/phase70_validation_expansion_20261006/train_070k.json'
    index_path=ROOT/'artifacts/experiment_readiness/reconstruction_phase70_20261006/train_070k.complete_only.index.json'
    manifest,index=load(selection_path),load(index_path)
    categories,train={},set()
    for entry in manifest['entries']:
        path=Path(manifest['data_root'])/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['parquet_sha256_reference']:
            raise RuntimeError('Materialized source changed')
        for row in pq.read_table(path, columns=['event_uid','source_category','source_file']).to_pylist():
            uid=str(row['event_uid']);assert row['source_category']==entry['category'] and row['source_file']==entry['source_file']
            assert uid not in train and uid not in categories
            if entry['split']=='train':train.add(uid)
            elif entry['split']=='validation':categories[uid]=entry['category']
            else:raise RuntimeError('Sealed or unknown role')
    universe=set(categories)
    assert len(universe)==160000 and len(train)==70000 and not train & universe and used<=universe
    assert index['normalizer_state']==load(ROOT/p['data_binding']['dataset_index'])['normalizer_state']
    pools={cat:ranked({u for u,c in categories.items() if c==cat}-used,20261007) for cat in CATEGORIES}
    if any(len(v)<2000 for v in pools.values()):
        raise RuntimeError('Insufficient independent per-category capacity: '+str({k:len(v) for k,v in pools.items()}))
    groups={cat:values[:2000] for cat,values in pools.items()}
    primary=[uid for cat in CATEGORIES for uid in groups[cat]]
    selection=ranked(universe-used-set(primary),20261007)[:1000]
    diagnostic=ranked(set(primary),20261007)[:100]
    assert len(set(primary+selection))==13000 and not set(primary+selection)&used
    PRIVATE.mkdir(parents=True)
    write(PRIVATE/'universe.json',{'event_uids':sorted(universe),'event_uids_sha256':uid_set_sha256(universe)})
    write(PRIVATE/'history.json',{'event_uids':sorted(used),'event_uids_sha256':uid_set_sha256(used),'registry_binding':binding(registry_path)})
    plan={'version':'phase71-primary-policy-plan-v1','status':'RESERVED_NOT_PROCESSED','event_uids':primary,'by_category':groups,
          'required_distinct_processed_per_category':2000,'scopes':['full','half'],'actual_processed_per_arm':0,
          'sealed_test_accessed':False,'checkpoint_selection_uses_primary_cohort':False,
          'exclusion_uid_set_sha256':uid_set_sha256(used),'seed':20261007,
          'endpoints':['exact_per_b_correct','inclusive_per_b_correct','event_any_correct','event_both_correct','coherent_pair','continuum_exact_inclusive_recovery','fake_b_background'],
          'checkpoint':'selection-only best micro_complete_target_efficiency; freeze its digest before primary inference',
          'evaluation_resources':{'tasks':50,'primary_chunks':48,'diagnostic_beam_tasks':2,'maximum_concurrency':16,'cpus_per_task':1,'memory_gib':16,'hours_per_task':4,'requeue':False},
          'small_native_suite':'100-event diagnostic plus20-event beam; never complete final evaluation',
          'no_automatic_successor':True,'available_before_reservation':{cat:len(v) for cat,v in pools.items()}}
    write(PRIVATE/'policy-plan.json',plan)
    c=copy.deepcopy(old)
    c.pop('remaining_untouched_after_phase70',None)
    c.update(manifest_version='hypertagging-reconstruction-phase71-cohort-v1',study_id='phase71-auxiliary-teacher-objective-20261006',
             created_at=datetime.now(timezone.utc).isoformat(),seed=20261007,selection_original_seed=20261007,strict_selection_seed=20261007,
             remaining_untouched_after_phase71=62000,historical_used_event_uid_count=85000,
             checkpoint_selection_event_uids=selection,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),
             evaluation_event_uids=diagnostic,event_uids=diagnostic,evaluation_event_uids_sha256=uid_sequence_sha256(diagnostic),event_uids_sha256=uid_sequence_sha256(diagnostic),
             validation_exclusion_event_uid_count=159000,validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selection)),
             selection_reuse_policy='Fresh1000 selection; fresh12000 stratified primary reservation;100 native diagnostic rows are a subset of primary.',
             overlap_audit={'strict_vs_history':0,'selection_vs_history':0,'strict_vs_selection':0},
             policy_evaluation_plan=binding(PRIVATE/'policy-plan.json'),dataset_index=str(index_path.relative_to(ROOT)),
             selection_manifest_sha256=binding(selection_path)['sha256'])
    c['source_bindings']={'validation_universe':binding(PRIVATE/'universe.json'),'previous_validation_universe':binding(PRIVATE/'history.json')}
    write(PRIVATE/'cohort.json',c)
    write(PRIVATE/'parent-closeout.json',load(REVIEW/'native/closeout-reviewed.json'))
    write(PRIVATE/'geometry.json',load(REVIEW/'native/geometry.json'))
    p['geometry_admission'].update(binding(PRIVATE/'geometry.json'))
    write(PRIVATE/'review-decision.json',decision)
    write(PRIVATE/'parent-policy-aggregate.json',aggregate)
    p.update(study_id=c['study_id'],preregistration_version='hypertagging-reconstruction-phase71-preregistration-v1',created_at=c['created_at'],
             pilot_classification='TEACHER_AUXILIARY_OBJECTIVE_PILOT',scientific_question='Does increasing auxiliary complete-target teacher supervision from0.5 to1.0 improve exact and inclusive tagging while retaining mixed-context exposure and controlling continuum fake-B background?')
    p['parent_phase70']=binding(prior)
    base=copy.deepcopy(p['arms'][0]);p['arms']=[]
    for role,weight in [('aux_teacher_050',0.5),('aux_teacher_100',1.0)]:
        arm=copy.deepcopy(base);arm.update(role=role,label=role,hypothesis='Increasing complete-target auxiliary supervision can help recursive efficiency, but may worsen exposure mismatch or background.')
        arm['overrides'].update(type_conditioned_daughter_relation_bias=True,auxiliary_teacher_weight=weight);p['arms'].append(arm)
    p['exposure_expectations']={role:copy.deepcopy(p['exposure_expectations']['type_bias']) for role in ('aux_teacher_050','aux_teacher_100')}
    p['pretraining_refinement']={'mode':'fixed_encoder_teacher_auxiliary_objective','checkpoint_for_reconstruction':'arm_bound_parameter_transfer','additional_pretraining_steps':0}
    p['scientific_source_boundary']['limitation']='Auxiliary teacher objective contrast within mixed-context training; not a data-size or pretraining-duration comparison. Shared enabled-bias architecture is a reference, not a promoted winner.'
    for key in ('phase69_closeout_basis','phase69_retained_metric_basis'):p.pop(key)
    p['phase70_closeout_basis']={**binding(PRIVATE/'parent-closeout.json'),'classification':'completed_pair_one_passes_original_gates','selected_next_factor':'auxiliary_teacher_weight','sealed_test_accessed':False}
    p['phase70_retained_metric_basis']={**binding(PRIVATE/'parent-closeout.json'),'version':'phase70-closeout-v1'}
    p['review_decision']=binding(PRIVATE/'review-decision.json');p['parent_policy_aggregate']=binding(PRIVATE/'parent-policy-aggregate.json')
    common=p['common_training_contract'];common['seed']=20261007;common['balanced_level_replay_contract']['seed']=20261007
    common['max_cardinality']=17;common['max_cardinality_by_level']=[[1,12],[2,17],[3,15],[4,15],[5,14],[6,2]]
    p['data_binding'].update(selection_manifest=str(selection_path.relative_to(ROOT)),selection_manifest_sha256=binding(selection_path)['sha256'],dataset_index=str(index_path.relative_to(ROOT)),dataset_index_sha256=binding(index_path)['sha256'],validation_events=160000)
    capacity=production_capacity_report(index,global_n_queries=MODEL_PRESETS[common['model_preset']].n_queries,global_max_cardinality=17,n_queries_by_level=dict(common['n_queries_by_level']),max_cardinality_by_level=dict(common['max_cardinality_by_level']),target_policy='complete_only')
    assert capacity['production_training_allowed'] and capacity['query_overflow_count']==capacity['cardinality_overflow_count']==0
    p['capacity_admission']={'dataset_index':str(index_path.relative_to(ROOT)),'dataset_index_sha256':binding(index_path)['sha256'],'reports_by_arm':{a['role']:copy.deepcopy(capacity) for a in p['arms']}}
    p['shared_capacity_repair']={'reason':'Expanded identity-safe validation contains four complete targets beyond old limits: level1 cardinalities10 and12, level2 cardinality17 twice. Minimal shared repair preserves every target.',
        'old_global_limit':16,'new_global_limit':17,'old_level1':9,'new_level1':12,'old_level2':16,'new_level2':17,'overflow_before':4,'overflow_after':0,
        'applies_to_both_arms':True,'cross_phase_causal_comparison_valid':False,'limits_selected_for_capacity_not_efficiency':True}
    stats={'status':'PASS','normalizer_scope':'train','sealed_test_accessed':False,'index_path':str(index_path.relative_to(ROOT)),
           'index_sha256':binding(index_path)['sha256'],'source_files':{str(f.relative_to(ROOT)):binding(f)['sha256'] for f in (selection_path,index_path)},'training_normalizer_identical_to_phase70':True}
    write(PRIVATE/'statistics.json',stats);p['fresh_statistics_binding']=binding(PRIVATE/'statistics.json')
    p['validation_cohort'].update(binding(PRIVATE/'cohort.json'),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget']={'previously_used':85000,'remaining_untouched_after_phase71':62000,'newly_reserved':13000,'selection_events':1000,'rollout_selection_events':1000,'strict_events':100,'primary_policy_events':12000,'strict_is_diagnostic_subset':True,'reconstruction_excluded_events':159000,'validation_role_events':160000,'actual_primary_processed':0}
    p['policy_evaluation_plan']=binding(PRIVATE/'policy-plan.json')
    p['efficiency_effect_target']={'absolute_per_b_gain':0.0005,'additional_proven_successes_per_8000':4,'interpretation':decision['interpretation']}
    p['decision_rules'].update(authority='User authorized one next bounded study on2026-10-06; no successor.',primary='Exact retained and inclusive FSP-group per-B correct-tag efficiency; any-B, both-B and coherent pair; all full/half category-sized endpoints.',
        confirmation='Require joint exact/inclusive efficiency improvement with paired category-stratified collision uncertainty and no continuum fake-B degradation; zero successes or only shallow proxy gains do not establish efficiency benefit. Preserve every original diagnostic gate.',
        historical_basis=decision['rationale'],execution_check='Both arms keep the enabled relation-bias architecture, mixed exposure and identical initialization. Verify saved auxiliary objective weights0.5/1.0, positive native auxiliary losses, target counts and actual model forwards.',budget_boundary='Two4376-update/280064-slot tasks, zero extra pretraining, oneH100NVL each,8CPU/64GiB/36h, no requeue. Same architecture, not guaranteed equal realized compute.',stop_rule='Review this single bounded objective pair once. No dose sweep, automatic successor, sealed test or model promotion.',
        uncertainty='Paired category-stratified collision intervals; report sparse-success bounds, unavailable truth and training-seed limitations. No cross-cohort pooling.')
    p['runtime_variation_boundary']['policy']='Shared enabled-bias architecture and seed20261007; only auxiliary_teacher_weight differs. Generated contexts can diverge after objective-dependent updates; no bitwise-training-prefix or equal-FLOP claim. Shared capacity repair bars causal Phase70/71 comparison.'
    p['evaluation_contract']['coverage_classification']='HISTORICAL_STYLE_DIAGNOSTIC_NOT_FINAL_POLICY_COVERAGE'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase71_20261006.json',p)
    # Single combined reservation includes selection, so future users cannot accidentally reuse it.
    reserved=PRIVATE/'reservation.json';write(reserved,{'event_uids':primary+selection,'primary_policy_events':12000,'checkpoint_selection_events':1000,'by_category':groups,'sealed_test_accessed':False})
    print(json.dumps({'preregistration':str(ROOT/'configs/reconstruction/ht_reconstruction_phase71_20261006.json'),'reservation':binding(reserved),'remaining':62000}))


if __name__=='__main__':build()
