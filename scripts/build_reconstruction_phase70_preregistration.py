"""Reserve one bounded decoder relation-bias ablation after Phase69 review."""
from pathlib import Path
import copy, datetime, hashlib, json, sys, subprocess, getpass
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256, uid_set_sha256

def binding(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def build():
    queue=subprocess.check_output(['squeue','--noheader','--user',getpass.getuser(),'--format=%j|%k'],text=True)
    if 'phase70' in queue: raise RuntimeError('Concurrent Phase70 campaign already exists')
    if list((ROOT/'artifacts/codex').glob('*phase70*submission*')): raise RuntimeError('Phase70 submission already recorded')
    prior=ROOT/'configs/reconstruction/ht_reconstruction_phase69_20261005.json'
    p=json.loads(prior.read_text());oldpath=ROOT/p['validation_cohort']['path'];old=json.loads(oldpath.read_text())
    history=ROOT/old['source_bindings']['previous_validation_universe']['path']
    used=set(json.loads(history.read_text())['event_uids'])|set(old['event_uids'])|set(old['checkpoint_selection_event_uids'])
    assert len(used)==59900
    universe=set(json.loads((ROOT/old['source_bindings']['validation_universe']['path']).read_text())['event_uids'])
    fresh=ranked(universe-used,20261006);selection,strict=fresh[:1000],fresh[1000:1100]
    assert len(set(selection+strict))==1100 and not set(selection+strict)&used
    ledger=ROOT/'configs/reconstruction/ht_reconstruction_phase70_used_validation_20261005.json'
    write(ledger,{'version':'validation-selection-ledger-v1','event_uids':sorted(used),'event_uid_count':len(used),'event_uids_sha256':uid_set_sha256(used),'source_bindings':[binding(history),binding(oldpath)],'sealed_test_accessed':False})
    c=copy.deepcopy(old);c.pop('remaining_untouched_after_phase69')
    c.update(manifest_version='hypertagging-reconstruction-phase70-cohort-v1',study_id='phase70-type-relation-bias-20261005',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261006,selection_original_seed=20261006,strict_selection_seed=20261006,remaining_untouched_after_phase70=39000,historical_used_event_uid_count=59900,checkpoint_selection_event_uids=selection,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),evaluation_event_uids=strict,event_uids=strict,evaluation_event_uids_sha256=uid_sequence_sha256(strict),event_uids_sha256=uid_sequence_sha256(strict),validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selection)),selection_reuse_policy='Fresh 1000 selection and 100 strict; excludes all 59900 prior reservations.',overlap_audit={'strict_vs_history':len(set(strict)&used),'selection_vs_history':len(set(selection)&used),'strict_vs_selection':len(set(strict)&set(selection))})
    c['source_bindings']['previous_validation_universe']=binding(ledger);c['source_bindings']['phase69_scored_cohort']=binding(oldpath)
    cp=ROOT/'configs/reconstruction/ht_reconstruction_phase70_validation_cohort_20261005.json';write(cp,c)
    p.update(study_id=c['study_id'],preregistration_version='hypertagging-reconstruction-phase70-preregistration-v1',created_at=c['created_at'],scientific_question='Does the type-conditioned daughter relation-bias module improve strict recursive assembly at fixed refined encoder, PID, normalization, data, updates and inference rules?',pilot_classification='DECODER_RELATION_BIAS_PILOT')
    p['parent_phase69']=binding(prior)
    p['common_training_contract']['seed']=20261006;p['common_training_contract']['balanced_level_replay_contract']['seed']=20261006
    close=ROOT/'artifacts/codex/reconstruction_phase69_closeout_20261005.json'
    p.pop('phase68_closeout_basis');p.pop('phase68_retained_metric_basis');p.pop('encoder_intervention')
    p['phase69_closeout_basis']={**binding(close),'classification':'completed_pair_one_passes_original_gates','selected_next_factor':'decoder_type_relation_bias','sealed_test_accessed':False}
    p['phase69_retained_metric_basis']={**binding(close),'version':'phase69-closeout-v1'}
    p['validation_cohort'].update(binding(cp),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget'].pop('remaining_untouched_after_phase69');p['validation_budget'].update(previously_used=59900,remaining_untouched_after_phase70=39000)
    base=copy.deepcopy(p['arms'][0]);p['arms']=[]
    for role,enabled in [('type_bias',True),('no_type_bias',False)]:
        arm=copy.deepcopy(base);arm.update(role=role,label=role,hypothesis='The type-conditioned relation-bias module helps legal recursive assembly.' if enabled else 'Removing the module avoids a misleading learned compatibility prior; any gain must survive strict topology and source gates.')
        arm['overrides']['type_conditioned_daughter_relation_bias']=enabled;p['arms'].append(arm)
    p['capacity_admission']['reports_by_arm']={role:copy.deepcopy(p['capacity_admission']['reports_by_arm']['refined_encoder']) for role in ['type_bias','no_type_bias']}
    p['exposure_expectations']={role:copy.deepcopy(p['exposure_expectations']['refined_encoder']) for role in ['type_bias','no_type_bias']}
    p['pretraining_refinement']={'mode':'fixed_encoder_decoder_bias_ablation','checkpoint_for_reconstruction':'arm_bound_parameter_transfer','additional_pretraining_steps':0}
    p['decision_rules']={
      'authority':'User authorized exactly one bounded next campaign on 2026-10-05.',
      'primary':'micro_complete_target_efficiency',
      'confirmation':'Pass every original gate; require joint improvement in primary full retained LCAG, exact nontrivial components, nontrivial source precision and recall, and nontrivial coherent forests; report depth>=2. Auxiliary-only, oracle-only, isolated-leaf or single-metric gains do not establish usefulness.',
      'uncertainty':'Paired event-cluster intervals conditional on two fitted models; exploratory single seed without multiplicity correction or training-seed uncertainty. No cross-cohort pooling.',
      'dataset_size':'Hold70000; no controlled size effect or comparative superiority of pretraining is established.',
      'budget_boundary':'4376 updates and280064 replay slots each, zero new pretraining, two1-H100-NVL tasks capped at36h each, no requeue. Architecture changes parameter count and FLOPs; not compute-equivalent.',
      'execution_check':'Same refined encoder and PID hashes; only resolved type_conditioned_daughter_relation_bias differs. Verify module absent/present in architecture and checkpoint; preserve all runtime normalization, policy and truth-free source guards.',
      'stop_rule':'Review this pair once. No further campaign, automatic repetition, dose sweep, sealed test or scientific promotion.',
      'historical_basis':'Stage A relation-bias gain was below its threshold and lacked modern complete strict evidence. Phase68-69 hybrid refinement has no replicated joint recursive benefit; no candidate in Phase69 is exact at depth>=2. This tests decoder use of representations, not more pretraining or a selected quality winner.'}
    p['scientific_source_boundary']['limitation']='Decoder module architecture ablation. Both arms share historical refined pretraining and PID. It cannot establish all pretraining benefit, a pure geometric effect, or a data-size effect.'
    p['runtime_variation_boundary']['policy']='Shared seed20261006 and replay schedule, but module removal changes decoder parameter count and random-number consumption. No bitwise common-prefix or equal-FLOP claim. The estimand is the complete on/off architecture intervention, not one isolated bias tensor.'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase70_20261005.json',p)

if __name__=='__main__':build()
