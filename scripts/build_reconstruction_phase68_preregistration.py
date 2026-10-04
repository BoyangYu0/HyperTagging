"""Reserve a bounded context-exposure diagnostic after the completed Phase67 pair."""
from pathlib import Path
import copy, datetime, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256, uid_set_sha256


def binding(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def build():
    oldpath=ROOT/'configs/reconstruction/ht_reconstruction_phase67_20261004.json'
    p=json.loads(oldpath.read_text())
    oldcp=ROOT/p['validation_cohort']['path']; old=json.loads(oldcp.read_text())
    hist=ROOT/old['source_bindings']['previous_validation_universe']['path']
    used=set(json.loads(hist.read_text())['event_uids'])|set(old['event_uids'])|set(old['checkpoint_selection_event_uids'])
    assert len(used)==57700
    universe=set(json.loads((ROOT/old['source_bindings']['validation_universe']['path']).read_text())['event_uids'])
    fresh=ranked(universe-used,20261004); selection,strict=fresh[:1000],fresh[1000:1100]
    ledger=ROOT/'configs/reconstruction/ht_reconstruction_phase68_used_validation_20261004.json'
    write(ledger,{'version':'validation-selection-ledger-v1','event_uids':sorted(used),'event_uid_count':len(used),'event_uids_sha256':uid_set_sha256(used),'source_bindings':[binding(hist),binding(oldcp)],'sealed_test_accessed':False})
    c=copy.deepcopy(old); c.pop('remaining_untouched_after_phase67',None)
    study='phase68-encoder-transfer-20261004'
    c.update(manifest_version='hypertagging-reconstruction-phase68-cohort-v1',study_id=study,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261004,selection_original_seed=20261004,strict_selection_seed=20261004,remaining_untouched_after_phase68=41200,historical_used_event_uid_count=57700,checkpoint_selection_event_uids=selection,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),evaluation_event_uids=strict,event_uids=strict,evaluation_event_uids_sha256=uid_sequence_sha256(strict),event_uids_sha256=uid_sequence_sha256(strict),validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selection)),selection_reuse_policy='Fresh 1000 selection and 100 strict; excludes all 57700 prior reservations.',overlap_audit={'strict_vs_history':len(set(strict)&used),'selection_vs_history':len(set(selection)&used),'strict_vs_selection':len(set(strict)&set(selection))})
    c['source_bindings']['previous_validation_universe']=binding(ledger)
    c['source_bindings']['phase67_scored_cohort']=binding(oldcp)
    cp=ROOT/'configs/reconstruction/ht_reconstruction_phase68_validation_cohort_20261004.json';write(cp,c)
    p.update(study_id=study,preregistration_version='hypertagging-reconstruction-phase68-preregistration-v1',created_at=c['created_at'],pilot_classification='ENCODER_TRANSFER_PILOT',scientific_question='Does incremental Phase64 encoder refinement improve strict recursive reconstruction relative to its train-rescaled pre-refinement encoder parameters, at identical refined PID, normalization, data and downstream training?')
    for key in ('parent_phase64','phase64_closeout_basis','phase64_retained_metric_basis'): p.pop(key,None)
    p.pop('parent_phase65',None)
    p.pop('phase65_closeout_basis',None)
    p.pop('phase65_retained_metric_basis',None)
    p['parent_phase67']=binding(oldpath)
    common=p['common_training_contract']; common['seed']=20261004;common['balanced_level_replay_contract']['seed']=20261004
    # Legacy nominal slot metadata used half of the actual replay budget. Replace
    # it with per-arm expectations calculated from all 4376*64 replay slots.
    for key in list(common):
        if key.startswith('nominal_expected_'): common.pop(key)
    template=copy.deepcopy(p['arms'][0]);capacity=copy.deepcopy(p['capacity_admission']['reports_by_arm']['scheduled_mixed'])
    p['arms']=[]
    for role,prob in [('refined_encoder',0.5),('pre_refinement_encoder',0.5)]:
        arm=copy.deepcopy(template);arm.update(role=role,label=role,hypothesis='Mechanistic context-exposure diagnostic, not evidence that teacher forcing or pretraining is a quality winner.')
        arm['overrides']['scheduled_sampling_probability']=prob
        arm['overrides']['freeze_pretrained_encoder_steps']=2188
        if role == 'pre_refinement_encoder':
            receipt=json.loads((ROOT/'artifacts/codex/phase68_encoder_ablation_lineage_20261004.json').read_text())
            arm['checkpoint']='runtime_inputs/reconstruction_phase68_20261004/pre-refinement-encoder-shared-pid.pt'
            arm['checkpoint_sha256']=receipt['checkpoint_sha256']
            arm['checkpoint_step']=0
        arm['hypothesis']='Incremental encoder-refinement contribution at fixed refined PID and train-only buffers; no unpretrained or hyperbolic-versus-Euclidean claim.'
        p['arms'].append(arm)
    p['capacity_admission']['reports_by_arm']={a['role']:copy.deepcopy(capacity) for a in p['arms']}
    p['exposure_expectations']={}
    for a in p['arms']:
        prob=a['overrides']['scheduled_sampling_probability']
        predicted=sum(prob*min(step/2188,1.0)*64 for step in range(4376))
        p['exposure_expectations'][a['role']]={'replay_slots':280064,'nominal_predicted_slots':predicted,'nominal_teacher_slots':280064-predicted,'scope':'All primary event-level replay slots, before any context eligibility effect.'}
    close=ROOT/'artifacts/codex/reconstruction_phase67_closeout_20261004.json'
    p['phase67_closeout_basis']={**binding(close),'classification':'completed_pair_both_fail_original_gates','selected_next_factor':'encoder_refinement_transfer','sealed_test_accessed':False}
    p['phase67_retained_metric_basis']={**binding(close),'version':'phase67-closeout-v1'}
    p['validation_cohort'].update(binding(cp),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget'].pop('remaining_untouched_after_phase67',None);p['validation_budget'].update(previously_used=57700,remaining_untouched_after_phase68=41200)
    p['pretraining_refinement']={'mode':'encoder_refinement_transfer_ablation','checkpoint_for_reconstruction':'arm_bound_parameter_transfer','additional_pretraining_steps':0}
    p['encoder_intervention']=binding(ROOT/'artifacts/codex/phase68_encoder_ablation_lineage_20261004.json')
    p['decision_rules'].update(authority='User explicitly authorized one bounded next campaign on 2026-10-04.',stop_dose_tuning='Close context-regime tuning. Exactly two encoder-transfer jobs; no automatic successor, sweep, promotion, sealed test or longer budget.',representation_beneficial='Require all original gates plus joint nontrivial topology/source improvement and nontrivial coherent forests; report depth>=2 and paired uncertainty. Auxiliary-only or isolated-leaf gains do not validate transfer.',context_hypotheses='Both arms use mixed probability0.5 solely as a held-fixed reference, not a selected quality winner.',execution_check='Both execute mixed contexts and all4376 finite updates; verify initial encoder intervention and identical PID/buffers.',budget_boundary='Equal4376 updates/280064 replay slots and zero new pretraining. Not equal FLOPs; historical refinement compute is sunk and explicitly differs between source encoders.')
    p['scientific_source_boundary']['limitation']='Single-seed incremental encoder-refinement ablation with shared refined PID and normalization; not pretraining versus no pretraining, not a data-growth comparison, and not independent training-seed replication.'
    p['runtime_variation_boundary']['policy']='Seed20261004 shared within pair with fresh cohort. No exact common-prefix claim. Encoder parameters differ at initialization by design; all decoder/PID/policy factors remain fixed.'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase68_20261004.json',p)

if __name__=='__main__':build()
