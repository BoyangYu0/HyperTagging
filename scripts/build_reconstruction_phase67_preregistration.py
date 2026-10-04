"""Reserve a bounded context-exposure diagnostic after the completed Phase66 pair."""
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
    oldpath=ROOT/'configs/reconstruction/ht_reconstruction_phase66_20261003.json'
    p=json.loads(oldpath.read_text())
    oldcp=ROOT/p['validation_cohort']['path']; old=json.loads(oldcp.read_text())
    hist=ROOT/old['source_bindings']['previous_validation_universe']['path']
    used=set(json.loads(hist.read_text())['event_uids'])|set(old['event_uids'])|set(old['checkpoint_selection_event_uids'])
    assert len(used)==56600
    universe=set(json.loads((ROOT/old['source_bindings']['validation_universe']['path']).read_text())['event_uids'])
    fresh=ranked(universe-used,20261004); selection,strict=fresh[:1000],fresh[1000:1100]
    ledger=ROOT/'configs/reconstruction/ht_reconstruction_phase67_used_validation_20261004.json'
    write(ledger,{'version':'validation-selection-ledger-v1','event_uids':sorted(used),'event_uid_count':len(used),'event_uids_sha256':uid_set_sha256(used),'source_bindings':[binding(hist),binding(oldcp)],'sealed_test_accessed':False})
    c=copy.deepcopy(old); c.pop('remaining_untouched_after_phase66',None)
    study='phase67-context-exposure-20261004'
    c.update(manifest_version='hypertagging-reconstruction-phase67-cohort-v1',study_id=study,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261004,selection_original_seed=20261004,strict_selection_seed=20261004,remaining_untouched_after_phase67=42300,historical_used_event_uid_count=56600,checkpoint_selection_event_uids=selection,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),evaluation_event_uids=strict,event_uids=strict,evaluation_event_uids_sha256=uid_sequence_sha256(strict),event_uids_sha256=uid_sequence_sha256(strict),validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selection)),selection_reuse_policy='Fresh 1000 selection and 100 strict; excludes all 56600 prior reservations.',overlap_audit={'strict_vs_history':len(set(strict)&used),'selection_vs_history':len(set(selection)&used),'strict_vs_selection':len(set(strict)&set(selection))})
    c['source_bindings']['previous_validation_universe']=binding(ledger)
    c['source_bindings']['phase66_scored_cohort']=binding(oldcp)
    cp=ROOT/'configs/reconstruction/ht_reconstruction_phase67_validation_cohort_20261004.json';write(cp,c)
    p.update(study_id=study,preregistration_version='hypertagging-reconstruction-phase67-preregistration-v1',created_at=c['created_at'],pilot_classification='CONTEXT_EXPOSURE_PILOT',scientific_question='Does the Phase66 context-regime precision/recall tradeoff replicate under seed20261004 and fresh validation, and does either regime establish joint nontrivial recursive improvement?')
    for key in ('parent_phase64','phase64_closeout_basis','phase64_retained_metric_basis'): p.pop(key,None)
    p.pop('parent_phase65',None)
    p.pop('phase65_closeout_basis',None)
    p.pop('phase65_retained_metric_basis',None)
    p['parent_phase66']=binding(oldpath)
    common=p['common_training_contract']; common['seed']=20261004;common['balanced_level_replay_contract']['seed']=20261004
    # Legacy nominal slot metadata used half of the actual replay budget. Replace
    # it with per-arm expectations calculated from all 4376*64 replay slots.
    for key in list(common):
        if key.startswith('nominal_expected_'): common.pop(key)
    template=copy.deepcopy(p['arms'][0]);capacity=copy.deepcopy(p['capacity_admission']['reports_by_arm']['scheduled_mixed'])
    p['arms']=[]
    for role,prob in [('scheduled_mixed',0.5),('teacher_only',0.0)]:
        arm=copy.deepcopy(template);arm.update(role=role,label=role,hypothesis='Mechanistic context-exposure diagnostic, not evidence that teacher forcing or pretraining is a quality winner.')
        arm['overrides']['scheduled_sampling_probability']=prob
        arm['overrides']['freeze_pretrained_encoder_steps']=2188
        p['arms'].append(arm)
    p['capacity_admission']['reports_by_arm']={a['role']:copy.deepcopy(capacity) for a in p['arms']}
    p['exposure_expectations']={}
    for a in p['arms']:
        prob=a['overrides']['scheduled_sampling_probability']
        predicted=sum(prob*min(step/2188,1.0)*64 for step in range(4376))
        p['exposure_expectations'][a['role']]={'replay_slots':280064,'nominal_predicted_slots':predicted,'nominal_teacher_slots':280064-predicted,'scope':'All primary event-level replay slots, before any context eligibility effect.'}
    close=ROOT/'artifacts/codex/reconstruction_phase66_closeout_20261004.json'
    p['phase66_closeout_basis']={**binding(close),'classification':'completed_pair_both_fail_original_gates','selected_next_factor':'training_context_exposure','sealed_test_accessed':False}
    p['phase66_retained_metric_basis']={**binding(close),'version':'phase66-closeout-v1'}
    p['validation_cohort'].update(binding(cp),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget'].pop('remaining_untouched_after_phase66',None);p['validation_budget'].update(previously_used=56600,remaining_untouched_after_phase67=42300)
    p['decision_rules'].update(authority='User explicitly authorized one bounded next campaign on 2026-10-04.',stop_dose_tuning='Exactly two confirmation jobs; stop this context-regime campaign after this pair. No automatic successor, sweep, promotion, sealed test or longer budget.',representation_beneficial='No representation or data-size causal claim. Require all original gates plus joint nontrivial topology/source gains and nontrivial coherent forests, report depth>=2 and paired uncertainty.',context_hypotheses='Teacher-only may preserve targets lost from generated contexts, but may worsen inference exposure mismatch. Report both outcomes without post-hoc selection.',execution_check='Teacher-only must log zero sampled predicted contexts and zero rollout/auxiliary teacher branches. Mixed must execute predicted contexts. Count representable targets, finite steps and actual forwards.',budget_boundary='Equal optimizer steps and replay slots, not equal FLOPs: teacher-only avoids rollout and the conditional auxiliary teacher branch. Auxiliary weight stays 0.5 but is inactive in teacher-only; this is intrinsic to the context-regime contrast.')
    p['scientific_source_boundary']['limitation']='Single-seed context-regime diagnostic; no causal comparison of data growth or pretraining improvements; no bitwise common-prefix claim.'
    p['runtime_variation_boundary']['policy']='Same authenticated initial encoder, fixed PID head, fresh decoder/training/replay/cohort seed20261004; conditional paired event intervals omit seed variation.'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase67_20261004.json',p)

if __name__=='__main__':build()
