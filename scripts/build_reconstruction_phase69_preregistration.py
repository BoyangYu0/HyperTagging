"""Reserve one independent-seed confirmation of the Phase68 hybrid encoder contrast."""
from pathlib import Path
import copy, datetime, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256, uid_set_sha256

def binding(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def build():
    prior=ROOT/'configs/reconstruction/ht_reconstruction_phase68_20261004.json'
    p=json.loads(prior.read_text());oldpath=ROOT/p['validation_cohort']['path'];old=json.loads(oldpath.read_text())
    history=ROOT/old['source_bindings']['previous_validation_universe']['path']
    used=set(json.loads(history.read_text())['event_uids'])|set(old['event_uids'])|set(old['checkpoint_selection_event_uids'])
    assert len(used)==58800
    universe=set(json.loads((ROOT/old['source_bindings']['validation_universe']['path']).read_text())['event_uids'])
    fresh=ranked(universe-used,20261005);selection,strict=fresh[:1000],fresh[1000:1100]
    ledger=ROOT/'configs/reconstruction/ht_reconstruction_phase69_used_validation_20261005.json'
    write(ledger,{'version':'validation-selection-ledger-v1','event_uids':sorted(used),'event_uid_count':len(used),'event_uids_sha256':uid_set_sha256(used),'source_bindings':[binding(history),binding(oldpath)],'sealed_test_accessed':False})
    c=copy.deepcopy(old);c.pop('remaining_untouched_after_phase68')
    c.update(manifest_version='hypertagging-reconstruction-phase69-cohort-v1',study_id='phase69-encoder-transfer-20261005',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261005,selection_original_seed=20261005,strict_selection_seed=20261005,remaining_untouched_after_phase69=40100,historical_used_event_uid_count=58800,checkpoint_selection_event_uids=selection,checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selection),evaluation_event_uids=strict,event_uids=strict,evaluation_event_uids_sha256=uid_sequence_sha256(strict),event_uids_sha256=uid_sequence_sha256(strict),validation_exclusion_event_uids_sha256=uid_set_sha256(universe-set(selection)),selection_reuse_policy='Fresh 1000 selection and 100 strict; excludes all 58800 prior reservations.',overlap_audit={'strict_vs_history':len(set(strict)&used),'selection_vs_history':len(set(selection)&used),'strict_vs_selection':len(set(strict)&set(selection))})
    c['source_bindings']['previous_validation_universe']=binding(ledger);c['source_bindings']['phase68_scored_cohort']=binding(oldpath)
    cp=ROOT/'configs/reconstruction/ht_reconstruction_phase69_validation_cohort_20261005.json';write(cp,c)
    p.update(study_id=c['study_id'],preregistration_version='hypertagging-reconstruction-phase69-preregistration-v1',created_at=c['created_at'],scientific_question='Does the incremental encoder-refinement contrast reproduce across an independent training seed and fresh validation cohort, including its exploratory nontrivial source-recall signal, without conflating shared refined PID with no pretraining?')
    p['parent_phase68']=binding(prior)
    p['common_training_contract']['seed']=20261005;p['common_training_contract']['balanced_level_replay_contract']['seed']=20261005
    close=ROOT/'artifacts/codex/reconstruction_phase68_closeout_20261005.json'
    p.pop('phase67_closeout_basis');p.pop('phase67_retained_metric_basis')
    p['phase68_closeout_basis']={**binding(close),'classification':'completed_pair_both_fail_original_gates','selected_next_factor':'encoder_refinement_transfer','sealed_test_accessed':False}
    p['phase68_retained_metric_basis']={**binding(close),'version':'phase68-closeout-v1'}
    p['validation_cohort'].update(binding(cp),checkpoint_selection_event_uids_sha256=c['checkpoint_selection_event_uids_sha256'],evaluation_event_uids_sha256=c['event_uids_sha256'])
    p['validation_budget'].pop('remaining_untouched_after_phase68');p['validation_budget'].update(previously_used=58800,remaining_untouched_after_phase69=40100)
    p['decision_rules'].update(authority='User explicitly authorized one bounded next campaign on 2026-10-05.',stop_dose_tuning='Exactly one two-arm independent-seed hybrid encoder confirmation. No automatic successor, dose sweep, promotion, sealed test or longer budget.',execution_check='Both execute mixed contexts and 4376 finite updates; verify encoder-only initial contrast with identical PID/buffers. No bitwise common-prefix claim.',confirmation='Primary full retained LCAG and exact nontrivial components must improve jointly with nontrivial source precision/recall and nontrivial coherent forests, pass all original gates, and report depth>=2. Otherwise no useful-transfer claim. Auxiliary-only and isolated-leaf gains do not qualify. End this confirmation campaign after review; no automatic repetition.')
    p['scientific_source_boundary']['limitation']='Second seed/cohort hybrid encoder contrast; shared refined PID and normalization. Does not establish all pretraining benefit or identify a data-size effect. Cross-cohort outcomes are descriptive, never pooled as one causal effect.'
    p['runtime_variation_boundary']['policy']='Seed20261005 shared within pair; independent of Phase68 seed20261004. No exact common-prefix claim. Identical downstream configuration with differing encoder parameters only.'
    write(ROOT/'configs/reconstruction/ht_reconstruction_phase69_20261005.json',p)

if __name__=='__main__':build()
