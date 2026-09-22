#!/usr/bin/env python3
"""Preregister matched late leaf-PID objective balance, with unchanged fail guards."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous=ROOT/'configs/reconstruction/ht_reconstruction_phase56_20260921.json'
    cohort_path=ROOT/'configs/reconstruction/ht_reconstruction_phase57_validation_cohort_20260922.json'
    evidence_path=ROOT/'artifacts/codex/reconstruction_phase56_closeout_20260922.json'
    retained_path=ROOT/'artifacts/codex/reconstruction_phase56_retained_metrics_20260922.json'
    p,cohort,evidence,retained=[json.loads(q.read_text()) for q in (previous,cohort_path,evidence_path,retained_path)]
    assert evidence['status']=='INCOMPLETE_COMPARISON' and evidence['metric_completeness']=='COMPLETE_AVAILABLE_ARTIFACTS'
    assert retained['all_returned_beam_candidates_checked'] and not evidence['comparison_available']
    for key in ('parent_phase55','phase55_closeout_basis','phase55_retained_metric_basis','untouched_validation_cohort'):p.pop(key)
    p.update(study_id='phase57-pretraining-objective-balance-20260922',preregistration_version='hypertagging-reconstruction-phase57-preregistration-v1',created_at=datetime.now(timezone.utc).isoformat(),scientific_question='Does reducing late leaf-PID objective weight from 0.4 to 0.2 improve balanced representation and downstream reconstruction at fixed parent weight 2, data and compute?')
    c=p['common_training_contract'];c['seed']=20260924;c['balanced_level_replay_contract']['seed']=20260924
    pre=p['pretraining_refinement'];pre['config']['seed']=20260924;pre['config']['parent_ranking_weight']=2.0;pre['single_arm_difference']='leaf_pid_phase_weights_after_step_1094'
    pre['comparison_limitation']='Matched objective balance study; parent weight 2 is a completed reference, not a proven winner. Selection reuses exact Phase56 1000 events; strict 100 are fresh. No claim of independent cross-phase selection replication.'
    old_roles=[a['role'] for a in p['arms']]
    roles=['pretraining_balance_control','lower_late_pid_pretraining']
    for arm,role,late in zip(p['arms'],roles,[0.4,0.2]):
        arm['role']=role
        arm['pretraining_parent_ranking_weight']=2.0
        arm['pretraining_leaf_pid_phase_weights']=[1.0,1.0,late,late]
        arm['label']=role
        arm['hypothesis']=f'Late leaf-PID phase weight {late} with parent weight 2, followed by identical reconstruction.'
    p['capacity_admission']['reports_by_arm']={new:copy.deepcopy(p['capacity_admission']['reports_by_arm'][old]) for old,new in zip(old_roles,roles)}
    p['scientific_source_boundary']['limitation']='Both arms initialize original encoder81096 parameters with fresh train-only normalization, optimizer, schedule, RNG and memory. Only late pretraining leaf-PID phase weight differs. Parent weight is 2 in both, fixed final refined checkpoints feed identical reconstruction. No physical momentum-resolution or representability repair claim.'
    p['parent_phase56']=binding(previous)
    p['phase56_closeout_basis']={**binding(evidence_path),'classification':'incomplete_control_objective_preflight_failure_no_promotion','selected_next_factor':'late_pretraining_leaf_pid_phase_weights','sealed_test_accessed':False}
    p['phase56_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].update(stop_dose_tuning='Stop parent-dose tuning: Phase56 control attrition prevents replication. This is one bounded objective-balance test, not an automatic campaign.',dataset_size='Hold 70000 training events; no controlled size learning curve establishes an immediate growth benefit. Plan additional validation capacity separately.',pretraining='Late leaf-PID phase weights 0.4 versus 0.2 with parent weight 2 in both arms; all other objectives, steps and reconstruction frozen. Dominance threshold 20 with fail action is unchanged.',authority='User authorized Phase56 review, publication and one next bounded training pair on 2026-09-22. No promotion, sealed-test access or automatic chain.')
    p['validation_budget']={'validation_role_events':50000,'previously_excluded':49609,'newly_reserved':100,'reused_selection_events':1000,'selection_events':1000,'rollout_selection_events':1000,'strict_events':100,'remaining_untouched_after_phase57':291,'limitation':'Exact Phase56 selection reused with fresh strict cohort; not independent selection replication.'}
    p['validation_cohort']={**binding(cohort_path),**{k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero','permitted_selection_reuse_count')},'sealed_test_role_access':'forbidden'}
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase57_20260922.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
