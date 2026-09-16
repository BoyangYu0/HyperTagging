#!/usr/bin/env python3
"""Preregister one bounded recovery-objective weight comparison."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase51_20260916.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase52_validation_cohort_20260917.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase51_closeout_20260917.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase51_retained_metrics_20260917.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert not evidence['arms']['late_pid_adaptation']['all_gates_passed']
    assert not evidence['arms']['late_adaptation_control']['all_gates_passed']
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    for key in ('parent_phase50','phase50_closeout_basis','phase50_retained_metric_basis'):
        p.pop(key)
    p.update(study_id='phase52-recovery-objective-20260917',
        preregistration_version='hypertagging-reconstruction-phase52-preregistration-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question='At fixed 70000 events and frozen PID, does increasing the existing recovery-objective weight from 2 to 4 improve recursive full/half reconstruction without degrading source precision?')
    p['gradient_execution_contract'] = {'autocast_weight_cache': False, 'trainable_pid_supervision_requires_gradient': True, 'phase48_pid_adaptation_effect_was_not_executed': True}
    p['common_training_contract']['seed'] = 20260919
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260919
    control=copy.deepcopy(p['arms'][0]); control['overrides']['freeze_leaf_pid_head_steps']=4376
    control['hypothesis']='Reference late encoder adaptation with the historical PID head frozen.'
    candidate=copy.deepcopy(control); candidate['role']='stronger_recovery'
    candidate['overrides']['recovery_objective_weight']=4.0
    candidate['hypothesis']='Double the existing object-presence recovery term for targets missing from predicted context. This tests a downstream objective bottleneck; it cannot repair structurally incompatible targets.'
    p['arms']=[control,candidate]
    capacity = p['capacity_admission']['reports_by_arm']
    capacity['stronger_recovery'] = capacity.pop('late_pid_adaptation')
    p['parent_phase51']=binding(previous)
    p['phase51_closeout_basis']={**binding(evidence_path),'classification':'completed_corrected_source_no_promotion','selected_next_factor':'recovery_objective_weight','sealed_test_accessed':False}
    p['phase51_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].pop('pid_adaptation_beneficial',None)
    p['decision_rules'].update(
        recovery_beneficial='Jointly inspect primary recovery, original and retained full/half topology, exact nontrivial components, source precision/recall and coherent forests. Unchanged strict gates remain necessary; no automatic promotion.',
        uncertainty='Paired event bootstrap conditional on trained models. Seed and cohort differ across campaigns, so compare within-campaign effects. This is exploratory, with no multiplicity correction.',
        dataset_size='Hold 70000 events: previous scaling confounded data, steps and cohort and supplies no controlled learning curve.',
        pretraining='Longer pretraining has no established benefit. Corrected representation pretraining remains a worthwhile later controlled experiment, but existing studies do not establish that it is more beneficial. Test the cheaper downstream recovery-objective contrast first.',
        authority='User authorized Phase51 review, publication and exactly one next bounded campaign on 2026-09-17; two 4376-step reconstruction arms, no campaign chain.')
    p['scientific_source_boundary']['limitation']='Same corrected source, train-only statistics, historical encoder81096 and late encoder adaptation in both arms. PID remains frozen in both. Only recovery-objective weight differs (2 versus 4). No physical mother momentum-resolution or repaired representability claim.'
    p['untouched_validation_cohort']={k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path),sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed']==cohort['seed']
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase52_20260917.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
