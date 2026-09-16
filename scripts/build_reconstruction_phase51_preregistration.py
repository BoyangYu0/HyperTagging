#!/usr/bin/env python3
"""Preregister one bounded late PID-head adaptation comparison."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase50_20260915.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase51_validation_cohort_20260916.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase50_closeout_20260916.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase50_retained_metrics_20260916.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert not evidence['arms']['late_pid_adaptation']['all_gates_passed']
    assert evidence['arms']['late_adaptation_control']['all_gates_passed']
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    for key in ('parent_phase49','phase49_closeout_basis','phase49_retained_metric_basis'):
        p.pop(key)
    p.update(study_id='phase51-low-rate-pid-adaptation-20260916',
        preregistration_version='hypertagging-reconstruction-phase51-preregistration-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question='At fixed 70000 events, does a tenfold lower PID learning-rate multiplier of 0.1 when unfreezing the transferred detector PID head after step 2188 improve strict topology and primary source-set plus mother-PID recovery versus keeping it frozen, with the same late encoder adaptation?')
    p['gradient_execution_contract'] = {'autocast_weight_cache': False, 'trainable_pid_supervision_requires_gradient': True, 'phase48_pid_adaptation_effect_was_not_executed': True}
    p['common_training_contract']['seed'] = 20260918
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260918
    control=copy.deepcopy(p['arms'][0]); control['overrides']['freeze_leaf_pid_head_steps']=4376
    control['hypothesis']='Reference late encoder adaptation with the historical PID head frozen.'
    candidate=copy.deepcopy(control); candidate['role']='late_pid_adaptation'
    candidate['overrides']['freeze_leaf_pid_head_steps']=2188
    candidate['overrides']['leaf_pid_lr_multiplier']=0.1
    candidate['hypothesis']='Adapt the PID head after decoder warmup at multiplier 0.1 to test whether a smaller update preserves primary recovery while retaining topology gains. This is an exploratory dose study, not a proven remedy.'
    p['arms']=[control,candidate]
    p['parent_phase50']=binding(previous)
    p['phase50_closeout_basis']={**binding(evidence_path),'classification':'completed_corrected_source_no_promotion','selected_next_factor':'late_pid_head_adaptation','sealed_test_accessed':False}
    p['phase50_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].pop('adaptation_beneficial',None)
    p['decision_rules'].update(
        pid_adaptation_beneficial='Jointly inspect primary recovery, detector PID diagnostics, all-retained full/half LCAG and exact nontrivial components, source precision/recall, and coherent forests. All unchanged strict gates remain necessary before promotion consideration; no automatic promotion.',
        uncertainty='Prospective lower-dose PID adaptation versus frozen PID with paired event bootstrap within this campaign. Cross-phase seed and validation cohort both differ; compare within-campaign effects, not raw cross-phase rates. Cross-phase comparisons cannot isolate learning-rate dose because cohort and seed also change.',
        dataset_size='Hold 70000 events and authenticated train-only statistics. Earlier scaling confounded data, steps and cohort; no controlled learning curve demonstrates data limitation.',
        pretraining='Do not increase pretraining epochs: Phases34/42 do not establish a benefit. Corrected pretraining remains unmeasured. This bounded dose study tests gentler downstream PID adaptation, not corrected pretraining or a remedy for structural target incompatibility.',
        authority='User authorized Phase50 complete review, publication and exactly one next bounded campaign on 2026-09-16; two 4376-step reconstruction arms, no campaign chain.')
    p['scientific_source_boundary']['limitation']='Same gradient-safe source, train-only statistics and historical encoder81096 as Phase50; effective PID gradients were verified in Phase50. Both arms use late encoder adaptation. The candidate unfreezes PID at step 2188 with multiplier 0.1; control PID stays frozen. The contrast is the complete low-rate PID adaptation regimen; it does not independently identify timing and learning-rate effects. No physical mother p4-resolution or repaired representability claim.'
    p['untouched_validation_cohort']={k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path),sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed']==cohort['seed']
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase51_20260916.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
