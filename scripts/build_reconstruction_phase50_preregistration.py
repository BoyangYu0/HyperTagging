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
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase49_20260914.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase50_validation_cohort_20260915.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase49_closeout_20260915.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase49_retained_metrics_20260915.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert evidence['arms']['late_pid_adaptation']['all_gates_passed']
    assert not evidence['arms']['late_adaptation_control']['all_gates_passed']
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    for key in ('parent_phase48','phase48_closeout_basis','phase48_retained_metric_basis'):
        p.pop(key)
    p.update(study_id='phase50-pid-adaptation-replication-20260915',
        preregistration_version='hypertagging-reconstruction-phase50-preregistration-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question='Does the Phase49 late PID adaptation signal replicate with a new training seed and untouched validation cohort: does unfreezing the transferred detector PID head after step 2188 improve strict topology and primary source-set plus mother-PID recovery versus keeping it frozen, with the same late encoder adaptation?')
    p['gradient_execution_contract'] = {'autocast_weight_cache': False, 'trainable_pid_supervision_requires_gradient': True, 'phase48_pid_adaptation_effect_was_not_executed': True}
    p['common_training_contract']['seed'] = 20260917
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260917
    control=copy.deepcopy(p['arms'][0]); control['overrides']['freeze_leaf_pid_head_steps']=4376
    control['hypothesis']='Reference late encoder adaptation with the historical PID head frozen.'
    candidate=copy.deepcopy(control); candidate['role']='late_pid_adaptation'
    candidate['overrides']['freeze_leaf_pid_head_steps']=2188
    candidate['hypothesis']='Adapt the historical PID head to corrected downstream detector context after decoder warmup; test whether PID-dependent quantities and strict topology improve.'
    p['arms']=[control,candidate]
    p['parent_phase49']=binding(previous)
    p['phase49_closeout_basis']={**binding(evidence_path),'classification':'completed_corrected_source_no_promotion','selected_next_factor':'late_pid_head_adaptation','sealed_test_accessed':False}
    p['phase49_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].pop('adaptation_beneficial',None)
    p['decision_rules'].update(
        pid_adaptation_beneficial='Jointly inspect primary recovery, detector PID diagnostics, all-retained full/half LCAG and exact nontrivial components, source precision/recall, and coherent forests. All unchanged strict gates remain necessary before promotion consideration; no automatic promotion.',
        uncertainty='Independent-seed replication of the Phase49 factor with paired event bootstrap within each campaign. Cross-phase seed and validation cohort both differ; compare within-campaign effects, not raw cross-phase rates. Two seeds still cannot establish robust training-seed uncertainty.',
        dataset_size='Hold 70000 events and authenticated train-only statistics. Earlier scaling confounded data, steps and cohort; no controlled learning curve demonstrates data limitation.',
        pretraining='Do not increase pretraining epochs: Phases34/42 do not establish a benefit. Corrected pretraining remains unmeasured. This bounded independent-seed replication tests the repeatability of downstream PID adaptation, not corrected pretraining or a remedy for structural target incompatibility.',
        authority='User authorized Phase49 complete review, publication and exactly one next bounded campaign on 2026-09-15; two 4376-step reconstruction arms, no campaign chain.')
    p['scientific_source_boundary']['limitation']='Same gradient-safe source, train-only statistics and historical encoder81096 as Phase49; effective PID gradients were verified in Phase49. This independent-seed replication keeps both arm contracts unchanged. Both arms use late encoder adaptation; only PID-head unfreeze timing differs. No physical mother p4-resolution or repaired representability claim.'
    p['untouched_validation_cohort']={k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path),sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed']==cohort['seed']
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase50_20260915.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
