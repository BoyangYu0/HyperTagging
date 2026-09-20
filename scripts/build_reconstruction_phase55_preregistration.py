#!/usr/bin/env python3
"""Preregister one matched parent-ranking pretraining-quality experiment."""
import copy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.build_reconstruction_phase46_preregistration import binding
from hypertagging.training.pretrain_trainer import PretrainConfig


def main():
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase54_20260918.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase55_validation_cohort_20260920.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase54_closeout_20260920.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase54_retained_metrics_20260920.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert all(not a['all_gates_passed'] for a in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    for key in ('parent_phase53','phase53_closeout_basis','phase53_retained_metric_basis'): p.pop(key)
    p.update(study_id='phase55-pretraining-parent-quality-20260920',
        preregistration_version='hypertagging-reconstruction-phase55-preregistration-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question='At fixed 70000 events and matched compute, does increasing direct-parent ranking weight during parameter-initialized pretraining refinement improve downstream recursive reconstruction?')
    p['common_training_contract']['seed'] = 20260922
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260922
    control=copy.deepcopy(p['arms'][0]); control['pretraining_parent_ranking_weight']=1.0
    control['hypothesis']='Reference parent-ranking weight 1 during matched corrected pretraining refinement, followed by identical late encoder adaptation.'
    candidate=copy.deepcopy(control); candidate['role']='stronger_parent_pretraining'; candidate['pretraining_parent_ranking_weight']=2.0
    candidate['hypothesis']='Increase direct-parent ranking weight to 2 during pretraining refinement to test whether more direct structural supervision transfers to recursive assembly.'
    p['arms']=[control,candidate]
    capacity=p['capacity_admission']['reports_by_arm']; capacity['stronger_parent_pretraining']=capacity.pop('weaker_recovery')
    raw=yaml.safe_load((ROOT/'configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml').read_text())
    allowed={f.name for f in fields(PretrainConfig)}
    c=asdict(PretrainConfig(data=p['data_binding']['selection_manifest'],output_dir='runtime_bound'))
    c.update({k:v for k,v in raw.items() if k in allowed})
    c.update(data=p['data_binding']['selection_manifest'], dataset_index=p['data_binding']['dataset_index'],
        output_dir='runtime_bound', max_steps=2188, lr_schedule_total_steps=2188, batch_size=32, seed=20260922,
        learning_rate=0.00005, warmup_fraction=0.05, min_lr_ratio=0.05, log_every=1,
        curriculum_phase_steps=[547]*4, checkpoint_every=1094, validate_every=1094,
        validation_events=1000, validation_batches=32, objective_gradient_diagnostics_every=547,
        resume=None, weights_initialization_checkpoint=None, validation_event_uids=[], parent_ranking_weight=1.0)
    p['pretraining_refinement']={'config':c,'checkpoint_for_reconstruction':'fixed_final_step_2188',
        'initialization':'parameters_only_fresh_train_normalization_optimizer_schedule_rng_and_memory',
        'training_presentations':70016,'validation_population':'first_1000_of_shared_checkpoint_selection_cohort',
        'single_arm_difference':'parent_ranking_weight','initial_checkpoint_step':81096,
        'comparison_limitation':'Both arms receive refinement; this does not compare refinement with no refinement. One training seed and a small strict cohort do not establish general superiority.'}
    p['parent_phase54']=binding(previous)
    p['phase54_closeout_basis']={**binding(evidence_path),'classification':'completed_corrected_source_no_promotion','selected_next_factor':'pretraining_parent_ranking_weight','sealed_test_accessed':False}
    p['phase54_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].pop('recovery_beneficial',None)
    p['decision_rules'].update(
        representation_beneficial='Compare matched arms on primary recovery, original and retained full/half topology, exact nontrivial components, source precision/recall and nontrivial coherent forests. Original gates remain necessary; no automatic promotion. Pretraining loss alone cannot establish transfer benefit.',
        uncertainty='Paired event bootstrap conditional on selected models; one seed, exploratory and no multiplicity correction. Different phase cohorts prevent causal cross-phase rate comparisons.',
        dataset_size='Hold 70000 events in both refinement and reconstruction. No controlled learning curve establishes an immediate data-growth benefit.',
        pretraining='Stop recovery-dose tuning after Phase54 failed to reproduce joint benefit. Test one pretraining-quality factor with matched 2188-step refinement and identical 4376-step downstream reconstruction. Do not merely extend historical pretraining duration.',
        authority='User authorized Phase54 review, publication and exactly one next bounded campaign on 2026-09-20; two jobs each containing 2188 pretraining and 4376 reconstruction steps, no campaign chain.')
    p['scientific_source_boundary'].pop('historical_pretrained_weights_unchanged', None)
    p['scientific_source_boundary']['historical_source_checkpoint_file_unchanged'] = True
    p['validation_budget'] = {'validation_role_events':50000, 'previously_excluded':46409, 'newly_reserved':2100, 'remaining_untouched_after_phase55':1491, 'another_fresh_2100_event_cohort_available':False}
    p['scientific_source_boundary']['limitation']='Both arms initialize parameters from encoder81096 with fresh corrected train-only normalization and fresh optimizer, schedule, RNG and channel memory. Only pretraining parent-ranking weight differs. Both use fixed final refined checkpoints and identical reconstruction configs, including recovery weight 2 and frozen transferred PID. No repaired representability or physical mother momentum-resolution claim.'
    p['untouched_validation_cohort']={k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path),sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed']==cohort['seed']
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase55_20260920.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
