#!/usr/bin/env python3
"""Preregister corrected-source late adaptation versus a frozen encoder."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hypertagging.data.capacity import production_capacity_report  # noqa: E402
from hypertagging.training.model_config import MODEL_PRESETS  # noqa: E402


def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase45_20260911.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase46_validation_cohort_20260912.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase45_closeout_20260912.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase45_retained_metrics_20260912.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert all(not a['all_gates_passed'] for a in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    p['study_id'] = 'phase46-corrected-frozen-encoder-20260912'
    p['preregistration_version'] = 'hypertagging-reconstruction-phase46-preregistration-v1'
    p['created_at'] = datetime.now(timezone.utc).isoformat()
    p['common_training_contract']['seed'] = 20260913
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260913
    index_path = ROOT/'artifacts/experiment_readiness/reconstruction_phase46_20260912/train_070k.complete_only.index.json'
    index = json.loads(index_path.read_text())
    assert index['normalizer_scope'] == 'train' and index['event_identity_validation']['sealed_test_opened'] is False
    p['data_binding']['dataset_index'] = str(index_path.relative_to(ROOT))
    p['data_binding']['dataset_index_sha256'] = binding(index_path)['sha256']
    p['scientific_question'] = 'With corrected reconstructed targets and fresh train-only statistics, does late encoder adaptation improve strict hierarchy reconstruction relative to keeping the same pretrained encoder frozen?'
    for key in ('phase44_closeout_basis', 'phase44_retained_metric_basis', 'parent_phase44'):
        p.pop(key)
    p['parent_phase45'] = binding(previous)
    p['phase45_closeout_basis'] = {**binding(evidence_path), 'classification': 'completed_pre_audit_source_no_promotion', 'selected_next_factor': 'corrected_late_adaptation_vs_frozen_encoder', 'sealed_test_accessed': False}
    p['phase45_retained_metric_basis'] = {**binding(retained_path), 'version': retained['version'], 'evaluator_revision': retained['evaluator_revision']}
    p['scientific_source_boundary'] = {'integrated_audit_revision': 'b67223c', 'both_arms_use_corrected_training_and_inference': True, 'fresh_train_only_normalization_required': True, 'resume_forbidden': True, 'historical_pretrained_weights_unchanged': True, 'cross_phase_causal_comparison_valid': False, 'limitation': 'Historical encoder weights were learned before target/charge/corruption audit repairs. This measures their downstream adaptation under corrected code, not a corrected pretraining objective.'}
    p['fresh_statistics_binding'] = binding(ROOT/'artifacts/experiment_readiness/reconstruction_phase46_20260912/fresh-statistics-audit.json')
    p['decision_rules'] = {
        'primary': 'micro_complete_target_efficiency',
        'adaptation_beneficial': 'Late adaptation must improve the primary and pass every unchanged strict gate; review retained topology jointly. Neither arm is promoted automatically.',
        'retained_tree_review': 'All explicit full/half roots and all returned beam candidates must be scored; incompatible roots are failed trials, not skipped.',
        'uncertainty': 'Single-seed paired exploratory comparison, with independent selection and strict cohorts; new source and cohort prevent causal cross-phase claims.',
        'dataset_size': 'Hold 70000 events. No controlled learning curve establishes data limitation; repair correctness before scaling.',
        'pretraining': 'Hold historical checkpoint81096 to isolate adaptation. Correcting target/physics/corruption supervision is higher priority than more epochs. Longer pretraining alone was not supported by Phase42; benefit from a newly trained corrected encoder remains unmeasured.',
        'authority': 'User requested Phase45 review, publication and next trainings on 2026-09-12; two bounded reconstruction arms only.'}
    template = copy.deepcopy(p['arms'][0])
    common = p['common_training_contract']
    preset = MODEL_PRESETS[common['model_preset']]
    capacity = production_capacity_report(index, global_n_queries=preset.n_queries,
        global_max_cardinality=common['max_cardinality'],
        n_queries_by_level=dict(common['n_queries_by_level']),
        max_cardinality_by_level=dict(common['max_cardinality_by_level']), target_policy='complete_only')
    assert capacity['production_training_allowed'] and capacity['query_overflow_count'] == capacity['cardinality_overflow_count'] == 0
    p['capacity_admission'].update(dataset_index=str(index_path.relative_to(ROOT)), dataset_index_sha256=binding(index_path)['sha256'])
    p['arms'], p['capacity_admission']['reports_by_arm'] = [], {}
    for role, frozen in [('late_adaptation_control',2188), ('frozen_encoder',4376)]:
        arm = copy.deepcopy(template)
        arm.update(role=role, hypothesis='Measure task-specific encoder adaptation under corrected supervision.' if frozen == 2188 else 'Measure fixed pretrained representation under the same corrected supervision.')
        arm['overrides'].update(encoder_lr_multiplier=0.05, freeze_pretrained_encoder_steps=frozen)
        p['arms'].append(arm)
        p['capacity_admission']['reports_by_arm'][role] = copy.deepcopy(capacity)
    p['untouched_validation_cohort'] = {k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path), sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed'] == p['common_training_contract']['balanced_level_replay_contract']['seed'] == cohort['seed']
    (ROOT/'configs/reconstruction/ht_reconstruction_phase46_20260912.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
