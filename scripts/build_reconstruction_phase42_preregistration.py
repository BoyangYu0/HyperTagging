#!/usr/bin/env python3
"""Preregister a bounded downstream test of additional pretraining at fixed data size."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = ROOT / 'configs/reconstruction/ht_reconstruction_phase41_20260909.json'
    cohort_path = ROOT / 'configs/reconstruction/ht_reconstruction_phase42_validation_cohort_20260909.json'
    evidence_path = ROOT / 'artifacts/codex/reconstruction_phase41_closeout_20260909.json'
    cohort = json.loads(cohort_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    assert evidence['metric_completeness'] == 'COMPLETE'
    assert all(not arm['all_gates_passed'] for arm in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    p = json.loads(previous.read_text())
    p['study_id'] = 'phase42-pretraining-transfer-20260909'
    p['preregistration_version'] = 'hypertagging-reconstruction-phase42-preregistration-v1'
    p['created_at'] = datetime.now(timezone.utc).isoformat()
    p['scientific_question'] = 'Does extending pretraining from 81096 to 108128 steps improve strict downstream reconstruction with the current decoder, fixed 70k train data and identical 4376-step budgets?'
    p.pop('phase40r1_closeout_basis')
    p.pop('parent_phase40r1')
    p['parent_phase41'] = {'path': str(previous.relative_to(ROOT)), 'sha256': sha(previous)}
    p['phase41_closeout_basis'] = {
        'classification': 'completed_controlled_study_no_promotion',
        'selected_next_factor': 'pretraining_checkpoint_81096_vs_108128',
        'sealed_test_accessed': False,
        'path': str(evidence_path.relative_to(ROOT)), 'sha256': sha(evidence_path)}
    p['decision_rules'] = {
        'primary': 'micro_complete_target_efficiency',
        'additional_pretraining_beneficial': 'Candidate improves primary and passes every unchanged strict hierarchy gate; otherwise do not infer benefit from pretraining loss.',
        'uncertainty': 'Single training seed and 100 strict / 20 beam validation events; require later replication before promotion.',
        'dataset_size': 'Hold at 70000; previous doubling also doubled optimization steps and changed cohort, so it is not an isolated dataset-size effect.',
        'authority': 'User requested next trainings on 2026-09-09; bounded two-arm validation study only.'}
    p['arms'] = []
    capacity = copy.deepcopy(p['capacity_admission']['reports_by_arm']['pointer32_control'])
    p['capacity_admission']['reports_by_arm'] = {}
    for role, step in [('pretrain81096_control', 81096), ('pretrain108128', 108128)]:
        checkpoint = ROOT / f'runtime_inputs/reconstruction_phase34_20260904/checkpoint-step-{step}.pt'
        p['arms'].append({'role': role, 'checkpoint': str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256': sha(checkpoint), 'checkpoint_step': step,
            'hypothesis': 'Additional pretrained encoder and PID state improve downstream topology.' if step == 108128 else 'Reproduce the current pretraining transfer control.',
            'overrides': {'freeze_pretrained_encoder_steps': 2188, 'pointer_positive_weight': 32.0}})
        p['capacity_admission']['reports_by_arm'][role] = copy.deepcopy(capacity)
    p['untouched_validation_cohort'] = {k: cohort[k] for k in (
        'checkpoint_selection_event_uid_count', 'checkpoint_selection_event_uids_sha256',
        'evaluation_event_uid_count', 'evaluation_event_uids_sha256', 'all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update({'path': str(cohort_path.relative_to(ROOT)), 'sha256': sha(cohort_path), 'sealed_test_role_access': 'forbidden'})
    # Retain the exact paired training seed, replay order, thresholds and optimization schedule.
    assert p['common_training_contract']['seed'] == cohort['seed']
    p['evaluation_contract']['beam_search']['evaluated_scopes'] = ['full', 'half']
    output = ROOT / 'configs/reconstruction/ht_reconstruction_phase42_20260909.json'
    output.write_text(json.dumps(p, indent=2, sort_keys=True) + '\n')
    print(output)


if __name__ == '__main__':
    main()
