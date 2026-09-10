#!/usr/bin/env python3
"""Preregister early versus late task-specific encoder adaptation at fixed data."""
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
    previous = ROOT / 'configs/reconstruction/ht_reconstruction_phase42_20260909.json'
    cohort_path = ROOT / 'configs/reconstruction/ht_reconstruction_phase43_validation_cohort_20260910.json'
    evidence_path = ROOT / 'artifacts/codex/reconstruction_phase42_closeout_20260910.json'
    cohort = json.loads(cohort_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    assert evidence['metric_completeness'] == 'COMPLETE'
    assert all(not arm['all_gates_passed'] for arm in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    p = json.loads(previous.read_text())
    p['study_id'] = 'phase43-early-encoder-adaptation-20260910'
    p['preregistration_version'] = 'hypertagging-reconstruction-phase43-preregistration-v1'
    p['created_at'] = datetime.now(timezone.utc).isoformat()
    p['scientific_question'] = 'Does task-specific encoder adaptation from step zero improve strict reconstruction versus the current 2188-step freeze, at fixed pretrained checkpoint, 70k data, and 4376-step budget?'
    p.pop('phase41_closeout_basis')
    p.pop('parent_phase41')
    p['parent_phase42'] = {'path': str(previous.relative_to(ROOT)), 'sha256': sha(previous)}
    p['phase42_closeout_basis'] = {
        'classification': 'completed_controlled_study_no_promotion',
        'selected_next_factor': 'encoder_freeze_steps_2188_vs_0',
        'sealed_test_accessed': False,
        'path': str(evidence_path.relative_to(ROOT)), 'sha256': sha(evidence_path)}
    p['decision_rules'] = {
        'primary': 'micro_complete_target_efficiency',
        'early_adaptation_beneficial': 'Candidate improves primary and passes every unchanged strict hierarchy gate; otherwise no promotion and no claimed benefit.',
        'uncertainty': 'Single training seed and 100 strict / 20 beam events; later replication required. Equal total steps do not imply equal encoder update counts or exact FLOPs.',
        'dataset_size': 'Hold at 70000; data-limited behavior is unproven and prior scaling confounded data with optimization budget.',
        'pretraining': 'Do not extend pretraining duration now; Phase42 later checkpoint failed to improve the primary or satisfy all hierarchy gates.',
        'authority': 'User requested the same closeout and next-training workflow on 2026-09-10; bounded two-arm validation study only.'}
    checkpoint = copy.deepcopy(p['arms'][0])
    assert checkpoint['checkpoint_step'] == 81096
    assert sha(ROOT / checkpoint['checkpoint']) == checkpoint['checkpoint_sha256']
    capacity = copy.deepcopy(p['capacity_admission']['reports_by_arm']['pretrain81096_control'])
    p['arms'] = []
    p['capacity_admission']['reports_by_arm'] = {}
    for role, freeze in [('late_adaptation_control', 2188), ('early_adaptation', 0)]:
        arm = copy.deepcopy(checkpoint)
        arm.update(role=role, hypothesis='Earlier task-specific representation adaptation improves topology.' if freeze == 0 else 'Reproduce the current late-adaptation control.')
        arm['overrides']['freeze_pretrained_encoder_steps'] = freeze
        p['arms'].append(arm)
        p['capacity_admission']['reports_by_arm'][role] = copy.deepcopy(capacity)
    p['untouched_validation_cohort'] = {k: cohort[k] for k in (
        'checkpoint_selection_event_uid_count', 'checkpoint_selection_event_uids_sha256',
        'evaluation_event_uid_count', 'evaluation_event_uids_sha256', 'all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(path=str(cohort_path.relative_to(ROOT)), sha256=sha(cohort_path), sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed'] == cohort['seed']
    output = ROOT / 'configs/reconstruction/ht_reconstruction_phase43_20260910.json'
    output.write_text(json.dumps(p, indent=2, sort_keys=True) + '\n')
    print(output)


if __name__ == '__main__':
    main()
