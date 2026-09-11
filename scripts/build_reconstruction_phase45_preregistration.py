#!/usr/bin/env python3
"""Preregister a fixed-data comparison of late encoder adaptation learning rates."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = ROOT / 'configs/reconstruction/ht_reconstruction_phase44_20260911.json'
    cohort_path = ROOT / 'configs/reconstruction/ht_reconstruction_phase45_validation_cohort_20260911.json'
    evidence_path = ROOT / 'artifacts/codex/reconstruction_phase44_closeout_20260911.json'
    retained_path = ROOT / 'artifacts/codex/reconstruction_phase44_retained_metrics_20260911.json'
    retained = json.loads(retained_path.read_text())
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked'] and retained['legacy_metrics_unchanged']
    cohort = json.loads(cohort_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    assert evidence['metric_completeness'] == 'COMPLETE'
    assert evidence['status'] == 'RECOVERED_DIAGNOSTIC'
    assert all(not arm['all_gates_passed'] for arm in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    p = json.loads(previous.read_text())
    p['study_id'] = 'phase45-late-encoder-learning-rate-20260911'
    p['preregistration_version'] = 'hypertagging-reconstruction-phase45-preregistration-v1'
    p['created_at'] = datetime.now(timezone.utc).isoformat()
    p['common_training_contract']['seed'] = 20260912
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260912
    p['evaluation_contract']['retained_tree_checks'] = {'version': 'retained-direct-tree-checks-v1', 'required': True, 'scopes': ['full', 'half'], 'all_returned_beam_candidates': True, 'population': 'all_explicit_retained_roots_including_outside_training_policy', 'replaces_original_gates': False}
    p['scientific_question'] = 'At fixed 70k data, pretrained checkpoint 81096, 2188 frozen encoder steps and 4376 total steps, does increasing the encoder learning-rate multiplier from 0.05 to 0.10 improve strict downstream hierarchy reconstruction?'
    p.pop('phase43_closeout_basis')
    p.pop('parent_phase43')
    p['parent_phase44'] = {'path': str(previous.relative_to(ROOT)), 'sha256': sha(previous)}
    p['phase44_closeout_basis'] = {
        'classification': 'recovered_diagnostic_study_no_promotion',
        'selected_next_factor': 'late_encoder_lr_multiplier_005_vs_010',
        'sealed_test_accessed': False,
        'path': str(evidence_path.relative_to(ROOT)), 'sha256': sha(evidence_path),
        'limitation': 'Original Phase44 jobs failed after training because nested replay seed was stale; independently recovered evaluations are post-hoc diagnostics, not a clean confirmatory replication.'}
    p['phase44_retained_metric_basis'] = {'path': str(retained_path.relative_to(ROOT)), 'sha256': sha(retained_path), 'version': retained['version'], 'evaluator_revision': retained['evaluator_revision']}
    p['decision_rules'] = {
        'primary': 'micro_complete_target_efficiency',
        'stronger_adaptation_beneficial': 'Candidate improves primary and passes every unchanged strict hierarchy gate; otherwise no promotion and no claimed benefit.',
        'retained_tree_review': 'Review all-retained full/half LCAG, mother coverage and coherent forest agreement alongside the unchanged gates. Passing original gates alone does not establish full-event reconstruction; discordant retained outcomes require further study. No automatic promotion.',
        'uncertainty': 'Exploratory single-seed paired comparison on a fresh cohort. No causal cross-phase comparison; replication required for any positive signal.',
        'dataset_size': 'Hold at 70000; prior scaling confounded data with optimization budget and no controlled learning curve establishes a data-limited regime.',
        'pretraining': 'Hold checkpoint81096. Longer duration was not supported by Phase42. Stronger task-specific representation adaptation is a hypothesis, not a change to pretraining objectives.',
        'authority': 'User requested Phase44 review and next trainings on 2026-09-11; bounded two-arm validation study only.'}
    checkpoint = copy.deepcopy(p['arms'][0])
    assert checkpoint['checkpoint_step'] == 81096
    assert sha(ROOT / checkpoint['checkpoint']) == checkpoint['checkpoint_sha256']
    capacity = copy.deepcopy(p['capacity_admission']['reports_by_arm']['late_adaptation_control'])
    p['arms'] = []
    p['capacity_admission']['reports_by_arm'] = {}
    for role, multiplier in [('encoder_lr005_control', 0.05), ('encoder_lr010', 0.10)]:
        arm = copy.deepcopy(checkpoint)
        arm.update(role=role, hypothesis='Stronger late task-specific encoder adaptation improves topology.' if multiplier == 0.10 else 'Reproduce the late-adaptation learning-rate control.')
        arm['overrides']['freeze_pretrained_encoder_steps'] = 2188
        arm['overrides']['encoder_lr_multiplier'] = multiplier
        p['arms'].append(arm)
        p['capacity_admission']['reports_by_arm'][role] = copy.deepcopy(capacity)
    p['untouched_validation_cohort'] = {k: cohort[k] for k in (
        'checkpoint_selection_event_uid_count', 'checkpoint_selection_event_uids_sha256',
        'evaluation_event_uid_count', 'evaluation_event_uids_sha256', 'all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(path=str(cohort_path.relative_to(ROOT)), sha256=sha(cohort_path), sealed_test_role_access='forbidden')
    assert p['common_training_contract']['seed'] == p['common_training_contract']['balanced_level_replay_contract']['seed'] == cohort['seed']
    output = ROOT / 'configs/reconstruction/ht_reconstruction_phase45_20260911.json'
    output.write_text(json.dumps(p, indent=2, sort_keys=True) + '\n')
    print(output)


if __name__ == '__main__':
    main()
