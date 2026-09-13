#!/usr/bin/env python3
"""Preregister a second-seed corrected-source adaptation comparison."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous = ROOT/'configs/reconstruction/ht_reconstruction_phase46_20260912.json'
    cohort_path = ROOT/'configs/reconstruction/ht_reconstruction_phase47_validation_cohort_20260913.json'
    evidence_path = ROOT/'artifacts/codex/reconstruction_phase46_closeout_20260913.json'
    retained_path = ROOT/'artifacts/codex/reconstruction_phase46_retained_metrics_20260913.json'
    p, cohort, evidence, retained = [json.loads(q.read_text()) for q in (previous, cohort_path, evidence_path, retained_path)]
    assert evidence['status'] == 'COMPLETED' and evidence['metric_completeness'] == 'COMPLETE'
    assert retained['status'] == 'COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert all(not arm['all_gates_passed'] for arm in evidence['arms'].values())
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    old_common = copy.deepcopy(p['common_training_contract'])
    for key in ('parent_phase45','phase45_closeout_basis','phase45_retained_metric_basis'):
        p.pop(key)
    p.update(study_id='phase47-corrected-adaptation-replication-20260913',
        preregistration_version='hypertagging-reconstruction-phase47-preregistration-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scientific_question='Does the Phase46 tradeoff between primary source-set recovery and strict retained topology repeat with a second training/replay seed and a fresh disjoint validation cohort?')
    p['common_training_contract']['seed'] = 20260914
    p['common_training_contract']['balanced_level_replay_contract']['seed'] = 20260914
    p['parent_phase46'] = binding(previous)
    p['phase46_closeout_basis'] = {**binding(evidence_path), 'classification':'completed_corrected_source_no_promotion', 'selected_next_factor':'corrected_adaptation_second_seed', 'sealed_test_accessed':False}
    p['phase46_retained_metric_basis'] = {**binding(retained_path), 'version':retained['version'], 'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].update(
        adaptation_beneficial='Replicate the direction of the primary and jointly inspect strict retained topology. Every unchanged strict gate must pass before any promotion consideration; no automatic promotion.',
        uncertainty='Second-seed exploratory replication. A fresh validation cohort also changes event composition; do not interpret cross-phase deltas as isolated seed effects or pool validation cohorts as independent training replications.',
        dataset_size='Hold 70000 events and the exact authenticated Phase46 train-only index. No controlled learning curve demonstrates data limitation.',
        pretraining='Historical encoder81096 remains fixed to isolate downstream adaptation. Corrected pretraining quality remains a plausible unmeasured intervention; more epochs alone are not supported by Phase42. Resolve the primary/topology tradeoff before allocating a larger pretraining campaign.',
        authority='User requested Phase46 review, publication and next trainings on 2026-09-13; exactly two bounded reconstruction arms.')
    p['scientific_source_boundary']['limitation'] = 'Same corrected scientific implementation and authenticated train-only statistics as Phase46. Historical pretrained weights unchanged. This replication tests downstream adaptation, not newly corrected pretraining objectives.'
    p['statistics_reuse'] = {'phase46_index_unchanged':True, 'justification':'Same training records, policy, feature adapter and normalizer implementation; source hashes checked by every contract.', 'new_fit_required_if_source_or_train_population_changes':True}
    p['untouched_validation_cohort'] = {k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path), sealed_test_role_access='forbidden')
    check=copy.deepcopy(p['common_training_contract']); check['seed']=old_common['seed'];check['balanced_level_replay_contract']['seed']=old_common['balanced_level_replay_contract']['seed']
    assert check==old_common
    assert p['common_training_contract']['seed']==cohort['seed']
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase47_20260913.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__': main()
