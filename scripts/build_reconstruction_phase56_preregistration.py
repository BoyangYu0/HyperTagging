#!/usr/bin/env python3
"""Preregister a bounded replication of parent-ranking pretraining weights."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase46_preregistration import binding


def main():
    previous=ROOT/'configs/reconstruction/ht_reconstruction_phase55_20260920.json'
    cohort_path=ROOT/'configs/reconstruction/ht_reconstruction_phase56_validation_cohort_20260921.json'
    evidence_path=ROOT/'artifacts/codex/reconstruction_phase55_closeout_20260921.json'
    retained_path=ROOT/'artifacts/codex/reconstruction_phase55_retained_metrics_20260921.json'
    p,cohort,evidence,retained=[json.loads(q.read_text()) for q in (previous,cohort_path,evidence_path,retained_path)]
    assert evidence['status']=='COMPLETED' and evidence['metric_completeness']=='COMPLETE'
    assert evidence['original_job_status']=='FAILED_POST_TRAINING_EVALUATION'
    assert retained['status']=='COMPLETE' and retained['all_returned_beam_candidates_checked']
    assert cohort['all_required_overlaps_zero'] and not any(cohort['overlap_audit'].values())
    for key in ('parent_phase54','phase54_closeout_basis','phase54_retained_metric_basis'):p.pop(key)
    p.update(study_id='phase56-pretraining-parent-replication-20260921',preregistration_version='hypertagging-reconstruction-phase56-preregistration-v1',created_at=datetime.now(timezone.utc).isoformat(),scientific_question='Does the mixed Phase55 parent-ranking weight contrast reproduce with a fresh seed and untouched validation cohort at the same training data and compute?')
    c=p['common_training_contract'];c['seed']=20260923;c['balanced_level_replay_contract']['seed']=20260923
    c['max_validation_events']=1000
    pre=p['pretraining_refinement'];pre['config']['seed']=20260923
    pre['training_presentations']=70000;pre['nominal_batch_capacity']=70016
    pre['presentation_accounting']='2187 full batches of 32 plus one final batch of 16; verify sum of batch_events'
    pre['comparison_limitation']='Matched parent-ranking objective comparison, not refinement versus no refinement. Phase56 selection cohort is smaller than Phase55: 1000 versus 2000, with 1000 rollout unchanged. Compare arms within phase; cross-phase changes are not causal.'
    p['parent_phase55']=binding(previous)
    p['phase55_closeout_basis']={**binding(evidence_path),'classification':'recovered_evaluation_corrected_lineage_no_promotion','selected_next_factor':'pretraining_parent_ranking_weight','sealed_test_accessed':False}
    p['phase55_retained_metric_basis']={**binding(retained_path),'version':retained['version'],'evaluator_revision':retained['evaluator_revision']}
    p['decision_rules'].update(
        representation_beneficial='Require a reproducible joint improvement in primary recovery, original and retained full/half topology, exact nontrivial components, source precision/recall and nontrivial coherent forests. Original gates remain necessary. No automatic promotion.',
        stop_dose_tuning='One replication only. If joint benefit is not reproduced, stop parent-ranking dose tuning and diagnose representation/assembly errors before choosing another factor.',
        dataset_size='Hold 70000 training events. No controlled learning curve establishes an immediate training-data-growth benefit. Validation-pool exhaustion is separate.',
        pretraining='Replicate parent-ranking weights 1 versus 2 at matched 2188 refinement steps and identical 4376 reconstruction steps, with corrected checkpoint handoff.',
        authority='User authorized Phase55 review, publication and one next bounded training pair on 2026-09-21. No campaign chain, automatic promotion or sealed-test access.')
    p['validation_budget']={'validation_role_events':50000,'previously_excluded':48509,'newly_reserved':1100,'selection_events':1000,'rollout_selection_events':1000,'strict_events':100,'remaining_untouched_after_phase56':391,'selection_size_change':'Reduced 2000 to 1000, matched across arms; larger selection uncertainty is an explicit limitation. No historical selection or strict UID reused.'}
    p['untouched_validation_cohort']={k:cohort[k] for k in ('checkpoint_selection_event_uid_count','checkpoint_selection_event_uids_sha256','evaluation_event_uid_count','evaluation_event_uids_sha256','all_required_overlaps_zero')}
    p['untouched_validation_cohort'].update(**binding(cohort_path),sealed_test_role_access='forbidden')
    assert cohort['seed']==20260923 and cohort['checkpoint_selection_event_uid_count']==1000
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase56_20260921.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
