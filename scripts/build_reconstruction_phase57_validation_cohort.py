#!/usr/bin/env python3
"""Reuse only Phase56 selection; reserve 100 previously untouched strict events."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reconstruction_phase35_evaluation_cohort import atomic_json, ranked_validation_uids, sha256, uid_sequence_sha256, uid_set_sha256
from scripts.build_reconstruction_phase56_validation_cohort import role_uids
from scripts.run_reconstruction_phase56 import validation_exclusions


def main():
    previous_path=ROOT/'configs/reconstruction/ht_reconstruction_phase56_validation_cohort_20260921.json'
    old=json.loads(previous_path.read_text())
    manifest=json.loads((ROOT/'configs/training_selection/production_1m_20260812/train_070k_phase40.json').read_text())
    selected=old['checkpoint_selection_event_uids']
    historical=set(validation_exclusions(old))|set(selected)|set(old['event_uids'])
    assert len(historical)==49609 and len(set(selected))==1000
    train=set(role_uids(manifest,'train'))
    assert len(train)==70000
    rows=ranked_validation_uids(manifest,excluded=historical|train,seed=20260924,limit=100)
    strict=[uid for uid,_ in rows]
    exclusions=historical-set(selected)
    p=dict(old)
    p.update(manifest_version='hypertagging-reconstruction-phase57-cohort-v1',study_id='phase57-pretraining-objective-balance-20260922',created_at=datetime.now(timezone.utc).isoformat(),seed=20260924,
        selection_algorithm='exact_previous_selection_and_fresh_strict_uid_hash_v1',selection_original_seed=20260923,strict_selection_seed=20260924,
        selection_reuse_policy='Exact Phase56 selection reused; strict evaluation is fresh. Not an independent cross-phase selection replication.',permitted_selection_reuse_count=1000,
        historical_used_event_uid_count=len(historical),remaining_untouched_after_phase57=291,
        validation_exclusion_event_uid_count=len(exclusions),validation_exclusion_event_uids_sha256=uid_set_sha256(list(exclusions)),
        evaluation_event_uids=strict,event_uids=strict,evaluation_event_uids_sha256=uid_sequence_sha256(strict),event_uids_sha256=uid_sequence_sha256(strict),
        evaluation_source_category_counts=dict(sorted(Counter(c for _,c in rows).items())),
        overlap_audit={'strict_vs_all_historical':len(set(strict)&historical),'train_vs_selection':len(train&set(selected)),'train_vs_strict':len(train&set(strict)),'strict_vs_selection':len(set(strict)&set(selected)),'selection_vs_forbidden_history':len(set(selected)&exclusions)},all_required_overlaps_zero=True)
    p['source_bindings']={**old['source_bindings'],'phase56_cohort':{'path':str(previous_path.relative_to(ROOT)),'sha256':sha256(previous_path)}}
    assert not any(p['overlap_audit'].values())
    output=ROOT/'configs/reconstruction/ht_reconstruction_phase57_validation_cohort_20260922.json'
    if output.exists():raise FileExistsError(output)
    atomic_json(output,p)
    print(json.dumps({'output':str(output),'overlaps':p['overlap_audit'],'selection_reused':1000,'fresh_strict':100,'untouched_remaining':291}))

if __name__=='__main__':main()
