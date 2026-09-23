#!/usr/bin/env python3
"""Fix the selection eligibility set before reserving fresh Phase58 strict events."""

import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_reconstruction_phase57 import validation_exclusions as old_exclusions
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    sha256,
    uid_sequence_sha256,
    uid_set_sha256,
)
from hypertagging.training.fixed_validation import FIXED_VALIDATION_VERSION


def ranked(uids, seed):
    return sorted(
        uids,
        key=lambda uid: (
            hashlib.sha256(
                f"{FIXED_VALIDATION_VERSION}:{seed}:{uid}".encode()
            ).digest(),
            uid,
        ),
    )


def main():
    old_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase57_validation_cohort_20260922.json"
    )
    old = json.loads(old_path.read_text())
    universe_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_validation_universe_20260923.json"
    )
    universe = json.loads(universe_path.read_text())
    all_uids = set(universe["event_uids"])
    observed_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase57_observed_validation_20260923.json"
    )
    observed = json.loads(observed_path.read_text())
    historical = (
        set(old_exclusions(old))
        | set(old["checkpoint_selection_event_uids"])
        | set(old["event_uids"])
        | set(observed["checkpoint_selection_event_uids"])
    )
    assert len(all_uids) == 50000 and len(all_uids - historical) == 125
    selected = ranked(old["checkpoint_selection_event_uids"], 20260925)
    strict = ranked(all_uids - historical, 20260925)[:100]
    excluded = all_uids - set(selected)
    assert (
        len(selected) == 1000
        and len(strict) == 100
        and len(excluded) == 49000
        and not set(strict) & set(selected)
    )
    c = dict(old)
    c.update(
        manifest_version="hypertagging-reconstruction-phase58-cohort-v1",
        study_id="phase58-corrected-pretraining-objective-balance-20260923",
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=20260925,
        selection_original_seed=20260925,
        strict_selection_seed=20260925,
        selection_algorithm="exact_1000_eligible_uid_set_ranked_with_training_seed",
        selection_reuse_policy="Reuse Phase56 selection set, reordered with seed20260925; exclude every other validation UID before reconstruction selection.",
        permitted_selection_reuse_count=1000,
        remaining_untouched_after_phase58=25,
        historical_used_event_uid_count=len(historical),
        validation_exclusion_event_uid_count=len(excluded),
        validation_exclusion_event_uids_sha256=uid_set_sha256(list(excluded)),
        checkpoint_selection_event_uids=selected,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        overlap_audit={
            "strict_vs_history": len(set(strict) & historical),
            "strict_vs_selection": len(set(strict) & set(selected)),
            "selection_vs_exclusions": len(set(selected) & excluded),
        },
        all_required_overlaps_zero=True,
    )
    c.pop("remaining_untouched_after_phase57", None)
    c.pop("evaluation_source_category_counts", None)
    c["source_bindings"] = {
        **old["source_bindings"],
        "phase57_cohort": {
            "path": str(old_path.relative_to(ROOT)),
            "sha256": sha256(old_path),
        },
        "phase57_observed_selection": {
            "path": str(observed_path.relative_to(ROOT)),
            "sha256": sha256(observed_path),
        },
        "validation_universe": {
            "path": str(universe_path.relative_to(ROOT)),
            "sha256": sha256(universe_path),
        },
    }
    output = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase58_validation_cohort_20260923.json"
    )
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
    print("PASS:1000eligible,49000excluded,100freshstrict,25untouched remain")


if __name__ == "__main__":
    main()
