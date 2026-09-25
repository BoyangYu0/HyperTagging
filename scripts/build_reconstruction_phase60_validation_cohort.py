#!/usr/bin/env python3
"""Reserve fresh selection and strict cohorts exclusively from added validation sources."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_reconstruction_phase56_validation_cohort import role_uids
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    sha256,
    uid_sequence_sha256,
    uid_set_sha256,
)

SEED = 20260927
STUDY = "phase60-fresh-validation-pretraining-replication-20260925"


def main():
    old_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase59_validation_cohort_20260924.json"
    )
    old = json.loads(old_path.read_text())
    old_universe_path = ROOT / old["source_bindings"]["validation_universe"]["path"]
    old_uids = set(json.loads(old_universe_path.read_text())["event_uids"])
    selection = (
        ROOT
        / "configs/training_selection/phase60_validation_expansion_20260925/train_070k.json"
    )
    universe_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_validation_universe_20260925.json"
    )
    if universe_path.exists():
        existing = json.loads(universe_path.read_text())
        assert existing["source_manifest_sha256"] == sha256(selection)
        uids = existing["event_uids"]
        assert existing["event_uids_sha256"] == uid_set_sha256(uids)
    else:
        uids = role_uids(json.loads(selection.read_text()), "validation")
    assert len(uids) == len(set(uids)) == 100000 and old_uids <= set(uids)
    fresh = set(uids) - old_uids
    assert len(fresh) == 50000
    ordered = ranked(fresh, SEED)
    selected = ordered[:1000]
    strict = ordered[1000:1100]
    excluded = set(uids) - set(selected)
    universe_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_validation_universe_20260925.json"
    )
    if not universe_path.exists():
        universe_path.write_text(
            json.dumps(
                {
                    "version": "validation-role-universe-v1",
                    "role": "validation",
                    "event_uid_count": len(uids),
                    "event_uids_sha256": uid_set_sha256(uids),
                    "event_uids": sorted(uids),
                    "source_manifest": str(selection.relative_to(ROOT)),
                    "source_manifest_sha256": sha256(selection),
                    "sealed_test_accessed": False,
                },
                indent=2,
            )
            + "\n"
        )
    c = dict(old)
    c.pop("remaining_untouched_after_phase59")
    c.update(
        manifest_version="hypertagging-reconstruction-phase60-cohort-v1",
        study_id=STUDY,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        selection_original_seed=SEED,
        strict_selection_seed=SEED,
        remaining_untouched_after_phase60=48900,
        historical_used_event_uid_count=50000,
        checkpoint_selection_event_uids=selected,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uid_count=100,
        event_uid_count=100,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        selection_manifest_sha256=sha256(selection),
        validation_exclusion_event_uid_count=len(excluded),
        validation_exclusion_event_uids_sha256=uid_set_sha256(list(excluded)),
        permitted_selection_reuse_count=0,
        selection_algorithm="exact_1000_eligible_uid_set_ranked_with_training_seed",
        selection_reuse_policy="No reuse:1000 fresh selection events and100 disjoint strict events from new source files.",
        evaluation_limitation="Exploratory replication on fresh sources; no promotion and no causal cross-phase comparison.",
        overlap_audit={
            "strict_vs_history": len(set(strict) & old_uids),
            "strict_vs_selection": len(set(strict) & set(selected)),
            "selection_vs_history": len(set(selected) & old_uids),
            "selection_vs_exclusions": len(set(selected) & excluded),
        },
        all_required_overlaps_zero=True,
    )
    bind = lambda p: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
    c["source_bindings"] = {
        "validation_universe": bind(universe_path),
        "previous_validation_universe": bind(old_universe_path),
        "phase59_cohort": bind(old_path),
        "validation_expansion": bind(selection.parent / "expansion-audit.json"),
    }
    # Replace stale bindings inherited from the original 50k universe.
    for key in ("selection_manifest", "dataset_index"):
        if key in c:
            path = (
                selection
                if key == "selection_manifest"
                else ROOT
                / "artifacts/experiment_readiness/reconstruction_phase60_20260925/train_070k.complete_only.index.json"
            )
            c[key] = str(path.relative_to(ROOT))
            c[key + "_sha256"] = sha256(path)
    p = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase60_validation_cohort_20260925.json"
    )
    assert not p.exists()
    p.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
    print("PASS fresh selection1000 strict100 remaining48900")


if __name__ == "__main__":
    main()
