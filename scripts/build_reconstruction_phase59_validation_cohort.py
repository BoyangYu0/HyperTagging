"""Reserve the last 25 untouched validation events for a bounded feasibility pilot."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_reconstruction_phase58_validation_cohort import ranked
from scripts.run_reconstruction_phase57 import validation_exclusions
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    sha256,
    uid_sequence_sha256,
)


def main():
    old_path = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase58_validation_cohort_20260923.json"
    )
    old = json.loads(old_path.read_text())
    prior = json.loads(
        (ROOT / old["source_bindings"]["phase57_cohort"]["path"]).read_text()
    )
    observed = json.loads(
        (
            ROOT / old["source_bindings"]["phase57_observed_selection"]["path"]
        ).read_text()
    )
    universe = json.loads(
        (ROOT / old["source_bindings"]["validation_universe"]["path"]).read_text()
    )
    historical = (
        set(validation_exclusions(prior))
        | set(prior["checkpoint_selection_event_uids"])
        | set(prior["event_uids"])
        | set(observed["checkpoint_selection_event_uids"])
        | set(old["event_uids"])
    )
    strict = ranked(set(universe["event_uids"]) - historical, 20260926)
    selected = ranked(old["checkpoint_selection_event_uids"], 20260926)
    assert (
        len(strict) == 25 and len(selected) == 1000 and not set(strict) & set(selected)
    )
    c = dict(old)
    c.pop("remaining_untouched_after_phase58", None)
    c.update(
        manifest_version="hypertagging-reconstruction-phase59-cohort-v1",
        study_id="phase59-pretraining-stability-pilot-20260924",
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=20260926,
        selection_original_seed=20260926,
        strict_selection_seed=20260926,
        remaining_untouched_after_phase59=0,
        historical_used_event_uid_count=len(historical),
        checkpoint_selection_event_uids=selected,
        checkpoint_selection_event_uids_sha256=uid_sequence_sha256(selected),
        evaluation_event_uids=strict,
        event_uids=strict,
        evaluation_event_uid_count=25,
        event_uid_count=25,
        evaluation_event_uids_sha256=uid_sequence_sha256(strict),
        event_uids_sha256=uid_sequence_sha256(strict),
        selection_reuse_policy="Exact Phase56 selection set, reordered with seed20260926; all other49000 excluded.",
        evaluation_limitation="Last25 untouched events; feasibility pilot only, no confirmatory quality claim or promotion. Expand independent validation before another campaign.",
    )
    c["source_bindings"] = {
        **old["source_bindings"],
        "phase58_cohort": {
            "path": str(old_path.relative_to(ROOT)),
            "sha256": sha256(old_path),
        },
    }
    p = (
        ROOT
        / "configs/reconstruction/ht_reconstruction_phase59_validation_cohort_20260924.json"
    )
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
