#!/usr/bin/env python3
"""Build the untouched phase-36 validation-selection and evaluation cohorts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reconstruction_phase35_evaluation_cohort import (  # noqa: E402
    atomic_json,
    ranked_validation_uids,
    sha256,
    uid_sequence_sha256,
    uid_set_sha256,
)
from hypertagging.preprocessing.schema_v4 import iter_event_records_v4  # noqa: E402


PHASE35_PREREGISTRATION = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase35_improvement_20260904.json"
)
PHASE35_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase35_evaluation_cohort_20260904.json"
)
PHASE35_HISTORICAL_EXCLUSIONS = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase35_validation_exclusions_20260904.json"
)
MANIFEST_VERSION = "hypertagging-reconstruction-phase36-cohort-v1"
STUDY_ID = "phase36-hierarchy-aware-reconstruction-20260905"
SELECTION_SEED = 20260905
SELECTION_COUNT = 2_000
EVALUATION_COUNT = 100


def role_uids(manifest: dict[str, object], role: str) -> list[str]:
    data_root = Path(str(manifest["data_root"])).resolve(strict=True)
    result: list[str] = []
    for raw_entry in manifest["entries"]:  # type: ignore[index]
        entry = dict(raw_entry)
        if entry.get("split") != role:
            continue
        shard = (data_root / str(entry["path"])).resolve(strict=True)
        result.extend(
            str(record["event_uid"])
            for record in iter_event_records_v4(shard)
        )
    if len(result) != len(set(result)):
        raise RuntimeError(f"{role} role contains duplicate event UIDs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase35 = json.loads(PHASE35_PREREGISTRATION.read_text(encoding="utf-8"))
    data = dict(phase35["data_binding"])
    selection_path = (ROOT / str(data["selection_manifest"])).resolve(strict=True)
    index_path = (ROOT / str(data["dataset_index"])).resolve(strict=True)
    if (
        sha256(selection_path) != data["selection_manifest_sha256"]
        or sha256(index_path) != data["dataset_index_sha256"]
    ):
        raise RuntimeError("phase35 data binding changed")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("selection_contract", {}).get("included_splits")
        != ["train", "validation"]
        or index.get("event_identity_validation", {}).get("sealed_test_opened")
        is not False
        or index.get("split_counts")
        not in (
            {"train": 35_000, "validation": 50_000},
            {"train": 35_000, "validation": 50_000, "test": 0},
        )
    ):
        raise RuntimeError("dataset index does not prove train/validation-only access")

    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    train_uids = role_uids(selection_manifest, "train")
    if len(train_uids) != 35_000:
        raise RuntimeError("phase35 training role no longer contains 35,000 events")

    historical = json.loads(
        PHASE35_HISTORICAL_EXCLUSIONS.read_text(encoding="utf-8")
    )
    historical_uids = [str(uid) for uid in historical["event_uids"]]
    phase35_cohort = json.loads(PHASE35_COHORT.read_text(encoding="utf-8"))
    phase35_selection_uids = [
        str(uid) for uid in phase35_cohort["checkpoint_selection_event_uids"]
    ]
    phase35_evaluation_uids = [
        str(uid) for uid in phase35_cohort["event_uids"]
    ]
    if len(phase35_selection_uids) != 2_000 or len(phase35_evaluation_uids) != 100:
        raise RuntimeError("phase35 cohort sizes changed")

    validation_exclusions = set(historical_uids)
    validation_exclusions.update(phase35_selection_uids)
    validation_exclusions.update(phase35_evaluation_uids)
    selected = ranked_validation_uids(
        selection_manifest,
        excluded=validation_exclusions,
        seed=SELECTION_SEED,
        limit=SELECTION_COUNT + EVALUATION_COUNT,
    )
    selection_rows = selected[:SELECTION_COUNT]
    evaluation_rows = selected[SELECTION_COUNT:]
    selection_uids = [uid for uid, _category in selection_rows]
    evaluation_uids = [uid for uid, _category in evaluation_rows]

    train_set = set(train_uids)
    historical_set = set(historical_uids)
    phase35_selection_set = set(phase35_selection_uids)
    phase35_evaluation_set = set(phase35_evaluation_uids)
    phase36_selection_set = set(selection_uids)
    phase36_evaluation_set = set(evaluation_uids)
    overlap_audit = {
        "phase35_train_vs_phase36_selection": len(train_set & phase36_selection_set),
        "phase35_train_vs_phase36_evaluation": len(train_set & phase36_evaluation_set),
        "historical_validation_vs_phase36_selection": len(
            historical_set & phase36_selection_set
        ),
        "historical_validation_vs_phase36_evaluation": len(
            historical_set & phase36_evaluation_set
        ),
        "phase35_selection_vs_phase36_selection": len(
            phase35_selection_set & phase36_selection_set
        ),
        "phase35_selection_vs_phase36_evaluation": len(
            phase35_selection_set & phase36_evaluation_set
        ),
        "phase35_evaluation_vs_phase36_selection": len(
            phase35_evaluation_set & phase36_selection_set
        ),
        "phase35_evaluation_vs_phase36_evaluation": len(
            phase35_evaluation_set & phase36_evaluation_set
        ),
        "phase36_selection_vs_phase36_evaluation": len(
            phase36_selection_set & phase36_evaluation_set
        ),
    }
    if any(overlap_audit.values()):
        raise RuntimeError(f"phase36 cohort is not untouched: {overlap_audit}")

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "role": "validation",
        "sealed_test_role_access": "forbidden",
        "selection_algorithm": "manifest-role-uid-hash-v1",
        "seed": SELECTION_SEED,
        "selection_manifest": str(selection_path.relative_to(ROOT)),
        "selection_manifest_sha256": sha256(selection_path),
        "dataset_index": str(index_path.relative_to(ROOT)),
        "dataset_index_sha256": sha256(index_path),
        "source_bindings": {
            "phase35_preregistration": {
                "path": str(PHASE35_PREREGISTRATION.relative_to(ROOT)),
                "sha256": sha256(PHASE35_PREREGISTRATION),
            },
            "historical_validation_exclusions": {
                "path": str(PHASE35_HISTORICAL_EXCLUSIONS.relative_to(ROOT)),
                "sha256": sha256(PHASE35_HISTORICAL_EXCLUSIONS),
                "event_uid_count": len(historical_uids),
                "event_uids_sha256": uid_set_sha256(historical_uids),
            },
            "phase35_cohort": {
                "path": str(PHASE35_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE35_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase35_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase35_selection_uids
                ),
                "evaluation_event_uid_count": len(phase35_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase35_evaluation_uids
                ),
            },
            "phase35_training_role": {
                "event_uid_count": len(train_uids),
                "event_uids_sha256": uid_set_sha256(train_uids),
                "event_uids_hash_scheme": "sha256-u64be-length-prefixed-utf8-v1",
            },
        },
        "validation_exclusion_event_uid_count": len(validation_exclusions),
        "validation_exclusion_event_uids_sha256": uid_set_sha256(
            list(validation_exclusions)
        ),
        "checkpoint_selection_event_uid_count": len(selection_uids),
        "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
            selection_uids
        ),
        "checkpoint_selection_event_uids": selection_uids,
        "evaluation_event_uid_count": len(evaluation_uids),
        "evaluation_event_uids_sha256": uid_sequence_sha256(evaluation_uids),
        "evaluation_event_uids": evaluation_uids,
        # Compatibility aliases consumed by the offline full-decay evaluator.
        "event_uid_count": len(evaluation_uids),
        "event_uids_sha256": uid_sequence_sha256(evaluation_uids),
        "event_uids": evaluation_uids,
        "evaluation_source_category_counts": dict(
            sorted(Counter(category for _uid, category in evaluation_rows).items())
        ),
        "overlap_audit": overlap_audit,
        "all_required_overlaps_zero": True,
    }
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("cohort output must remain in the repository") from error
    if output.exists():
        raise RuntimeError(f"refusing to overwrite cohort manifest: {output}")
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "checkpoint_selection_events": len(selection_uids),
                "evaluation_events": len(evaluation_uids),
                "overlap_audit": overlap_audit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
