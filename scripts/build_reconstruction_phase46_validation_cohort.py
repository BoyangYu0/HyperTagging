#!/usr/bin/env python3
"""Build the untouched phase-46 validation-selection and evaluation cohorts."""

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
PHASE36_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase36_validation_cohort_20260905.json"
)
PHASE37_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase37_validation_cohort_20260906.json"
)
PHASE38_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase38_validation_cohort_20260906.json"
)
PHASE39_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase39_validation_cohort_20260907.json"
)
PHASE40_COHORT = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase40_validation_cohort_20260907.json"
)
SELECTION = (
    ROOT / "configs/training_selection/production_1m_20260812/train_070k_phase40.json"
)
DATASET_INDEX = (
    ROOT
    / "artifacts/experiment_readiness/reconstruction_phase40_20260907/train_070k.complete_only.index.json"
)
MANIFEST_VERSION = "hypertagging-reconstruction-phase46-cohort-v1"
STUDY_ID = "phase46-corrected-frozen-encoder-20260912"
SELECTION_SEED = 20260913
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

    selection_path = SELECTION.resolve(strict=True)
    index_path = DATASET_INDEX.resolve(strict=True)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("selection_contract", {}).get("included_splits")
        != ["train", "validation"]
        or index.get("event_identity_validation", {}).get("sealed_test_opened")
        is not False
        or index.get("split_counts")
        not in (
            {"train": 70_000, "validation": 50_000},
            {"train": 70_000, "validation": 50_000, "test": 0},
        )
    ):
        raise RuntimeError("dataset index does not prove train/validation-only access")

    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    train_uids = role_uids(selection_manifest, "train")
    if len(train_uids) != 70_000:
        raise RuntimeError("phase46 training role does not contain 70,000 events")

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

    phase36_cohort = json.loads(PHASE36_COHORT.read_text(encoding="utf-8"))
    phase36_selection_uids = [
        str(uid) for uid in phase36_cohort["checkpoint_selection_event_uids"]
    ]
    phase36_evaluation_uids = [
        str(uid) for uid in phase36_cohort["evaluation_event_uids"]
    ]
    if len(phase36_selection_uids) != 2_000 or len(phase36_evaluation_uids) != 100:
        raise RuntimeError("phase36 cohort sizes changed")

    phase37_cohort = json.loads(PHASE37_COHORT.read_text(encoding="utf-8"))
    phase37_selection_uids = [
        str(uid) for uid in phase37_cohort["checkpoint_selection_event_uids"]
    ]
    phase37_evaluation_uids = [
        str(uid) for uid in phase37_cohort["evaluation_event_uids"]
    ]
    if len(phase37_selection_uids) != 2_000 or len(phase37_evaluation_uids) != 100:
        raise RuntimeError("phase37 cohort sizes changed")

    phase38_cohort = json.loads(PHASE38_COHORT.read_text(encoding="utf-8"))
    phase38_selection_uids = [
        str(uid) for uid in phase38_cohort["checkpoint_selection_event_uids"]
    ]
    phase38_evaluation_uids = [
        str(uid) for uid in phase38_cohort["evaluation_event_uids"]
    ]
    if len(phase38_selection_uids) != 2_000 or len(phase38_evaluation_uids) != 100:
        raise RuntimeError("phase38 cohort sizes changed")

    phase39_cohort = json.loads(PHASE39_COHORT.read_text(encoding="utf-8"))
    phase39_selection_uids = [
        str(uid) for uid in phase39_cohort["checkpoint_selection_event_uids"]
    ]
    phase39_evaluation_uids = [
        str(uid) for uid in phase39_cohort["evaluation_event_uids"]
    ]
    if len(phase39_selection_uids) != 2_000 or len(phase39_evaluation_uids) != 100:
        raise RuntimeError("phase39 cohort sizes changed")

    phase40_cohort = json.loads(PHASE40_COHORT.read_text(encoding="utf-8"))
    phase40_selection_uids = [
        str(uid) for uid in phase40_cohort["checkpoint_selection_event_uids"]
    ]
    phase40_evaluation_uids = [
        str(uid) for uid in phase40_cohort["evaluation_event_uids"]
    ]
    if len(phase40_selection_uids) != 2_000 or len(phase40_evaluation_uids) != 100:
        raise RuntimeError("phase40 cohort sizes changed")

    phase41_path = ROOT / "configs/reconstruction/ht_reconstruction_phase41_validation_cohort_20260909.json"
    phase41 = json.loads(phase41_path.read_text())
    phase41_uids = set(phase41["checkpoint_selection_event_uids"]) | set(phase41["evaluation_event_uids"])
    phase43_path = ROOT / "configs/reconstruction/ht_reconstruction_phase43_validation_cohort_20260910.json"
    phase43 = json.loads(phase43_path.read_text())
    phase43_uids = set(phase43["checkpoint_selection_event_uids"]) | set(phase43["evaluation_event_uids"])
    phase42_path = ROOT / "configs/reconstruction/ht_reconstruction_phase42_validation_cohort_20260909.json"
    phase42 = json.loads(phase42_path.read_text())
    phase42_uids = set(phase42["checkpoint_selection_event_uids"]) | set(phase42["evaluation_event_uids"])
    phase44_path = ROOT / "configs/reconstruction/ht_reconstruction_phase44_validation_cohort_20260911.json"
    phase44 = json.loads(phase44_path.read_text())
    phase44_uids = set(phase44["checkpoint_selection_event_uids"]) | set(phase44["evaluation_event_uids"])
    phase45_path = ROOT / "configs/reconstruction/ht_reconstruction_phase45_validation_cohort_20260911.json"
    phase45 = json.loads(phase45_path.read_text())
    phase45_uids = set(phase45["checkpoint_selection_event_uids"]) | set(phase45["evaluation_event_uids"])
    validation_exclusions = phase45_uids | phase44_uids | set(historical_uids) | phase41_uids | phase43_uids | phase42_uids
    validation_exclusions.update(phase35_selection_uids)
    validation_exclusions.update(phase35_evaluation_uids)
    validation_exclusions.update(phase36_selection_uids)
    validation_exclusions.update(phase36_evaluation_uids)
    validation_exclusions.update(phase37_selection_uids)
    validation_exclusions.update(phase37_evaluation_uids)
    validation_exclusions.update(phase38_selection_uids)
    validation_exclusions.update(phase38_evaluation_uids)
    validation_exclusions.update(phase39_selection_uids)
    validation_exclusions.update(phase39_evaluation_uids)
    validation_exclusions.update(phase40_selection_uids)
    validation_exclusions.update(phase40_evaluation_uids)
    audit_path = ROOT / "configs/reconstruction/ht_reconstruction_phase46_audit_exclusions_20260912.json"
    audit = json.loads(audit_path.read_text())
    assert audit['role'] == 'validation' and audit['sealed_test_accessed'] is False
    audit_uids = set(audit['event_uids'])
    validation_exclusions.update(audit_uids)
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
    phase36_selection_set = set(phase36_selection_uids)
    phase36_evaluation_set = set(phase36_evaluation_uids)
    phase37_selection_set = set(phase37_selection_uids)
    phase37_evaluation_set = set(phase37_evaluation_uids)
    phase38_selection_set = set(phase38_selection_uids)
    phase38_evaluation_set = set(phase38_evaluation_uids)
    phase39_selection_set = set(phase39_selection_uids)
    phase39_evaluation_set = set(phase39_evaluation_uids)
    phase40_selection_set = set(phase40_selection_uids)
    phase40_evaluation_set = set(phase40_evaluation_uids)
    phase46_selection_set = set(selection_uids)
    phase46_evaluation_set = set(evaluation_uids)
    overlap_audit = {
        "phase35_train_vs_phase46_selection": len(train_set & phase46_selection_set),
        "phase35_train_vs_phase46_evaluation": len(train_set & phase46_evaluation_set),
        "historical_validation_vs_phase46_selection": len(
            historical_set & phase46_selection_set
        ),
        "historical_validation_vs_phase46_evaluation": len(
            historical_set & phase46_evaluation_set
        ),
        "phase35_selection_vs_phase46_selection": len(
            phase35_selection_set & phase46_selection_set
        ),
        "phase35_selection_vs_phase46_evaluation": len(
            phase35_selection_set & phase46_evaluation_set
        ),
        "phase35_evaluation_vs_phase46_selection": len(
            phase35_evaluation_set & phase46_selection_set
        ),
        "phase35_evaluation_vs_phase46_evaluation": len(
            phase35_evaluation_set & phase46_evaluation_set
        ),
        "phase36_selection_vs_phase46_selection": len(
            phase36_selection_set & phase46_selection_set
        ),
        "phase36_selection_vs_phase46_evaluation": len(
            phase36_selection_set & phase46_evaluation_set
        ),
        "phase36_evaluation_vs_phase46_selection": len(
            phase36_evaluation_set & phase46_selection_set
        ),
        "phase36_evaluation_vs_phase46_evaluation": len(
            phase36_evaluation_set & phase46_evaluation_set
        ),
        "phase37_selection_vs_phase46_selection": len(
            phase37_selection_set & phase46_selection_set
        ),
        "phase37_selection_vs_phase46_evaluation": len(
            phase37_selection_set & phase46_evaluation_set
        ),
        "phase37_evaluation_vs_phase46_selection": len(
            phase37_evaluation_set & phase46_selection_set
        ),
        "phase37_evaluation_vs_phase46_evaluation": len(
            phase37_evaluation_set & phase46_evaluation_set
        ),
        "phase38_selection_vs_phase46_selection": len(
            phase38_selection_set & phase46_selection_set
        ),
        "phase38_selection_vs_phase46_evaluation": len(
            phase38_selection_set & phase46_evaluation_set
        ),
        "phase38_evaluation_vs_phase46_selection": len(
            phase38_evaluation_set & phase46_selection_set
        ),
        "phase38_evaluation_vs_phase46_evaluation": len(
            phase38_evaluation_set & phase46_evaluation_set
        ),
        "phase39_selection_vs_phase46_selection": len(
            phase39_selection_set & phase46_selection_set
        ),
        "phase39_selection_vs_phase46_evaluation": len(
            phase39_selection_set & phase46_evaluation_set
        ),
        "phase39_evaluation_vs_phase46_selection": len(
            phase39_evaluation_set & phase46_selection_set
        ),
        "phase39_evaluation_vs_phase46_evaluation": len(
            phase39_evaluation_set & phase46_evaluation_set
        ),
        "phase40_selection_vs_phase46_selection": len(
            phase40_selection_set & phase46_selection_set
        ),
        "phase40_selection_vs_phase46_evaluation": len(
            phase40_selection_set & phase46_evaluation_set
        ),
        "phase40_evaluation_vs_phase46_selection": len(
            phase40_evaluation_set & phase46_selection_set
        ),
        "phase40_evaluation_vs_phase46_evaluation": len(
            phase40_evaluation_set & phase46_evaluation_set
        ),
        "phase46_selection_vs_phase46_evaluation": len(
            phase46_selection_set & phase46_evaluation_set
        ),
    }
    overlap_audit["phase42_vs_phase46"] = len(phase42_uids & (phase46_selection_set | phase46_evaluation_set))
    overlap_audit["phase43_vs_phase46"] = len(phase43_uids & (phase46_selection_set | phase46_evaluation_set))
    overlap_audit["phase41_vs_phase46"] = len(phase41_uids & (phase46_selection_set | phase46_evaluation_set))
    overlap_audit["independent_audit_vs_phase46"] = len(audit_uids & (phase46_selection_set | phase46_evaluation_set))
    overlap_audit["phase45_vs_phase46"] = len(phase45_uids & (phase46_selection_set | phase46_evaluation_set))
    overlap_audit["phase44_vs_phase46"] = len(phase44_uids & (phase46_selection_set | phase46_evaluation_set))
    if any(overlap_audit.values()):
        raise RuntimeError(f"phase46 cohort is not untouched: {overlap_audit}")

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
            "independent_audit": {"path": str(audit_path.relative_to(ROOT)), "sha256": sha256(audit_path)},
            "phase45_cohort": {"path": str(phase45_path.relative_to(ROOT)), "sha256": sha256(phase45_path)},
            "phase44_cohort": {"path": str(phase44_path.relative_to(ROOT)), "sha256": sha256(phase44_path)},
            "phase42_cohort": {"path": str(phase42_path.relative_to(ROOT)), "sha256": sha256(phase42_path)},
            "phase43_cohort": {"path": str(phase43_path.relative_to(ROOT)), "sha256": sha256(phase43_path)},
            "phase41_cohort": {"path": str(phase41_path.relative_to(ROOT)), "sha256": sha256(phase41_path)},
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
            "phase36_cohort": {
                "path": str(PHASE36_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE36_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase36_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase36_selection_uids
                ),
                "evaluation_event_uid_count": len(phase36_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase36_evaluation_uids
                ),
            },
            "phase37_cohort": {
                "path": str(PHASE37_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE37_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase37_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase37_selection_uids
                ),
                "evaluation_event_uid_count": len(phase37_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase37_evaluation_uids
                ),
            },
            "phase38_cohort": {
                "path": str(PHASE38_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE38_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase38_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase38_selection_uids
                ),
                "evaluation_event_uid_count": len(phase38_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase38_evaluation_uids
                ),
            },
            "phase39_cohort": {
                "path": str(PHASE39_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE39_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase39_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase39_selection_uids
                ),
                "evaluation_event_uid_count": len(phase39_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase39_evaluation_uids
                ),
            },
            "phase40_cohort": {
                "path": str(PHASE40_COHORT.relative_to(ROOT)),
                "sha256": sha256(PHASE40_COHORT),
                "checkpoint_selection_event_uid_count": len(
                    phase40_selection_uids
                ),
                "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
                    phase40_selection_uids
                ),
                "evaluation_event_uid_count": len(phase40_evaluation_uids),
                "evaluation_event_uids_sha256": uid_sequence_sha256(
                    phase40_evaluation_uids
                ),
            },
            "phase46_training_role": {
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
