#!/usr/bin/env python3
"""Build the fresh validation-only phase-35 full-decay evaluation cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase35_improvement_20260904.json"
)
FIXED_VALIDATION_VERSION = "manifest-role-uid-hash-v1"
MANIFEST_VERSION = "hypertagging-reconstruction-evaluation-cohort-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uid_sequence_sha256(event_uids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"hypertagging-evaluation-event-uids-v1\0")
    for uid in event_uids:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def uid_set_sha256(event_uids: list[str]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(set(event_uids)):
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def ranked_validation_uids(
    manifest: dict[str, Any],
    *,
    excluded: set[str],
    seed: int,
    limit: int,
) -> list[tuple[str, str]]:
    from hypertagging.preprocessing.schema_v4 import iter_event_records_v4

    data_root = Path(str(manifest["data_root"])).resolve(strict=True)
    ranked: list[tuple[int, int, str, str]] = []
    stream_index = 0
    for entry in manifest["entries"]:
        if entry.get("split") != "validation":
            continue
        shard = (data_root / str(entry["path"])).resolve(strict=True)
        for record in iter_event_records_v4(shard):
            uid = str(record["event_uid"])
            if uid in excluded:
                stream_index += 1
                continue
            rank = int.from_bytes(
                hashlib.sha256(
                    f"{FIXED_VALIDATION_VERSION}:{seed}:{uid}".encode()
                ).digest(),
                byteorder="big",
            )
            category = str(record.get("source_category", entry["category"]))
            candidate = (-rank, stream_index, uid, category)
            if len(ranked) < limit:
                heapq.heappush(ranked, candidate)
            elif rank < -ranked[0][0]:
                heapq.heapreplace(ranked, candidate)
            stream_index += 1
    if stream_index != 50_000:
        raise RuntimeError(
            f"validation-only scan found {stream_index} events, expected 50000"
        )
    chosen = sorted(ranked, key=lambda item: (-item[0], item[2]))
    if len(chosen) != limit:
        raise RuntimeError("validation role is too small for reserved cohorts")
    return [(item[2], item[3]) for item in chosen]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    data = prereg["data_binding"]
    selection_path = (ROOT / data["selection_manifest"]).resolve(strict=True)
    index_path = (ROOT / data["dataset_index"]).resolve(strict=True)
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
    exclusion_binding = prereg["validation_exclusion"]
    exclusion_path = (ROOT / exclusion_binding["manifest"]).resolve(strict=True)
    if sha256(exclusion_path) != exclusion_binding["manifest_sha256"]:
        raise RuntimeError("historical validation exclusion manifest changed")
    exclusions = json.loads(exclusion_path.read_text(encoding="utf-8"))[
        "event_uids"
    ]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    seed = int(prereg["common_training_contract"]["seed"])
    selected = ranked_validation_uids(
        manifest,
        excluded=set(map(str, exclusions)),
        seed=seed,
        limit=2_100,
    )
    training_selection = [uid for uid, _ in selected[:2_000]]
    evaluation = [uid for uid, _ in selected[2_000:]]
    if set(training_selection) & set(evaluation):
        raise RuntimeError("phase35 selection and evaluation cohorts overlap")
    if (set(training_selection) | set(evaluation)) & set(exclusions):
        raise RuntimeError("phase35 cohorts reuse historically inspected events")
    categories = Counter(category for _, category in selected[2_000:])
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_id": prereg["study_id"],
        "role": "validation",
        "sealed_test_role_access": "forbidden",
        "selection_algorithm": FIXED_VALIDATION_VERSION,
        "seed": seed,
        "selection_manifest": data["selection_manifest"],
        "selection_manifest_sha256": data["selection_manifest_sha256"],
        "dataset_index": data["dataset_index"],
        "dataset_index_sha256": data["dataset_index_sha256"],
        "historical_exclusion_count": len(exclusions),
        "historical_exclusion_uids_sha256": uid_set_sha256(exclusions),
        "checkpoint_selection_event_uid_count": len(training_selection),
        "checkpoint_selection_event_uids_sha256": uid_sequence_sha256(
            training_selection
        ),
        "checkpoint_selection_event_uids": training_selection,
        "event_uid_count": len(evaluation),
        "event_uids_sha256": uid_sequence_sha256(evaluation),
        "event_uids": evaluation,
        "source_category_counts": dict(sorted(categories.items())),
        "overlap_audit": {
            "historical_exclusions_vs_checkpoint_selection": 0,
            "historical_exclusions_vs_evaluation": 0,
            "checkpoint_selection_vs_evaluation": 0,
        },
    }
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("cohort manifest output must remain in repository") from error
    if output.exists():
        raise RuntimeError(f"refusing to overwrite cohort manifest: {output}")
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "checkpoint_selection_events": len(training_selection),
                "evaluation_events": len(evaluation),
                "source_category_counts": dict(sorted(categories.items())),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
