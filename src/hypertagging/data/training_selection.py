"""Immutable, source-safe training selections built from published shard metadata.

The selection contract is intentionally shard-level.  It validates the small
publication sidecars, completion markers, and parquet footer without reading
the parquet payload.  The marker's parquet digest is retained as a trusted
content reference until the later full dataset-index build revalidates it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from hypertagging.preprocessing.schema_v4 import COMPLETION_MARKER_VERSION


INVENTORY_VERSION = "hypertagging-training-inventory-v1"
ROLE_MANIFEST_VERSION = "hypertagging-source-role-manifest-v1"
SELECTION_MANIFEST_VERSION = "hypertagging-training-selection-v1"
SUMMARY_VERSION = "hypertagging-training-selection-summary-v1"
HASH_FIELD = "manifest_hash"
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class LoadedTrainingSelection:
    manifest_path: Path
    manifest_hash: str
    paths: tuple[Path, ...]
    source_split_overrides: dict[str, str]
    split_counts: dict[str, int]
    split_shard_counts: dict[str, int]
    included_splits: tuple[str, ...]
    source_expectations: dict[str, dict[str, Any]]


def canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != HASH_FIELD}
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_hashed_manifest(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document[HASH_FIELD] = canonical_manifest_hash(document)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def load_hashed_manifest(
    path: str | Path,
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {source}")
    if payload.get(HASH_FIELD) != canonical_manifest_hash(payload):
        raise ValueError(f"manifest hash mismatch: {source}")
    if (
        expected_version is not None
        and payload.get("manifest_version") != expected_version
    ):
        raise ValueError(f"unsupported manifest version in {source}")
    return payload


def is_training_selection_manifest(path: str | Path) -> bool:
    candidate = Path(path)
    if candidate.suffix != ".json" or not candidate.is_file():
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("manifest_version") == SELECTION_MANIFEST_VERSION
    )


def load_training_selection(
    path: str | Path,
    *,
    include_splits: Iterable[str] | None = None,
) -> LoadedTrainingSelection:
    source = Path(path).resolve()
    payload = load_hashed_manifest(source, expected_version=SELECTION_MANIFEST_VERSION)
    included = tuple(
        dict.fromkeys(include_splits or ("train", "validation", "test"))
    )
    invalid_included = set(included) - {"train", "validation", "test"}
    if invalid_included or not included:
        raise ValueError(f"invalid included selection splits: {sorted(invalid_included)}")
    root = Path(payload["data_root"])
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("training selection contains no entries")
    paths: list[Path] = []
    source_roles: dict[str, str] = {}
    split_counts = Counter()
    split_shards = Counter()
    task_splits: dict[int, str] = {}
    source_expectations: dict[str, dict[str, Any]] = {}
    for entry in entries:
        split = str(entry.get("split", ""))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid selection split {split!r}")
        source_file = str(entry.get("source_file", ""))
        if not source_file:
            raise ValueError("selection entry is missing source_file")
        task_id = int(entry["task_id"])
        previous_source = source_roles.setdefault(source_file, split)
        if previous_source != split:
            raise ValueError(f"source {source_file!r} occurs in multiple splits")
        previous_task = task_splits.setdefault(task_id, split)
        if previous_task != split:
            raise ValueError(f"task {task_id} occurs in multiple splits")
        if split not in included:
            continue
        relative = Path(str(entry["path"]))
        parquet = relative if relative.is_absolute() else root / relative
        parquet = parquet.resolve()
        if not parquet.is_file():
            raise FileNotFoundError(f"missing selected parquet shard: {parquet}")
        _validate_selection_publication(parquet, entry)
        paths.append(parquet)
        source_expectations[source_file] = {
            "category": str(entry["category"]),
            "task_id": task_id,
            "path": str(parquet),
            "split": split,
            "event_count": int(entry["event_count"]),
            "task_record_hash": str(entry["task_record_hash"]),
        }
        split_counts[split] += int(entry["event_count"])
        split_shards[split] += 1
    if len(paths) != len(set(paths)):
        raise ValueError("selection contains duplicate parquet paths")
    expected_counts = {
        name: (int(payload.get("split_counts", {}).get(name, 0)) if name in included else 0)
        for name in ("train", "validation", "test")
    }
    expected_shards = {
        name: (
            int(payload.get("split_shard_counts", {}).get(name, 0))
            if name in included
            else 0
        )
        for name in ("train", "validation", "test")
    }
    actual_counts = {
        name: int(split_counts.get(name, 0))
        for name in ("train", "validation", "test")
    }
    actual_shards = {
        name: int(split_shards.get(name, 0))
        for name in ("train", "validation", "test")
    }
    if actual_counts != expected_counts:
        raise ValueError("selection split event counts disagree with entries")
    if actual_shards != expected_shards:
        raise ValueError("selection split shard counts disagree with entries")
    if payload.get("uid_validation", {}).get("status") != "pending_full_index_build":
        raise ValueError("selection UID-validation gate is missing or ambiguous")
    return LoadedTrainingSelection(
        manifest_path=source,
        manifest_hash=str(payload[HASH_FIELD]),
        paths=tuple(paths),
        source_split_overrides=source_roles,
        split_counts=expected_counts,
        split_shard_counts=expected_shards,
        included_splits=included,
        source_expectations=source_expectations,
    )


def inventory_publications(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root).resolve()
    parquet_paths = sorted(root.glob("*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no parquet shards found in {root}")
    entries = [_inventory_entry(path, root) for path in parquet_paths]
    _validate_unique_sources_and_tasks(entries)
    campaigns = sorted({str(entry["campaign_id"]) for entry in entries})
    source_commits = sorted({str(entry["source_git_commit"]) for entry in entries})
    source_trees = sorted({str(entry["source_git_tree"]) for entry in entries})
    schemas = sorted({str(entry["schema_version"]) for entry in entries})
    return {
        "manifest_version": INVENTORY_VERSION,
        "data_root": str(root),
        "content_validation_scope": {
            "parquet_payload": "trusted_completion_marker_sha256_reference_not_rehashed",
            "parquet_footer": "row_count_and_schema_checked",
            "sidecar": "sha256_checked_against_completion_marker",
            "completion_marker": "sha256_recorded_and_contract_checked",
        },
        "uid_validation": {
            "status": "pending_full_index_build",
            "gate": "required_before_scientific_training",
        },
        "campaigns": campaigns,
        "source_git_commits": source_commits,
        "source_git_trees": source_trees,
        "schema_versions": schemas,
        "shard_count": len(entries),
        "event_count": sum(int(entry["event_count"]) for entry in entries),
        "category_shard_counts": dict(
            sorted(Counter(entry["category"] for entry in entries).items())
        ),
        "category_event_counts": dict(
            sorted(
                Counter(
                    {
                        category: sum(
                            int(entry["event_count"])
                            for entry in entries
                            if entry["category"] == category
                        )
                        for category in {entry["category"] for entry in entries}
                    }
                ).items()
            )
        ),
        "entries": entries,
    }


def assign_source_roles(
    inventory: Mapping[str, Any],
    *,
    seed: int,
    validation_quotas: Mapping[str, int],
    test_quotas: Mapping[str, int],
    stress_quotas: Mapping[str, int],
) -> dict[str, Any]:
    entries = [dict(entry) for entry in inventory["entries"]]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_category.setdefault(str(entry["category"]), []).append(entry)
    assignments: list[dict[str, Any]] = []
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda entry: (
                _rank(seed, category, entry),
                int(entry["task_id"]),
                str(entry["source_file"]),
            ),
        )
        quotas = {
            "validation": int(validation_quotas.get(category, 0)),
            "test": int(test_quotas.get(category, 0)),
            "stress": int(stress_quotas.get(category, 0)),
        }
        if sum(quotas.values()) > len(ranked):
            raise ValueError(f"held-out quotas exceed available {category} shards")
        offset = 0
        for role in ("validation", "test", "stress"):
            for entry in ranked[offset : offset + quotas[role]]:
                assignments.append(
                    _role_entry(entry, role, _rank(seed, category, entry))
                )
            offset += quotas[role]
        for entry in ranked[offset:]:
            assignments.append(
                _role_entry(entry, "training_pool", _rank(seed, category, entry))
            )
    assignments.sort(
        key=lambda entry: (
            entry["role"],
            entry["category"],
            entry["rank"],
            entry["task_id"],
        )
    )
    _validate_role_isolation(assignments)
    return {
        "manifest_version": ROLE_MANIFEST_VERSION,
        "inventory_hash": inventory[HASH_FIELD],
        "selection_seed": int(seed),
        "allocation_order": ["validation", "test", "stress", "training_pool"],
        "quota_shards": {
            "validation": dict(sorted(validation_quotas.items())),
            "test": dict(sorted(test_quotas.items())),
            "stress": dict(sorted(stress_quotas.items())),
        },
        "role_shard_counts": dict(
            sorted(Counter(entry["role"] for entry in assignments).items())
        ),
        "role_event_counts": dict(
            sorted(
                {
                    role: sum(
                        int(entry["event_count"])
                        for entry in assignments
                        if entry["role"] == role
                    )
                    for role in {entry["role"] for entry in assignments}
                }.items()
            )
        ),
        "category_role_shard_counts": _category_role_counts(assignments),
        "source_role_isolation": "validated",
        "task_role_isolation": "validated",
        "entries": assignments,
    }


def build_training_selection(
    inventory: Mapping[str, Any],
    roles: Mapping[str, Any],
    *,
    selection_name: str,
    training_quotas: Mapping[str, int],
    include_test: bool = True,
) -> dict[str, Any]:
    inventory_entries = {
        str(entry["inventory_entry_hash"]): entry for entry in inventory["entries"]
    }
    role_entries = list(roles["entries"])
    selected: list[dict[str, Any]] = []
    held_out_roles = [("validation", "validation")]
    if include_test:
        held_out_roles.append(("test", "test"))
    for role, split in held_out_roles:
        selected.extend(
            _selection_entry(inventory_entries[entry["inventory_entry_hash"]], split)
            for entry in role_entries
            if entry["role"] == role
        )
    for category, quota in sorted(training_quotas.items()):
        candidates = sorted(
            (
                entry
                for entry in role_entries
                if entry["role"] == "training_pool" and entry["category"] == category
            ),
            key=lambda entry: (entry["rank"], entry["task_id"]),
        )
        if int(quota) > len(candidates):
            raise ValueError(f"training quota exceeds available {category} shards")
        selected.extend(
            _selection_entry(inventory_entries[entry["inventory_entry_hash"]], "train")
            for entry in candidates[: int(quota)]
        )
    selected.sort(
        key=lambda entry: (entry["split"], entry["category"], entry["task_id"])
    )
    _validate_selection_isolation(selected)
    split_counts = Counter()
    split_shards = Counter()
    for entry in selected:
        split_counts[entry["split"]] += int(entry["event_count"])
        split_shards[entry["split"]] += 1
    if not include_test:
        # Keep the excluded role explicit in the machine-readable contract.
        split_counts["test"] = 0
        split_shards["test"] = 0
    return {
        "manifest_version": SELECTION_MANIFEST_VERSION,
        "selection_name": selection_name,
        "data_root": inventory["data_root"],
        "inventory_hash": inventory[HASH_FIELD],
        "roles_hash": roles[HASH_FIELD],
        "selection_seed": roles["selection_seed"],
        "training_category_shard_quotas": dict(sorted(training_quotas.items())),
        "selection_mode": "explicit_whole_shard_source_roles",
        **(
            {
                "selection_includes_test": False,
                "excluded_roles": ["stress", "test"],
            }
            if not include_test
            else {}
        ),
        "max_events_prefix_allowed": False,
        "normalizer_scope": "train_split_only",
        "uid_validation": {
            "status": "pending_full_index_build",
            "gate": "required_before_scientific_training",
        },
        "split_counts": dict(sorted(split_counts.items())),
        "split_shard_counts": dict(sorted(split_shards.items())),
        "category_split_shard_counts": _category_split_counts(selected),
        "source_split_isolation": "validated",
        "task_split_isolation": "validated",
        "entries": selected,
    }


def validate_nested_selections(selections: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(selections, key=lambda item: int(item["split_counts"]["train"]))
    previous: set[str] = set()
    held_out: set[tuple[str, str]] | None = None
    for selection in ordered:
        train = {
            str(entry["inventory_entry_hash"])
            for entry in selection["entries"]
            if entry["split"] == "train"
        }
        if not previous.issubset(train):
            raise ValueError("training selections are not nested")
        previous = train
        current_held_out = {
            (str(entry["inventory_entry_hash"]), str(entry["split"]))
            for entry in selection["entries"]
            if entry["split"] in {"validation", "test"}
        }
        if held_out is None:
            held_out = current_held_out
        elif current_held_out != held_out:
            raise ValueError("validation/test pools differ between selections")


def _inventory_entry(path: Path, root: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    marker = path.with_suffix(path.suffix + ".complete")
    if not sidecar.is_file() or not marker.is_file():
        raise ValueError(f"incomplete shard publication: {path}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        completion = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid publication JSON for {path}") from error
    if completion.get("marker_schema_version") != COMPLETION_MARKER_VERSION:
        raise ValueError(f"unsupported completion marker schema for {path}")
    for marker_key, metadata_key in (
        ("schema_version", "schema_version"),
        ("event_count", "event_count"),
        ("feature_spec_hash", "feature_spec_hash"),
        ("model_feature_contract_hash", "model_feature_contract_hash"),
        ("campaign_id", "campaign_id"),
        ("source_git_commit", "source_git_commit"),
        ("source_git_tree", "source_git_tree"),
        ("task_record_hash", "task_record_hash"),
        ("task_id", "task_id"),
        ("source_file", "source_file"),
        ("physics_category", "physics_category"),
    ):
        if completion.get(marker_key) != metadata.get(metadata_key):
            raise ValueError(f"marker {marker_key} disagrees with sidecar for {path}")
    sidecar_hash = _sha256_file(sidecar)
    if completion.get("sidecar_sha256") != sidecar_hash:
        raise ValueError(f"completion marker sidecar digest mismatch for {path}")
    parquet_hash = str(completion.get("parquet_sha256", ""))
    if not _is_sha256(parquet_hash):
        raise ValueError(f"completion marker parquet digest is invalid for {path}")
    parquet_file = pq.ParquetFile(path)
    event_count = int(metadata.get("event_count", -1))
    if parquet_file.metadata.num_rows != event_count:
        raise ValueError(f"parquet footer row count disagrees with sidecar for {path}")
    if "event_json" not in parquet_file.schema_arrow.names:
        raise ValueError(f"expected schema-v4 event_json field in {path}")
    entry: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sidecar_path": str(sidecar.relative_to(root)),
        "completion_marker_path": str(marker.relative_to(root)),
        "schema_version": str(metadata["schema_version"]),
        "campaign_id": str(metadata["campaign_id"]),
        "campaign_config_digest": str(metadata.get("campaign_config_digest", "")),
        "source_git_commit": str(metadata["source_git_commit"]),
        "source_git_tree": str(metadata["source_git_tree"]),
        "source_state": str(metadata.get("source_state", "")),
        "task_id": int(metadata["task_id"]),
        "task_record_hash": str(metadata["task_record_hash"]),
        "category": str(metadata.get("physics_category") or metadata.get("category")),
        "source_file": str(metadata["source_file"]),
        "source_file_identity": str(metadata.get("source_file_identity", "")),
        "source_file_sha256": str(metadata.get("source_file_sha256", "")),
        "entry_start": metadata.get("entry_start"),
        "entry_stop_exclusive": metadata.get("entry_stop_exclusive"),
        "event_count": event_count,
        "planned_events": int(metadata.get("planned_events", event_count)),
        "klm_training_scope": str(metadata.get("klm_training_scope", "")),
        "parquet_size_bytes": path.stat().st_size,
        "parquet_sha256_reference": parquet_hash,
        "sidecar_sha256": sidecar_hash,
        "completion_marker_sha256": _sha256_file(marker),
        "completion_marker_schema_version": str(completion["marker_schema_version"]),
    }
    entry["inventory_entry_hash"] = hashlib.sha256(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return entry


def _validate_selection_publication(path: Path, entry: Mapping[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    marker = path.with_suffix(path.suffix + ".complete")
    if not sidecar.is_file() or not marker.is_file():
        raise ValueError(f"selected shard publication is incomplete: {path}")
    if _sha256_file(sidecar) != entry.get("sidecar_sha256"):
        raise ValueError(f"selected shard sidecar hash mismatch: {path}")
    if _sha256_file(marker) != entry.get("completion_marker_sha256"):
        raise ValueError(f"selected shard completion-marker hash mismatch: {path}")
    try:
        completion = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid selected shard completion marker: {path}") from error
    if completion.get("parquet_sha256") != entry.get("parquet_sha256_reference"):
        raise ValueError(f"selected shard parquet hash reference mismatch: {path}")


def _rank(seed: int, category: str, entry: Mapping[str, Any]) -> str:
    value = "|".join(
        (
            str(seed),
            category,
            str(entry["task_id"]),
            str(entry["source_file"]),
            str(entry["task_record_hash"]),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _role_entry(entry: Mapping[str, Any], role: str, rank: str) -> dict[str, Any]:
    return {
        "inventory_entry_hash": entry["inventory_entry_hash"],
        "task_id": entry["task_id"],
        "source_file": entry["source_file"],
        "category": entry["category"],
        "event_count": entry["event_count"],
        "role": role,
        "rank": rank,
    }


def _selection_entry(entry: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "inventory_entry_hash": entry["inventory_entry_hash"],
        "path": entry["path"],
        "schema_version": entry["schema_version"],
        "campaign_id": entry["campaign_id"],
        "source_git_commit": entry["source_git_commit"],
        "source_git_tree": entry["source_git_tree"],
        "task_id": entry["task_id"],
        "task_record_hash": entry["task_record_hash"],
        "source_file": entry["source_file"],
        "category": entry["category"],
        "event_count": entry["event_count"],
        "parquet_sha256_reference": entry["parquet_sha256_reference"],
        "sidecar_sha256": entry["sidecar_sha256"],
        "completion_marker_sha256": entry["completion_marker_sha256"],
        "split": split,
    }


def _validate_unique_sources_and_tasks(entries: Iterable[Mapping[str, Any]]) -> None:
    sources: set[str] = set()
    tasks: set[int] = set()
    for entry in entries:
        source = str(entry["source_file"])
        task = int(entry["task_id"])
        if source in sources:
            raise ValueError(f"duplicate source_file in inventory: {source}")
        if task in tasks:
            raise ValueError(f"duplicate task_id in inventory: {task}")
        sources.add(source)
        tasks.add(task)


def _validate_role_isolation(entries: Iterable[Mapping[str, Any]]) -> None:
    source_roles: dict[str, str] = {}
    task_roles: dict[int, str] = {}
    for entry in entries:
        role = str(entry["role"])
        source = str(entry["source_file"])
        task = int(entry["task_id"])
        if source_roles.setdefault(source, role) != role:
            raise ValueError(f"source {source!r} leaks across roles")
        if task_roles.setdefault(task, role) != role:
            raise ValueError(f"task {task} leaks across roles")


def _validate_selection_isolation(entries: Iterable[Mapping[str, Any]]) -> None:
    source_splits: dict[str, str] = {}
    task_splits: dict[int, str] = {}
    for entry in entries:
        split = str(entry["split"])
        source = str(entry["source_file"])
        task = int(entry["task_id"])
        if source_splits.setdefault(source, split) != split:
            raise ValueError(f"source {source!r} leaks across splits")
        if task_splits.setdefault(task, split) != split:
            raise ValueError(f"task {task} leaks across splits")


def _category_role_counts(
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for entry in entries:
        counts.setdefault(str(entry["category"]), Counter())[str(entry["role"])] += 1
    return {
        category: dict(sorted(values.items()))
        for category, values in sorted(counts.items())
    }


def _category_split_counts(
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for entry in entries:
        counts.setdefault(str(entry["category"]), Counter())[str(entry["split"])] += 1
    return {
        category: dict(sorted(values.items()))
        for category, values in sorted(counts.items())
    }


def _is_sha256(value: str) -> bool:
    return len(value) == SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "INVENTORY_VERSION",
    "ROLE_MANIFEST_VERSION",
    "SELECTION_MANIFEST_VERSION",
    "SUMMARY_VERSION",
    "LoadedTrainingSelection",
    "assign_source_roles",
    "build_training_selection",
    "canonical_manifest_hash",
    "inventory_publications",
    "is_training_selection_manifest",
    "load_hashed_manifest",
    "load_training_selection",
    "validate_nested_selections",
    "write_hashed_manifest",
]
