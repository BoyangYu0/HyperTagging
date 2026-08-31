"""Immutable, source-safe training selections built from published shard metadata.

The selection contract is intentionally shard-level.  It validates the small
publication sidecars, completion markers, and parquet footer without reading
the parquet payload.  The marker's parquet digest is retained as a trusted
content reference until the later full dataset-index build revalidates it.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
import unicodedata

import pyarrow.parquet as pq

from hypertagging.preprocessing.schema_v4 import (
    COMPLETION_MARKER_VERSION,
    SCHEMA_VERSION_V4,
)


INVENTORY_VERSION = "hypertagging-training-inventory-v1"
ROLE_MANIFEST_VERSION = "hypertagging-source-role-manifest-v1"
SELECTION_MANIFEST_VERSION = "hypertagging-training-selection-v1"
SUMMARY_VERSION = "hypertagging-training-selection-summary-v1"
HASH_FIELD = "manifest_hash"
SHA256_HEX_LENGTH = 64
SPLIT_VOCABULARY = ("train", "validation", "test")
SELECTION_ENTRY_SPLIT_ORDER = tuple(sorted(SPLIT_VOCABULARY))
SELECTION_ENTRY_KEYS = frozenset(
    {
        "inventory_entry_hash",
        "path",
        "schema_version",
        "campaign_id",
        "campaign_config_digest",
        "source_git_commit",
        "source_git_tree",
        "task_id",
        "task_record_hash",
        "source_file",
        "category",
        "event_count",
        "parquet_sha256_reference",
        "sidecar_sha256",
        "completion_marker_sha256",
        "split",
    }
)
_SELECTION_MANIFEST_TRUE_TEST_KEYS = frozenset(
    {
        "manifest_version",
        "selection_name",
        "data_root",
        "inventory_hash",
        "roles_hash",
        "selection_seed",
        "training_category_shard_quotas",
        "selection_mode",
        "max_events_prefix_allowed",
        "normalizer_scope",
        "uid_validation",
        "split_counts",
        "split_shard_counts",
        "category_split_shard_counts",
        "source_split_isolation",
        "task_split_isolation",
        "entries",
        HASH_FIELD,
    }
)
_SELECTION_MANIFEST_FALSE_TEST_KEYS = _SELECTION_MANIFEST_TRUE_TEST_KEYS | frozenset(
    {"selection_includes_test", "excluded_roles"}
)
_MANIFEST_BINDING_PROVENANCE = object()


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


@dataclass(frozen=True)
class _AuthenticatedManifestBinding:
    """Private one-read manifest provenance token."""

    source: Path
    canonical_bytes: bytes
    payload: dict[str, Any]
    manifest_hash: str
    _provenance: object


@dataclass(frozen=True)
class _ValidatedSelectionMetadata:
    """Canonical lexical values produced by pure manifest validation."""

    included_splits: tuple[str, ...]
    data_root: str
    entry_paths: tuple[str, ...]


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
    if not isinstance(path, (str, Path)):
        raise ValueError("manifest path must be a string or Path")
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {source}") from error
    return _validate_hashed_manifest_payload(
        payload, source=source, expected_version=expected_version
    )


def _authenticated_manifest_binding(
    source: Path, payload: object, canonical_bytes: bytes, *, expected_version: str
) -> _AuthenticatedManifestBinding:
    authenticated = _validate_hashed_manifest_payload(
        payload, source=source, expected_version=expected_version
    )
    pinned = deepcopy(authenticated)
    return _AuthenticatedManifestBinding(
        source=source,
        canonical_bytes=canonical_bytes,
        payload=pinned,
        manifest_hash=str(pinned[HASH_FIELD]),
        _provenance=_MANIFEST_BINDING_PROVENANCE,
    )


def _load_selection_manifest_binding(path: str | Path) -> _AuthenticatedManifestBinding:
    if not isinstance(path, (str, Path)):
        raise ValueError("training selection path must be a string or Path")
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {source}") from error
    return _authenticated_manifest_binding(
        source,
        payload,
        text.encode("utf-8"),
        expected_version=SELECTION_MANIFEST_VERSION,
    )


def _load_selection_manifest_binding_or_none(
    path: str | Path,
) -> _AuthenticatedManifestBinding | None:
    """Read one JSON document, rejecting invalid selection manifests.

    Generic JSON shard lists remain supported, but a document identifying itself
    as a selection manifest is never downgraded to that legacy format after an
    authentication or metadata failure.
    """

    if not isinstance(path, (str, Path)):
        raise ValueError("training selection path must be a string or Path")
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {source}") from error
    manifest_shaped = isinstance(payload, dict) and (
        "manifest_version" in payload
        or any(
            key in payload
            for key in (
                "selection_mode",
                "selection_name",
                "selection_seed",
                "uid_validation",
                "roles_hash",
            )
        )
    )
    if manifest_shaped:
        if not isinstance(payload, dict) or payload.get("manifest_version") != SELECTION_MANIFEST_VERSION:
            raise ValueError("unsupported training selection manifest version")
        authenticated = _validate_hashed_manifest_payload(
            payload,
            source=source,
            expected_version=SELECTION_MANIFEST_VERSION,
        )
        return _AuthenticatedManifestBinding(
            source=source,
            canonical_bytes=text.encode("utf-8"),
            payload=deepcopy(authenticated),
            manifest_hash=str(authenticated[HASH_FIELD]),
            _provenance=_MANIFEST_BINDING_PROVENANCE,
        )
    return None


def _validate_hashed_manifest_payload(
    payload: object,
    *,
    source: str | Path,
    expected_version: str | None = None,
) -> dict[str, Any]:
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
    if candidate.suffix != ".json":
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("manifest_version") == SELECTION_MANIFEST_VERSION
    )


def _validated_training_selection_metadata(
    payload: Mapping[str, Any],
    *,
    include_splits: Iterable[str] | None = None,
    required_splits: Iterable[str] | None = None,
) -> _ValidatedSelectionMetadata:
    """Validate an authenticated selection manifest without touching sources."""

    if not isinstance(payload, Mapping):
        raise ValueError("training selection manifest must be an object")
    if payload.get("manifest_version") != SELECTION_MANIFEST_VERSION:
        raise ValueError("unsupported training selection manifest version")
    payload_keys = frozenset(payload)
    if payload_keys == _SELECTION_MANIFEST_TRUE_TEST_KEYS:
        false_test_variant = False
    elif payload_keys == _SELECTION_MANIFEST_FALSE_TEST_KEYS:
        false_test_variant = True
        if (
            payload.get("selection_includes_test") is not False
            or payload.get("excluded_roles") != ["stress", "test"]
        ):
            raise ValueError("false-test selection variant has invalid test markers")
    else:
        raise ValueError("training selection top-level schema is invalid")
    for field in ("inventory_hash", "roles_hash", HASH_FIELD):
        if not _is_sha256(payload.get(field)):
            raise ValueError(f"training selection {field} is not lowercase SHA-256")
    if payload.get("selection_mode") != "explicit_whole_shard_source_roles":
        raise ValueError("training selection mode is invalid")
    if payload.get("max_events_prefix_allowed") is not False:
        raise ValueError("training selection max-events gate is invalid")
    if payload.get("normalizer_scope") != "train_split_only":
        raise ValueError("training selection normalizer scope is invalid")
    selection_name = payload.get("selection_name")
    if (
        not isinstance(selection_name, str)
        or not selection_name
        or selection_name != selection_name.strip()
        or "\x00" in selection_name
    ):
        raise ValueError("training selection name is invalid")
    selection_seed = payload.get("selection_seed")
    if (
        not isinstance(selection_seed, int)
        or isinstance(selection_seed, bool)
    ):
        raise ValueError("training selection seed is invalid")
    quotas = payload.get("training_category_shard_quotas")
    if not isinstance(quotas, Mapping) or not quotas or any(
        not isinstance(category, str)
        or not category
        or not isinstance(quota, int)
        or isinstance(quota, bool)
        or quota <= 0
        for category, quota in quotas.items()
    ) or list(quotas) != sorted(quotas):
        raise ValueError("training selection category quotas are invalid")
    if payload.get("source_split_isolation") != "validated":
        raise ValueError("training selection source-split isolation is invalid")
    if payload.get("task_split_isolation") != "validated":
        raise ValueError("training selection task-split isolation is invalid")
    uid_validation = payload.get("uid_validation")
    if not isinstance(uid_validation, Mapping) or set(uid_validation) != {
        "status",
        "gate",
    } or (
        uid_validation.get("status") != "pending_full_index_build"
        or uid_validation.get("gate") != "required_before_scientific_training"
    ):
        raise ValueError("training selection UID-validation gate is invalid")
    data_root = _validate_data_root_lexical_path(payload.get("data_root"))

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("training selection contains no entries")
    lexical_paths: set[str] = set()
    canonical_entry_paths: list[str] = []
    source_files: set[str] = set()
    task_ids: set[int] = set()
    split_counts = Counter({split: 0 for split in SPLIT_VOCABULARY})
    split_shard_counts = Counter({split: 0 for split in SPLIT_VOCABULARY})
    previous_order: tuple[int, str, int] | None = None
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"training selection entry {position} must be an object")
        path = _validate_projected_entry(entry, position)
        split = entry.get("split")
        if not isinstance(split, str) or split not in SPLIT_VOCABULARY:
            raise ValueError(f"invalid training selection split at entry {position}")
        if path in lexical_paths:
            raise ValueError("training selection contains duplicate lexical paths")
        lexical_paths.add(path)
        canonical_entry_paths.append(path)
        source_file = entry.get("source_file")
        if (
            not isinstance(source_file, str)
            or not source_file
            or source_file != source_file.strip()
            or "\x00" in source_file
        ):
            raise ValueError(
                f"invalid source_file at training selection entry {position}"
            )
        if source_file in source_files:
            raise ValueError("training selection contains duplicate source_file values")
        source_files.add(source_file)
        task_id = entry.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError(f"invalid task_id at training selection entry {position}")
        if task_id in task_ids:
            raise ValueError("training selection contains duplicate task_id values")
        task_ids.add(task_id)
        category = entry["category"]
        order = (SELECTION_ENTRY_SPLIT_ORDER.index(split), str(category), task_id)
        if previous_order is not None and order < previous_order:
            raise ValueError(
                "training selection entries are not sorted by split, category, task_id"
            )
        previous_order = order
        event_count = entry.get("event_count")
        if (
            not isinstance(event_count, int)
            or isinstance(event_count, bool)
            or event_count <= 0
        ):
            raise ValueError(
                f"invalid event_count at training selection entry {position}"
            )
        split_counts[split] += event_count
        split_shard_counts[split] += 1

    expected_counts = _validate_split_aggregates(
        payload.get("split_counts"),
        field="split_counts",
        allow_omitted_test=false_test_variant,
    )
    expected_shards = _validate_split_aggregates(
        payload.get("split_shard_counts"), field="split_shard_counts"
    )
    actual_counts = {split: split_counts[split] for split in SPLIT_VOCABULARY}
    actual_shards = {split: split_shard_counts[split] for split in SPLIT_VOCABULARY}
    if expected_counts != actual_counts:
        raise ValueError("training selection split_counts disagree with entries")
    if expected_shards != actual_shards:
        raise ValueError("training selection split_shard_counts disagree with entries")
    _validate_category_split_shard_counts(
        payload.get("category_split_shard_counts"), entries
    )
    actual_train_quotas = dict(
        sorted(
            Counter(
                str(entry["category"])
                for entry in entries
                if entry["split"] == "train"
            ).items()
        )
    )
    if dict(quotas) != actual_train_quotas:
        raise ValueError("training selection category quotas disagree with entries")
    positive_splits = tuple(
        split
        for split in SPLIT_VOCABULARY
        if actual_counts[split] > 0 and actual_shards[split] > 0
    )
    includes_test = not false_test_variant
    if "test" in positive_splits:
        if false_test_variant:
            raise ValueError(
                "true-test selection variant must omit test markers; "
                "test-role contract is invalid"
            )
    elif not false_test_variant:
        raise ValueError("false-test selection variant has invalid test markers")
    if includes_test != ("test" in positive_splits):
        raise ValueError("training selection test-role contract is inconsistent")
    included = _validate_requested_splits(
        positive_splits if include_splits is None else include_splits,
        field="included selection splits",
    )
    missing_included = [split for split in included if split not in positive_splits]
    if missing_included:
        raise ValueError(
            "included selection splits have no positive entries/counts/shards: "
            f"{missing_included}"
        )
    if required_splits is not None:
        required = _validate_requested_splits(
            required_splits, field="required selection splits"
        )
        missing_required = [split for split in required if split not in included]
        if missing_required:
            raise ValueError(
                "required selection splits are not included with positive "
                f"entries/counts/shards: {missing_required}"
            )
    return _ValidatedSelectionMetadata(
        included_splits=included,
        data_root=data_root,
        entry_paths=tuple(canonical_entry_paths),
    )


def validate_training_selection_metadata(
    payload: Mapping[str, Any],
    *,
    include_splits: Iterable[str] | None = None,
    required_splits: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate an authenticated selection manifest without touching sources."""

    return _validated_training_selection_metadata(
        payload,
        include_splits=include_splits,
        required_splits=required_splits,
    ).included_splits


def _validate_requested_splits(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f"{field} must be an ordered sequence")
    try:
        requested = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field} must be an ordered sequence") from error
    if not requested:
        raise ValueError(f"{field} must be nonempty")
    previous_index = -1
    seen: list[str] = []
    for split in requested:
        if not isinstance(split, str) or split not in SPLIT_VOCABULARY:
            raise ValueError(f"{field} contains an invalid split")
        if split in seen:
            raise ValueError(f"{field} contains a duplicate split")
        index = SPLIT_VOCABULARY.index(split)
        if index <= previous_index:
            raise ValueError(f"{field} is not an order-preserving subsequence")
        previous_index = index
        seen.append(split)
    return requested


def _validate_data_root_lexical_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or "\\" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("training selection data_root is not a safe lexical path")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("training selection data_root is not a safe lexical path") from error
    lexical = PurePosixPath(value)
    if not lexical.is_absolute() or str(lexical) != value:
        raise ValueError("training selection data_root must be absolute and canonical")
    components = value.split("/")[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ValueError("training selection data_root is not a safe lexical path")
    if not lexical.name:
        raise ValueError("training selection data_root is not a safe lexical path")
    return str(lexical)


def _validate_entry_relative_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or "\\" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(
            "training selection entry path is not a safe builder-relative path"
        )
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(
            "training selection entry path is not a safe builder-relative path"
        ) from error
    lexical = PurePosixPath(value)
    raw_components = value.split("/")
    if ".." in raw_components or ".." in lexical.parts:
        raise ValueError("training selection entry path contains parent traversal")
    if (
        lexical.is_absolute()
        or not lexical.parts
        or lexical == PurePosixPath(".")
        or len(raw_components) != 1
        or raw_components[0] in {"", ".", ".."}
    ):
        raise ValueError(
            "training selection entry path is not a safe one-component "
            "builder-relative path"
        )
    if "." in raw_components or ".." in lexical.parts:
        raise ValueError(
            "training selection entry path contains parent traversal"
        )
    normalized = str(lexical)
    if normalized != value or not normalized.endswith(".parquet"):
        raise ValueError(
            "training selection entry path is not a canonical one-component parquet name"
        )
    return normalized


def _validate_projected_entry(entry: Mapping[str, Any], position: int) -> str:
    if set(entry) != SELECTION_ENTRY_KEYS:
        raise ValueError(f"training selection entry {position} is missing builder metadata")
    schema = entry.get("schema_version")
    if schema != SCHEMA_VERSION_V4:
        raise ValueError(f"training selection entry {position} schema version is invalid")
    for field in ("campaign_id", "source_git_commit", "source_git_tree"):
        value = entry.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"training selection entry {position} {field} is invalid")
    for field in ("source_git_commit", "source_git_tree"):
        if not _is_lower_hex(entry[field], length=40):
            raise ValueError(f"training selection entry {position} {field} is invalid")
    source_file = entry.get("source_file")
    if (
        not isinstance(source_file, str)
        or not source_file
        or source_file != source_file.strip()
        or "\x00" in source_file
    ):
        raise ValueError(f"invalid source_file at training selection entry {position}")
    category = entry.get("category")
    if (
        not isinstance(category, str)
        or not category
        or category != category.strip()
        or "\x00" in category
    ):
        raise ValueError(f"invalid category at training selection entry {position}")
    task_id = entry.get("task_id")
    if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
        raise ValueError(f"invalid task_id at training selection entry {position}")
    event_count = entry.get("event_count")
    if (
        not isinstance(event_count, int)
        or isinstance(event_count, bool)
        or event_count <= 0
    ):
        raise ValueError(f"invalid event_count at training selection entry {position}")
    for field in (
        "inventory_entry_hash",
        "campaign_config_digest",
        "task_record_hash",
        "parquet_sha256_reference",
        "sidecar_sha256",
        "completion_marker_sha256",
    ):
        if not _is_sha256(entry.get(field)):
            raise ValueError(
                f"training selection entry {position} {field} is not lowercase SHA-256"
            )
    path = _validate_entry_relative_path(entry.get("path"))
    return path


def _is_lower_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_split_aggregates(
    value: object, *, field: str, allow_omitted_test: bool = False
) -> dict[str, int]:
    allowed_keys = [set(SPLIT_VOCABULARY)]
    if allow_omitted_test:
        allowed_keys.append({"train", "validation"})
    if not isinstance(value, Mapping) or set(value) not in allowed_keys:
        raise ValueError(
            f"training selection {field} must cover exact split vocabulary"
        )
    output: dict[str, int] = {}
    for split in SPLIT_VOCABULARY:
        count = value.get(split, 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"training selection {field} contains an invalid count")
        output[split] = count
    return output


def _validate_category_split_shard_counts(
    value: object, entries: Iterable[Mapping[str, Any]]
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("training selection category split shard counts are invalid")
    for category, counts in value.items():
        if (
            not isinstance(category, str)
            or not category
            or not isinstance(counts, Mapping)
        ):
            raise ValueError(
                "training selection category split shard counts are invalid"
            )
        for split, count in counts.items():
            if (
                not isinstance(split, str)
                or split not in SPLIT_VOCABULARY
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise ValueError(
                    "training selection category split shard counts are invalid"
                )
    if value != _category_split_counts(entries):
        raise ValueError(
            "training selection category_split_shard_counts disagree with entries"
        )


def load_training_selection(
    path: str | Path,
    *,
    include_splits: Iterable[str] | None = None,
    required_splits: Iterable[str] | None = None,
) -> LoadedTrainingSelection:
    return _load_training_selection_bound(
        _load_selection_manifest_binding(path),
        include_splits=include_splits,
        required_splits=required_splits,
    )


def _load_training_selection_bound(
    binding: _AuthenticatedManifestBinding,
    *,
    include_splits: Iterable[str] | None = None,
    required_splits: Iterable[str] | None = None,
    _index_binding: Any | None = None,
) -> LoadedTrainingSelection:
    _require_authenticated_manifest_binding(binding)
    source = binding.source
    payload = binding.payload
    bound_index_payload = None
    index_resolved_paths: tuple[Path, ...] | None = None
    if _index_binding is not None:
        from hypertagging.data.dataset_index import _require_resolved_index_binding

        bound_index_payload, index_resolved_paths, _resolved_shard_paths = (
            _require_resolved_index_binding(_index_binding)
        )
        validated = _validate_training_selection_index_metadata_pure(
            payload,
            bound_index_payload,
            include_splits=include_splits,
            required_splits=required_splits,
        )
    else:
        validated = _validated_training_selection_metadata(
            payload,
            include_splits=include_splits,
            required_splits=required_splits,
        )
    included = validated.included_splits
    entries = payload["entries"]
    selected_entries_and_paths = [
        (entry, path)
        for entry, path in zip(entries, validated.entry_paths, strict=True)
        if entry["split"] in included
    ]
    selected_entries = [entry for entry, _path in selected_entries_and_paths]
    source = source.resolve()
    root = Path(validated.data_root).resolve()
    resolved_paths = [
        (root / path).resolve() for _entry, path in selected_entries_and_paths
    ]
    if any(not _is_relative_to(parquet, root) for parquet in resolved_paths):
        raise ValueError("selected parquet shard resolves outside data_root")
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("selection contains duplicate resolved parquet paths")
    if index_resolved_paths is not None:
        _validate_selection_against_index_resolved_paths(
            resolved_paths, index_resolved_paths
        )
    paths: list[Path] = []
    source_roles: dict[str, str] = {}
    split_counts = Counter()
    split_shards = Counter()
    source_expectations: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source_roles[entry["source_file"]] = entry["split"]
    file_checks = [parquet.is_file() for parquet in resolved_paths]
    for parquet, is_file in zip(resolved_paths, file_checks, strict=True):
        if not is_file:
            raise FileNotFoundError(f"missing selected parquet shard: {parquet}")
    stats = [parquet.stat() for parquet in resolved_paths]
    inodes = [(stat.st_dev, stat.st_ino) for stat in stats]
    if len(inodes) != len(set(inodes)):
        raise ValueError("selection contains duplicate hardlink parquet inodes")
    publication_files = [
        (
            parquet.with_suffix(parquet.suffix + ".metadata.json"),
            parquet.with_suffix(parquet.suffix + ".complete"),
        )
        for parquet in resolved_paths
    ]
    sidecar_checks = [sidecar.is_file() for sidecar, _marker in publication_files]
    marker_checks = [marker.is_file() for _sidecar, marker in publication_files]
    for (sidecar, _marker), sidecar_present, marker_present in zip(
        publication_files, sidecar_checks, marker_checks, strict=True
    ):
        if not sidecar_present or not marker_present:
            raise ValueError(f"selected shard publication is incomplete: {sidecar}")
    for entry, parquet in zip(selected_entries, resolved_paths, strict=True):
        split = entry["split"]
        source_file = entry["source_file"]
        task_id = entry["task_id"]
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
    expected_counts = {
        name: (payload["split_counts"][name] if name in included else 0)
        for name in SPLIT_VOCABULARY
    }
    expected_shards = {
        name: (payload["split_shard_counts"][name] if name in included else 0)
        for name in SPLIT_VOCABULARY
    }
    actual_counts = {name: int(split_counts.get(name, 0)) for name in SPLIT_VOCABULARY}
    actual_shards = {name: int(split_shards.get(name, 0)) for name in SPLIT_VOCABULARY}
    if actual_counts != expected_counts:
        raise ValueError("selection split event counts disagree with entries")
    if actual_shards != expected_shards:
        raise ValueError("selection split shard counts disagree with entries")
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


def _require_authenticated_manifest_binding(binding: object) -> None:
    if (
        type(binding) is not _AuthenticatedManifestBinding
        or binding._provenance is not _MANIFEST_BINDING_PROVENANCE
    ):
        raise ValueError("training selection provenance binding is invalid")
    try:
        pinned_payload = json.loads(binding.canonical_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training selection provenance binding is invalid") from error
    if (
        pinned_payload != binding.payload
        or binding.manifest_hash != binding.payload.get(HASH_FIELD)
    ):
        raise ValueError("training selection provenance binding was mutated")
    _validate_hashed_manifest_payload(
        binding.payload,
        source=binding.source,
        expected_version=SELECTION_MANIFEST_VERSION,
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_selection_against_index_metadata_pure(
    payload: Mapping[str, Any],
    included: tuple[str, ...],
    selected_entries: list[Mapping[str, Any]],
    lexical_paths: list[str],
    index_payload: Mapping[str, Any],
) -> None:
    """Bind manifest projections to an already authenticated index payload."""

    selection = index_payload.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise ValueError("dataset index selection contract is missing")
    if selection.get("selection_manifest_hash") != payload.get(HASH_FIELD):
        raise ValueError("dataset index training-selection hash mismatch")
    if selection.get("included_splits") != list(included):
        raise ValueError("dataset index included-role contract mismatch")
    index_paths = index_payload.get("paths")
    if not isinstance(index_paths, list) or index_paths != lexical_paths:
        raise ValueError("dataset index shard paths do not match training selection")
    expected_counts = {
        split: sum(
            int(entry["event_count"])
            for entry in selected_entries
            if entry["split"] == split
        )
        for split in SPLIT_VOCABULARY
    }
    index_counts = index_payload.get("split_counts")
    if not isinstance(index_counts, Mapping):
        raise ValueError("dataset index split counts disagree with training selection")
    for split in SPLIT_VOCABULARY:
        count = index_counts.get(split, 0)
        if not isinstance(count, int) or isinstance(count, bool) or count != expected_counts[split]:
            raise ValueError("dataset index split counts disagree with training selection")
    expected_groups = {
        entry["source_file"]: entry["split"] for entry in selected_entries
    }
    index_groups = index_payload.get("source_groups")
    if index_groups != expected_groups:
        raise ValueError("dataset index source groups disagree with training selection")
    expected_shard_counts = {
        split: (
            payload["split_shard_counts"][split] if split in included else 0
        )
        for split in SPLIT_VOCABULARY
    }
    indexed_shard_counts = Counter(
        index_groups[entry["source_file"]] for entry in selected_entries
    )
    if {
        split: int(indexed_shard_counts.get(split, 0))
        for split in SPLIT_VOCABULARY
    } != expected_shard_counts:
        raise ValueError(
            "dataset index split shard counts disagree with training selection"
        )
    descriptors = index_payload.get("shards")
    if not isinstance(descriptors, list) or len(descriptors) != len(selected_entries):
        raise ValueError("dataset index shard descriptors do not match selection")
    for entry, path, descriptor in zip(
        selected_entries, lexical_paths, descriptors, strict=True
    ):
        if not isinstance(descriptor, Mapping):
            raise ValueError("dataset index shard descriptor is invalid")
        if descriptor.get("path") != str(path):
            raise ValueError("dataset index shard paths do not match training selection")
        if descriptor.get("event_count") != entry["event_count"]:
            raise ValueError("dataset index shard counts disagree with selection")
        if descriptor.get("schema") != entry["schema_version"]:
            raise ValueError("dataset index shard schema disagrees with selection")
        if descriptor.get("source_digest") != entry["parquet_sha256_reference"]:
            raise ValueError("dataset index parquet digest disagrees with selection")
        if descriptor.get("sidecar_hash") != entry["sidecar_sha256"]:
            raise ValueError("dataset index sidecar digest disagrees with selection")
        if descriptor.get("completion_marker_hash") != entry[
            "completion_marker_sha256"
        ]:
            raise ValueError(
                "dataset index completion-marker digest disagrees with selection"
            )


def _validate_training_selection_index_metadata_pure(
    payload: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    *,
    include_splits: Iterable[str] | None,
    required_splits: Iterable[str] | None,
) -> _ValidatedSelectionMetadata:
    """Purely bind all manifest projections to authenticated index metadata."""

    validated = _validated_training_selection_metadata(
        payload,
        include_splits=include_splits,
        required_splits=required_splits,
    )
    included = validated.included_splits
    entries = payload["entries"]
    selected_entries_and_paths = [
        (entry, path)
        for entry, path in zip(entries, validated.entry_paths, strict=True)
        if entry["split"] in included
    ]
    selected_entries = [entry for entry, _path in selected_entries_and_paths]
    lexical_root = PurePosixPath(validated.data_root)
    lexical_paths = [
        str(lexical_root / path) for _entry, path in selected_entries_and_paths
    ]
    _validate_selection_against_index_metadata_pure(
        payload, included, selected_entries, lexical_paths, index_payload
    )
    return validated


def _validate_selection_against_index_resolved_paths(
    resolved_paths: list[Path], index_resolved_paths: tuple[Path, ...]
) -> None:
    if tuple(resolved_paths) != index_resolved_paths:
        raise ValueError("dataset index shard paths do not match training selection")


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
    if not isinstance(include_test, bool):
        raise ValueError("include_test must be boolean")
    if (
        not isinstance(training_quotas, Mapping)
        or not training_quotas
        or any(
            not isinstance(category, str)
            or not category
            or not isinstance(quota, int)
            or isinstance(quota, bool)
            or quota <= 0
            for category, quota in training_quotas.items()
        )
    ):
        raise ValueError("training selection category quotas are invalid")
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
        key=lambda entry: (
            SELECTION_ENTRY_SPLIT_ORDER.index(entry["split"]),
            entry["category"],
            entry["task_id"],
        )
    )
    _validate_selection_isolation(selected)
    split_counts = Counter()
    split_shards = Counter()
    for entry in selected:
        split_counts[entry["split"]] += int(entry["event_count"])
        split_shards[entry["split"]] += 1
    if include_test and (
        split_counts.get("test", 0) <= 0 or split_shards.get("test", 0) <= 0
    ):
        raise ValueError("true-test selection requires positive test counts and shards")
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
    campaign_config_digest = entry.get("campaign_config_digest")
    if not _is_sha256(campaign_config_digest):
        raise ValueError("inventory campaign_config_digest is not lowercase SHA-256")
    return {
        "inventory_entry_hash": entry["inventory_entry_hash"],
        "path": entry["path"],
        "schema_version": entry["schema_version"],
        "campaign_id": entry["campaign_id"],
        "campaign_config_digest": campaign_config_digest,
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
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
    "validate_training_selection_metadata",
    "validate_nested_selections",
    "write_hashed_manifest",
]
