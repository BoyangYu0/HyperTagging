"""Versioned one-pass dataset index and mergeable sufficient statistics."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import torch

from hypertagging.data.heterogeneous import heterogeneous_event_from_record
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.data.streaming import StreamingMaskedFeatureNormalizer
from hypertagging.preprocessing.schema_v4 import (
    COMPLETION_MARKER_VERSION,
    FEATURE_SPEC_REVISION_V4,
    LEAF_MODE_TO_ID,
    SCHEMA_VERSION_V4,
    TARGET_COMPOSITE_METADATA_INDICES,
    feature_spec_v4,
    iter_event_records_v4,
)
from hypertagging.preprocessing.schema_v2 import SCHEMA_VERSION_V1, SCHEMA_VERSION_V2
from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PID_VOCABULARY_VERSION


DATASET_INDEX_VERSION = "hypertagging-dataset-index-v3"
SUPPORTED_SCHEMAS = {
    SCHEMA_VERSION_V1,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
}
FEATURE_BLOCKS = ("common", "track", "cluster", "composite")
MAX_CHANNEL_FREQUENCY_SLICE_SIGNATURES = 4096
MAX_ALLOWED_TYPE_LEVEL = 32
_COMMON_INDEX_KEYS = frozenset(
    {
        "index_version",
        "paths",
        "event_count",
        "node_count",
        "schema_versions",
        "feature_spec_hashes",
        "track_fit_policies",
        "pid_vocabulary_version",
        "split_config",
        "split_counts",
        "source_groups",
        "category_counts",
        "legacy_fraction",
        "normalizer_state",
        "normalizer_scope",
        "allowed_types_by_level",
        "mother_count_histograms_by_level",
        "daughter_cardinality_histogram",
        "daughter_cardinality_histograms_by_level",
        "depth_distribution",
        "target_policy",
        "target_policy_counts",
        "policy_capacity_statistics",
        "shards",
        "feature_spec_revision",
        "feature_spec_hash",
        "supported_schema_set",
        "selection_contract",
        "index_hash",
    }
)
_FULL_INDEX_KEYS = _COMMON_INDEX_KEYS | frozenset(
    {
        "capacity_slices_by_level",
        "channel_frequency_histogram",
        "channel_frequency_slice_coverage",
        "full_truth_to_reconstructable_channel_collisions",
        "event_identity_validation",
    }
)
_SIDECAR_INDEX_KEYS = _COMMON_INDEX_KEYS | frozenset({"index_source"})
_INDEX_BINDING_PROVENANCE = object()
_RESOLVED_INDEX_BINDING_PROVENANCE = object()


@dataclass(frozen=True)
class _AuthenticatedIndexBinding:
    """Private one-read dataset-index provenance token."""

    source: Path
    canonical_bytes: bytes
    payload: dict[str, Any]
    index_hash: str
    _provenance: object


@dataclass(frozen=True)
class _ResolvedIndexBinding:
    """Private authenticated index token with a pinned path-resolution phase."""

    authenticated: _AuthenticatedIndexBinding
    resolved_paths: tuple[Path, ...]
    resolved_shard_paths: tuple[Path, ...]
    _provenance: object


def build_dataset_index(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    split_config: SourceAwareSplitConfig | None = None,
    target_policy: str = "complete_only",
    max_events: int | None = None,
    source_split_overrides: Mapping[str, str] | None = None,
    selection_manifest_hash: str | None = None,
    selection_included_splits: Iterable[str] | None = None,
    source_expectations: Mapping[str, Mapping[str, Any]] | None = None,
    require_event_identity_validation: bool = False,
) -> Path:
    """Scan once, then persist all startup statistics needed by trainers."""

    config = split_config or SourceAwareSplitConfig()
    source_split_overrides = dict(source_split_overrides or {})
    source_expectations = {
        str(source): dict(expectation)
        for source, expectation in dict(source_expectations or {}).items()
    }
    included_splits = tuple(
        selection_included_splits
        or (("train", "validation", "test") if selection_manifest_hash else ())
    )
    if require_event_identity_validation and (
        selection_manifest_hash is None or not source_expectations
    ):
        raise ValueError(
            "scientific event-identity validation requires selection source expectations"
        )
    if require_event_identity_validation and included_splits != (
        "train",
        "validation",
    ):
        raise ValueError(
            "scientific event-identity indexing permits exactly train and validation"
        )
    if selection_manifest_hash is not None and max_events is not None:
        raise ValueError("source-role selection cannot be combined with max_events")
    normalizers = {name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS}
    all_split_normalizers = {
        name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS
    }
    split_counts = Counter()
    category_counts = Counter()
    legacy_nodes = total_nodes = event_count = 0
    allowed_types: dict[int, set[int]] = {}
    mother_count_histograms: dict[int, Counter[int]] = {}
    daughter_cardinality = Counter()
    daughter_cardinality_by_level: dict[int, Counter[int]] = {}
    depth_distribution = Counter()
    target_counts = Counter()
    source_groups: dict[str, str] = {}
    schema_versions = set()
    feature_spec_hashes = set()
    track_fit_policies = set()
    shards: list[dict[str, Any]] = []
    policy_capacity = {
        policy: Counter()
        for policy in ("complete_only", "reconstructable_partial", "diagnostic_all")
    }
    capacity_slices: dict[int, dict[str, dict[str, dict[str, int]]]] = {}
    channel_frequency: Counter[str] = Counter()
    channel_capacity: dict[int, dict[str, dict[str, int]]] = {}
    channel_frequency_slice_overflow_events = 0
    channel_projection_groups: dict[int, set[int]] = {}
    channel_projection_event_counts: Counter[int] = Counter()
    channel_projection_mechanisms: dict[int, Counter[str]] = {}
    seen_event_uids: set[str] = set()
    event_uid_digest = hashlib.sha256()
    resolved = [Path(path).resolve() for path in paths]
    for path in resolved:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        sidecar_hash = _sha256_file(sidecar) if sidecar.exists() else ""
        marker_hash = _sha256_file(marker) if marker.exists() else ""
        shard_start_count = event_count
        shard_metadata: dict[str, Any] = {}
        shard_schema_versions: set[str] = set()
        marker_payload: dict[str, Any] | None = None
        if sidecar.exists():
            shard_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if shard_metadata.get("schema_version") == SCHEMA_VERSION_V4:
                marker_payload = _validated_completion_marker(path, shard_metadata)
            if source_expectations:
                metadata_source = str(shard_metadata.get("source_file", ""))
                expectation = source_expectations.get(metadata_source)
                if expectation is None:
                    raise ValueError(
                        f"shard source {metadata_source!r} is absent from selection expectations"
                    )
                if str(path) != str(expectation["path"]):
                    raise ValueError("selection source is bound to a different parquet path")
                if int(shard_metadata.get("task_id", -1)) != int(expectation["task_id"]):
                    raise ValueError("selection task_id disagrees with shard metadata")
                if str(shard_metadata.get("task_record_hash", "")) != str(
                    expectation["task_record_hash"]
                ):
                    raise ValueError("selection task record hash disagrees with shard metadata")
                metadata_category = str(
                    shard_metadata.get("category", shard_metadata.get("physics_category", ""))
                )
                if metadata_category != str(expectation["category"]):
                    raise ValueError("selection category disagrees with shard metadata")
        feature_spec_hashes.add(str(shard_metadata.get("feature_spec_hash", "")))
        track_fit_policies.add(str(shard_metadata.get("track_fit_policy", "")))
        for record in iter_event_records_v4(path):
            if max_events is not None and event_count >= max_events:
                break
            event_count += 1
            event_uid = str(record.get("event_uid", ""))
            if not event_uid:
                raise ValueError("event record has an empty event_uid")
            if event_uid in seen_event_uids:
                raise ValueError(f"duplicate event_uid in selected records: {event_uid!r}")
            seen_event_uids.add(event_uid)
            encoded_uid = event_uid.encode("utf-8")
            event_uid_digest.update(len(encoded_uid).to_bytes(8, "big"))
            event_uid_digest.update(encoded_uid)
            source = str(record.get("source_file", ""))
            if (
                selection_manifest_hash is not None
                and source not in source_split_overrides
            ):
                raise ValueError(
                    f"record source {source!r} is absent from source-role selection"
                )
            if source_expectations:
                expectation = source_expectations[source]
                if str(record.get("source_category", "")) != str(
                    expectation["category"]
                ):
                    raise ValueError(
                        f"record category disagrees with selection for source {source!r}"
                    )
            split = source_split_overrides.get(
                source, stable_split_name(record, config)
            )
            split_counts[split] += 1
            category_counts[str(record.get("source_category", ""))] += 1
            source = source or str(record["event_uid"])
            previous = source_groups.setdefault(source, split)
            if previous != split:
                raise ValueError(f"source group {source!r} leaks across splits")
            record_schema = str(
                record.get(
                    "source_schema_version", record.get("schema_version", "")
                )
            )
            schema_versions.add(record_schema)
            shard_schema_versions.add(record_schema)
            event = heterogeneous_event_from_record(record)
            for side in ("b1", "b2"):
                full_id = int(getattr(event, f"{side}_full_truth_channel_id"))
                reconstructable_id = int(
                    getattr(event, f"{side}_reconstructable_channel_id")
                )
                if full_id > 0 and reconstructable_id > 0:
                    channel_projection_groups.setdefault(reconstructable_id, set()).add(
                        full_id
                    )
                    channel_projection_event_counts[reconstructable_id] += 1
                    mechanism_counts = channel_projection_mechanisms.setdefault(
                        reconstructable_id, Counter()
                    )
                    nodes = record.get("nodes", [])
                    if any(bool(node.get("contracted_intermediate")) for node in nodes):
                        mechanism_counts["contracted_intermediate_present"] += 1
                    if any(bool(node.get("copied")) for node in nodes):
                        mechanism_counts["copied_node_present"] += 1
                    if bool(record.get("charge_conjugate_normalization")):
                        mechanism_counts["charge_conjugate_normalization"] += 1
            channel_key = ":".join(
                map(
                    str,
                    sorted(
                        (
                            int(event.b1_reconstructable_channel_id),
                            int(event.b2_reconstructable_channel_id),
                        )
                    ),
                )
            )
            retain_channel_slice = (
                channel_key in channel_frequency
                or len(channel_frequency) < MAX_CHANNEL_FREQUENCY_SLICE_SIGNATURES
            )
            if retain_channel_slice:
                channel_frequency[channel_key] += 1
            else:
                channel_frequency_slice_overflow_events += 1
            total_nodes += int(event.active.sum())
            legacy_nodes += int(
                (
                    event.leaf_kinematics_mode_ids
                    == LEAF_MODE_TO_ID["legacy_conflated"]
                ).sum()
            )
            depth_distribution[int(event.level_ids[event.active].max())] += 1
            if split == "train":
                for block in FEATURE_BLOCKS:
                    availability = getattr(event, f"{block}_availability")
                    if block == "composite":
                        availability = availability.clone()
                        availability[:, list(TARGET_COMPOSITE_METADATA_INDICES)] = False
                    normalizers[block].update(
                        getattr(event, f"{block}_features"),
                        availability,
                    )
            for block in FEATURE_BLOCKS:
                availability = getattr(event, f"{block}_availability")
                if block == "composite":
                    availability = availability.clone()
                    availability[:, list(TARGET_COMPOSITE_METADATA_INDICES)] = False
                all_split_normalizers[block].update(
                    getattr(event, f"{block}_features"),
                    availability,
                )
            for level in sorted(
                {int(x) for x in event.level_ids[event.active].tolist() if int(x) > 0}
            ):
                eligible = event.active & (event.level_ids == level)
                if target_policy != "diagnostic_all":
                    eligible &= event.valid_reconstruction_target
                if target_policy == "complete_only":
                    eligible &= event.recursive_reconstructable_complete
                mothers = eligible.nonzero(as_tuple=False).flatten()
                cardinalities = [
                    int(event.daughter_adjacency[mother].sum())
                    for mother in mothers.tolist()
                ]
                neutral_multiplicity = int(
                    (event.active & (event.level_ids == 0) & (event.charge == 0)).sum()
                )
                for dimension, value in (
                    (
                        "source_category",
                        str(record.get("source_category", "")) or "unknown",
                    ),
                    ("event_multiplicity", str(int(event.active.sum()))),
                    ("neutral_multiplicity", str(neutral_multiplicity)),
                ):
                    _update_capacity_slice(
                        capacity_slices,
                        level=level,
                        dimension=dimension,
                        value=value,
                        mother_count=int(mothers.numel()),
                        maximum_cardinality=max(cardinalities, default=0),
                    )
                if retain_channel_slice:
                    row = channel_capacity.setdefault(level, {}).setdefault(
                        channel_key,
                        {
                            "event_count": 0,
                            "maximum_mothers": 0,
                            "maximum_daughter_cardinality": 0,
                        },
                    )
                    row["event_count"] += 1
                    row["maximum_mothers"] = max(
                        row["maximum_mothers"], int(mothers.numel())
                    )
                    row["maximum_daughter_cardinality"] = max(
                        row["maximum_daughter_cardinality"],
                        max(cardinalities, default=0),
                    )
                mother_count_histograms.setdefault(level, Counter())[
                    int(mothers.numel())
                ] += 1
                for mother in mothers.tolist():
                    daughter_cardinality[
                        int(event.daughter_adjacency[mother].sum())
                    ] += 1
                    daughter_cardinality_by_level.setdefault(level, Counter())[
                        int(event.daughter_adjacency[mother].sum())
                    ] += 1
                    target_counts[f"level_{level}"] += 1
                    if split == "train":
                        allowed_types.setdefault(level, set()).add(
                            int(event.pid_target_labels[mother])
                        )
            for policy in policy_capacity:
                eligible = event.active & (event.level_ids > 0)
                if policy != "diagnostic_all":
                    eligible &= event.valid_reconstruction_target
                if policy == "complete_only":
                    eligible &= event.recursive_reconstructable_complete
                policy_capacity[policy]["eligible_targets"] += int(eligible.sum())
                for level in event.level_ids[eligible].tolist():
                    policy_capacity[policy][f"level_{int(level)}"] += 1
            target_counts["partial"] += int(event.partial_missing_daughters.sum())
            target_counts["recursive_complete"] += int(
                event.recursive_reconstructable_complete.sum()
            )
        if max_events is not None and event_count >= max_events:
            pass
        descriptor_schema = str(shard_metadata.get("schema_version", ""))
        if not descriptor_schema:
            if len(shard_schema_versions) != 1:
                raise ValueError(f"shard contains ambiguous schema metadata: {path}")
            descriptor_schema = next(iter(shard_schema_versions))
        shards.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "source_digest": _sha256_file(path),
                "sidecar_hash": sidecar_hash,
                "completion_marker_hash": marker_hash,
                "event_count": event_count - shard_start_count,
                "schema": descriptor_schema,
                "feature_hash": str(shard_metadata.get("feature_spec_hash", "")),
                "pid_vocabulary": str(shard_metadata.get("pid_vocabulary_version", "")),
                "track_fit_policy": str(shard_metadata.get("track_fit_policy", "")),
                "source_entry_range": _source_entry_range(
                    shard_metadata, event_count - shard_start_count
                ),
                "completion_marker_content": marker_payload,
            }
        )
        if max_events is not None and event_count >= max_events:
            break
    if not event_count:
        raise ValueError("cannot index an empty dataset")
    fitted_normalizers = (
        normalizers if split_counts.get("train", 0) else all_split_normalizers
    )
    normalizer_state = {
        block: {
            key: value.tolist()
            for key, value in normalizer.state_dict().items()
            if key in {"count", "mean", "m2"}
        }
        for block, normalizer in fitted_normalizers.items()
    }
    for level, signatures in channel_capacity.items():
        for signature, row in signatures.items():
            frequency = str(channel_frequency[signature])
            aggregate = (
                capacity_slices.setdefault(level, {})
                .setdefault("channel_frequency", {})
                .setdefault(
                    frequency,
                    {
                        "event_count": 0,
                        "maximum_mothers": 0,
                        "maximum_daughter_cardinality": 0,
                    },
                )
            )
            aggregate["event_count"] += row["event_count"]
            aggregate["maximum_mothers"] = max(
                aggregate["maximum_mothers"], row["maximum_mothers"]
            )
            aggregate["maximum_daughter_cardinality"] = max(
                aggregate["maximum_daughter_cardinality"],
                row["maximum_daughter_cardinality"],
            )
    payload = {
        "index_version": DATASET_INDEX_VERSION,
        "paths": [str(path) for path in resolved],
        "event_count": event_count,
        "node_count": total_nodes,
        "schema_versions": sorted(schema_versions),
        "feature_spec_hashes": sorted(feature_spec_hashes),
        "track_fit_policies": sorted(track_fit_policies),
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "split_config": config.__dict__,
        "split_counts": dict(split_counts),
        "source_groups": dict(sorted(source_groups.items())),
        "category_counts": dict(category_counts),
        "legacy_fraction": legacy_nodes / max(total_nodes, 1),
        "normalizer_state": normalizer_state,
        "normalizer_scope": "train"
        if split_counts.get("train", 0)
        else "all_events_no_train_split_diagnostic",
        "allowed_types_by_level": {
            str(level): sorted(tokens) for level, tokens in allowed_types.items()
        },
        "mother_count_histograms_by_level": {
            str(level): {str(k): v for k, v in sorted(hist.items())}
            for level, hist in sorted(mother_count_histograms.items())
        },
        "daughter_cardinality_histogram": {
            str(k): v for k, v in sorted(daughter_cardinality.items())
        },
        "daughter_cardinality_histograms_by_level": {
            str(level): {str(k): v for k, v in sorted(histogram.items())}
            for level, histogram in sorted(daughter_cardinality_by_level.items())
        },
        "depth_distribution": {
            str(k): v for k, v in sorted(depth_distribution.items())
        },
        "target_policy": target_policy,
        "target_policy_counts": dict(target_counts),
        "policy_capacity_statistics": {
            policy: dict(counts) for policy, counts in policy_capacity.items()
        },
        "capacity_slices_by_level": {
            str(level): dimensions
            for level, dimensions in sorted(capacity_slices.items())
        },
        "channel_frequency_histogram": {
            str(frequency): count
            for frequency, count in sorted(Counter(channel_frequency.values()).items())
        },
        "channel_frequency_slice_coverage": {
            "maximum_tracked_signatures": MAX_CHANNEL_FREQUENCY_SLICE_SIGNATURES,
            "tracked_signatures": len(channel_frequency),
            "overflow_events": channel_frequency_slice_overflow_events,
            "exact": channel_frequency_slice_overflow_events == 0,
        },
        "full_truth_to_reconstructable_channel_collisions": {
            "distinct_reconstructable_channels": len(channel_projection_groups),
            "collision_group_count": sum(
                len(full_ids) > 1 for full_ids in channel_projection_groups.values()
            ),
            "groups": [
                {
                    "reconstructable_channel_id": reconstructable_id,
                    "full_truth_channel_ids": sorted(full_ids),
                    "distinct_full_truth_channels": len(full_ids),
                    "event_branch_count": channel_projection_event_counts[
                        reconstructable_id
                    ],
                    # These are co-occurrence diagnostics, not causal labels.
                    "possible_mechanism_event_counts": dict(
                        channel_projection_mechanisms.get(reconstructable_id, {})
                    ),
                }
                for reconstructable_id, full_ids in sorted(
                    channel_projection_groups.items()
                )
                if len(full_ids) > 1
            ],
            "mechanism_scope": (
                "co-occurrence only: PID reduction, skipped topology, charge-conjugate "
                "normalization, and copied-node deduplication require signature-level "
                "follow-up before causal attribution"
            ),
        },
        "shards": shards,
        "feature_spec_revision": FEATURE_SPEC_REVISION_V4,
        "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
        "supported_schema_set": sorted(SUPPORTED_SCHEMAS),
        "selection_contract": _selection_contract(
            resolved,
            max_events=max_events,
            selection_manifest_hash=selection_manifest_hash,
            included_splits=included_splits,
        ),
        "event_identity_validation": {
            "status": "passed",
            "validation_scope": "all_opened_train_and_validation_event_records",
            "validated_events": event_count,
            "unique_event_uids": len(seen_event_uids),
            "duplicate_event_uids": 0,
            "source_mismatches": 0,
            "category_mismatches": 0,
            "task_binding": (
                "selection_to_sidecar_to_completion_marker_validated"
                if source_expectations
                else "not_requested_legacy_index"
            ),
            "event_uid_stream_sha256": event_uid_digest.hexdigest(),
            "sealed_test_opened": "test" in included_splits,
        },
    }
    payload["index_hash"] = _index_hash(payload)
    _validate_dataset_index_metadata_payload(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def _load_dataset_index_binding(path: str | Path) -> _AuthenticatedIndexBinding:
    if not isinstance(path, (str, Path)):
        raise ValueError("dataset index path must be a string or Path")
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dataset index JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset index must be a JSON object")
    _validate_dataset_index_metadata_payload(payload)
    pinned = deepcopy(payload)
    return _AuthenticatedIndexBinding(
        source=source,
        canonical_bytes=text.encode("utf-8"),
        payload=pinned,
        index_hash=str(pinned["index_hash"]),
        _provenance=_INDEX_BINDING_PROVENANCE,
    )


def load_dataset_index_metadata(path: str | Path) -> dict[str, Any]:
    """Authenticate index metadata without resolving or opening shard paths."""

    return dict(_load_dataset_index_binding(path).payload)


def _validate_dataset_index_metadata_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("dataset index must be a JSON object")
    if payload.get("index_version") != DATASET_INDEX_VERSION:
        raise ValueError("unsupported dataset index version")
    stored_hash = payload.get("index_hash")
    if stored_hash != _index_hash(payload):
        raise ValueError("dataset index hash mismatch")
    if set(payload) not in {_FULL_INDEX_KEYS, _SIDECAR_INDEX_KEYS}:
        raise ValueError("dataset index top-level schema is invalid")
    if "index_source" in payload and payload.get("index_source") != "merged_shard_sidecars":
        raise ValueError("dataset index source schema is invalid")
    paths = _validate_index_no_source_gates(payload)
    _validate_index_shard_descriptors(payload.get("shards"), paths, payload)
    _validate_index_aggregate_invariants(payload)


def _validate_index_aggregate_invariants(payload: Mapping[str, Any]) -> None:
    descriptors = payload.get("shards")
    if not isinstance(descriptors, list):
        raise ValueError("dataset index shard descriptors are invalid")
    if payload.get("event_count") != sum(
        descriptor["event_count"] for descriptor in descriptors
    ):
        raise ValueError("dataset index event count disagrees with shard descriptors")
    split_counts = payload.get("split_counts")
    if not isinstance(split_counts, Mapping) or sum(split_counts.values()) != payload.get(
        "event_count"
    ):
        raise ValueError("dataset index split counts disagree with event count")
    category_counts = payload.get("category_counts")
    if (
        not isinstance(category_counts, Mapping)
        or any(
            not isinstance(category, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for category, count in category_counts.items()
        )
        or sum(category_counts.values()) != payload.get("event_count")
    ):
        raise ValueError("dataset index category counts disagree with event count")
    expected_normalizer_scope = (
        "train"
        if split_counts.get("train", 0) > 0
        else "all_events_no_train_split_diagnostic"
    )
    if payload.get("normalizer_scope") != expected_normalizer_scope:
        raise ValueError("dataset index normalizer scope disagrees with train count")
    for field in ("schema_versions", "feature_spec_hashes", "track_fit_policies"):
        if field == "schema_versions":
            descriptor_values = [descriptor["schema"] for descriptor in descriptors]
        elif field == "feature_spec_hashes":
            descriptor_values = [descriptor["feature_hash"] for descriptor in descriptors]
        else:
            descriptor_values = [
                descriptor["track_fit_policy"] for descriptor in descriptors
            ]
        if payload.get(field) != sorted(set(descriptor_values)):
            raise ValueError(f"dataset index {field} disagree with shard descriptors")
    normalizer_state = payload.get("normalizer_state")
    expected_widths = {"common": 12, "track": 16, "cluster": 9, "composite": 13}
    if not isinstance(normalizer_state, Mapping) or set(normalizer_state) != set(
        expected_widths
    ):
        raise ValueError("dataset index normalizer blocks are invalid")
    for block, width in expected_widths.items():
        state = normalizer_state.get(block)
        if not isinstance(state, Mapping) or set(state) != {"count", "mean", "m2"}:
            raise ValueError("dataset index normalizer blocks are invalid")
        for field in ("count", "mean", "m2"):
            values = state.get(field)
            if not isinstance(values, list) or len(values) != width:
                raise ValueError("dataset index normalizer shapes are invalid")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or (field in {"count", "m2"} and float(value) < 0.0)
                for value in values
            ):
                raise ValueError("dataset index normalizer values are invalid")
    identity = payload.get("event_identity_validation")
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise ValueError(
                "dataset index event-identity metadata is invalid; "
                "identity/task-binding gate is not valid"
            )
        if set(identity) != {
            "status",
            "validation_scope",
            "validated_events",
            "unique_event_uids",
            "duplicate_event_uids",
            "source_mismatches",
            "category_mismatches",
            "task_binding",
            "event_uid_stream_sha256",
            "sealed_test_opened",
        }:
            raise ValueError(
                "dataset index event-identity metadata is invalid; "
                "identity/task-binding gate is not valid"
            )
        for field in (
            "validated_events",
            "unique_event_uids",
            "duplicate_event_uids",
            "source_mismatches",
            "category_mismatches",
        ):
            value = identity.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    "dataset index event-identity metadata is invalid; "
                    "identity/task-binding gate is not valid"
                )
        if (
            identity["validated_events"] != payload["event_count"]
            or identity["unique_event_uids"] != payload["event_count"]
            or identity["duplicate_event_uids"] != 0
            or identity["source_mismatches"] != 0
            or identity["category_mismatches"] != 0
            or identity.get("status") != "passed"
            or identity.get("sealed_test_opened") is not False
            or identity.get("validation_scope")
            != "all_opened_train_and_validation_event_records"
            or not _is_sha256_hex(identity.get("event_uid_stream_sha256"))
            or identity.get("task_binding")
            not in {
                "selection_to_sidecar_to_completion_marker_validated",
                "not_requested_legacy_index",
            }
        ):
            raise ValueError(
                "dataset index event-identity metadata is invalid; "
                "identity/task-binding gate is not valid"
            )


def _validate_index_no_source_gates(payload: Mapping[str, Any]) -> list[str]:
    """Validate every stored-only index contract and return lexical paths."""

    if payload.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
        raise ValueError("dataset index PID vocabulary mismatch")
    schema_versions = payload.get("schema_versions")
    if (
        not isinstance(schema_versions, list)
        or not schema_versions
        or any(
            not isinstance(schema, str) or schema not in SUPPORTED_SCHEMAS
            for schema in schema_versions
        )
    ):
        raise ValueError("dataset index contains unsupported schemas")
    if payload.get("supported_schema_set") != sorted(SUPPORTED_SCHEMAS):
        raise ValueError("dataset index supported-schema contract mismatch")
    if payload.get("feature_spec_revision") != FEATURE_SPEC_REVISION_V4:
        raise ValueError("dataset index feature-spec revision mismatch")
    feature_hash = payload.get("feature_spec_hash")
    if not _is_sha256_hex(feature_hash) or feature_hash != feature_spec_v4()[
        "feature_spec_hash"
    ]:
        raise ValueError("dataset index feature-spec hash mismatch")
    for field in ("event_count", "node_count"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"dataset index {field} is invalid")
    for field in ("schema_versions", "feature_spec_hashes", "track_fit_policies"):
        value = payload.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"dataset index {field} is invalid")
    if any(
        not isinstance(item, str)
        or (item and not _is_sha256_hex(item))
        for item in payload["feature_spec_hashes"]
    ):
        raise ValueError("dataset index feature_spec_hashes is invalid")
    if payload.get("target_policy") not in {
        "complete_only",
        "reconstructable_partial",
        "diagnostic_all",
    }:
        raise ValueError("dataset index target policy is invalid")
    if payload.get("normalizer_scope") not in {
        "train",
        "all_events_no_train_split_diagnostic",
    }:
        raise ValueError("dataset index normalizer scope is invalid")
    legacy_fraction = payload.get("legacy_fraction")
    if (
        not isinstance(legacy_fraction, (int, float))
        or isinstance(legacy_fraction, bool)
        or not 0.0 <= float(legacy_fraction) <= 1.0
    ):
        raise ValueError("dataset index legacy fraction is invalid")
    _validate_index_split_config(payload.get("split_config"))
    split_counts = payload.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("dataset index split_counts is invalid")
    for split, count in split_counts.items():
        if (
            not isinstance(split, str)
            or split not in {"train", "validation", "test"}
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError("dataset index split_counts is invalid")
    source_groups = payload.get("source_groups")
    if not isinstance(source_groups, Mapping) or any(
        not isinstance(source, str)
        or not isinstance(split, str)
        or split not in {"train", "validation", "test"}
        for source, split in source_groups.items()
    ):
        raise ValueError("dataset index source_groups is invalid")
    _validate_allowed_types_by_level(payload)
    validate_dataset_index_selection_contract(payload.get("selection_contract"))
    paths = _validate_index_paths(payload.get("paths"))
    selection = payload["selection_contract"]
    expected_fingerprint = _selection_fingerprint_from_strings(
        paths,
        mode=selection["mode"],
        max_events=selection["max_events"],
        selection_manifest_hash=selection["selection_manifest_hash"],
    )
    if selection["fingerprint"] != expected_fingerprint:
        raise ValueError("dataset index selection fingerprint mismatch")
    return paths


def _validate_allowed_types_by_level(payload: Mapping[str, Any]) -> None:
    """Validate the loader-consumed level/token map without touching sources."""

    value = payload.get("allowed_types_by_level")
    if not isinstance(value, dict):
        raise ValueError("dataset index allowed-types mapping is invalid")
    level_keys = list(value)
    parsed_levels: list[int] = []
    for key in level_keys:
        if not isinstance(key, str) or not key or not key.isascii() or not key.isdecimal():
            raise ValueError("dataset index allowed-types level key is invalid")
        level = int(key)
        if key != str(level) or not 1 <= level <= MAX_ALLOWED_TYPE_LEVEL:
            raise ValueError("dataset index allowed-types level key is invalid")
        parsed_levels.append(level)
    if parsed_levels != sorted(parsed_levels):
        raise ValueError("dataset index allowed-types levels are not ordered")

    for tokens in value.values():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("dataset index allowed-types token list is invalid")
        if any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token < len(PDG_TOKENS)
            for token in tokens
        ):
            raise ValueError("dataset index allowed-types token vocabulary is invalid")
        if tokens != sorted(set(tokens)):
            raise ValueError("dataset index allowed-types tokens are not ordered and unique")

    mother_histograms = payload.get("mother_count_histograms_by_level")
    daughter_histograms = payload.get("daughter_cardinality_histograms_by_level")
    if not isinstance(mother_histograms, Mapping) or not isinstance(
        daughter_histograms, Mapping
    ):
        raise ValueError("dataset index allowed-types cross-fields are invalid")
    if any(
        key not in mother_histograms or key not in daughter_histograms
        for key in level_keys
    ):
        raise ValueError("dataset index allowed-types cross-fields are invalid")
    target_counts = payload.get("target_policy_counts")
    if not isinstance(target_counts, Mapping):
        raise ValueError("dataset index allowed-types cross-fields are invalid")
    for key in level_keys:
        count_key = f"level_{key}"
        if count_key in target_counts:
            count = target_counts[count_key]
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("dataset index allowed-types cross-fields are invalid")


def _validate_index_paths(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("dataset index paths are invalid")
    paths: list[str] = []
    for path in value:
        canonical = _validate_canonical_index_path(path)
        if canonical in paths:
            raise ValueError("dataset index contains duplicate canonical paths")
        paths.append(canonical)
    return paths


def _validate_canonical_index_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("dataset index path is invalid")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("dataset index path is invalid") from error
    canonical = PurePosixPath(value)
    if (
        not canonical.is_absolute()
        or value.startswith("//")
        or str(canonical) != value
    ):
        raise ValueError("dataset index path is not canonical")
    if any(part in {".", ".."} for part in canonical.parts):
        raise ValueError("dataset index path is not canonical")
    return value


def _validate_index_split_config(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "seed",
        "group_by_source_file",
        "group_by_category",
    }:
        raise ValueError("dataset index split configuration is invalid")
    fractions = []
    for field in ("train_fraction", "validation_fraction", "test_fraction"):
        fraction = value.get(field)
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("dataset index split configuration is invalid")
        fractions.append(float(fraction))
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("dataset index split configuration is invalid")
    if not isinstance(value.get("seed"), int) or isinstance(value.get("seed"), bool):
        raise ValueError("dataset index split configuration is invalid")
    for field in ("group_by_source_file", "group_by_category"):
        if not isinstance(value.get(field), bool):
            raise ValueError("dataset index split configuration is invalid")


def _validate_index_shard_descriptors(
    value: object, paths: list[str], payload: Mapping[str, Any]
) -> None:
    if not isinstance(value, list) or not value or len(value) > len(paths):
        raise ValueError("dataset index shard descriptor cardinality is invalid")
    selection = payload.get("selection_contract")
    truncated = isinstance(selection, Mapping) and selection.get("max_events") is not None
    if not truncated and len(value) != len(paths):
        raise ValueError("dataset index shard descriptor cardinality is invalid")
    required = {
        "path",
        "size",
        "source_digest",
        "sidecar_hash",
        "completion_marker_hash",
        "event_count",
        "schema",
        "feature_hash",
        "pid_vocabulary",
        "track_fit_policy",
        "source_entry_range",
        "completion_marker_content",
    }
    descriptor_paths: list[str] = []
    for position, descriptor in enumerate(value):
        if not isinstance(descriptor, Mapping) or set(descriptor) != required:
            raise ValueError(f"dataset index shard descriptor {position} is invalid")
        path = _validate_canonical_index_path(descriptor.get("path"))
        if path != paths[position]:
            raise ValueError("dataset index shard/path metadata disagree")
        descriptor_paths.append(path)
        for field in ("size", "event_count"):
            number = descriptor.get(field)
            if not isinstance(number, int) or isinstance(number, bool) or number < 0:
                raise ValueError(f"dataset index shard {field} is invalid")
        schema = descriptor.get("schema")
        if not isinstance(schema, str) or schema not in SUPPORTED_SCHEMAS:
            raise ValueError("dataset index shard schema is invalid")
        if not _is_sha256_hex(descriptor.get("source_digest")):
            raise ValueError("dataset index shard source digest is invalid")
        for field in ("sidecar_hash", "completion_marker_hash", "feature_hash"):
            digest = descriptor.get(field)
            if schema == SCHEMA_VERSION_V4:
                if not _is_sha256_hex(digest):
                    raise ValueError(f"dataset index v4 shard {field} is invalid")
            elif not isinstance(digest, str) or (
                digest and not _is_sha256_hex(digest)
            ):
                raise ValueError(f"dataset index shard {field} is invalid")
        pid_vocabulary = descriptor.get("pid_vocabulary")
        if not isinstance(pid_vocabulary, str):
            raise ValueError("dataset index shard PID vocabulary is invalid")
        if schema == SCHEMA_VERSION_V4 and pid_vocabulary != PID_VOCABULARY_VERSION:
            raise ValueError("dataset index v4 shard PID vocabulary is invalid")
        if not isinstance(descriptor.get("track_fit_policy"), str):
            raise ValueError("dataset index shard track-fit policy is invalid")
        source_range = descriptor.get("source_entry_range")
        if (
            not isinstance(source_range, list)
            or len(source_range) != 2
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item < 0
                for item in source_range
            )
            or source_range[1] <= source_range[0]
        ):
            raise ValueError("dataset index shard source-entry range is invalid")
        marker_content = descriptor.get("completion_marker_content")
        if schema == SCHEMA_VERSION_V4:
            _validate_index_v4_marker_content(marker_content, descriptor)
        elif marker_content is not None:
            raise ValueError("dataset index non-v4 completion marker metadata is invalid")
    if descriptor_paths != paths[: len(descriptor_paths)]:
        raise ValueError("dataset index shard/path metadata disagree")


def _validate_index_v4_marker_content(
    value: object, descriptor: Mapping[str, Any]
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("dataset index v4 completion marker metadata is invalid")
    required = {
        "marker_schema_version",
        "schema_version",
        "event_count",
        "feature_spec_hash",
        "model_feature_contract_hash",
        "parquet_sha256",
        "sidecar_sha256",
        "entry_start",
        "entry_stop_exclusive",
        "track_fit_policy",
    }
    if not required.issubset(value):
        raise ValueError("dataset index v4 completion marker metadata is invalid")
    event_count = value.get("event_count")
    if (
        value.get("marker_schema_version") != COMPLETION_MARKER_VERSION
        or value.get("schema_version") != SCHEMA_VERSION_V4
        or not isinstance(event_count, int)
        or isinstance(event_count, bool)
        or event_count != descriptor.get("event_count")
        or value.get("feature_spec_hash") != descriptor.get("feature_hash")
        or not _is_sha256_hex(value.get("feature_spec_hash"))
        or not _is_sha256_hex(value.get("model_feature_contract_hash"))
        or value.get("parquet_sha256") != descriptor.get("source_digest")
        or value.get("sidecar_sha256") != descriptor.get("sidecar_hash")
    ):
        raise ValueError("dataset index v4 completion marker metadata is invalid")
    if any(
        isinstance(item, (Mapping, list, tuple, set, frozenset))
        for item in value.values()
    ):
        raise ValueError("dataset index v4 completion marker metadata is invalid")
    marker_start = value.get("entry_start")
    marker_stop = value.get("entry_stop_exclusive")
    if (
        not isinstance(marker_start, int)
        or isinstance(marker_start, bool)
        or marker_start < 0
        or not isinstance(marker_stop, int)
        or isinstance(marker_stop, bool)
        or marker_stop <= marker_start
        or [marker_start, marker_stop] != descriptor.get("source_entry_range")
    ):
        raise ValueError("dataset index v4 completion marker source range is invalid")
    marker_policy = value.get("track_fit_policy")
    descriptor_policy = descriptor.get("track_fit_policy")
    if (
        not isinstance(marker_policy, str)
        or marker_policy != descriptor_policy
    ):
        raise ValueError("dataset index v4 completion marker policy is invalid")


def validate_dataset_index_selection_contract(selection: object) -> None:
    """Validate dataset-index selection metadata without touching sources."""

    if not isinstance(selection, Mapping):
        raise ValueError("dataset index selection contract must be an object")
    if set(selection) != {
        "mode",
        "max_events",
        "selection_manifest_hash",
        "included_splits",
        "fingerprint",
    }:
        raise ValueError("dataset index selection contract keys are invalid")
    mode = selection.get("mode")
    if not isinstance(mode, str) or mode not in (
        "all",
        "ordered_prefix",
        "source_role_manifest",
    ):
        raise ValueError("dataset index selection mode is invalid")
    if not _is_sha256_hex(selection.get("fingerprint")):
        raise ValueError("dataset index selection fingerprint is invalid")
    included_splits = selection.get("included_splits")
    if not isinstance(included_splits, list):
        raise ValueError("dataset index included-split contract is invalid")
    split_vocabulary = ("train", "validation", "test")
    previous_index = -1
    seen: list[str] = []
    for split in included_splits:
        if not isinstance(split, str) or split not in split_vocabulary:
            raise ValueError("dataset index included-split contract is invalid")
        if split in seen:
            raise ValueError("dataset index included-split contract is invalid")
        split_index = split_vocabulary.index(split)
        if split_index <= previous_index:
            raise ValueError("dataset index included-split contract is invalid")
        previous_index = split_index
        seen.append(split)
    manifest_hash = selection.get("selection_manifest_hash")
    max_events = selection.get("max_events")
    if mode == "source_role_manifest":
        if (
            not included_splits
            or not _is_sha256_hex(manifest_hash)
            or max_events is not None
        ):
            raise ValueError("dataset index source-role hash contract is invalid")
    elif manifest_hash is not None or included_splits != []:
        raise ValueError("dataset index raw-selection hash contract is invalid")
    elif mode == "all" and max_events is not None:
        raise ValueError("dataset index all-events selection is invalid")
    elif mode == "ordered_prefix" and (
        not isinstance(max_events, int)
        or isinstance(max_events, bool)
        or max_events <= 0
    ):
        raise ValueError("dataset index ordered-prefix selection is invalid")
def load_dataset_index(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    return _load_dataset_index_bound(
        _load_dataset_index_binding(path), verify_sources=verify_sources
    )


def _load_dataset_index_bound(
    binding: _AuthenticatedIndexBinding, *, verify_sources: bool = True
) -> dict[str, Any]:
    return _load_resolved_dataset_index_bound(
        _resolve_dataset_index_binding(binding), verify_sources=verify_sources
    )


def _load_resolved_dataset_index_bound(
    binding: _ResolvedIndexBinding, *, verify_sources: bool = True
) -> dict[str, Any]:
    payload, resolved_paths, resolved_shard_paths = _require_resolved_index_binding(
        binding
    )
    if verify_sources:
        _verify_indexed_shards(
            payload,
            resolved_paths=resolved_paths,
            resolved_shard_paths=resolved_shard_paths,
        )
    return payload


def _resolve_dataset_index_binding(
    binding: _AuthenticatedIndexBinding,
) -> _ResolvedIndexBinding:
    """Resolve and rebind every authenticated index path without source reads."""

    payload = _require_authenticated_index_binding(binding)
    resolved_paths = tuple(Path(path).resolve() for path in payload["paths"])
    resolved_shard_paths = tuple(
        Path(shard["path"]).resolve() for shard in payload["shards"]
    )
    selection = payload["selection_contract"]
    resolved_fingerprint = _selection_fingerprint_from_strings(
        (str(path) for path in resolved_paths),
        mode=selection["mode"],
        max_events=selection["max_events"],
        selection_manifest_hash=selection["selection_manifest_hash"],
    )
    if resolved_fingerprint != selection["fingerprint"]:
        raise ValueError("dataset index resolved selection fingerprint mismatch")
    truncated = selection["max_events"] is not None
    paths_match = (
        resolved_paths[: len(resolved_shard_paths)] == resolved_shard_paths
        if truncated
        else resolved_paths == resolved_shard_paths
    )
    if not paths_match:
        raise ValueError("dataset index resolved shard/path list mismatch")
    return _ResolvedIndexBinding(
        authenticated=binding,
        resolved_paths=resolved_paths,
        resolved_shard_paths=resolved_shard_paths,
        _provenance=_RESOLVED_INDEX_BINDING_PROVENANCE,
    )


def _require_resolved_index_binding(
    binding: object,
) -> tuple[dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    if (
        type(binding) is not _ResolvedIndexBinding
        or binding._provenance is not _RESOLVED_INDEX_BINDING_PROVENANCE
    ):
        raise ValueError("dataset index resolved provenance binding is invalid")
    payload = _require_authenticated_index_binding(binding.authenticated)
    return payload, binding.resolved_paths, binding.resolved_shard_paths


def _require_authenticated_index_binding(
    binding: object,
) -> dict[str, Any]:
    if (
        type(binding) is not _AuthenticatedIndexBinding
        or binding._provenance is not _INDEX_BINDING_PROVENANCE
    ):
        raise ValueError("dataset index provenance binding is invalid")
    try:
        pinned_payload = json.loads(binding.canonical_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset index provenance binding is invalid") from error
    if pinned_payload != binding.payload or binding.index_hash != binding.payload.get(
        "index_hash"
    ):
        raise ValueError("dataset index provenance binding was mutated")
    _validate_dataset_index_metadata_payload(binding.payload)
    return binding.payload


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _update_capacity_slice(
    slices: dict[int, dict[str, dict[str, dict[str, int]]]],
    *,
    level: int,
    dimension: str,
    value: str,
    mother_count: int,
    maximum_cardinality: int,
) -> None:
    row = (
        slices.setdefault(level, {})
        .setdefault(dimension, {})
        .setdefault(
            value,
            {"event_count": 0, "maximum_mothers": 0, "maximum_daughter_cardinality": 0},
        )
    )
    row["event_count"] += 1
    row["maximum_mothers"] = max(row["maximum_mothers"], mother_count)
    row["maximum_daughter_cardinality"] = max(
        row["maximum_daughter_cardinality"], maximum_cardinality
    )


def build_dataset_index_from_sidecars(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    split_config: SourceAwareSplitConfig | None = None,
    target_policy: str = "complete_only",
    source_split_overrides: Mapping[str, str] | None = None,
    selection_manifest_hash: str | None = None,
    selection_included_splits: Iterable[str] | None = None,
    source_expectations: Mapping[str, Mapping[str, Any]] | None = None,
    require_event_identity_validation: bool = False,
) -> Path:
    """Merge shard sufficient statistics without opening event payloads.

    This path requires source-file grouping because an event-level split cannot
    be reconstructed from aggregate metadata alone.
    """

    config = split_config or SourceAwareSplitConfig()
    source_split_overrides = dict(source_split_overrides or {})
    source_expectations = {
        str(source): dict(expectation)
        for source, expectation in dict(source_expectations or {}).items()
    }
    if require_event_identity_validation:
        raise ValueError(
            "sidecar indexing cannot satisfy full event UID/source validation"
        )
    if not config.group_by_source_file:
        raise ValueError("metadata-only indexing requires source-file grouping")
    resolved = [Path(path).resolve() for path in paths]
    split_counts = Counter()
    category_counts = Counter()
    source_groups: dict[str, str] = {}
    schema_versions: set[str] = set()
    feature_hashes: set[str] = set()
    track_fit_policies: set[str] = set()
    normalizers = {name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS}
    all_split_normalizers = {
        name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS
    }
    capacity = Counter()
    completeness = Counter()
    total_nodes = legacy_nodes = 0
    shards: list[dict[str, Any]] = []
    policy_capacity = {
        policy: Counter()
        for policy in ("complete_only", "reconstructable_partial", "diagnostic_all")
    }
    for path in resolved:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        if not sidecar.exists() or not marker.exists():
            raise ValueError(f"incomplete shard publication for {path}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        marker_payload = _validated_completion_marker(path, metadata)
        source = str(metadata.get("source_file", ""))
        if not source:
            raise ValueError(
                f"metadata-only indexing requires source_file in {sidecar}"
            )
        if source_expectations:
            expectation = source_expectations.get(source)
            if expectation is None:
                raise ValueError(f"sidecar source {source!r} is absent from selection")
            if int(metadata.get("task_id", -1)) != int(expectation["task_id"]):
                raise ValueError("selection task_id disagrees with shard metadata")
            if str(metadata.get("task_record_hash", "")) != str(
                expectation["task_record_hash"]
            ):
                raise ValueError("selection task record hash disagrees with shard metadata")
        if selection_manifest_hash is not None and source not in source_split_overrides:
            raise ValueError(
                f"sidecar source {source!r} is absent from source-role selection"
            )
        pseudo_event = {
            "event_uid": f"sidecar:{path.name}",
            "source_file": source,
            "source_category": str(metadata.get("category", "")),
        }
        split = source_split_overrides.get(
            source, stable_split_name(pseudo_event, config)
        )
        previous = source_groups.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source group {source!r} leaks across splits")
        event_count = int(metadata.get("event_count", 0))
        split_counts[split] += event_count
        category_counts[pseudo_event["source_category"]] += event_count
        schema_versions.add(str(metadata.get("schema_version", "")))
        feature_hashes.add(str(metadata.get("feature_spec_hash", "")))
        track_fit_policies.add(str(metadata.get("track_fit_policy", "")))
        shard_capacity = Counter(
            {
                str(key): int(value)
                for key, value in metadata.get(
                    "aggregate_capacity_statistics", {}
                ).items()
            }
        )
        capacity.update(shard_capacity)
        completeness.update(
            {
                str(key): int(value)
                for key, value in metadata.get(
                    "aggregate_completeness_statistics", {}
                ).items()
            }
        )
        total_nodes += int(shard_capacity.get("nodes", 0))
        legacy_nodes += int(shard_capacity.get("leaf_mode_legacy_conflated", 0))
        for block, state in metadata.get("aggregate_feature_welford", {}).items():
            if block == "ecl_cluster":
                block = "cluster"
            shard = StreamingMaskedFeatureNormalizer()
            shard.load_state_dict(
                {
                    "count": torch.tensor(state["count"], dtype=torch.float32),
                    "mean": torch.tensor(state["mean"], dtype=torch.float32),
                    "m2": torch.tensor(state["m2"], dtype=torch.float32),
                }
            )
            all_split_normalizers[block].merge(shard)
            if split == "train":
                normalizers[block].merge(shard)
        supplied_policy = metadata.get("policy_capacity_statistics", {})
        for policy in policy_capacity:
            if policy in supplied_policy:
                policy_capacity[policy].update(
                    {
                        str(key): int(value)
                        for key, value in supplied_policy[policy].items()
                    }
                )
        shards.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "source_digest": _sha256_file(path),
                "sidecar_hash": _sha256_file(sidecar),
                "completion_marker_hash": _sha256_file(marker),
                "event_count": event_count,
                "schema": str(metadata.get("schema_version", "")),
                "feature_hash": str(metadata.get("feature_spec_hash", "")),
                "pid_vocabulary": str(metadata.get("pid_vocabulary_version", "")),
                "track_fit_policy": str(metadata.get("track_fit_policy", "")),
                "source_entry_range": _source_entry_range(metadata, event_count),
                "completion_marker_content": marker_payload,
            }
        )
    if not capacity.get("events", 0):
        raise ValueError("cannot index empty shard metadata")
    fitted_normalizers = (
        normalizers if split_counts.get("train", 0) else all_split_normalizers
    )
    normalizer_state = {
        block: {
            key: value.tolist()
            for key, value in normalizer.state_dict().items()
            if key in {"count", "mean", "m2"}
        }
        for block, normalizer in fitted_normalizers.items()
    }
    mother_hist: dict[str, dict[str, int]] = {}
    allowed: dict[str, set[int]] = {}
    daughter_hist: dict[str, int] = {}
    daughter_hist_by_level: dict[str, dict[str, int]] = {}
    depth_hist: dict[str, int] = {}
    for key, value in capacity.items():
        if key.startswith("mother_count_level_"):
            level, count = key.removeprefix("mother_count_level_").split("_value_")
            mother_hist.setdefault(level, {})[count] = int(value)
        elif key.startswith("target_type_level_"):
            level, token = key.removeprefix("target_type_level_").split("_token_")
            allowed.setdefault(level, set()).add(int(token))
        elif key.startswith("daughter_cardinality_level_"):
            level, count = key.removeprefix("daughter_cardinality_level_").split(
                "_value_"
            )
            daughter_hist_by_level.setdefault(level, {})[count] = int(value)
        elif key.startswith("daughter_cardinality_"):
            daughter_hist[key.removeprefix("daughter_cardinality_")] = int(value)
        elif key.startswith("depth_"):
            depth_hist[key.removeprefix("depth_")] = int(value)
    payload = {
        "index_version": DATASET_INDEX_VERSION,
        "index_source": "merged_shard_sidecars",
        "paths": [str(path) for path in resolved],
        "event_count": int(capacity["events"]),
        "node_count": total_nodes,
        "schema_versions": sorted(schema_versions),
        "feature_spec_hashes": sorted(feature_hashes),
        "track_fit_policies": sorted(track_fit_policies),
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "split_config": config.__dict__,
        "split_counts": dict(split_counts),
        "source_groups": dict(sorted(source_groups.items())),
        "category_counts": dict(category_counts),
        "legacy_fraction": legacy_nodes / max(total_nodes, 1),
        "normalizer_state": normalizer_state,
        "normalizer_scope": "train"
        if split_counts.get("train", 0)
        else "all_events_no_train_split_diagnostic",
        "allowed_types_by_level": {
            level: sorted(allowed[level])
            for level in sorted(allowed, key=int)
        },
        "mother_count_histograms_by_level": mother_hist,
        "daughter_cardinality_histogram": daughter_hist,
        "daughter_cardinality_histograms_by_level": daughter_hist_by_level,
        "depth_distribution": depth_hist,
        "target_policy": target_policy,
        "target_policy_counts": {
            "partial": int(completeness.get("partial_targets", 0)),
            "recursive_complete": int(completeness.get("recursive_complete", 0)),
            "valid": int(completeness.get("valid_targets", 0)),
        },
        "policy_capacity_statistics": {
            policy: dict(counts) for policy, counts in policy_capacity.items()
        },
        "shards": shards,
        "feature_spec_revision": FEATURE_SPEC_REVISION_V4,
        "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
        "supported_schema_set": sorted(SUPPORTED_SCHEMAS),
        "selection_contract": _selection_contract(
            resolved,
            max_events=None,
            selection_manifest_hash=selection_manifest_hash,
            included_splits=tuple(
                selection_included_splits
                or (("train", "validation", "test") if selection_manifest_hash else ())
            ),
        ),
    }
    payload["index_hash"] = _index_hash(payload)
    _validate_dataset_index_metadata_payload(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def tensor_normalizer_state(
    index: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        block: {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in state.items()
        }
        for block, state in index["normalizer_state"].items()
    }


def _index_hash(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "index_hash"}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_entry_range(metadata: Mapping[str, Any], event_count: int) -> list[int]:
    start = metadata.get("entry_start")
    stop = metadata.get("entry_stop_exclusive")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(stop, int)
        and not isinstance(stop, bool)
        and stop > start
    ):
        return [start, stop]
    return [0, max(int(event_count), 1)]


def _selection_fingerprint(
    paths: Iterable[Path],
    max_events: int | None,
    *,
    selection_manifest_hash: str | None = None,
    mode: str | None = None,
) -> str:
    resolved_paths = [str(Path(path).resolve()) for path in paths]
    resolved_mode = mode or (
        "source_role_manifest"
        if selection_manifest_hash is not None
        else ("ordered_prefix" if max_events is not None else "all")
    )
    return _selection_fingerprint_from_strings(
        resolved_paths,
        mode=resolved_mode,
        max_events=max_events,
        selection_manifest_hash=selection_manifest_hash,
    )


def _selection_fingerprint_from_strings(
    paths: Iterable[str],
    *,
    mode: str,
    max_events: int | None,
    selection_manifest_hash: str | None = None,
) -> str:
    """Hash exact authenticated path strings without filesystem operations."""

    stored_paths = list(paths)
    if any(not isinstance(path, str) for path in stored_paths):
        raise ValueError("selection fingerprint paths must be strings")
    payload = {
        "paths": stored_paths,
        "max_events": max_events,
        "event_selection": mode,
        "selection_manifest_hash": selection_manifest_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _selection_contract(
    paths: Iterable[Path],
    *,
    max_events: int | None,
    selection_manifest_hash: str | None,
    included_splits: Iterable[str] = (),
) -> dict[str, Any]:
    resolved = list(paths)
    mode = (
        "source_role_manifest"
        if selection_manifest_hash is not None
        else ("ordered_prefix" if max_events is not None else "all")
    )
    return {
        "mode": mode,
        "max_events": max_events,
        "selection_manifest_hash": selection_manifest_hash,
        "included_splits": list(included_splits),
        "fingerprint": _selection_fingerprint(
            resolved,
            max_events,
            selection_manifest_hash=selection_manifest_hash,
            mode=mode,
        ),
    }


def _validated_completion_marker(
    path: Path, metadata: dict[str, Any], *, _assume_present: bool = False
) -> dict[str, Any]:
    marker = path.with_suffix(path.suffix + ".complete")
    if not _assume_present and not marker.exists():
        raise ValueError(f"missing completion marker for {path}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid completion marker for {path}") from error
    if payload.get("marker_schema_version") != COMPLETION_MARKER_VERSION:
        raise ValueError(f"unsupported completion marker schema for {path}")
    for marker_key, metadata_key in (
        ("schema_version", "schema_version"),
        ("event_count", "event_count"),
        ("feature_spec_hash", "feature_spec_hash"),
        ("model_feature_contract_hash", "model_feature_contract_hash"),
    ):
        if payload.get(marker_key) != metadata.get(metadata_key):
            raise ValueError(
                f"completion marker {marker_key} disagrees with sidecar for {path}"
            )
    if payload.get("parquet_sha256") != _sha256_file(path):
        raise ValueError(f"completion marker parquet digest mismatch for {path}")
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if payload.get("sidecar_sha256") != _sha256_file(sidecar):
        raise ValueError(f"completion marker sidecar digest mismatch for {path}")
    for key in (
        "campaign_id",
        "campaign_config_digest",
        "source_git_commit",
        "source_git_tree",
        "source_state",
        "task_record_hash",
        "task_id",
        "source_file",
        "source_file_size",
        "source_file_mtime_ns",
        "source_file_identity",
        "source_file_sha256",
        "entry_start",
        "entry_stop_exclusive",
        "planned_events",
        "campaign_stage",
        "klm_training_scope",
        "production_readiness_report_sha256",
        "physics_category",
        "output_file",
        "leaf_kinematics_mode",
        "track_fit_policy",
        "charge_conjugate_normalization",
        "event_buffer_size",
        "row_group_size",
    ):
        if key in payload and payload.get(key) != metadata.get(key):
            raise ValueError(
                f"completion marker {key} disagrees with sidecar for {path}"
            )
    source_range = _source_entry_range(metadata, int(metadata.get("event_count", 0)))
    payload["entry_start"], payload["entry_stop_exclusive"] = source_range
    payload["track_fit_policy"] = str(metadata.get("track_fit_policy", ""))
    return payload


def _verify_indexed_shards(
    index: dict[str, Any],
    *,
    resolved_paths: tuple[Path, ...],
    resolved_shard_paths: tuple[Path, ...],
) -> None:
    resolved_shards = list(zip(resolved_shard_paths, index["shards"], strict=True))
    file_checks = [path.is_file() for path, _shard in resolved_shards]
    for (path, _shard), is_file in zip(resolved_shards, file_checks, strict=True):
        if not is_file:
            raise FileNotFoundError(f"missing indexed dataset shard: {path}")
    stats = [path.stat() for path, _shard in resolved_shards]
    inodes = [(stat.st_dev, stat.st_ino) for stat in stats]
    if len(inodes) != len(set(inodes)):
        raise ValueError("dataset index contains duplicate hardlink shard inodes")
    publication_files = [
        (
            path.with_suffix(path.suffix + ".metadata.json"),
            path.with_suffix(path.suffix + ".complete"),
            shard,
        )
        for path, shard in resolved_shards
        if shard.get("schema") == SCHEMA_VERSION_V4
    ]
    sidecar_checks = [
        sidecar.is_file() for sidecar, _marker, _shard in publication_files
    ]
    marker_checks = [
        marker.is_file() for _sidecar, marker, _shard in publication_files
    ]
    for (sidecar, _marker, _shard), sidecar_present, marker_present in zip(
        publication_files, sidecar_checks, marker_checks, strict=True
    ):
        if not sidecar_present or not marker_present:
            raise ValueError(f"incomplete indexed v4 shard {sidecar}")
    for (path, shard), stat in zip(resolved_shards, stats, strict=True):
        if stat.st_size != shard["size"]:
            raise ValueError(f"stale dataset index source size for {path}")
        if _sha256_file(path) != shard.get("source_digest"):
            raise ValueError(f"stale dataset index source digest for {path}")
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        schema = shard.get("schema")
        if schema == SCHEMA_VERSION_V4:
            if _sha256_file(sidecar) != shard.get("sidecar_hash"):
                raise ValueError(f"stale dataset index sidecar for {path}")
            if _sha256_file(marker) != shard.get("completion_marker_hash"):
                raise ValueError(f"stale dataset index completion marker for {path}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            marker_payload = _validated_completion_marker(
                path, metadata, _assume_present=True
            )
            if marker_payload != shard.get("completion_marker_content"):
                raise ValueError(f"completion marker content changed for {path}")
            if int(shard.get("event_count", -1)) != int(
                metadata.get("event_count", -2)
            ):
                raise ValueError(
                    f"indexed event count disagrees with sidecar for {path}"
                )
            if shard.get("feature_hash") != metadata.get("feature_spec_hash"):
                raise ValueError(f"indexed shard feature hash changed for {path}")
            if shard.get("pid_vocabulary") != metadata.get("pid_vocabulary_version"):
                raise ValueError(f"indexed shard PID vocabulary changed for {path}")
            if metadata.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
                raise ValueError(f"indexed shard PID vocabulary mismatch for {path}")


__all__ = [
    "DATASET_INDEX_VERSION",
    "build_dataset_index",
    "build_dataset_index_from_sidecars",
    "load_dataset_index",
    "load_dataset_index_metadata",
    "tensor_normalizer_state",
    "validate_dataset_index_selection_contract",
]
