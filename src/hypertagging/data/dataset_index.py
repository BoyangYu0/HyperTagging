"""Versioned one-pass dataset index and mergeable sufficient statistics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from hypertagging.data.heterogeneous import heterogeneous_event_from_record
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.data.streaming import StreamingMaskedFeatureNormalizer
from hypertagging.preprocessing.schema_v4 import (
    FEATURE_SPEC_REVISION_V4, LEAF_MODE_TO_ID, SCHEMA_VERSION_V4,
    TARGET_COMPOSITE_METADATA_INDICES, feature_spec_v4, iter_event_records_v4,
)
from hypertagging.preprocessing.schema_v2 import SCHEMA_VERSION_V1, SCHEMA_VERSION_V2
from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION


DATASET_INDEX_VERSION = "hypertagging-dataset-index-v2"
SUPPORTED_SCHEMAS = {SCHEMA_VERSION_V1, SCHEMA_VERSION_V2, SCHEMA_VERSION_V3, SCHEMA_VERSION_V4}
FEATURE_BLOCKS = ("common", "track", "cluster", "composite")


def build_dataset_index(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    split_config: SourceAwareSplitConfig | None = None,
    target_policy: str = "complete_only",
    max_events: int | None = None,
) -> Path:
    """Scan once, then persist all startup statistics needed by trainers."""

    config = split_config or SourceAwareSplitConfig()
    normalizers = {
        name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS
    }
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
    shards: list[dict[str, Any]] = []
    policy_capacity = {
        policy: Counter() for policy in (
            "complete_only", "reconstructable_partial", "diagnostic_all"
        )
    }
    resolved = [Path(path).resolve() for path in paths]
    for path in resolved:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        sidecar_hash = _sha256_file(sidecar) if sidecar.exists() else ""
        marker_hash = _sha256_file(marker) if marker.exists() else ""
        shard_start_count = event_count
        shard_metadata: dict[str, Any] = {}
        marker_payload: dict[str, Any] | None = None
        if sidecar.exists():
            shard_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if shard_metadata.get("feature_spec_hash"):
                feature_spec_hashes.add(str(shard_metadata["feature_spec_hash"]))
            if shard_metadata.get("schema_version") == SCHEMA_VERSION_V4:
                marker_payload = _validated_completion_marker(path, shard_metadata)
        for record in iter_event_records_v4(path):
            if max_events is not None and event_count >= max_events:
                break
            event_count += 1
            split = stable_split_name(record, config)
            split_counts[split] += 1
            category_counts[str(record.get("source_category", ""))] += 1
            source = str(record.get("source_file", "")) or str(record["event_uid"])
            previous = source_groups.setdefault(source, split)
            if previous != split:
                raise ValueError(f"source group {source!r} leaks across splits")
            schema_versions.add(
                str(record.get("source_schema_version", record.get("schema_version", "")))
            )
            event = heterogeneous_event_from_record(record)
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
                mother_count_histograms.setdefault(level, Counter())[int(mothers.numel())] += 1
                for mother in mothers.tolist():
                    daughter_cardinality[int(event.daughter_adjacency[mother].sum())] += 1
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
        shards.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "source_digest": _sha256_file(path),
                "sidecar_hash": sidecar_hash,
                "completion_marker_hash": marker_hash,
                "event_count": event_count - shard_start_count,
                "schema": str(shard_metadata.get("schema_version", "")),
                "feature_hash": str(shard_metadata.get("feature_spec_hash", "")),
                "pid_vocabulary": str(shard_metadata.get("pid_vocabulary_version", "")),
                "source_entry_range": [
                    shard_metadata.get("entry_start"),
                    shard_metadata.get("entry_stop_exclusive"),
                ],
                "completion_marker_content": marker_payload,
            }
        )
        if max_events is not None and event_count >= max_events:
            break
    if not event_count:
        raise ValueError("cannot index an empty dataset")
    fitted_normalizers = normalizers if split_counts.get("train", 0) else all_split_normalizers
    normalizer_state = {
        block: {
            key: value.tolist()
            for key, value in normalizer.state_dict().items()
            if key in {"count", "mean", "m2"}
        }
        for block, normalizer in fitted_normalizers.items()
    }
    payload = {
        "index_version": DATASET_INDEX_VERSION,
        "paths": [str(path) for path in resolved],
        "event_count": event_count,
        "node_count": total_nodes,
        "schema_versions": sorted(schema_versions),
        "feature_spec_hashes": sorted(feature_spec_hashes),
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "split_config": config.__dict__,
        "split_counts": dict(split_counts),
        "source_groups": dict(sorted(source_groups.items())),
        "category_counts": dict(category_counts),
        "legacy_fraction": legacy_nodes / max(total_nodes, 1),
        "normalizer_state": normalizer_state,
        "normalizer_scope": "train" if split_counts.get("train", 0) else "all_events_no_train_split_diagnostic",
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
        "depth_distribution": {str(k): v for k, v in sorted(depth_distribution.items())},
        "target_policy": target_policy,
        "target_policy_counts": dict(target_counts),
        "policy_capacity_statistics": {
            policy: dict(counts) for policy, counts in policy_capacity.items()
        },
        "shards": shards,
        "feature_spec_revision": FEATURE_SPEC_REVISION_V4,
        "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
        "supported_schema_set": sorted(SUPPORTED_SCHEMAS),
        "selection_contract": {
            "mode": "ordered_prefix",
            "max_events": max_events,
            "fingerprint": _selection_fingerprint(resolved, max_events),
        },
    }
    payload["index_hash"] = _index_hash(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def load_dataset_index(path: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("index_version") != DATASET_INDEX_VERSION:
        raise ValueError("unsupported dataset index version")
    stored_hash = payload.get("index_hash")
    if stored_hash != _index_hash(payload):
        raise ValueError("dataset index hash mismatch")
    if payload.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
        raise ValueError("dataset index PID vocabulary mismatch")
    if not set(payload.get("schema_versions", ())).issubset(SUPPORTED_SCHEMAS):
        raise ValueError("dataset index contains unsupported schemas")
    if payload.get("supported_schema_set") != sorted(SUPPORTED_SCHEMAS):
        raise ValueError("dataset index supported-schema contract mismatch")
    if payload.get("feature_spec_revision") != FEATURE_SPEC_REVISION_V4:
        raise ValueError("dataset index feature-spec revision mismatch")
    if payload.get("feature_spec_hash") != feature_spec_v4()["feature_spec_hash"]:
        raise ValueError("dataset index feature-spec hash mismatch")
    selection = payload.get("selection_contract", {})
    resolved = [Path(value).resolve() for value in payload.get("paths", ())]
    if selection.get("fingerprint") != _selection_fingerprint(
        resolved, selection.get("max_events")
    ):
        raise ValueError("dataset index selection fingerprint mismatch")
    if verify_sources:
        _verify_indexed_shards(payload)
    return payload


def build_dataset_index_from_sidecars(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    split_config: SourceAwareSplitConfig | None = None,
    target_policy: str = "complete_only",
) -> Path:
    """Merge shard sufficient statistics without opening event payloads.

    This path requires source-file grouping because an event-level split cannot
    be reconstructed from aggregate metadata alone.
    """

    config = split_config or SourceAwareSplitConfig()
    if not config.group_by_source_file:
        raise ValueError("metadata-only indexing requires source-file grouping")
    resolved = [Path(path).resolve() for path in paths]
    split_counts = Counter()
    category_counts = Counter()
    source_groups: dict[str, str] = {}
    schema_versions: set[str] = set()
    feature_hashes: set[str] = set()
    normalizers = {
        name: StreamingMaskedFeatureNormalizer() for name in FEATURE_BLOCKS
    }
    capacity = Counter()
    completeness = Counter()
    total_nodes = legacy_nodes = 0
    shards: list[dict[str, Any]] = []
    policy_capacity = {
        policy: Counter() for policy in (
            "complete_only", "reconstructable_partial", "diagnostic_all"
        )
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
            raise ValueError(f"metadata-only indexing requires source_file in {sidecar}")
        pseudo_event = {
            "event_uid": f"sidecar:{path.name}",
            "source_file": source,
            "source_category": str(metadata.get("category", "")),
        }
        split = stable_split_name(pseudo_event, config)
        previous = source_groups.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source group {source!r} leaks across splits")
        event_count = int(metadata.get("event_count", 0))
        split_counts[split] += event_count
        category_counts[pseudo_event["source_category"]] += event_count
        schema_versions.add(str(metadata.get("schema_version", "")))
        feature_hashes.add(str(metadata.get("feature_spec_hash", "")))
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
        if split == "train":
            for block, state in metadata.get(
                "aggregate_feature_welford", {}
            ).items():
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
                normalizers[block].merge(shard)
        supplied_policy = metadata.get("policy_capacity_statistics", {})
        for policy in policy_capacity:
            if policy in supplied_policy:
                policy_capacity[policy].update(
                    {str(key): int(value) for key, value in supplied_policy[policy].items()}
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
                "source_entry_range": [metadata.get("entry_start"), metadata.get("entry_stop_exclusive")],
                "completion_marker_content": marker_payload,
            }
        )
    if not capacity.get("events", 0):
        raise ValueError("cannot index empty shard metadata")
    normalizer_state = {
        block: {
            key: value.tolist()
            for key, value in normalizer.state_dict().items()
            if key in {"count", "mean", "m2"}
        }
        for block, normalizer in normalizers.items()
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
            level, count = key.removeprefix(
                "daughter_cardinality_level_"
            ).split("_value_")
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
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "split_config": config.__dict__,
        "split_counts": dict(split_counts),
        "source_groups": dict(sorted(source_groups.items())),
        "category_counts": dict(category_counts),
        "legacy_fraction": legacy_nodes / max(total_nodes, 1),
        "normalizer_state": normalizer_state,
        "allowed_types_by_level": {
            level: sorted(tokens) for level, tokens in allowed.items()
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
        "selection_contract": {
            "mode": "all",
            "max_events": None,
            "fingerprint": _selection_fingerprint(resolved, None),
        },
    }
    payload["index_hash"] = _index_hash(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def tensor_normalizer_state(index: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
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


def _selection_fingerprint(paths: Iterable[Path], max_events: int | None) -> str:
    payload = {
        "paths": [str(Path(path).resolve()) for path in paths],
        "max_events": max_events,
        "event_selection": "ordered_prefix",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_completion_marker(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.exists():
        raise ValueError(f"missing completion marker for {path}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid completion marker for {path}") from error
    for marker_key, metadata_key in (
        ("schema_version", "schema_version"),
        ("event_count", "event_count"),
        ("feature_spec_hash", "feature_spec_hash"),
    ):
        if payload.get(marker_key) != metadata.get(metadata_key):
            raise ValueError(
                f"completion marker {marker_key} disagrees with sidecar for {path}"
            )
    if payload.get("parquet_sha256") and payload["parquet_sha256"] != _sha256_file(path):
        raise ValueError(f"completion marker parquet digest mismatch for {path}")
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if payload.get("sidecar_sha256") and payload["sidecar_sha256"] != _sha256_file(sidecar):
        raise ValueError(f"completion marker sidecar digest mismatch for {path}")
    return payload


def _verify_indexed_shards(index: dict[str, Any]) -> None:
    expected_paths = [str(Path(path).resolve()) for path in index.get("paths", ())]
    shard_paths = [str(Path(shard["path"]).resolve()) for shard in index.get("shards", ())]
    truncated = index.get("selection_contract", {}).get("max_events") is not None
    paths_match = (
        expected_paths[: len(shard_paths)] == shard_paths
        if truncated else expected_paths == shard_paths
    )
    if not paths_match:
        raise ValueError("dataset index shard/path list mismatch")
    for shard in index.get("shards", ()):
        path = Path(shard["path"])
        if not path.is_file() or path.stat().st_size != int(shard["size"]):
            raise ValueError(f"stale dataset index source size for {path}")
        if _sha256_file(path) != shard.get("source_digest"):
            raise ValueError(f"stale dataset index source digest for {path}")
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        schema = shard.get("schema")
        if schema == SCHEMA_VERSION_V4:
            if not sidecar.exists() or not marker.exists():
                raise ValueError(f"incomplete indexed v4 shard {path}")
            if _sha256_file(sidecar) != shard.get("sidecar_hash"):
                raise ValueError(f"stale dataset index sidecar for {path}")
            if _sha256_file(marker) != shard.get("completion_marker_hash"):
                raise ValueError(f"stale dataset index completion marker for {path}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            marker_payload = _validated_completion_marker(path, metadata)
            if marker_payload != shard.get("completion_marker_content"):
                raise ValueError(f"completion marker content changed for {path}")
            if int(shard.get("event_count", -1)) != int(metadata.get("event_count", -2)):
                raise ValueError(f"indexed event count disagrees with sidecar for {path}")
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
    "tensor_normalizer_state",
]
