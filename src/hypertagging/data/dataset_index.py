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
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID, iter_event_records_v4
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION


DATASET_INDEX_VERSION = "hypertagging-dataset-index-v1"
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
    split_counts = Counter()
    category_counts = Counter()
    legacy_nodes = total_nodes = event_count = 0
    allowed_types: dict[int, set[int]] = {}
    mother_count_histograms: dict[int, Counter[int]] = {}
    daughter_cardinality = Counter()
    depth_distribution = Counter()
    target_counts = Counter()
    source_groups: dict[str, str] = {}
    schema_versions = set()
    feature_spec_hashes = set()
    resolved = [Path(path).resolve() for path in paths]
    for path in resolved:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if sidecar.exists():
            shard_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if shard_metadata.get("feature_spec_hash"):
                feature_spec_hashes.add(str(shard_metadata["feature_spec_hash"]))
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
                    normalizers[block].update(
                        getattr(event, f"{block}_features"),
                        getattr(event, f"{block}_availability"),
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
                    target_counts[f"level_{level}"] += 1
                    if split == "train":
                        allowed_types.setdefault(level, set()).add(
                            int(event.pid_target_labels[mother])
                        )
            target_counts["partial"] += int(event.partial_missing_daughters.sum())
            target_counts["recursive_complete"] += int(
                event.recursive_reconstructable_complete.sum()
            )
        if max_events is not None and event_count >= max_events:
            break
    if not event_count:
        raise ValueError("cannot index an empty dataset")
    normalizer_state = {
        block: {
            key: value.tolist()
            for key, value in normalizer.state_dict().items()
            if key in {"count", "mean", "m2"}
        }
        for block, normalizer in normalizers.items()
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
        "depth_distribution": {str(k): v for k, v in sorted(depth_distribution.items())},
        "target_policy": target_policy,
        "target_policy_counts": dict(target_counts),
    }
    payload["index_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def load_dataset_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("index_version") != DATASET_INDEX_VERSION:
        raise ValueError("unsupported dataset index version")
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
    for path in resolved:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        if not sidecar.exists() or not marker.exists():
            raise ValueError(f"incomplete shard publication for {path}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
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
    depth_hist: dict[str, int] = {}
    for key, value in capacity.items():
        if key.startswith("mother_count_level_"):
            level, count = key.removeprefix("mother_count_level_").split("_value_")
            mother_hist.setdefault(level, {})[count] = int(value)
        elif key.startswith("target_type_level_"):
            level, token = key.removeprefix("target_type_level_").split("_token_")
            allowed.setdefault(level, set()).add(int(token))
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
        "depth_distribution": depth_hist,
        "target_policy": target_policy,
        "target_policy_counts": {
            "partial": int(completeness.get("partial_targets", 0)),
            "recursive_complete": int(completeness.get("recursive_complete", 0)),
            "valid": int(completeness.get("valid_targets", 0)),
        },
    }
    payload["index_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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


__all__ = [
    "DATASET_INDEX_VERSION",
    "build_dataset_index",
    "build_dataset_index_from_sidecars",
    "load_dataset_index",
    "tensor_normalizer_state",
]
