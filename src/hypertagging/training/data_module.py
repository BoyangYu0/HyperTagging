"""Lazy parquet/shard loading, source-safe splits, and online normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence
import warnings

import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, IterableDataset

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    collate_heterogeneous_events,
    heterogeneous_event_from_record,
)
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.data.streaming import StreamingMaskedFeatureNormalizer
from hypertagging.preprocessing.schema_v4 import (
    TARGET_COMPOSITE_METADATA_INDICES,
    iter_event_records_v4,
)


FEATURE_BLOCKS = ("common", "track", "cluster", "composite")


@dataclass
class RealDataModule:
    input_paths: tuple[str, ...]
    normalizers: dict[str, StreamingMaskedFeatureNormalizer]
    split_manifest: dict[str, object]
    split_manifest_hash: str
    overflow_counters: dict[str, int]
    seed: int
    split_config: SourceAwareSplitConfig
    max_events: int | None = None
    max_nodes: int | None = None
    max_nodes_overflow: str = "raise"
    shuffle_buffer_size: int = 1024
    allow_legacy_conflated: bool = False
    split_overrides: dict[str, str] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    legacy_conflated_fraction: float = 0.0
    allowed_types_by_level: dict[int, tuple[int, ...]] = field(default_factory=dict)
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    source_schema_versions: tuple[str, ...] = ()
    dataset_index: dict[str, object] | None = None
    _materialized_splits: dict[str, list[HeterogeneousEvent]] | None = field(
        default=None, init=False, repr=False
    )

    def iter_events(
        self,
        split: str,
        *,
        shuffle: bool = False,
        epoch: int = 0,
    ) -> Iterator[HeterogeneousEvent]:
        records: Iterator[dict] = self._records()
        if shuffle and self.shuffle_buffer_size > 0:
            from hypertagging.data.streaming import BoundedShuffleBuffer

            records = iter(
                BoundedShuffleBuffer(
                    records,
                    size=self.shuffle_buffer_size,
                    seed=self.seed + epoch,
                )
            )
        for record in records:
            assigned = self.split_overrides.get(
                str(record["event_uid"]),
                stable_split_name(record, self.split_config),
            )
            if assigned != split:
                continue
            event = heterogeneous_event_from_record(record)
            if self.max_nodes is not None and event.common_features.shape[0] > self.max_nodes:
                if self.max_nodes_overflow == "drop":
                    self.overflow_counters["max_nodes_dropped"] += 1
                    continue
                raise OverflowError(
                    f"event {event.event_uid} has {event.common_features.shape[0]} nodes, "
                    f"exceeding max_nodes={self.max_nodes}"
                )
            yield event

    def _records(self) -> Iterator[dict]:
        emitted = 0
        for input_path in self.input_paths:
            for record in iter_event_records_v4(input_path):
                if self.max_events is not None and emitted >= self.max_events:
                    return
                emitted += 1
                yield record

    def batches(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        epoch: int = 0,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers > 0:
            dataset = _StreamingHeterogeneousDataset(self, split, shuffle, epoch)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_heterogeneous_events,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
            for batch in loader:
                yield self.normalize_batch(batch)
            return
        pending: list[HeterogeneousEvent] = []
        for event in self.iter_events(split, shuffle=shuffle, epoch=epoch):
            pending.append(event)
            if len(pending) == batch_size:
                yield self.normalize_batch(collate_heterogeneous_events(pending))
                pending.clear()
        if pending:
            yield self.normalize_batch(collate_heterogeneous_events(pending))

    def normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for block, normalizer in self.normalizers.items():
            # Common/composite values remain raw until the authoritative
            # model-side runtime normalizer. Track/cluster blocks are static.
            if block in {"common", "composite"}:
                continue
            values_key = f"{block}_features"
            availability_key = f"{block}_availability"
            result[values_key] = normalizer.transform(
                result[values_key], result[availability_key]
            )
        result["node_features"] = result["common_features"]
        result["runtime_features_are_raw"] = torch.tensor(True)
        return result

    def normalization_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: normalizer.state_dict() for name, normalizer in self.normalizers.items()
        }

    @property
    def splits(self) -> dict[str, list[HeterogeneousEvent]]:
        """Diagnostic compatibility view; trainers intentionally do not use it."""

        if self.max_events is None or self.max_events > 10_000:
            raise RuntimeError(
                "materializing RealDataModule.splits is disabled for large datasets; "
                "use iter_events or an explicit max_events<=10000 diagnostic pilot"
            )
        if self._materialized_splits is None:
            self._materialized_splits = {
                name: list(self.iter_events(name, shuffle=False))
                for name in ("train", "validation", "test")
            }
        return self._materialized_splits

    @property
    def events(self) -> list[HeterogeneousEvent]:
        return [
            event
            for name in ("train", "validation", "test")
            for event in self.splits[name]
        ]


def build_real_data_module(
    data: str | Path | Sequence[str | Path],
    *,
    max_events: int | None = None,
    max_nodes: int | None = None,
    max_nodes_overflow: str = "raise",
    split_config: SourceAwareSplitConfig | None = None,
    seed: int = 20260730,
    pilot_split_repair: bool = False,
    allow_legacy_conflated: bool = False,
    shuffle_buffer_size: int = 1024,
    normalization_state: dict[str, dict[str, torch.Tensor]] | None = None,
    required_splits: tuple[str, ...] = ("train",),
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    dataset_index: str | Path | None = None,
    rescan_dataset: bool = False,
    target_policy: str = "complete_only",
    allow_incomplete_v4_publication: bool = False,
) -> RealDataModule:
    """Build a restartable streaming data module without retaining event lists."""

    paths = resolve_data_paths(data)
    if not allow_incomplete_v4_publication:
        _require_complete_v4_publications(paths)
    config = split_config or SourceAwareSplitConfig(seed=seed)
    if dataset_index is not None and not rescan_dataset:
        from hypertagging.data.dataset_index import (
            load_dataset_index,
            tensor_normalizer_state,
        )

        index = load_dataset_index(dataset_index)
        if index.get("target_policy") != target_policy:
            raise ValueError(
                "dataset index target policy does not match trainer target policy; "
                "request an explicit rescan to change policy"
            )
        indexed_max_events = index.get("selection_contract", {}).get("max_events")
        if indexed_max_events != max_events:
            raise ValueError(
                "dataset index event-selection/max-events fingerprint mismatch"
            )
        indexed_paths = {str(Path(path).resolve()) for path in index["paths"]}
        if indexed_paths != {str(path.resolve()) for path in paths}:
            raise ValueError("dataset index shard paths do not match requested data")
        if index["split_config"] != config.__dict__:
            raise ValueError("dataset index split configuration mismatch")
        legacy_fraction = float(index["legacy_fraction"])
        if legacy_fraction and not allow_legacy_conflated:
            raise ValueError("dataset index reports legacy-conflated nodes")
        split_counts = {
            name: int(index["split_counts"].get(name, 0))
            for name in ("train", "validation", "test")
        }
        missing = [name for name in required_splits if split_counts[name] == 0]
        if missing:
            raise ValueError(f"indexed dataset has empty required split(s) {missing}")
        split_manifest = {
            "seed": seed,
            "groups": index["source_groups"],
            "split_counts": split_counts,
            "pilot_split_repair": False,
            "overrides": {},
            "dataset_index_hash": index["index_hash"],
        }
        manifest_json = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
        module = RealDataModule(
            input_paths=tuple(str(path) for path in paths),
            normalizers={},
            split_manifest=split_manifest,
            split_manifest_hash=hashlib.sha256(manifest_json.encode()).hexdigest(),
            overflow_counters={"max_nodes_dropped": 0, "invalid_events": 0},
            seed=seed,
            split_config=config,
            max_events=max_events,
            max_nodes=max_nodes,
            max_nodes_overflow=max_nodes_overflow,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_legacy_conflated=allow_legacy_conflated,
            split_counts=split_counts,
            legacy_conflated_fraction=legacy_fraction,
            allowed_types_by_level={
                int(level): tuple(int(token) for token in tokens)
                for level, tokens in index["allowed_types_by_level"].items()
            },
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            source_schema_versions=tuple(index["schema_versions"]),
            dataset_index=index,
        )
        state = normalization_state or tensor_normalizer_state(index)
        for block, block_state in state.items():
            normalizer = StreamingMaskedFeatureNormalizer()
            normalizer.load_state_dict(block_state)
            module.normalizers[block] = normalizer
        return module
    split_counts = {"train": 0, "validation": 0, "test": 0}
    group_splits: dict[str, str] = {}
    source_counts: dict[str, dict[str, int]] = {
        "train": {}, "validation": {}, "test": {}
    }
    first_uid = ""
    first_split = ""
    total_nodes = 0
    legacy_nodes = 0
    scanned = 0
    allowed_types: dict[int, set[int]] = {}
    source_schema_versions: set[str] = set()
    for path in paths:
        for record in iter_event_records_v4(path):
            if max_events is not None and scanned >= max_events:
                break
            scanned += 1
            uid = str(record["event_uid"])
            source_schema_versions.add(
                str(record.get("source_schema_version", record.get("schema_version", "")))
            )
            split = stable_split_name(record, config)
            if not first_uid:
                first_uid, first_split = uid, split
            split_counts[split] += 1
            source = str(record.get("source_file", ""))
            source_counts[split][source] = source_counts[split].get(source, 0) + 1
            group = source if config.group_by_source_file and source else uid
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"source group {group!r} leaks across splits")
            for node in record.get("nodes", []):
                total_nodes += 1
                legacy_nodes += int(
                    str(node.get("leaf_kinematics_mode")) == "legacy_conflated"
                )
                if (
                    split == "train"
                    and int(node.get("level", 0)) > 0
                    and bool(node.get("valid_reconstruction_target", False))
                ):
                    allowed_types.setdefault(int(node["level"]), set()).add(
                        int(node.get("pid_target_token", 0))
                    )
        if max_events is not None and scanned >= max_events:
            break
    if scanned == 0:
        raise ValueError("no events were loaded from the supplied parquet data")
    legacy_fraction = legacy_nodes / max(total_nodes, 1)
    if legacy_nodes and not allow_legacy_conflated:
        raise ValueError(
            "legacy-conflated v1/v2/v3 data are rejected for real training; "
            "use --allow-legacy-conflated only for diagnostic, non-data-compatible runs"
        )
    if legacy_nodes:
        warnings.warn(
            f"DIAGNOSTIC ONLY: {legacy_fraction:.2%} of nodes are legacy-conflated; "
            "data-compatible performance claims are disabled",
            RuntimeWarning,
            stacklevel=2,
        )
    overrides: dict[str, str] = {}
    missing = [name for name in required_splits if split_counts[name] == 0]
    if missing:
        if pilot_split_repair and missing == ["train"] and first_uid:
            overrides[first_uid] = "train"
            split_counts[first_split] -= 1
            split_counts["train"] += 1
        else:
            raise ValueError(
                f"source-aware split has empty required split(s) {missing}; "
                "change grouping/fractions or explicitly enable pilot_split_repair"
            )
    split_manifest = {
        "seed": seed,
        "groups": dict(sorted(group_splits.items())),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "pilot_split_repair": bool(pilot_split_repair),
        "overrides": overrides,
    }
    manifest_json = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    module = RealDataModule(
        input_paths=tuple(str(path) for path in paths),
        normalizers={},
        split_manifest=split_manifest,
        split_manifest_hash=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        overflow_counters={"max_nodes_dropped": 0, "invalid_events": 0},
        seed=seed,
        split_config=config,
        max_events=max_events,
        max_nodes=max_nodes,
        max_nodes_overflow=max_nodes_overflow,
        shuffle_buffer_size=shuffle_buffer_size,
        allow_legacy_conflated=allow_legacy_conflated,
        split_overrides=overrides,
        split_counts=split_counts,
        legacy_conflated_fraction=legacy_fraction,
        allowed_types_by_level={
            level: tuple(sorted(tokens)) for level, tokens in allowed_types.items()
        },
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        source_schema_versions=tuple(sorted(source_schema_versions)),
    )
    if normalization_state is None:
        module.normalizers = fit_training_normalizers(module.iter_events("train"))
    else:
        for block, state in normalization_state.items():
            normalizer = StreamingMaskedFeatureNormalizer()
            normalizer.load_state_dict(state)
            module.normalizers[block] = normalizer
    return module


def _require_complete_v4_publications(paths: Sequence[Path]) -> None:
    from hypertagging.data.dataset_index import _validated_completion_marker

    for path in paths:
        try:
            fields = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception as error:
            raise ValueError(f"cannot inspect parquet publication {path}") from error
        if "event_json" not in fields:
            # Legacy v1-v3 container compatibility remains available.
            continue
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        if not sidecar.is_file() or not marker.is_file():
            raise ValueError(
                f"incomplete schema-v4 publication {path}: parquet, metadata sidecar, "
                "and completion marker are required"
            )
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid schema-v4 metadata sidecar {sidecar}") from error
        _validated_completion_marker(path, metadata)


def fit_training_normalizers(
    events: Iterator[HeterogeneousEvent] | Sequence[HeterogeneousEvent],
) -> dict[str, StreamingMaskedFeatureNormalizer]:
    output = {
        block: StreamingMaskedFeatureNormalizer() for block in FEATURE_BLOCKS
    }
    count = 0
    for event in events:
        count += 1
        for block in FEATURE_BLOCKS:
            availability = getattr(event, f"{block}_availability")
            if block == "composite":
                availability = availability.clone()
                availability[:, list(TARGET_COMPOSITE_METADATA_INDICES)] = False
            output[block].update(
                getattr(event, f"{block}_features"),
                availability,
            )
    if count == 0:
        raise ValueError("cannot fit feature normalization without training events")
    return output


def resolve_data_paths(data: str | Path | Sequence[str | Path]) -> list[Path]:
    entries = [data] if isinstance(data, (str, Path)) else list(data)
    output: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            output.extend(sorted(path.glob("*.parquet")))
        elif path.suffix == ".parquet":
            output.append(path)
        elif path.suffix in {".jsonl", ".json"}:
            text = path.read_text(encoding="utf-8")
            records = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else json.loads(text)
            )
            if isinstance(records, dict):
                records = records.get("shards", records.get("entries", []))
            for record in records:
                candidate = (
                    record
                    if isinstance(record, str)
                    else (
                        record.get("output_file")
                        or record.get("output")
                        or record.get("path")
                        or record.get("parquet")
                    )
                )
                if candidate:
                    candidate_path = Path(candidate)
                    output.append(
                        candidate_path
                        if candidate_path.is_absolute()
                        else path.parent / candidate_path
                    )
        else:
            raise ValueError(f"unsupported data input: {path}")
    unique = sorted({path.resolve() for path in output})
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing parquet shard(s): {missing}")
    if not unique:
        raise ValueError("data input resolved to no parquet shards")
    return unique


class _StreamingHeterogeneousDataset(IterableDataset):
    def __init__(
        self,
        module: RealDataModule,
        split: str,
        shuffle: bool,
        epoch: int,
    ) -> None:
        super().__init__()
        self.paths = module.input_paths
        self.max_events = module.max_events
        self.max_nodes = module.max_nodes
        self.overflow = module.max_nodes_overflow
        self.split = split
        self.split_config = module.split_config
        self.split_overrides = module.split_overrides
        self.shuffle_buffer_size = module.shuffle_buffer_size if shuffle else 0
        self.seed = module.seed + epoch

    def __iter__(self):
        from hypertagging.data.streaming import ParquetEventIterableDataset

        records = ParquetEventIterableDataset(
            self.paths,
            max_events=self.max_events,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
            split_name=self.split,
            split_config=self.split_config,
            split_overrides=self.split_overrides,
        )
        for record in records:
            event = heterogeneous_event_from_record(record)
            if self.max_nodes is not None and event.common_features.shape[0] > self.max_nodes:
                if self.overflow == "drop":
                    continue
                raise OverflowError(
                    f"event {event.event_uid} exceeds max_nodes={self.max_nodes}"
                )
            yield event


__all__ = [
    "FEATURE_BLOCKS",
    "RealDataModule",
    "build_real_data_module",
    "fit_training_normalizers",
    "resolve_data_paths",
]
