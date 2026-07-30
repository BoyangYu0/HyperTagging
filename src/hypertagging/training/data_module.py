"""Real parquet/shard loading, stable splits, and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import torch

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    collate_heterogeneous_events,
    load_heterogeneous_events,
)
from hypertagging.data.splitting import (
    MaskedFeatureNormalizer,
    SourceAwareSplitConfig,
    stable_split_name,
)


FEATURE_BLOCKS = ("common", "track", "cluster", "composite")


@dataclass
class RealDataModule:
    events: list[HeterogeneousEvent]
    splits: dict[str, list[HeterogeneousEvent]]
    normalizers: dict[str, MaskedFeatureNormalizer]
    split_manifest: dict[str, str]
    split_manifest_hash: str
    input_paths: tuple[str, ...]
    overflow_counters: dict[str, int]
    seed: int

    def batches(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        epoch: int = 0,
    ) -> Iterator[dict[str, torch.Tensor]]:
        events = list(self.splits[split])
        if shuffle:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(events), generator=generator).tolist()
            events = [events[index] for index in order]
        for start in range(0, len(events), batch_size):
            batch = collate_heterogeneous_events(events[start : start + batch_size])
            yield self.normalize_batch(batch)

    def normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for block, normalizer in self.normalizers.items():
            values_key = f"{block}_features"
            availability_key = f"{block}_availability"
            result[values_key] = normalizer.transform(
                result[values_key],
                result[availability_key],
            )
        result["node_features"] = result["common_features"]
        return result

    def normalization_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: normalizer.state_dict() for name, normalizer in self.normalizers.items()}


def build_real_data_module(
    data: str | Path | Sequence[str | Path],
    *,
    max_events: int | None = None,
    max_nodes: int | None = None,
    max_nodes_overflow: str = "raise",
    split_config: SourceAwareSplitConfig | None = None,
    seed: int = 20260730,
) -> RealDataModule:
    """Load one parquet, shards, a directory, or a JSON/JSONL manifest."""

    paths = resolve_data_paths(data)
    events: list[HeterogeneousEvent] = []
    dropped = 0
    remaining = max_events
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        before = len(events)
        loaded = load_heterogeneous_events(
            path,
            limit=remaining,
            max_nodes=max_nodes,
            overflow_strategy=max_nodes_overflow,
        )
        events.extend(loaded)
        if remaining is not None:
            remaining -= len(loaded)
        dropped += max(0, before + len(loaded) - len(events))
    if not events:
        raise ValueError("no events were loaded from the supplied parquet data")
    uids = [event.event_uid for event in events]
    if len(set(uids)) != len(uids):
        duplicate = next(uid for uid in uids if uids.count(uid) > 1)
        raise ValueError(f"duplicate event_uid across input shards: {duplicate}")
    config = split_config or SourceAwareSplitConfig(seed=seed)
    splits = {"train": [], "validation": [], "test": []}
    split_manifest: dict[str, str] = {}
    for event in events:
        record = {
            "event_uid": event.event_uid,
            "event_id": event.event_id,
            "source_file": event.source_file,
            "source_category": event.source_category,
        }
        name = stable_split_name(record, config)
        splits[name].append(event)
        split_manifest[event.event_uid] = name
    # Very small pilot fixtures can hash entirely outside train. Make a
    # deterministic, recorded pilot-only repair so normalization is possible.
    if not splits["train"]:
        first = min(events, key=lambda event: event.event_uid)
        old = split_manifest[first.event_uid]
        splits[old].remove(first)
        splits["train"].append(first)
        split_manifest[first.event_uid] = "train"
    normalizers = fit_training_normalizers(splits["train"])
    manifest_json = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    return RealDataModule(
        events=events,
        splits=splits,
        normalizers=normalizers,
        split_manifest=split_manifest,
        split_manifest_hash=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        input_paths=tuple(str(path) for path in paths),
        overflow_counters={"max_nodes_dropped": dropped},
        seed=seed,
    )


def fit_training_normalizers(
    events: Sequence[HeterogeneousEvent],
) -> dict[str, MaskedFeatureNormalizer]:
    if not events:
        raise ValueError("cannot fit feature normalization without training events")
    output: dict[str, MaskedFeatureNormalizer] = {}
    for block in FEATURE_BLOCKS:
        values = torch.cat([getattr(event, f"{block}_features") for event in events], dim=0)
        availability = torch.cat(
            [getattr(event, f"{block}_availability") for event in events],
            dim=0,
        )
        output[block] = MaskedFeatureNormalizer().fit(values, availability)
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
                candidate = record if isinstance(record, str) else (
                    record.get("output")
                    or record.get("path")
                    or record.get("parquet")
                )
                if candidate:
                    candidate_path = Path(candidate)
                    output.append(
                        candidate_path if candidate_path.is_absolute() else path.parent / candidate_path
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


__all__ = [
    "FEATURE_BLOCKS",
    "RealDataModule",
    "build_real_data_module",
    "fit_training_normalizers",
    "resolve_data_paths",
]
