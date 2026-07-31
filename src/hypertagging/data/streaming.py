"""Lazy event-row parquet iteration and mergeable masked normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info

from hypertagging.preprocessing.schema_v4 import iter_event_records_v4


T = TypeVar("T")


class BoundedShuffleBuffer(Generic[T]):
    """Deterministic streaming shuffle whose memory is bounded by ``size``."""

    def __init__(self, source: Iterable[T], *, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("shuffle buffer size must be positive")
        self.source = source
        self.size = int(size)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[T]:
        rng = random.Random(self.seed)
        buffer: list[T] = []
        for value in self.source:
            if len(buffer) < self.size:
                buffer.append(value)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = value
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


@dataclass
class StreamingCursor:
    """Deterministic cursor using epoch replay and batch/event skipping.

    No physical parquet row-group cursor is claimed: exact trainer resume
    reconstructs the same single-worker shuffled iterator and replays through
    ``batch_index``; direct dataset iteration can replay ``events_consumed``.
    """

    epoch: int = 0
    events_consumed: int = 0
    batch_index: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "StreamingCursor":
        return cls(**{name: int(state.get(name, 0)) for name in cls.__dataclass_fields__})


class ParquetEventIterableDataset(IterableDataset):
    """Worker-safe lazy iteration over v4 event rows or adapted legacy shards."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        max_events: int | None = None,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        split_name: str | None = None,
        split_config: Any | None = None,
        split_overrides: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.paths = tuple(Path(path) for path in paths)
        self.max_events = max_events
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.split_name = split_name
        self.split_config = split_config
        self.split_overrides = dict(split_overrides or {})

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.iter_from_cursor(StreamingCursor())

    def iter_from_cursor(
        self, cursor: StreamingCursor
    ) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers

        def records() -> Iterator[dict[str, Any]]:
            global_index = 0
            # Files are the primary disjoint work unit. For fewer files than
            # workers, row groups are assigned directly by the schema iterator.
            assigned_paths = (
                self.paths
                if len(self.paths) < worker_count
                else self.paths[worker_id::worker_count]
            )
            for path in assigned_paths:
                for event in iter_event_records_v4(
                    path,
                    worker_id=worker_id if len(self.paths) < worker_count else 0,
                    worker_count=worker_count if len(self.paths) < worker_count else 1,
                ):
                    local_limit = (
                        None
                        if self.max_events is None
                        else max(
                            0,
                            (self.max_events + worker_count - 1 - worker_id)
                            // worker_count,
                        )
                    )
                    if local_limit is not None and global_index >= local_limit:
                        return
                    global_index += 1
                    if self.split_name is not None:
                        from hypertagging.data.splitting import stable_split_name

                        assigned = self.split_overrides.get(
                            str(event["event_uid"]),
                            stable_split_name(event, self.split_config),
                        )
                        if assigned != self.split_name:
                            continue
                    yield event

        source: Iterable[dict[str, Any]] = records()
        if self.shuffle_buffer_size > 0:
            source = BoundedShuffleBuffer(
                source,
                size=self.shuffle_buffer_size,
                seed=self.seed + worker_id,
            )
        skipped = 0
        for event in source:
            if skipped < cursor.events_consumed:
                skipped += 1
                continue
            yield event


class ShardManifestDataset(ParquetEventIterableDataset):
    """Semantic alias used by production training code."""


class StreamingMaskedFeatureNormalizer:
    """Masked population Welford statistics, updateable and mergeable."""

    def __init__(self) -> None:
        self.count: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    @property
    def std(self) -> torch.Tensor:
        if self.count is None or self.m2 is None:
            raise RuntimeError("streaming normalizer has no observations")
        return (self.m2 / self.count.clamp_min(1)).sqrt().clamp_min(1e-6)

    @property
    def standard_deviation(self) -> torch.Tensor:
        return self.std

    def update(
        self,
        values: torch.Tensor,
        availability: torch.Tensor,
    ) -> "StreamingMaskedFeatureNormalizer":
        if values.shape != availability.shape:
            raise ValueError("values and availability must have identical shapes")
        if values.ndim < 2:
            raise ValueError("feature values need a final feature dimension")
        clean = torch.nan_to_num(values)
        mask = availability.bool()
        flat_values = clean.reshape(-1, clean.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        batch_count = flat_mask.sum(dim=0).to(clean.dtype)
        batch_sum = torch.where(flat_mask, flat_values, 0).sum(dim=0)
        batch_mean = batch_sum / batch_count.clamp_min(1)
        centered = torch.where(
            flat_mask, flat_values - batch_mean, torch.zeros_like(flat_values)
        )
        batch_m2 = centered.square().sum(dim=0)
        self._merge_statistics(batch_count, batch_mean, batch_m2)
        return self

    def merge(
        self, other: "StreamingMaskedFeatureNormalizer"
    ) -> "StreamingMaskedFeatureNormalizer":
        if other.count is None or other.mean is None or other.m2 is None:
            return self
        self._merge_statistics(other.count, other.mean, other.m2)
        return self

    def _merge_statistics(
        self,
        count: torch.Tensor,
        mean: torch.Tensor,
        m2: torch.Tensor,
    ) -> None:
        if self.count is None:
            self.count = count.clone()
            self.mean = mean.clone()
            self.m2 = m2.clone()
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * count / total.clamp_min(1)
        self.m2 = self.m2 + m2 + delta.square() * self.count * count / total.clamp_min(1)
        self.count = total

    def transform(
        self, values: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("streaming normalizer must be fitted first")
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        normalized = (torch.nan_to_num(values) - mean) / std
        return torch.where(availability, normalized, torch.zeros_like(normalized))

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.count is None or self.mean is None or self.m2 is None:
            raise RuntimeError("streaming normalizer is not fitted")
        return {
            "count": self.count.clone(),
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "standard_deviation": self.std.clone(),
        }

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.count = state["count"].clone()
        self.mean = state["mean"].clone()
        if "m2" in state:
            self.m2 = state["m2"].clone()
        else:
            std = state["standard_deviation"]
            self.m2 = std.square() * self.count


class RuntimeFeatureNormalizer(nn.Module):
    """Normalize only raw fields rebuilt between contextual passes."""

    def __init__(
        self,
        *,
        common_mean: torch.Tensor,
        common_std: torch.Tensor,
        composite_mean: torch.Tensor,
        composite_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("common_mean", common_mean.detach().clone().float())
        self.register_buffer("common_std", common_std.detach().clone().float().clamp_min(1e-6))
        self.register_buffer("composite_mean", composite_mean.detach().clone().float())
        self.register_buffer(
            "composite_std", composite_std.detach().clone().float().clamp_min(1e-6)
        )

    @classmethod
    def identity(cls, common_width: int, composite_width: int) -> "RuntimeFeatureNormalizer":
        return cls(
            common_mean=torch.zeros(common_width),
            common_std=torch.ones(common_width),
            composite_mean=torch.zeros(composite_width),
            composite_std=torch.ones(composite_width),
        )

    def normalize_runtime(
        self,
        common: torch.Tensor,
        common_availability: torch.Tensor,
        composite: torch.Tensor,
        composite_availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from hypertagging.preprocessing.schema_v4 import (
            CATEGORICAL_COMMON_FEATURE_NAMES,
            CONTINUOUS_COMMON_INDICES,
            DYNAMIC_COMPOSITE_INDICES,
            TARGET_COMPOSITE_METADATA_INDICES,
            feature_spec_v4,
        )

        common_out = common.clone()
        common_mask = common_availability.clone()
        # Static common quantities arrive raw from the streaming data module,
        # while dynamic quantities may have just been rebuilt.  Applying the
        # same fitted transform to every continuous slot here prevents a
        # raw/normalized mixture in either contextual pass.
        for index in CONTINUOUS_COMMON_INDICES:
            common_out[..., index] = (
                common[..., index] - self.common_mean[index].to(common)
            ) / self.common_std[index].to(common)
        common_names = feature_spec_v4()["common"]
        for name in CATEGORICAL_COMMON_FEATURE_NAMES:
            index = common_names.index(name)
            common_out[..., index] = 0
            common_mask[..., index] = False
        common_out = torch.where(common_mask, common_out, torch.zeros_like(common_out))
        composite_out = composite.clone()
        for index in DYNAMIC_COMPOSITE_INDICES:
            if index < composite.shape[-1]:
                composite_out[..., index] = (
                    composite[..., index] - self.composite_mean[index].to(composite)
                ) / self.composite_std[index].to(composite)
        composite_mask = composite_availability.clone()
        # Defense in depth: target-only compatibility slots are unavailable
        # even before the encoder's versioned selection adapter.
        for index in TARGET_COMPOSITE_METADATA_INDICES:
            if index < composite_out.shape[-1]:
                composite_out[..., index] = 0
                composite_mask[..., index] = False
        composite_out = torch.where(
            composite_mask, composite_out, torch.zeros_like(composite_out)
        )
        return common_out, common_mask, composite_out, composite_mask


__all__ = [
    "BoundedShuffleBuffer",
    "ParquetEventIterableDataset",
    "ShardManifestDataset",
    "StreamingMaskedFeatureNormalizer",
    "RuntimeFeatureNormalizer",
    "StreamingCursor",
]
