"""Leakage-resistant source-aware splits and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import torch


@dataclass(frozen=True)
class SourceAwareSplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 20260730
    group_by_source_file: bool = True
    group_by_category: bool = False

    def __post_init__(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("split fractions must sum to one")


def stable_split_name(
    event: Mapping[str, Any],
    config: SourceAwareSplitConfig | None = None,
) -> str:
    """Assign a stable split while keeping configured source groups together."""

    config = config or SourceAwareSplitConfig()
    event_uid = str(event.get("event_uid", event.get("event_id", "")))
    if not event_uid:
        raise ValueError("stable source-aware splitting requires event_uid")
    group_parts = []
    if config.group_by_category and event.get("source_category"):
        group_parts.append(str(event["source_category"]))
    if config.group_by_source_file and event.get("source_file"):
        group_parts.append(str(event["source_file"]))
    group_parts.append(event_uid if not group_parts else "")
    key = "|".join(group_parts)
    digest = hashlib.sha256(f"{config.seed}|{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def split_records(
    events: Iterable[Mapping[str, Any]],
    config: SourceAwareSplitConfig | None = None,
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_uids: set[str] = set()
    for event in events:
        uid = str(event.get("event_uid", event.get("event_id", "")))
        if uid in seen_uids:
            raise ValueError(f"duplicate event_uid before splitting: {uid}")
        seen_uids.add(uid)
        output[stable_split_name(event, config)].append(event)
    return output


@dataclass
class MaskedFeatureNormalizer:
    """Per-feature train-split statistics that preserve missingness masks."""

    mean: torch.Tensor | None = None
    standard_deviation: torch.Tensor | None = None
    count: torch.Tensor | None = None

    def fit(self, values: torch.Tensor, availability: torch.Tensor) -> "MaskedFeatureNormalizer":
        if values.shape != availability.shape:
            raise ValueError("values and availability must have identical shapes")
        clean = torch.nan_to_num(values)
        weights = availability.to(clean.dtype)
        reduce_dims = tuple(range(values.ndim - 1))
        count = weights.sum(dim=reduce_dims)
        mean = (clean * weights).sum(dim=reduce_dims) / count.clamp_min(1)
        centered = torch.where(availability, clean - mean, torch.zeros_like(clean))
        variance = centered.square().sum(dim=reduce_dims) / count.clamp_min(1)
        self.mean = mean
        self.standard_deviation = variance.sqrt().clamp_min(1e-6)
        self.count = count
        return self

    def transform(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.standard_deviation is None:
            raise RuntimeError("normalizer must be fitted on the training split first")
        normalized = (torch.nan_to_num(values) - self.mean.to(values.device)) / self.standard_deviation.to(
            values.device
        )
        return torch.where(availability, normalized, torch.zeros_like(normalized))

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.mean is None or self.standard_deviation is None or self.count is None:
            raise RuntimeError("normalizer is not fitted")
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.mean = state["mean"].clone()
        self.standard_deviation = state["standard_deviation"].clone()
        self.count = state["count"].clone()


__all__ = [
    "MaskedFeatureNormalizer",
    "SourceAwareSplitConfig",
    "split_records",
    "stable_split_name",
]
