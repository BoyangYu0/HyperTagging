"""Dataset query/cardinality capacity accounting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from hypertagging.data.heterogeneous import HeterogeneousEvent


@dataclass(frozen=True)
class CapacityStatistics:
    maximum_mothers_per_level: dict[int, int]
    percentile_mothers_per_level: dict[int, dict[str, float]]
    daughter_cardinality_counts: dict[int, int]
    maximum_daughter_cardinality: int
    query_overflow_events: int
    cardinality_overflow_mothers: int
    query_overflow_rate: float
    cardinality_overflow_rate: float


def dataset_capacity_statistics(
    events: Iterable[HeterogeneousEvent],
    *,
    n_queries_by_level: Mapping[int, int] | None = None,
    max_cardinality_by_level: Mapping[int, int] | None = None,
    global_n_queries: int | None = None,
    global_max_cardinality: int | None = None,
    target_policy: str = "complete_only",
) -> CapacityStatistics:
    per_level: dict[int, list[int]] = {}
    cardinality_counts: Counter[int] = Counter()
    query_overflow = 0
    cardinality_overflow = 0
    total_level_events = 0
    total_mothers = 0
    for event in events:
        levels = sorted(
            {
                int(level)
                for level in event.level_ids[event.active].tolist()
                if int(level) > 0
            }
        )
        for level in levels:
            eligible = event.active & (event.level_ids == level)
            if target_policy != "diagnostic_all":
                eligible &= event.valid_reconstruction_target
            if target_policy == "complete_only":
                eligible &= event.recursive_reconstructable_complete
            elif target_policy != "reconstructable_partial":
                raise ValueError(f"unknown reconstruction target policy: {target_policy}")
            mothers = eligible.nonzero(as_tuple=False).flatten()
            count = int(mothers.numel())
            per_level.setdefault(level, []).append(count)
            total_level_events += 1
            capacity = (
                n_queries_by_level.get(level)
                if n_queries_by_level is not None and level in n_queries_by_level
                else global_n_queries
            )
            if capacity is not None and count > capacity:
                query_overflow += 1
            for mother in mothers.tolist():
                cardinality = int(event.daughter_adjacency[mother].sum())
                cardinality_counts[cardinality] += 1
                total_mothers += 1
                cardinality_capacity = (
                    max_cardinality_by_level.get(level)
                    if max_cardinality_by_level is not None and level in max_cardinality_by_level
                    else global_max_cardinality
                )
                if cardinality_capacity is not None and cardinality > cardinality_capacity:
                    cardinality_overflow += 1
    percentiles = {
        level: {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
        for level, values in per_level.items()
    }
    return CapacityStatistics(
        maximum_mothers_per_level={
            level: max(values) for level, values in per_level.items()
        },
        percentile_mothers_per_level=percentiles,
        daughter_cardinality_counts=dict(sorted(cardinality_counts.items())),
        maximum_daughter_cardinality=max(cardinality_counts, default=0),
        query_overflow_events=query_overflow,
        cardinality_overflow_mothers=cardinality_overflow,
        query_overflow_rate=query_overflow / max(total_level_events, 1),
        cardinality_overflow_rate=cardinality_overflow / max(total_mothers, 1),
    )


def require_capacity(
    statistics: CapacityStatistics,
    *,
    allow_query_overflow: bool = False,
    allow_cardinality_overflow: bool = False,
) -> None:
    if statistics.query_overflow_events and not allow_query_overflow:
        raise OverflowError(
            f"{statistics.query_overflow_events} event-level targets exceed query capacity"
        )
    if statistics.cardinality_overflow_mothers and not allow_cardinality_overflow:
        raise OverflowError(
            f"{statistics.cardinality_overflow_mothers} mothers exceed cardinality capacity"
        )


def capacity_statistics_from_index(
    index: Mapping[str, object],
    *,
    global_n_queries: int,
    global_max_cardinality: int,
    n_queries_by_level: Mapping[int, int] | None = None,
    max_cardinality_by_level: Mapping[int, int] | None = None,
) -> CapacityStatistics:
    """Reconstruct bounded capacity statistics from histogram sidecars."""

    histograms = index.get("mother_count_histograms_by_level", {})
    maxima: dict[int, int] = {}
    percentiles: dict[int, dict[str, float]] = {}
    query_overflow = total_level_events = 0
    for level_text, raw_hist in dict(histograms).items():
        level = int(level_text)
        hist = {int(key): int(value) for key, value in dict(raw_hist).items()}
        maxima[level] = max(hist, default=0)
        total = sum(hist.values())
        total_level_events += total
        query_capacity = (
            n_queries_by_level[level]
            if n_queries_by_level is not None and level in n_queries_by_level
            else global_n_queries
        )
        query_overflow += sum(v for k, v in hist.items() if k > query_capacity)
        percentiles[level] = {
            name: _histogram_percentile(hist, quantile)
            for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
        }
    cardinality = {
        int(key): int(value)
        for key, value in dict(index.get("daughter_cardinality_histogram", {})).items()
    }
    total_mothers = sum(cardinality.values())
    per_level_cardinality = dict(
        index.get("daughter_cardinality_histograms_by_level", {})
    )
    if per_level_cardinality:
        cardinality_overflow = 0
        for level_text, raw_hist in per_level_cardinality.items():
            level = int(level_text)
            capacity = (
                max_cardinality_by_level[level]
                if max_cardinality_by_level is not None
                and level in max_cardinality_by_level
                else global_max_cardinality
            )
            cardinality_overflow += sum(
                int(value)
                for count, value in dict(raw_hist).items()
                if int(count) > capacity
            )
    else:
        # Compatibility indexes lack level-resolved histograms. Conservatively
        # use the smallest configured limit rather than under-report overflow.
        effective_cardinality = min(
            [global_max_cardinality, *(max_cardinality_by_level or {}).values()]
        )
        cardinality_overflow = sum(
            value for count, value in cardinality.items()
            if count > effective_cardinality
        )
    return CapacityStatistics(
        maximum_mothers_per_level=maxima,
        percentile_mothers_per_level=percentiles,
        daughter_cardinality_counts=cardinality,
        maximum_daughter_cardinality=max(cardinality, default=0),
        query_overflow_events=query_overflow,
        cardinality_overflow_mothers=cardinality_overflow,
        query_overflow_rate=query_overflow / max(total_level_events, 1),
        cardinality_overflow_rate=cardinality_overflow / max(total_mothers, 1),
    )


def _histogram_percentile(histogram: Mapping[int, int], quantile: float) -> float:
    target = quantile * max(sum(histogram.values()) - 1, 0)
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative - 1 >= target:
            return float(value)
    return float(max(histogram, default=0))


__all__ = [
    "CapacityStatistics",
    "capacity_statistics_from_index",
    "dataset_capacity_statistics",
    "require_capacity",
]
