"""Dataset query/cardinality capacity accounting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

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
    events: Sequence[HeterogeneousEvent],
    *,
    n_queries_by_level: Mapping[int, int] | None = None,
    max_cardinality_by_level: Mapping[int, int] | None = None,
    global_n_queries: int | None = None,
    global_max_cardinality: int | None = None,
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
            mothers = (
                event.active & (event.level_ids == level)
            ).nonzero(as_tuple=False).flatten()
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


__all__ = ["CapacityStatistics", "dataset_capacity_statistics", "require_capacity"]
