"""Post-inference evaluation of ranked, coherent beam tree hypotheses.

Nothing in this module feeds reconstruction. Oracle statistics use truth only
after the search has returned its fixed, model-ranked candidate list. Unit
oracles may recover different truth components in different hypotheses; the
separate event oracle requires every component in one coherent hypothesis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from hypertagging.evaluation.full_decay_metrics import (
    DecayEvaluation,
    HalfDecayEvaluation,
    RatioMetric,
    summarize_decay_evaluations,
)


DecayResult = DecayEvaluation | HalfDecayEvaluation


@dataclass(frozen=True)
class BeamDecayEvaluation:
    """One event's immutable ranked evaluations and aggregation-safe oracles."""

    candidates: tuple[DecayResult, ...]
    scores: tuple[float, ...]
    oracle_at_k: dict[int, dict[str, Any]]

    @property
    def top1(self) -> DecayResult:
        return self.candidates[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_role": "post_inference_beam_comparison",
            "oracle_uses_truth": True,
            "oracle_is_deployable_performance": False,
            "candidate_count": len(self.candidates),
            "top1": self.top1.as_dict(),
            "candidates": [
                {"rank": rank, "score": score, "metrics": candidate.as_dict()}
                for rank, (score, candidate) in enumerate(
                    zip(self.scores, self.candidates), start=1
                )
            ],
            "oracle_at_k": {str(k): row for k, row in sorted(self.oracle_at_k.items())},
        }


def evaluate_ranked_decay_candidates(
    candidates: Sequence[DecayResult],
    *,
    scores: Sequence[float],
    oracle_ks: Sequence[int] | None = None,
) -> BeamDecayEvaluation:
    """Compare already evaluated candidates without reordering by truth.

    Every K uses at most the returned candidate count. Failed reconstructions
    retain the original truth denominator; unavailable targets retain zero.
    Scores are diagnostic and never participate in oracle alignment.
    """

    candidates = tuple(candidates)
    scores = tuple(float(score) for score in scores)
    if not candidates:
        raise ValueError("beam evaluation requires at least one candidate")
    if len(candidates) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("beam candidate scores must be finite and match candidates")
    requested_ks = tuple(oracle_ks if oracle_ks is not None else (1, len(candidates)))
    if not requested_ks or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in requested_ks
    ):
        raise ValueError("oracle_ks must contain positive integers")
    ks = tuple(sorted(set(requested_ks)))
    units_by_candidate = tuple(_units(candidate) for candidate in candidates)
    reference = units_by_candidate[0]
    expected = tuple(_truth_identity(unit) for unit in reference)
    for candidate, units in zip(candidates, units_by_candidate):
        if type(candidate) is not type(candidates[0]):
            raise ValueError("beam candidates must use the same evaluation scope")
        if tuple(_truth_identity(unit) for unit in units) != expected:
            raise ValueError("beam candidates must share identical ordered truth units")

    oracles: dict[int, dict[str, Any]] = {}
    for k in ks:
        selected = units_by_candidate[:k]
        unit_records: list[dict[str, Any]] = []
        totals: Counter[str] = Counter()
        for index, truth_unit in enumerate(reference):
            unit_candidates = tuple(units[index] for units in selected)
            eligible = truth_unit.perfect_lcag.denominator > 0
            exact_rank = next(
                (
                    rank
                    for rank, unit in enumerate(unit_candidates, 1)
                    if unit.perfectLCAG is True
                ),
                None,
            )
            recall = max(
                (unit.source_recall.numerator for unit in unit_candidates),
                default=0.0,
            )
            mother_pid_coverage = max(
                (unit.mother_pid_coverage.numerator for unit in unit_candidates),
                default=0.0,
            )
            totals["exact_numerator"] += int(exact_rank is not None)
            totals["exact_denominator"] += truth_unit.perfect_lcag.denominator
            totals["source_numerator"] += recall
            totals["source_denominator"] += truth_unit.source_recall.denominator
            totals["mother_numerator"] += mother_pid_coverage
            totals["mother_denominator"] += truth_unit.mother_pid_coverage.denominator
            totals["reciprocal_rank_numerator"] += (
                1.0 / exact_rank if exact_rank else 0.0
            )
            totals["reciprocal_rank_denominator"] += int(eligible)
            totals["rank_numerator"] += exact_rank or 0
            totals["rank_denominator"] += int(exact_rank is not None)
            unit_records.append(
                {
                    "scope": truth_unit.scope,
                    "unit_index": truth_unit.unit_index,
                    "truth_root_position": truth_unit.truth_root_position,
                    "available": truth_unit.available,
                    "first_exact_rank": exact_rank,
                    "first_exact_score": scores[exact_rank - 1] if exact_rank else None,
                }
            )
        event_eligible = bool(reference) and all(
            unit.perfect_lcag.denominator > 0 for unit in reference
        )
        event_rank = next(
            (
                rank
                for rank, units in enumerate(selected, 1)
                if event_eligible and all(unit.perfectLCAG is True for unit in units)
            ),
            None,
        )
        both_eligible = isinstance(candidates[0], HalfDecayEvaluation) and (
            candidates[0].both_halves_perfect_lcag.denominator > 0
        )
        both_rank = next(
            (
                rank
                for rank, candidate in enumerate(candidates[:k], 1)
                if isinstance(candidate, HalfDecayEvaluation)
                and candidate.both_halves_perfect_lcag.value == 1
            ),
            None,
        )
        oracles[k] = {
            "requested_k": k,
            "evaluated_candidate_count": len(selected),
            "perfect_lcag": _ratio(
                totals["exact_numerator"], totals["exact_denominator"]
            ),
            "source_recall": _ratio(
                totals["source_numerator"], totals["source_denominator"]
            ),
            "mother_pid_coverage": _ratio(
                totals["mother_numerator"], totals["mother_denominator"]
            ),
            "mean_reciprocal_exact_rank": _ratio(
                totals["reciprocal_rank_numerator"],
                totals["reciprocal_rank_denominator"],
            ),
            "mean_first_exact_rank_when_recovered": _ratio(
                totals["rank_numerator"], totals["rank_denominator"]
            ),
            "coherent_event_perfect_lcag": _ratio(
                int(event_rank is not None), int(event_eligible)
            ),
            "both_halves_perfect_lcag": _ratio(
                int(both_rank is not None), int(both_eligible)
            ),
            "first_coherent_event_exact_rank": event_rank,
            "first_coherent_event_exact_score": scores[event_rank - 1]
            if event_rank
            else None,
            "units": unit_records,
        }
    return BeamDecayEvaluation(candidates, scores, oracles)


def summarize_beam_decay_evaluations(
    evaluations: Iterable[BeamDecayEvaluation],
    *,
    include_event_metrics: bool = True,
) -> dict[str, Any]:
    """Micro-sum sufficient statistics, never average per-event rates."""

    rows = list(evaluations)
    oracle_keys = tuple(sorted(rows[0].oracle_at_k)) if rows else ()
    if any(tuple(sorted(row.oracle_at_k)) != oracle_keys for row in rows):
        raise ValueError("beam summaries require the same oracle K values")
    oracles = {}
    for k in oracle_keys:
        records = [row.oracle_at_k[k] for row in rows]
        summary: dict[str, Any] = {"requested_k": k}
        for name in (
            "perfect_lcag",
            "source_recall",
            "mother_pid_coverage",
            "mean_reciprocal_exact_rank",
            "mean_first_exact_rank_when_recovered",
            "coherent_event_perfect_lcag",
            "both_halves_perfect_lcag",
        ):
            if not include_event_metrics and name in {
                "coherent_event_perfect_lcag",
                "both_halves_perfect_lcag",
            }:
                continue
            summary[name] = _ratio(
                sum(record[name]["numerator"] for record in records),
                sum(record[name]["denominator"] for record in records),
            )
        summary["evaluated_candidate_count"] = sum(
            record["evaluated_candidate_count"] for record in records
        )
        oracles[str(k)] = summary
    return {
        ("event_count" if include_event_metrics else "unit_count"): len(rows),
        "aggregation_unit": "event" if include_event_metrics else "truth_decay_unit",
        "oracle_uses_truth": True,
        "oracle_is_deployable_performance": False,
        "candidate_count": sum(len(row.candidates) for row in rows),
        "candidate_count_distribution": dict(
            sorted(Counter(str(len(row.candidates)) for row in rows).items())
        ),
        "top1": summarize_decay_evaluations(row.top1 for row in rows),
        "oracle_at_k": oracles,
    }


def _units(candidate: DecayResult) -> tuple[DecayEvaluation, ...]:
    return (
        candidate.halves if isinstance(candidate, HalfDecayEvaluation) else (candidate,)
    )


def _truth_identity(unit: DecayEvaluation) -> tuple[Any, ...]:
    return (
        unit.scope,
        unit.unit_index,
        unit.available,
        unit.truth_root_position,
        unit.truth_sources,
        unit.truth_topology_mode,
        unit.perfect_lcag.denominator,
        unit.source_recall.denominator,
        unit.mother_pid_coverage.denominator,
    )


def _ratio(numerator: float, denominator: float) -> dict[str, float | None]:
    return RatioMetric(numerator, denominator).as_dict()


__all__ = [
    "BeamDecayEvaluation",
    "evaluate_ranked_decay_candidates",
    "summarize_beam_decay_evaluations",
]
