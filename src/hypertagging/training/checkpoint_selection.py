"""Explicit, serializable validation checkpoint-selection semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class CheckpointTrack:
    filename: str
    metric: str
    mode: str
    denominator_metric: str
    requires_rollout: bool = False


RECONSTRUCTION_CHECKPOINT_TRACKS: tuple[CheckpointTrack, ...] = (
    CheckpointTrack(
        "best_teacher_forced.pt",
        "validation_loss_total",
        "min",
        "validation_teacher_forced_terms",
    ),
    CheckpointTrack(
        "best_rollout_edge_f1.pt",
        "predicted_edge_f1",
        "max",
        "rollout_validation_events",
        requires_rollout=True,
    ),
    CheckpointTrack(
        "best_rollout_tree_validity.pt",
        "predicted_tree_validity_rate",
        "max",
        "rollout_validation_events",
        requires_rollout=True,
    ),
)

RECONSTRUCTION_TRACK_BY_METRIC = {
    track.metric: track for track in RECONSTRUCTION_CHECKPOINT_TRACKS
}

PRETRAIN_CHECKPOINT_TRACKS: tuple[CheckpointTrack, ...] = (
    CheckpointTrack(
        "best_principal_topology.pt",
        "validation_loss_lca",
        "min",
        "validation_active_denominator_lca",
    ),
    CheckpointTrack(
        "best_parent_ranking.pt",
        "validation_parent_ranking_accuracy",
        "max",
        "validation_parent_ranking_accuracy_denominator",
    ),
    CheckpointTrack(
        "best_tree_distance.pt",
        "validation_loss_tree_distance",
        "min",
        "validation_active_denominator_tree_distance",
    ),
    CheckpointTrack(
        "best_non_collapse_effective_rank.pt",
        "validation_effective_rank",
        "max",
        "validation_active_denominator_variance",
    ),
    CheckpointTrack(
        "best_channel_retrieval.pt",
        "validation_channel_retrieval_accuracy",
        "max",
        "validation_channel_retrieval_queries",
    ),
)


def initial_track_values(
    tracks: tuple[CheckpointTrack, ...] = RECONSTRUCTION_CHECKPOINT_TRACKS,
) -> dict[str, float]:
    return {
        track.metric: math.inf if track.mode == "min" else -math.inf
        for track in tracks
    }


def checkpoint_track_decisions(
    metrics: Mapping[str, float],
    previous: Mapping[str, float],
    tracks: tuple[CheckpointTrack, ...] = RECONSTRUCTION_CHECKPOINT_TRACKS,
) -> tuple[dict[str, float], tuple[CheckpointTrack, ...]]:
    """Update only tracks whose metric was genuinely evaluated this step."""

    updated = {**initial_track_values(tracks), **{key: float(value) for key, value in previous.items()}}
    selected: list[CheckpointTrack] = []
    rollout_events = float(metrics.get("rollout_validation_events", 0.0))
    for track in tracks:
        if track.requires_rollout and rollout_events <= 0:
            continue
        if track.metric not in metrics:
            continue
        value = float(metrics[track.metric])
        denominator = float(metrics.get(track.denominator_metric, 0.0))
        if not math.isfinite(value) or denominator <= 0:
            continue
        improved = value < updated[track.metric] if track.mode == "min" else value > updated[track.metric]
        if improved:
            updated[track.metric] = value
            selected.append(track)
    return updated, tuple(selected)


def reconstruction_selection_contract(
    *,
    best_metric: str,
    best_mode: str,
    max_validation_events: int,
    rollout_validation_events: int,
    rollout_validate_every: int,
    rollout_pid_kinematics_mode: str,
    rollout_pid_temperature: float,
    target_policy: str,
    constraint_policy: Mapping[str, object],
    eligibility_gates: Mapping[str, object] | None = None,
    scientific_mode: bool = False,
    validation_selection_manifest_hash: str = "",
) -> dict[str, object]:
    """Fields whose change makes checkpoint ranking incomparable on resume."""

    return {
        "version": "reconstruction-checkpoint-selection-v4",
        "primary_metric": best_metric,
        "primary_mode": best_mode,
        "tracks": [
            {
                "filename": track.filename,
                "metric": track.metric,
                "mode": track.mode,
                "denominator_metric": track.denominator_metric,
                "requires_rollout": track.requires_rollout,
            }
            for track in RECONSTRUCTION_CHECKPOINT_TRACKS
        ],
        "validation_event_limit": int(max_validation_events),
        "rollout_event_limit": int(rollout_validation_events),
        "rollout_validate_every": int(rollout_validate_every),
        "rollout_configuration": {
            "max_level": 8,
            "exclusive_resolution": "greedy",
            "bounded_weighted_set_packing": "diagnostic_only",
            "learned_confidence": True,
        },
        "constraint_policy": dict(constraint_policy),
        "pid_mode": rollout_pid_kinematics_mode,
        "pid_temperature": float(rollout_pid_temperature),
        "thresholds": {
            "object_probability": 0.5,
            "daughter_pointer_probability": 0.5,
            "confidence": 0.0,
            "type_probability": None,
        },
        "target_policy": target_policy,
        "primary_eligibility_gates": dict(eligibility_gates or {}),
        "validation_selection": {
            "version": (
                "manifest-role-uid-hash-v1"
                if scientific_mode else "ci-source-prefix-v1"
            ),
            "scientific_mode": bool(scientific_mode),
            "selection_manifest_hash": validation_selection_manifest_hash,
        },
    }


def rollout_checkpoint_eligibility(
    metrics: Mapping[str, float],
    gates: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate hard rollout gates independently of metric improvement."""

    failures: list[str] = []
    required_denominators = tuple(
        str(name) for name in gates.get("required_denominators", ())
    )
    for name in required_denominators:
        value = float(metrics.get(name, 0.0))
        if not math.isfinite(value) or value <= 0:
            failures.append(f"nonzero_denominator:{name}")
    checks = (
        (
            "predicted_tree_validity_rate",
            float(gates.get("minimum_tree_validity", 0.999)),
            "minimum",
        ),
        (
            "predicted_p4_closure_rate",
            float(gates.get("minimum_p4_closure", 1.0)),
            "minimum",
        ),
        (
            "predicted_recursive_source_conflicts",
            float(gates.get("maximum_recursive_source_conflicts", 0.0)),
            "maximum",
        ),
    )
    for metric, threshold, direction in checks:
        value = float(metrics.get(metric, float("nan")))
        if not math.isfinite(value):
            failures.append(f"finite:{metric}")
        elif direction == "minimum" and value < threshold:
            failures.append(f"minimum:{metric}")
        elif direction == "maximum" and value > threshold:
            failures.append(f"maximum:{metric}")
    return {
        "eligible": not failures,
        "failures": failures,
        "evaluated_metrics": {
            name: float(metrics.get(name, float("nan")))
            for name, _, _ in checks
        },
    }


def selection_reason(
    track: CheckpointTrack,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    return {
        "metric_name": track.metric,
        "mode": track.mode,
        "value": float(metrics[track.metric]),
        "denominator_name": track.denominator_metric,
        "denominator": float(metrics[track.denominator_metric]),
        "reason": f"new_{track.mode}_for_{track.metric}",
    }


__all__ = [
    "CheckpointTrack",
    "RECONSTRUCTION_CHECKPOINT_TRACKS",
    "RECONSTRUCTION_TRACK_BY_METRIC",
    "PRETRAIN_CHECKPOINT_TRACKS",
    "checkpoint_track_decisions",
    "initial_track_values",
    "reconstruction_selection_contract",
    "rollout_checkpoint_eligibility",
    "selection_reason",
]
