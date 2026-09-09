from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_reconstruction_phase35_full_decay import (
    ALLOWED_SELECTION_STEPS,
    _comparison_endpoints,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "configs/reconstruction/ht_reconstruction_phase35_improvement_20260904.json"
)


def _report(values: tuple[float, float, float, float]) -> dict:
    def metric(value: float, denominator: int) -> dict[str, float]:
        return {
            "numerator": value * denominator,
            "denominator": float(denominator),
            "value": value,
        }

    return {
        "summaries": {
            "full": {
                "inference": {
                    "configured_root_completion": metric(values[0], 100)
                },
                "decay_metrics": {
                    "source_recall": metric(values[1], 400),
                    "lcag_pair_accuracy": metric(values[2], 4000),
                    "mother_pid_coverage": metric(values[3], 200),
                },
            }
        }
    }


def test_endpoint_comparison_preserves_counts_and_reports_absolute_deltas() -> None:
    comparison = _comparison_endpoints(
        _report((0.0, 0.10, 0.0, 0.05)),
        _report((0.20, 0.25, 0.04, 0.15)),
    )

    assert set(comparison) == {
        "configured_root_completion",
        "full_decay_source_recall",
        "full_decay_lcag_pair_accuracy",
        "full_decay_mother_pid_coverage",
    }
    assert comparison["configured_root_completion"] == {
        "baseline": {"numerator": 0.0, "denominator": 100.0, "value": 0.0},
        "candidate": {"numerator": 20.0, "denominator": 100.0, "value": 0.2},
        "absolute_delta": pytest.approx(0.2),
    }
    assert comparison["full_decay_source_recall"]["absolute_delta"] == pytest.approx(
        0.15
    )


def test_full_decay_comparison_is_disjoint_from_checkpoint_selection() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / prereg["evaluation_cohort"]["manifest"]).read_text(encoding="utf-8")
    )
    selection = prereg["selection_and_evaluation"]["checkpoint_selection"]
    protocol = prereg["selection_and_evaluation"][
        "post_training_full_decay_protocol"
    ]

    assert set(manifest["event_uids"]).isdisjoint(
        manifest["checkpoint_selection_event_uids"]
    )
    assert selection["evaluation_cohort_used_for_checkpoint_selection"] is False
    assert selection["eligible_rollout_steps"] == list(ALLOWED_SELECTION_STEPS)
    assert protocol["candidate_checkpoint"].startswith("best.pt_selected_only")
    assert protocol["max_level"] == 6
    assert prereg["selection_and_evaluation"][
        "evaluation_cohort_arm_selection_authorized"
    ] is False


def test_runner_does_not_reselect_temporal_checkpoints_on_comparison_cohort() -> None:
    source = (
        ROOT / "scripts/run_reconstruction_phase35_full_decay.py"
    ).read_text(encoding="utf-8")

    assert 'training_dir / "best.pt"' in source
    assert "candidate_checkpoints" not in source
    assert "screen_score" not in source
    assert "max(eligible" not in source
    assert "baseline_checkpoint_direct" in source
    assert "candidate_checkpoint_direct" in source
