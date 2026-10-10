"""Post-PID tracing is observational and preserves the detector boundary."""

from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import RolloutConfig, level_rollout
from scripts.phase84_hierarchy_trace import capture_native_trace


def fixture():
    path = Path(__file__).with_name("test_hierarchical_inference_cpu.py")
    spec = importlib.util.spec_from_file_location("phase86_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._normalized_native_v4_batch(), module._TwoBThenUpsilonModel


def config():
    return RolloutConfig(
        max_level=3,
        root_types=(),
        continue_through_empty_levels=True,
        constraint_policy=ReconstructionConstraintPolicy(),
    )


def test_trace_preserves_predictions_and_scrubs_targets():
    batch, model = fixture()
    _, baseline, plain = capture_native_trace(model(), batch, config())
    enabled = replace(config(), capture_decode_trace=True)
    _, observed, trace = capture_native_trace(model(), batch, enabled)
    assert all(s.decode_trace is None for s in baseline.steps)
    assert [s.proposals for s in baseline.steps] == [
        s.proposals for s in observed.steps
    ]
    assert [s.accepted for s in baseline.steps] == [s.accepted for s in observed.steps]
    for key, value in baseline.batch.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, observed.batch[key], rtol=0, atol=0)
    changed = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    forbidden = (
        "truth_pid_labels",
        "pid_target_labels",
        "b_side",
        "parent_ids",
        "daughter_adjacency",
    )
    for key in forbidden:
        if key in changed:
            changed[key].zero_()
    _, _, mutated = capture_native_trace(model(), changed, enabled)
    assert trace == mutated
    for row in trace["steps"]:
        state = row["decode_trace"]["state"]
        assert not set(state) & {
            "truth_pid_labels",
            "pid_target_labels",
            "b_side",
            "evaluation_leaf_source_keys",
            "valid_reconstruction_target",
        }
        assert row["decode_trace"]["raw_proposals"] == row["proposals"]
        expected_context = [
            visible and parent < 0
            for visible, parent in zip(
                row["decode_trace"]["context_mask"], state["parent_ids"]
            )
        ]
        assert row["decode_trace"]["hard_decode_context_mask"] == expected_context
    for row in trace["steps"]:
        row.pop("decode_trace")
    trace["config"]["capture_decode_trace"] = False
    assert trace == plain


def test_snapshot_is_after_pid_update_and_does_not_alias(monkeypatch):
    batch, base = fixture()
    module = importlib.import_module("hypertagging.reconstruction.level_rollout")

    class WithPID(base):
        def __call__(self, *args, **kwargs):
            output = super().__call__(*args, **kwargs)
            return replace(
                output, leaf_pid_logits=torch.zeros(1, output.context_mask.shape[1], 2)
            )

    def rebuilt(state, _logits, **_kwargs):
        state = dict(state)
        state["p4"] = state["p4"].clone()
        state["p4"][..., 3] += 0.25
        return state

    monkeypatch.setattr(module, "_with_predicted_leaf_p4", rebuilt)
    _, result, trace = capture_native_trace(
        WithPID(), batch, replace(config(), capture_decode_trace=True)
    )
    before = torch.tensor(trace["steps"][0]["state_before"]["p4"])
    after = torch.tensor(trace["steps"][0]["decode_trace"]["state"]["p4"])
    torch.testing.assert_close(after[:, 3], before[:, 3] + 0.25)
    snapshot = result.steps[0].decode_trace["state"]["p4"]
    expected = snapshot.clone()
    with torch.inference_mode():
        result.batch["p4"].zero_()
    torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)


def test_trace_rejects_truth_guided_modes():
    batch, model = fixture()
    for mode in ("teacher_forced", "scheduled"):
        with pytest.raises(ValueError, match="predicted-only"):
            level_rollout(
                model(),
                batch,
                mode=mode,
                config=replace(config(), capture_decode_trace=True),
            )
