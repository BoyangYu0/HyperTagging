import json
import subprocess
import sys

import pytest

from hypertagging.training import (
    run_embedding_dry_run,
    run_gpt_dry_run,
    run_link_dry_run,
    run_reconstruction_dry_run,
)


@pytest.mark.parametrize(
    ("runner", "stage", "optimizer"),
    [
        (run_embedding_dry_run, "embedding", "Adam"),
        (run_link_dry_run, "link", "AdamW"),
        (run_reconstruction_dry_run, "reconstruction", "AdamW"),
        (run_gpt_dry_run, "gpt", "AdamW"),
    ],
)
def test_training_stage_dry_runs_on_cpu(runner, stage, optimizer):
    summary = runner(device="cpu")

    assert summary.stage == stage
    assert summary.device == "cpu"
    assert summary.optimizer_class == optimizer
    assert summary.loss == pytest.approx(summary.loss)
    assert summary.parameter_count > 0
    assert summary.backward_ran is True
    assert all(all(dim > 0 for dim in shape) for shape in summary.output_shapes)


@pytest.mark.parametrize(
    ("script", "stage"),
    [
        ("scripts/train_embedding.py", "embedding"),
        ("scripts/train_link.py", "link"),
        ("scripts/train_reconstruction.py", "reconstruction"),
        ("scripts/train_gpt_like.py", "gpt"),
    ],
)
def test_training_cli_dry_run_cpu(script, stage):
    completed = subprocess.run(
        [sys.executable, script, "--dry-run", "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["stage"] == stage
    assert summary["device"] == "cpu"
    assert summary["backward_ran"] is True
    assert summary["parameter_count"] > 0
