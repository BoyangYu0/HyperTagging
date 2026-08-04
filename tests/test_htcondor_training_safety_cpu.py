"""HTCondor safety and real-training entry-point tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import os
import subprocess

import pytest

from hypertagging.utils.gpu_safety import assert_full_training_requires_condor


REPOSITORY = Path(__file__).resolve().parents[1]


def _args(device: str, *, tiny: bool = False, allow: bool = False) -> Namespace:
    return Namespace(
        device=device,
        tiny=tiny,
        max_steps=2 if tiny else 1000,
        batch_size=1 if tiny else 64,
        allow_local_tiny_gpu_test=allow,
    )


def test_local_cpu_pilot_is_allowed(monkeypatch):
    monkeypatch.delenv("_CONDOR_SCRATCH_DIR", raising=False)
    monkeypatch.delenv("CONDOR_CLUSTER_ID", raising=False)
    assert_full_training_requires_condor(_args("cpu"))


def test_full_cuda_is_refused_outside_condor(monkeypatch):
    monkeypatch.delenv("_CONDOR_SCRATCH_DIR", raising=False)
    monkeypatch.delenv("CONDOR_CLUSTER_ID", raising=False)
    monkeypatch.delenv("CONDOR_PROCESS_ID", raising=False)
    with pytest.raises(RuntimeError, match="HTCondor"):
        assert_full_training_requires_condor(_args("cuda"))


def test_real_cuda_is_allowed_inside_mocked_condor(monkeypatch):
    monkeypatch.setenv("CONDOR_CLUSTER_ID", "31415")
    assert_full_training_requires_condor(_args("cuda"))


@pytest.mark.parametrize(
    ("script", "extra_environment", "trainer"),
    [
        (
            "submit_hyperbolic_pretrain.sh",
            {},
            "scripts/train_hyperbolic_pretrain.py",
        ),
        (
            "submit_level_reconstruction.sh",
            {"PRETRAINED_ENCODER": "/tmp/pretrained checkpoint.pt"},
            "scripts/train_level_reconstruction.py",
        ),
    ],
)
def test_condor_wrappers_render_real_data_trainers_without_submitting(
    script: str,
    extra_environment: dict[str, str],
    trainer: str,
):
    environment = {
        **os.environ,
        "DATA_MANIFEST": "/tmp/tiny manifest.jsonl",
        "OUTPUT_DIR": "/tmp/hypertagging condor output",
        **extra_environment,
    }
    result = subprocess.run(
        ["bash", f"scripts/condor/{script}", "--dry-run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert trainer in result.stdout
    assert "--data" in result.stdout
    assert "condor_submit " not in result.stdout
    assert "request_gpus = 1" in result.stdout


def test_mdst_worker_disables_nounset_while_sourcing_belle2_environment():
    launcher = (
        REPOSITORY / "scripts" / "condor" / "submit_mdst_production_10m.sh"
    ).read_text(encoding="utf-8")
    setup = "source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00"
    setup_position = launcher.index(setup)
    assert launcher.rindex("set +u", 0, setup_position) < setup_position
    assert launcher.index("set -u", setup_position) > setup_position
