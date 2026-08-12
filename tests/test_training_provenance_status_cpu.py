from pathlib import Path
import subprocess
import sys

from hypertagging.data.training_selection import load_hashed_manifest


ROOT = Path(__file__).resolve().parents[1]
STATUS = (
    ROOT
    / "configs"
    / "training_selection"
    / "production_1m_20260812"
    / "provenance_status.json"
)


def test_cpu_work_is_allowed_but_scientific_submission_fails_closed():
    status = load_hashed_manifest(
        STATUS,
        expected_version="hypertagging-training-provenance-status-v1",
    )
    assert status["cpu_implementation_allowed"] is True
    assert status["scientific_slurm_submission_allowed"] is False
    assert status["dataset_campaign"]["source_object_available"] is False
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_training_provenance.py"),
    ]
    assert subprocess.run(command, cwd=ROOT, check=False).returncode == 0
    blocked = subprocess.run(
        command + ["--require-scientific-slurm-ready"],
        cwd=ROOT,
        check=False,
    )
    assert blocked.returncode == 2
