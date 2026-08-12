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
READINESS = STATUS.with_name("training_readiness.json")


def test_cpu_work_is_allowed_but_scientific_submission_fails_closed():
    status = load_hashed_manifest(
        STATUS,
        expected_version="hypertagging-training-provenance-status-v1",
    )
    assert status["cpu_implementation_allowed"] is True
    assert status["scientific_slurm_submission_allowed"] is False
    assert status["dataset_campaign"]["source_object_available"] is False
    readiness = load_hashed_manifest(
        READINESS,
        expected_version="hypertagging-training-readiness-v1",
    )
    assert readiness["index_gate"]["status"] == "complete"
    assert readiness["index_gate"]["sealed_test_opened"] is False
    assert (
        readiness["capacity_gate"]["small_candidate"]["production_training_allowed"]
        is True
    )
    assert readiness["capacity_gate"]["gpu_debug"]["reconstruction_allowed"] is False
    assert status["final_non_gpu_readiness"]["trainer_tranche_complete"] is True
    assert status["final_non_gpu_readiness"]["slurm_tranche_complete"] is True
    assert readiness["local_gpu_microtest_gate"]["completion_receipt"] is None
    assert readiness["local_gpu_microtest_gate"]["required_renderer_flags"] == [
        "--local-admission-receipt",
        "--local-completion-receipt",
    ]
    assert readiness["scientific_submission_allowed"] is False
    assert readiness["slurm_tranche"]["launch_environment"] == (
        "export_nil_and_verified_contract_only"
    )
    assert readiness["slurm_tranche"]["rejected_export_policies"] == ["NONE", "ALL"]
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
