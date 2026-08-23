from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm.phase3_execution_authorization_v1 import (
    AUTHORIZATION_ARTIFACT,
    BASE_CONTRACT,
    EXPECTED_CONTRACT_FILE_SHA256,
    EXPECTED_GRES,
    PREFLIGHT_VERSION,
    canonical_sha256,
    file_sha256,
    verify_authorization_artifact,
)


AUTHORIZATION = ROOT / AUTHORIZATION_ARTIFACT
CONTRACT = ROOT / BASE_CONTRACT


def _write_preflight(path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "preflight_version": PREFLIGHT_VERSION,
        "status": "passed",
        "fresh_in_allocation": True,
        "gpu_pilot_completed": False,
        "never_cpu": True,
        "expected_gres": EXPECTED_GRES,
        "observed_gres": EXPECTED_GRES,
        "gpu": "NVIDIA H100 NVL",
        "slurm_job_id": "17000001",
        "slurm_gpus_on_node": "1",
        "cuda_visible_devices": "0",
        "timestamp": "2026-08-23T21:50:00+02:00",
        "execution_contract_file_sha256": file_sha256(CONTRACT),
        "authorization_artifact_file_sha256": file_sha256(AUTHORIZATION),
        "authorization_artifact_canonical_sha256": json.loads(
            AUTHORIZATION.read_text(encoding="utf-8")
        )["artifact_sha256"],
    }
    payload.update(overrides)
    payload["preflight_sha256"] = canonical_sha256(
        payload, digest_field="preflight_sha256"
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_new_authorization_is_distinct_from_old_false_report_and_self_hashed():
    payload = verify_authorization_artifact(AUTHORIZATION, contract_path=CONTRACT)
    assert payload["authorization_basis"] == "explicit_user_operator_instruction"
    assert payload["submission_authorized"] is True
    assert payload["submission_performed"] is False
    assert payload["gpu_pilot_completed"] is False
    assert payload["fresh_in_allocation_preflight_required"] is True
    assert payload["execution_role_required"] == "gpt-5.3-codex-spark medium"
    assert payload["immutable_parent_hashes"]["historical_failed_commit"] == (
        "93b71c5d7c1bc20181640aafb4e918abb9267362"
    )
    assert payload["retained_gates"]["objective_dominance"] == {
        "limit": 20.0,
        "violation_action": "fail",
        "fail_closed": True,
        "historical_observed_ratio": 22.894,
        "late_phase_leaf_pid_weight": 0.4,
        "projected_exact_ratio": 18.3152,
    }
    old_report = json.loads(
        (ROOT / "artifacts/codex/ht_pretraining_1m_phase3_recovery_20260823.json")
        .read_text(encoding="utf-8")
    )
    assert old_report["replacement"]["submission_authorized"] is False
    assert file_sha256(CONTRACT) == EXPECTED_CONTRACT_FILE_SHA256


def test_fresh_in_allocation_preflight_is_required_and_bound(tmp_path):
    preflight = tmp_path / "gpu-preflight.json"
    _write_preflight(preflight)
    verify_authorization_artifact(
        AUTHORIZATION,
        contract_path=CONTRACT,
        require_fresh_preflight=True,
        preflight_path=preflight,
    )


def test_missing_or_mismatched_preflight_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="preflight is required"):
        verify_authorization_artifact(
            AUTHORIZATION,
            contract_path=CONTRACT,
            require_fresh_preflight=True,
        )

    preflight = tmp_path / "gpu-preflight.json"
    _write_preflight(preflight, observed_gres="gpu:v100:1")
    with pytest.raises(RuntimeError, match="preflight GRES"):
        verify_authorization_artifact(
            AUTHORIZATION,
            contract_path=CONTRACT,
            require_fresh_preflight=True,
            preflight_path=preflight,
        )


def test_preflight_tampering_fails_even_when_the_contract_is_unchanged(tmp_path):
    preflight = tmp_path / "gpu-preflight.json"
    _write_preflight(preflight)
    preflight.write_text(
        preflight.read_text(encoding="utf-8").replace('"status": "passed"', '"status": "tampered"'),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="preflight hash mismatch"):
        verify_authorization_artifact(
            AUTHORIZATION,
            contract_path=CONTRACT,
            require_fresh_preflight=True,
            preflight_path=preflight,
        )


def test_execution_path_contains_versioned_authorization_and_preflight_barrier():
    renderer = (ROOT / "scripts/slurm/render_one_gpu_job.py").read_text()
    wrapper = (ROOT / "scripts/slurm/train_one_gpu.sbatch").read_text()
    verifier = (ROOT / "scripts/slurm/verify_job_contract.py").read_text()
    assert AUTHORIZATION_ARTIFACT in renderer
    assert "fresh_in_allocation_preflight_required" in renderer
    assert "verify_phase3_execution_authorization_v1.py" in wrapper
    assert "--require-fresh-in-allocation-preflight" in wrapper
    assert "phase3_execution_authorization_v1" in verifier
    assert "gpu_pilot_completed" in verifier or "gpu_pilot_completed" in (
        ROOT / "scripts/slurm/phase3_execution_authorization_v1.py"
    ).read_text()
