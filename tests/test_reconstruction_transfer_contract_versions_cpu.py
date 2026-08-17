from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm import (  # noqa: E402
    verify_reconstruction_transfer_probe_contract as verifier,
)
from scripts.slurm.finalize_reconstruction_transfer_probe_receipt import (  # noqa: E402
    result_satisfies_contract,
)


def _hash_contract(contract: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        **contract,
        "contract_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _contract(
    *,
    version: str,
    checkpoint_step: int,
    checkpoint: str = "checkpoint.pt",
    checkpoint_sha256: str = "0" * 64,
    gpu_environment: str = "/gpu-env",
) -> dict[str, object]:
    required = verifier.REQUIRED_PROBES_BY_CONTRACT_VERSION[version]
    return _hash_contract(
        {
            "contract_version": version,
            "mode": "frozen_pretrained_reconstruction_transfer_probe",
            "training_role": "train",
            "evaluation_role": "validation",
            "sealed_test_role_access": "forbidden",
            "checkpoint_step": checkpoint_step,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "selection_manifest": "selection.json",
            "dataset_index": "index.json",
            "study_output_base": (
                "artifacts/studies/reconstruction-transfer-probe/test"
            ),
            "gpu_environment": gpu_environment,
            "gres": "gpu:v100:1",
            "expected_git_sha": "a" * 40,
            "expected_git_tag": "test-tag",
            "optimizer_steps": required["max_steps"],
            "probe": dict(required),
            "hashed_inputs": [],
            "submission_authorized": True,
        }
    )


def _write_verifiable_contract(
    tmp_path: Path,
    *,
    version: str,
    checkpoint_step: int,
) -> Path:
    checkpoint = tmp_path / f"checkpoint-step-{checkpoint_step}.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    (tmp_path / "selection.json").write_text("{}\n")
    (tmp_path / "index.json").write_text("{}\n")
    hashed_input = tmp_path / "hashed-input.txt"
    hashed_input.write_text("bound\n")
    gpu_environment = tmp_path / "gpu-env"
    (gpu_environment / "bin").mkdir(parents=True)
    (gpu_environment / "bin" / "python").write_text("")
    contract = _contract(
        version=version,
        checkpoint_step=checkpoint_step,
        checkpoint=checkpoint.name,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        gpu_environment=str(gpu_environment),
    )
    contract["hashed_inputs"] = [
        {
            "path": hashed_input.name,
            "sha256": hashlib.sha256(hashed_input.read_bytes()).hexdigest(),
        }
    ]
    contract.pop("contract_sha256")
    contract = _hash_contract(contract)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path


def _fake_git(*args: str) -> str:
    if args in {
        ("rev-parse", "HEAD"),
        ("rev-list", "-n", "1", "test-tag"),
    }:
        return "a" * 40
    if args == ("status", "--porcelain", "--untracked-files=no"):
        return ""
    raise AssertionError(args)


def test_v2_profile_changes_only_optimizer_and_freeze_horizons():
    v1 = verifier.REQUIRED_PROBE
    v2 = verifier.HEADWARMUP_200_REQUIRED_PROBE
    differing = {name for name in v1 if v1[name] != v2[name]}
    assert differing == {
        "max_steps",
        "freeze_pretrained_encoder_steps",
        "freeze_leaf_pid_head_steps",
    }
    assert {name: v2[name] for name in differing} == {
        "max_steps": 200,
        "freeze_pretrained_encoder_steps": 200,
        "freeze_leaf_pid_head_steps": 200,
    }
    assert verifier.ALLOWED_STEPS_BY_CONTRACT_VERSION[
        verifier.CONTRACT_VERSION
    ] == {2188, 3282, 4376}
    assert verifier.ALLOWED_STEPS_BY_CONTRACT_VERSION[
        verifier.HEADWARMUP_200_CONTRACT_VERSION
    ] == {3282}


def test_v3_profile_changes_only_matched_binary_positive_balance():
    v1 = verifier.REQUIRED_PROBE
    v3 = verifier.QUERY_ACTIVATION_REQUIRED_PROBE
    assert {name for name in v3 if name not in v1} == {
        "object_positive_weight",
        "pointer_positive_weight",
    }
    assert v3["object_positive_weight"] == 16.0
    assert v3["pointer_positive_weight"] == 16.0
    assert {name: v3[name] for name in v1} == v1
    assert verifier.ALLOWED_STEPS_BY_CONTRACT_VERSION[
        verifier.QUERY_ACTIVATION_CONTRACT_VERSION
    ] == {3282}


@pytest.mark.parametrize(
    ("version", "changed_key", "expected_value"),
    [
        (verifier.OBJECT8_POINTER16_CONTRACT_VERSION, "object_positive_weight", 8.0),
        (verifier.OBJECT16_POINTER8_CONTRACT_VERSION, "pointer_positive_weight", 8.0),
        (verifier.OBJECT12_POINTER16_CONTRACT_VERSION, "object_positive_weight", 12.0),
    ],
)
def test_v4_profiles_change_exactly_one_factor_from_successful_v3(
    version,
    changed_key,
    expected_value,
):
    v3 = verifier.QUERY_ACTIVATION_REQUIRED_PROBE
    v4 = verifier.REQUIRED_PROBES_BY_CONTRACT_VERSION[version]
    assert {name for name in v3 if v3[name] != v4[name]} == {changed_key}
    assert v4[changed_key] == expected_value
    assert verifier.ALLOWED_STEPS_BY_CONTRACT_VERSION[version] == {3282}
    assert set(verifier.CALIBRATION_ARMS_BY_CONTRACT_VERSION) == {
        verifier.OBJECT8_POINTER16_CONTRACT_VERSION,
        verifier.OBJECT16_POINTER8_CONTRACT_VERSION,
        verifier.OBJECT12_POINTER16_CONTRACT_VERSION,
    }


def _calibration_preregistration() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": (
            verifier.POSITIVE_WEIGHT_CALIBRATION_PREREGISTRATION_VERSION
        ),
        "source_checkpoint": {
            "step": 3282,
            "sha256": verifier.POSITIVE_WEIGHT_CALIBRATION_CHECKPOINT_SHA256,
        },
        "immutable_baseline": {
            "job_id": "15774286",
            "contract_sha256": (
                verifier.POSITIVE_WEIGHT_CALIBRATION_BASELINE_CONTRACT_SHA256
            ),
        },
        "arms": [
            {
                "arm_id": "object8_pointer16",
                "contract_version": verifier.OBJECT8_POINTER16_CONTRACT_VERSION,
                "submission_order": 1,
                "object_positive_weight": 8.0,
                "pointer_positive_weight": 16.0,
            },
            {
                "arm_id": "object16_pointer8",
                "contract_version": verifier.OBJECT16_POINTER8_CONTRACT_VERSION,
                "submission_order": 2,
                "object_positive_weight": 16.0,
                "pointer_positive_weight": 8.0,
            },
        ],
        "submission_order": ["object8_pointer16", "object16_pointer8"],
        "sealed_test_role_access": "forbidden",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["preregistration_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


@pytest.mark.parametrize(
    "version",
    [
        verifier.OBJECT8_POINTER16_CONTRACT_VERSION,
        verifier.OBJECT16_POINTER8_CONTRACT_VERSION,
    ],
)
def test_v4_preregistration_binds_both_arms_and_submission_order(tmp_path, version):
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(_calibration_preregistration()))
    verified = verifier.verify_calibration_preregistration(
        path,
        contract_version=version,
    )
    expected = verifier.CALIBRATION_ARMS_BY_CONTRACT_VERSION[version]
    assert verified["arm_id"] == expected["arm_id"]
    assert verified["submission_order"] == expected["submission_order"]


def test_v5_midpoint_preregistration_binds_single_arm(tmp_path):
    payload = _calibration_preregistration()
    payload["arms"] = [
        {
            "arm_id": "object12_pointer16",
            "contract_version": verifier.OBJECT12_POINTER16_CONTRACT_VERSION,
            "submission_order": 1,
            "object_positive_weight": 12.0,
            "pointer_positive_weight": 16.0,
        },
    ]
    payload["submission_order"] = ["object12_pointer16"]
    payload.pop("preregistration_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["preregistration_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    path = tmp_path / "preregistration_midpoint.json"
    path.write_text(json.dumps(payload))
    verified = verifier.verify_calibration_preregistration(
        path,
        contract_version=verifier.OBJECT12_POINTER16_CONTRACT_VERSION,
    )
    expected = verifier.CALIBRATION_ARMS_BY_CONTRACT_VERSION[
        verifier.OBJECT12_POINTER16_CONTRACT_VERSION
    ]
    assert verified["arm_id"] == expected["arm_id"]
    assert verified["submission_order"] == expected["submission_order"]


def test_v4_preregistration_fails_closed_on_tampering(tmp_path):
    payload = _calibration_preregistration()
    payload["submission_order"] = ["object16_pointer8", "object8_pointer16"]
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verifier.verify_calibration_preregistration(
            path,
            contract_version=verifier.OBJECT8_POINTER16_CONTRACT_VERSION,
        )


@pytest.mark.parametrize(
    ("version", "checkpoint_step", "optimizer_steps"),
    [
        (verifier.CONTRACT_VERSION, 2188, "100"),
        (verifier.HEADWARMUP_200_CONTRACT_VERSION, 3282, "200"),
        (verifier.QUERY_ACTIVATION_CONTRACT_VERSION, 3282, "100"),
    ],
)
def test_verifier_accepts_exact_v1_and_v2_profiles(
    monkeypatch,
    tmp_path,
    version,
    checkpoint_step,
    optimizer_steps,
):
    path = _write_verifiable_contract(
        tmp_path,
        version=version,
        checkpoint_step=checkpoint_step,
    )
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "_git_output", _fake_git)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    contract, runtime = verifier.verify_contract(path)
    assert contract["contract_version"] == version
    assert runtime["checkpoint_step"] == str(checkpoint_step)
    assert runtime["optimizer_steps"] == optimizer_steps


def test_v2_verifier_rejects_non_step3282_and_horizon_drift():
    wrong_step = _contract(
        version=verifier.HEADWARMUP_200_CONTRACT_VERSION,
        checkpoint_step=4376,
    )
    with pytest.raises(RuntimeError, match="not approved"):
        verifier.required_probe_for_contract(wrong_step)
    wrong_horizon = dict(
        _contract(
            version=verifier.HEADWARMUP_200_CONTRACT_VERSION,
            checkpoint_step=3282,
        )
    )
    wrong_horizon["probe"] = {
        **wrong_horizon["probe"],
        "freeze_leaf_pid_head_steps": 199,
    }
    with pytest.raises(RuntimeError, match="registered contract profile"):
        verifier.required_probe_for_contract(wrong_horizon)

    wrong_activation_step = _contract(
        version=verifier.QUERY_ACTIVATION_CONTRACT_VERSION,
        checkpoint_step=2188,
    )
    with pytest.raises(RuntimeError, match="not approved"):
        verifier.required_probe_for_contract(wrong_activation_step)
    wrong_activation_weight = dict(
        _contract(
            version=verifier.QUERY_ACTIVATION_CONTRACT_VERSION,
            checkpoint_step=3282,
        )
    )
    wrong_activation_weight["probe"] = {
        **wrong_activation_weight["probe"],
        "object_positive_weight": 15.0,
    }
    with pytest.raises(RuntimeError, match="registered contract profile"):
        verifier.required_probe_for_contract(wrong_activation_weight)


@pytest.mark.parametrize(
    ("version", "checkpoint_step", "optimizer_steps"),
    [
        (verifier.CONTRACT_VERSION, 2188, 100),
        (verifier.HEADWARMUP_200_CONTRACT_VERSION, 3282, 200),
        (verifier.QUERY_ACTIVATION_CONTRACT_VERSION, 3282, 100),
        (verifier.OBJECT8_POINTER16_CONTRACT_VERSION, 3282, 100),
        (verifier.OBJECT16_POINTER8_CONTRACT_VERSION, 3282, 100),
    ],
)
def test_finalizer_validates_dynamic_versioned_optimizer_steps(
    version,
    checkpoint_step,
    optimizer_steps,
):
    contract = _contract(version=version, checkpoint_step=checkpoint_step)
    result = {
        "status": "completed",
        "optimizer_steps": optimizer_steps,
        "checkpoint_step": checkpoint_step,
        "probe": dict(verifier.REQUIRED_PROBES_BY_CONTRACT_VERSION[version]),
        "contract_sha256": contract["contract_sha256"],
        "source_checkpoint": {"unchanged": True},
        "output_checkpoint": {"all_model_tensors_finite": True},
        "data": {"split_counts": {"test": 0}},
    }
    assert result_satisfies_contract(result, contract, exit_status=0)
    result["optimizer_steps"] = 200 if optimizer_steps == 100 else 100
    assert not result_satisfies_contract(result, contract, exit_status=0)


def test_finalizer_fails_closed_on_profile_or_contract_mismatch():
    contract = _contract(
        version=verifier.HEADWARMUP_200_CONTRACT_VERSION,
        checkpoint_step=3282,
    )
    result = {
        "status": "completed",
        "optimizer_steps": 200,
        "checkpoint_step": 3282,
        "probe": dict(verifier.HEADWARMUP_200_REQUIRED_PROBE),
        "contract_sha256": contract["contract_sha256"],
        "source_checkpoint": {"unchanged": True},
        "output_checkpoint": {"all_model_tensors_finite": True},
        "data": {"split_counts": {"test": 0}},
    }
    wrong_probe = dict(result, probe={**result["probe"], "max_steps": 100})
    assert not result_satisfies_contract(wrong_probe, contract, exit_status=0)
    wrong_checkpoint = dict(result, checkpoint_step=2188)
    assert not result_satisfies_contract(wrong_checkpoint, contract, exit_status=0)
    tampered_contract = dict(contract, optimizer_steps=100)
    with pytest.raises(RuntimeError, match="contract hash mismatch"):
        result_satisfies_contract(result, tampered_contract, exit_status=0)
