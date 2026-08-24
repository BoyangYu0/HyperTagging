from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm.render_reconstruction_fullscale_job import build_config
from scripts.slurm.verify_reconstruction_fullscale_contract import (
    PREREGISTRATION,
    _require_common_fields,
    verify_contract_hash,
)
from hypertagging.training.reconstruction_trainer import _require_finite_payload


def _contract(mode: str, gres: str = "gpu:h100nvl:1") -> dict[str, object]:
    config = build_config(mode, gres, 64)
    preregistration_path = ROOT / PREREGISTRATION
    preregistration_hash = hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
    return {
        "contract_version": "hypertagging-reconstruction-fullscale-v1",
        "mode": mode,
        "training_role": "train",
        "evaluation_role": "validation",
        "sealed_test_role_access": "forbidden",
        "source_checkpoint_mutation": "forbidden",
        "gres": gres,
        "provenance_validation": {
            "transfer_classification": "exploratory_reconstruction_transfer",
            "pretraining_success_gate_passed": False,
        },
        "preregistration": {"path": PREREGISTRATION, "sha256": preregistration_hash},
        "config": config,
        "resources": {"max_restarts": 0 if mode == "calibration" else 1},
    }


def test_reconstruction_profiles_bind_dtype_scaler_and_presentation_horizon():
    h100 = build_config("production", "gpu:h100nvl:1", 64)
    v100 = build_config("production", "gpu:v100:1", 32)
    assert h100["max_steps"] == 1094
    assert h100["presentations_target"] == 70016
    assert h100["amp_dtype"] == "bfloat16"
    assert h100["grad_scaler_enabled"] is False
    assert v100["max_steps"] == 2188
    assert v100["amp_dtype"] == "float16"
    assert v100["grad_scaler_enabled"] is True


def test_contract_common_fields_reject_scientific_drift():
    config, resources = _require_common_fields(_contract("production"))
    assert config["object_positive_weight"] == 12.0
    assert config["pointer_positive_weight"] == 16.0
    assert resources["max_restarts"] == 1
    wrong = _contract("production")
    wrong["config"] = {**config, "freeze_pretrained_encoder_steps": 1}
    with pytest.raises(RuntimeError, match="frozen"):
        _require_common_fields(wrong)


def test_calibration_profile_is_validation_disabled_and_bounded():
    config, resources = _require_common_fields(_contract("calibration", "gpu:v100:1"))
    assert config["validation_enabled"] is False
    assert config["max_steps"] <= 256
    assert resources["max_restarts"] == 0


def test_finite_runtime_gate_rejects_nonfinite_tensor_and_accepts_nested_state():
    _require_finite_payload("nested", {"x": torch.ones(2), "numbers": [1.0, 2]})
    with pytest.raises(RuntimeError, match="non-finite"):
        _require_finite_payload("nested", {"x": torch.tensor([float("nan")])})


def test_contract_hash_is_canonical():
    contract = _contract("calibration")
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = __import__("hashlib").sha256(canonical.encode()).hexdigest()
    assert verify_contract_hash(contract) == contract["contract_sha256"]
