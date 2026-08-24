#!/usr/bin/env python3
"""Verify the immutable, one-GPU reconstruction transfer contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "hypertagging-reconstruction-fullscale-v1"
ALLOWED_GRES = {"gpu:h100nvl:1", "gpu:v100:1"}
SOURCE_CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
)
SELECTION_MANIFEST = "configs/training_selection/production_1m_20260812/train_035k.json"
DATASET_INDEX = (
    "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
    "train_035k.complete_only.index.json"
)
PREREGISTRATION = (
    "configs/reconstruction/ht_reconstruction_transfer_preregistration_20260824.json"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str, *, suffix: str | None = None) -> Path:
    candidate = (Path(value) if Path(value).is_absolute() else ROOT / value).resolve(
        strict=True
    )
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"contract path escapes repository: {value}") from error
    if suffix is not None and candidate.suffix != suffix:
        raise RuntimeError(f"contract path must end in {suffix}: {value}")
    return candidate


def verify_contract_hash(contract: dict[str, Any]) -> str:
    stored = str(contract.get("contract_sha256", ""))
    canonical = {key: value for key, value in contract.items() if key != "contract_sha256"}
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("reconstruction full-scale contract hash mismatch")
    return actual


def verify_hashed_inputs(items: list[dict[str, str]]) -> None:
    if not items:
        raise RuntimeError("reconstruction contract has no hashed inputs")
    for item in items:
        path = _repo_path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not HEX64.fullmatch(expected) or sha256(path) != expected:
            raise RuntimeError(f"hashed reconstruction input changed: {path}")


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, timeout=15
    )
    return result.stdout.strip()


def _require_common_fields(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("unsupported reconstruction full-scale contract")
    if contract.get("training_role") != "train":
        raise RuntimeError("reconstruction training role must be train")
    if contract.get("evaluation_role") != "validation":
        raise RuntimeError("reconstruction evaluation role must be validation")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("sealed-test access is not forbidden")
    if contract.get("source_checkpoint_mutation") != "forbidden":
        raise RuntimeError("source checkpoint mutation policy changed")
    provenance = contract.get("provenance_validation")
    if not isinstance(provenance, dict):
        raise RuntimeError("exploratory-transfer provenance is missing")
    if provenance.get("pretraining_success_gate_passed") is not False:
        raise RuntimeError("contract must explicitly record failed/incomplete pretraining")
    if provenance.get("transfer_classification") != "exploratory_reconstruction_transfer":
        raise RuntimeError("contract is not classified as exploratory reconstruction transfer")
    preregistration = contract.get("preregistration")
    if not isinstance(preregistration, dict):
        raise RuntimeError("versioned reconstruction preregistration is missing")
    if preregistration.get("path") != PREREGISTRATION:
        raise RuntimeError("reconstruction preregistration path changed")
    preregistration_path = _repo_path(PREREGISTRATION, suffix=".json")
    if preregistration.get("sha256") != sha256(preregistration_path):
        raise RuntimeError("reconstruction preregistration hash changed")
    if contract.get("gres") not in ALLOWED_GRES:
        raise RuntimeError("reconstruction contract has unsupported exact GRES")
    config = contract.get("config")
    resources = contract.get("resources")
    if not isinstance(config, dict) or not isinstance(resources, dict):
        raise RuntimeError("reconstruction config/resources are missing")
    if config.get("model_preset") != "small_candidate":
        raise RuntimeError("reconstruction transfer must use the compatible small_candidate preset")
    if config.get("max_cardinality") != 16:
        raise RuntimeError("reconstruction max_cardinality must remain 16")
    if config.get("object_positive_weight") != 12.0 or config.get("pointer_positive_weight") != 16.0:
        raise RuntimeError("reconstruction positive weights must remain 12/16")
    if config.get("target_policy") != "complete_only":
        raise RuntimeError("reconstruction target policy changed")
    if config.get("freeze_pretrained_encoder_steps") != config.get("max_steps"):
        raise RuntimeError("encoder is not frozen for the full optimizer horizon")
    if config.get("transfer_leaf_pid_head") is not True:
        raise RuntimeError("transferred leaf PID head is required")
    if config.get("freeze_leaf_pid_head_steps") != config.get("max_steps"):
        raise RuntimeError("transferred leaf PID head is not frozen for the full horizon")
    if config.get("batch_size", 0) <= 0 or config.get("max_steps", 0) <= 0:
        raise RuntimeError("batch_size and max_steps must be positive")
    if config.get("presentations_target") != config["batch_size"] * config["max_steps"]:
        raise RuntimeError("global-batch/presentation accounting is inconsistent")
    if config.get("lr_schedule_total_steps") != config.get("max_steps"):
        raise RuntimeError("LR schedule horizon must equal optimizer horizon")
    if config.get("validation_batch_size", 0) <= 0:
        raise RuntimeError("validation batch size must be positive")
    if config.get("mixed_precision") is not True:
        raise RuntimeError("production reconstruction must use explicit CUDA AMP")
    expected_dtype = "bfloat16" if contract["gres"] == "gpu:h100nvl:1" else "float16"
    if config.get("amp_dtype") != expected_dtype:
        raise RuntimeError(f"{contract['gres']} requires amp_dtype={expected_dtype}")
    if config.get("grad_scaler_enabled") is not (contract["gres"] == "gpu:v100:1"):
        raise RuntimeError("precision/scaler policy does not match exact GRES")
    return config, resources


def verify_contract(
    path: Path, *, require_slurm: bool = True, require_authorized: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract_hash = verify_contract_hash(contract)
    verify_hashed_inputs(list(contract.get("hashed_inputs", [])))
    config, resources = _require_common_fields(contract)
    mode = contract.get("mode")
    if mode not in {"calibration", "production"}:
        raise RuntimeError("mode must be calibration or production")
    if mode == "calibration":
        if config.get("max_steps", 0) > 256 or config.get("validation_enabled") is not False:
            raise RuntimeError("calibration is not bounded train-role-only execution")
        if int(contract.get("max_wall_seconds", 0)) > 900:
            raise RuntimeError("calibration exceeds 900-second bound")
        if int(resources.get("max_restarts", -1)) != 0:
            raise RuntimeError("calibration may not requeue")
    else:
        if config.get("validation_enabled") is not True:
            raise RuntimeError("production validation is disabled")
        if config.get("max_validation_events") != 2000 or config.get("rollout_validation_events") != 1000:
            raise RuntimeError("production validation cohorts changed")
        if int(resources.get("max_restarts", -1)) not in {0, 1}:
            raise RuntimeError("production requeue bound must be zero or one")
        if int(config.get("presentations_target", 0)) != 70016:
            raise RuntimeError("production presentation target must remain 70016")
    checkpoint = _repo_path(str(contract.get("checkpoint", "")), suffix=".pt")
    if str(checkpoint.relative_to(ROOT)) != SOURCE_CHECKPOINT:
        raise RuntimeError("source checkpoint path is not the authorized step-54064 checkpoint")
    if contract.get("checkpoint_step") != 54064 or contract.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint step/hash binding changed")
    if sha256(checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA256 does not match the authorized hash")
    selection = _repo_path(str(contract.get("selection_manifest", "")), suffix=".json")
    index = _repo_path(str(contract.get("dataset_index", "")), suffix=".json")
    if str(selection.relative_to(ROOT)) != SELECTION_MANIFEST or str(index.relative_to(ROOT)) != DATASET_INDEX:
        raise RuntimeError("selection/index path binding changed")
    output_root = Path(str(contract.get("output_root", "")))
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    expected_parent = (ROOT / "artifacts" / "runs" / "ht-reconstruction-transfer-fullscale-20260824").resolve()
    if expected_parent not in output_root.parents and output_root != expected_parent:
        expected_parent = (ROOT / "artifacts" / "runs" / "ht-reconstruction-calibration-20260824").resolve()
        if expected_parent not in output_root.parents:
            raise RuntimeError("reconstruction output root is outside the authorized study roots")
    gpu_environment = Path(str(contract.get("gpu_environment", "")))
    if not gpu_environment.is_absolute() or not (gpu_environment / "bin/python").is_file():
        raise RuntimeError("GPU environment is unavailable")
    expected_sha = str(contract.get("expected_git_sha", ""))
    expected_tag = str(contract.get("expected_git_tag", ""))
    if not HEX40.fullmatch(expected_sha) or _git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("reconstruction source Git SHA mismatch")
    if not expected_tag or _git("rev-list", "-n", "1", expected_tag) != expected_sha:
        raise RuntimeError("reconstruction implementation tag mismatch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean for reconstruction execution")
    if require_authorized and contract.get("submission_authorized") is not True:
        raise RuntimeError("reconstruction contract is not authorized")
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("reconstruction execution must run inside Slurm")
    runtime = {
        "contract_sha256": contract_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "checkpoint_step": "54064",
        "selection_manifest": str(selection),
        "dataset_index": str(index),
        "gpu_environment": str(gpu_environment),
        "output_root": str(output_root),
        "expected_gres": str(contract["gres"]),
        "expected_git_sha": expected_sha,
        "mode": str(mode),
        "max_steps": str(config["max_steps"]),
        "max_restarts": str(resources["max_restarts"]),
    }
    return contract, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--shell-output", type=Path, required=True)
    args = parser.parse_args()
    _, runtime = verify_contract(args.contract.resolve(strict=True))
    args.shell_output.write_text(
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
