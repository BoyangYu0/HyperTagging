"""Checkpoint helpers for new level-autoregressive training."""

from __future__ import annotations

from pathlib import Path
import os
import random
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import (
    RUNTIME_MODEL_CONTRACTS_V4,
    SCHEMA_VERSION_V4,
    feature_spec_v4,
)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    normalizer_state: dict[str, Any] | None = None,
    schema_version: str = SCHEMA_VERSION_V4,
    encoder: torch.nn.Module | None = None,
    scaler: Any | None = None,
    split_manifest_hash: str = "",
    confidence_head_trained: bool = False,
    schedule_state: dict[str, Any] | None = None,
    legacy_conflated_fraction: float = 0.0,
    streaming_cursor: dict[str, Any] | None = None,
    feature_contract: dict[str, Any] | None = None,
    data_order_contract: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": (
            encoder.state_dict()
            if encoder is not None
            else (
                getattr(model, "encoder").state_dict()
                if hasattr(model, "encoder")
                else {}
            )
        ),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": config or {},
        "metrics": metrics or {},
        "normalizer_state": normalizer_state or {},
        "preprocessing_schema_version": schema_version,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_specification": feature_spec_v4(),
        "runtime_model_contracts": dict(RUNTIME_MODEL_CONTRACTS_V4),
        "split_manifest_hash": split_manifest_hash,
        "git_commit": _git_commit(),
        "confidence_head_trained": bool(confidence_head_trained),
        "schedule_state": schedule_state or {},
        "legacy_conflated_fraction": float(legacy_conflated_fraction),
        "streaming_cursor": streaming_cursor or {},
        "feature_contract": feature_contract or {},
        "data_order_contract": data_order_contract or {},
        "architecture": architecture or {},
        "data_compatible_performance": not bool(legacy_conflated_fraction),
        "random_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Checkpoints contain optimizer and RNG metadata in addition to tensors;
    # callers must only load trusted experiment artifacts.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_random_states: bool = True,
    expected_schema_version: str | None = None,
    expected_feature_spec_hash: str | None = None,
    expected_split_manifest_hash: str | None = None,
    allow_contract_mismatch: bool = False,
    expected_data_order_contract: dict[str, Any] | None = None,
    expected_architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore model/training state and return the complete checkpoint metadata."""

    payload = load_training_checkpoint(path, map_location=map_location)
    mismatches = []
    if (
        expected_schema_version is not None
        and payload.get("preprocessing_schema_version") != expected_schema_version
    ):
        mismatches.append("schema version")
    stored_hash = payload.get("feature_specification", {}).get("feature_spec_hash")
    if expected_feature_spec_hash is not None and stored_hash != expected_feature_spec_hash:
        mismatches.append("feature specification")
    if (
        expected_split_manifest_hash is not None
        and payload.get("split_manifest_hash") != expected_split_manifest_hash
    ):
        mismatches.append("split manifest")
    if payload.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
        mismatches.append("PID vocabulary")
    for label, expected, stored in (
        ("data-order", expected_data_order_contract, payload.get("data_order_contract", {})),
        ("architecture", expected_architecture, payload.get("architecture", {})),
    ):
        if expected is not None:
            changed = [key for key, value in expected.items() if stored.get(key) != value]
            if changed:
                mismatches.append(f"{label} ({', '.join(sorted(changed))})")
    if mismatches and not allow_contract_mismatch:
        raise ValueError(f"checkpoint contract mismatch: {', '.join(mismatches)}")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_random_states and payload.get("random_states"):
        states = payload["random_states"]
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states.get("cuda"):
            torch.cuda.set_rng_state_all(states["cuda"])
    return payload


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"
