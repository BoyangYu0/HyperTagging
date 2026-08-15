#!/usr/bin/env python3
"""Run one source-bound frozen-encoder reconstruction transfer probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_reconstruction_transfer_probe_contract import (  # noqa: E402
    verify_contract,
)
from hypertagging.training.reconstruction_trainer import (  # noqa: E402
    ReconstructionConfig,
    train_level_reconstruction,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_model_tensor_receipt(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("probe output has no non-empty model_state_dict")
    tensor_count = 0
    element_count = 0
    nonfinite: list[str] = []
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        element_count += int(value.numel())
        if not torch.isfinite(value).all():
            nonfinite.append(str(name))
    if tensor_count == 0 or nonfinite:
        raise RuntimeError(
            "probe output model tensor integrity failed: "
            f"tensor_count={tensor_count}, nonfinite={nonfinite[:8]}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "all_model_tensors_finite": True,
    }


def _json_number(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("probe produced a non-finite reported metric")
        return value
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    output = args.output.resolve()
    training_output = args.training_output_dir.resolve()
    expected_root = Path(runtime["study_output_base"]).resolve()
    if expected_root not in output.parents or expected_root not in training_output.parents:
        raise RuntimeError("probe outputs must remain under the contract study root")
    if output.exists() or training_output.exists():
        raise RuntimeError("refusing to reuse a transfer-probe output")

    checkpoint = Path(runtime["checkpoint"])
    checkpoint_before = _sha256(checkpoint)
    probe = dict(contract["probe"])
    started = time.perf_counter()
    result = train_level_reconstruction(
        ReconstructionConfig(
            data=runtime["selection_manifest"],
            output_dir=str(training_output),
            pretrained_encoder=str(checkpoint),
            device="cuda",
            max_steps=int(probe["max_steps"]),
            lr_schedule_total_steps=int(probe["max_steps"]),
            batch_size=int(probe["batch_size"]),
            seed=int(probe["seed"]),
            checkpoint_every=int(probe["max_steps"]),
            validate_every=int(probe["validate_every"]),
            rollout_validate_every=int(probe["rollout_validate_every"]),
            freeze_pretrained_encoder_steps=int(
                probe["freeze_pretrained_encoder_steps"]
            ),
            transfer_leaf_pid_head=bool(probe["transfer_leaf_pid_head"]),
            freeze_leaf_pid_head_steps=int(probe["freeze_leaf_pid_head_steps"]),
            target_policy=str(probe["target_policy"]),
            dataset_index=runtime["dataset_index"],
            max_validation_events=int(probe["max_validation_events"]),
            rollout_validation_events=int(probe["rollout_validation_events"]),
            validation_batch_size=int(probe["validation_batch_size"]),
            model_preset=str(probe["model_preset"]),
            best_metric=str(probe["best_metric"]),
            best_mode=str(probe["best_mode"]),
            mixed_precision=True,
            scientific_mode=True,
            allow_tiny_bruteforce_matching=False,
            log_every=10,
            num_workers=0,
            pilot_allow_train_validation_fallback=False,
            minimum_encoder_transfer_coverage=0.9,
            allow_low_encoder_transfer_coverage=False,
        )
    )
    elapsed = time.perf_counter() - started
    if result.steps != int(probe["max_steps"]):
        raise RuntimeError(
            f"probe stopped at {result.steps}, expected {probe['max_steps']} steps"
        )
    if result.transfer_report is None:
        raise RuntimeError("probe did not produce a pretrained transfer report")
    if result.transfer_report.coverage < 0.9:
        raise RuntimeError("probe encoder transfer coverage fell below 90%")
    split_counts = dict(result.data_module.split_counts)
    if int(split_counts.get("test", 0)) != 0:
        raise RuntimeError("sealed-test rows appeared in the transfer probe")
    if int(split_counts.get("validation", 0)) < 2000:
        raise RuntimeError("transfer probe did not retain the fixed validation cohort")
    checkpoint_after = _sha256(checkpoint)
    if checkpoint_after != checkpoint_before or checkpoint_after != runtime["checkpoint_sha256"]:
        raise RuntimeError("source pretraining checkpoint changed during transfer probe")

    transfer = asdict(result.transfer_report)
    transfer["coverage"] = result.transfer_report.coverage
    receipt = {
        "status": "completed",
        "study": "frozen_pretrained_reconstruction_transfer_probe",
        "optimizer_steps": result.steps,
        "checkpoint_step": int(runtime["checkpoint_step"]),
        "source_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256_before": checkpoint_before,
            "sha256_after": checkpoint_after,
            "unchanged": True,
        },
        "output_checkpoint": _finite_model_tensor_receipt(result.checkpoint),
        "training_log": {
            "path": str(result.log_path.resolve()),
            "bytes": result.log_path.stat().st_size,
            "sha256": _sha256(result.log_path),
        },
        "probe": probe,
        "elapsed_seconds": elapsed,
        "final_loss": result.final_loss,
        "metrics": dict(result.metrics),
        "transfer_report": transfer,
        "data": {
            "split_counts": split_counts,
            "selection_manifest_hash": result.data_module.selection_manifest_hash,
            "split_manifest_hash": result.data_module.split_manifest_hash,
            "dataset_index_hash": (
                result.data_module.dataset_index.get("index_hash", "")
                if result.data_module.dataset_index
                else ""
            ),
            "sealed_test_role_access": "forbidden",
        },
        "contract_sha256": contract["contract_sha256"],
    }
    _atomic_json(output, _json_number(receipt))
    print(json.dumps({"status": "completed", "result": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
