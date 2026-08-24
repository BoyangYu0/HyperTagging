#!/usr/bin/env python3
"""Execute one contract-bound reconstruction calibration or production attempt."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
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

from scripts.slurm.verify_reconstruction_fullscale_contract import (  # noqa: E402
    verify_contract,
)
from hypertagging.training.reconstruction_trainer import (  # noqa: E402
    ReconstructionConfig,
    train_level_reconstruction,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_payload(payload: Any, path: str = "payload") -> list[str]:
    offenders: list[str] = []

    def visit(value: Any, name: str) -> None:
        if torch.is_tensor(value):
            if not torch.isfinite(value).all().item():
                offenders.append(name)
            return
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), f"{name}.{field.name}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]")
            return
        if isinstance(value, float) and not math.isfinite(value):
            offenders.append(name)

    visit(payload, path)
    return offenders


def finite_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    offenders = finite_payload(payload)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError("output checkpoint has no model_state_dict")
    model_tensors = [value for value in model_state.values() if torch.is_tensor(value)]
    if not model_tensors:
        raise RuntimeError("output checkpoint has no model tensors")
    if offenders:
        raise RuntimeError(f"output checkpoint contains non-finite state: {offenders[:8]}")

    def count_tensors(value: Any) -> int:
        if torch.is_tensor(value):
            return 1
        if isinstance(value, dict):
            return sum(count_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count_tensors(item) for item in value)
        return 0

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tensor_count_recursive": count_tensors(payload),
        "model_tensor_count": len(model_tensors),
        "all_checkpoint_tensors_finite": True,
        "all_model_tensors_finite": True,
    }


def json_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            json_finite(item, f"{path}[{index}]")


def first_twenty_gate(metrics_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        json_finite(row)
        if row.get("split") != "validation" and isinstance(row.get("step"), int):
            rows.append(row)
    rows = [row for row in rows if 1 <= int(row["step"]) <= 20]
    by_step = {int(row["step"]): row for row in rows}
    missing = sorted(set(range(1, 21)) - set(by_step))
    required = (
        "forward_finite",
        "loss_finite",
        "raw_gradient_finite",
        "model_finite",
        "optimizer_finite",
        "scheduler_finite",
    )
    failures = [
        f"step-{step}:{field}"
        for step, row in sorted(by_step.items())
        for field in required
        if row.get(field) != 1.0
    ]
    if missing or failures:
        raise RuntimeError(
            f"first-20 optimizer gate failed: missing={missing} failures={failures[:8]}"
        )
    return {
        "passed": True,
        "steps": 20,
        "required_fields": list(required),
        "min_step": 1,
        "max_step": 20,
        "rows": len(rows),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    output = args.output.resolve()
    training_output = args.training_output_dir.resolve()
    output_root = Path(runtime["output_root"]).resolve()
    if output_root not in output.parents or output_root not in training_output.parents:
        raise RuntimeError("reconstruction outputs escape the contract output root")
    if output.exists():
        raise RuntimeError("refusing to overwrite reconstruction result")
    if training_output.exists() and args.resume is None:
        raise RuntimeError("refusing to reuse a reconstruction training output")
    checkpoint = Path(runtime["checkpoint"]).resolve(strict=True)
    checkpoint_before = sha256(checkpoint)
    config = dict(contract["config"])
    started = time.perf_counter()
    result = train_level_reconstruction(
        ReconstructionConfig(
            data=runtime["selection_manifest"],
            output_dir=str(training_output),
            pretrained_encoder=str(checkpoint),
            device="cuda",
            max_steps=int(config["max_steps"]),
            lr_schedule_total_steps=int(config["lr_schedule_total_steps"]),
            learning_rate=float(config["learning_rate"]),
            warmup_fraction=float(config["warmup_fraction"]),
            warmup_steps=(
                None if config.get("warmup_steps") is None else int(config["warmup_steps"])
            ),
            max_warmup_steps=int(config["max_warmup_steps"]),
            min_lr_ratio=float(config["min_lr_ratio"]),
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
            checkpoint_every=int(config["checkpoint_every"]),
            validate_every=int(config["validate_every"]),
            rollout_validate_every=int(config["rollout_validate_every"]),
            resume=None if args.resume is None else str(args.resume.resolve(strict=True)),
            freeze_pretrained_encoder_steps=int(config["freeze_pretrained_encoder_steps"]),
            transfer_leaf_pid_head=bool(config["transfer_leaf_pid_head"]),
            freeze_leaf_pid_head_steps=int(config["freeze_leaf_pid_head_steps"]),
            target_policy=str(config["target_policy"]),
            dataset_index=runtime["dataset_index"],
            max_validation_events=int(config["max_validation_events"]),
            rollout_validation_events=int(config["rollout_validation_events"]),
            validation_batch_size=int(config["validation_batch_size"]),
            model_preset=str(config["model_preset"]),
            max_cardinality=int(config["max_cardinality"]),
            max_cardinality_by_level=tuple(
                (int(level), int(value))
                for level, value in config["max_cardinality_by_level"]
            ),
            object_positive_weight=float(config["object_positive_weight"]),
            pointer_positive_weight=float(config["pointer_positive_weight"]),
            best_metric=str(config["best_metric"]),
            best_mode=str(config["best_mode"]),
            initial_state_policy=str(config["initial_state_policy"]),
            scheduled_sampling_probability=float(config["scheduled_sampling_probability"]),
            scheduled_sampling_schedule=str(config["scheduled_sampling_schedule"]),
            scheduled_sampling_duration_steps=int(config["scheduled_sampling_duration_steps"]),
            mixed_precision=bool(config["mixed_precision"]),
            amp_dtype=str(config["amp_dtype"]),
            grad_scaler_enabled=bool(config["grad_scaler_enabled"]),
            validation_enabled=bool(config["validation_enabled"]),
            scientific_mode=bool(config["scientific_mode"]),
            allow_tiny_bruteforce_matching=False,
            log_every=int(config["log_every"]),
            num_workers=0,
            pilot_allow_train_validation_fallback=False,
            minimum_encoder_transfer_coverage=0.9,
            allow_low_encoder_transfer_coverage=False,
            rollout_pid_kinematics_mode=str(config["rollout_pid_kinematics_mode"]),
            rollout_pid_temperature=float(config["rollout_pid_temperature"]),
        )
    )
    elapsed = time.perf_counter() - started
    if result.steps != int(config["max_steps"]):
        raise RuntimeError(f"reconstruction stopped at {result.steps}, expected {config['max_steps']}")
    if result.transfer_report is None or result.transfer_report.coverage < 0.9:
        raise RuntimeError("reconstruction encoder transfer coverage is below 90%")
    split_counts = dict(result.data_module.split_counts)
    if int(split_counts.get("test", 0)) != 0:
        raise RuntimeError("sealed-test rows appeared in reconstruction execution")
    checkpoint_after = sha256(checkpoint)
    if checkpoint_after != checkpoint_before or checkpoint_after != runtime["checkpoint_sha256"]:
        raise RuntimeError("source pretraining checkpoint changed during reconstruction")
    metrics_path = result.log_path.resolve(strict=True)
    first20 = first_twenty_gate(metrics_path)
    output_checkpoint = finite_checkpoint(result.checkpoint.resolve(strict=True))
    result_payload: dict[str, Any] = {
        "status": "completed",
        "mode": runtime["mode"],
        "experiment": contract["experiment"],
        "contract_sha256": runtime["contract_sha256"],
        "optimizer_steps": result.steps,
        "presentations": result.steps * int(config["batch_size"]),
        "elapsed_seconds": elapsed,
        "source_checkpoint": {
            "path": str(checkpoint),
            "step": int(runtime["checkpoint_step"]),
            "sha256_before": checkpoint_before,
            "sha256_after": checkpoint_after,
            "unchanged": True,
        },
        "output_checkpoint": output_checkpoint,
        "training_log": {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256(metrics_path),
        },
        "first_20_optimizer_steps": first20,
        "config": config,
        "metrics": dict(result.metrics),
        "transfer_report": {
            **asdict(result.transfer_report),
            "coverage": result.transfer_report.coverage,
        },
        "data": {
            "split_counts": split_counts,
            "selection_manifest_hash": result.data_module.selection_manifest_hash,
            "split_manifest_hash": result.data_module.split_manifest_hash,
            "dataset_index_hash": result.data_module.dataset_index.get("index_hash", "")
            if result.data_module.dataset_index
            else "",
            "training_role": "train",
            "evaluation_role": "validation",
            "validation_accessed": bool(config["validation_enabled"]),
            "sealed_test_role_access": "forbidden",
        },
    }
    if runtime["mode"] == "calibration" and elapsed > float(contract["max_wall_seconds"]):
        raise RuntimeError("calibration exceeded its 900-second wall-clock bound")
    json_finite(result_payload)
    atomic_json(output, result_payload)
    print(json.dumps({"status": "completed", "result": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
