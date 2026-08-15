"""Read-only fixed-cohort validation for trusted pretraining checkpoints."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import torch

from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.data_module import RealDataModule, build_real_data_module
from hypertagging.training.fixed_validation import (
    FIXED_VALIDATION_VERSION,
    select_validation_events,
)
from hypertagging.training.model_config import ModelArchitecture
from hypertagging.training.pretrain_trainer import (
    ContextualPretrainingModel,
    PretrainConfig,
    _validate_pretraining,
)


READ_ONLY_VALIDATION_VERSION = "pretraining-read-only-validation-v1"
FIXED_VALIDATION_EVENTS = 2_000
EVALUATION_ROLE = "validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _assert_finite_tensors(values: Iterable[tuple[str, torch.Tensor]]) -> None:
    nonfinite: list[str] = []
    for name, tensor in values:
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"checkpoint contains nonfinite tensors: {nonfinite[:5]}")


def _iter_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, item in value.items():
            child = f"{prefix}.{name}" if prefix else str(name)
            yield from _iter_tensors(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{index}]")


def _checkpoint_config(
    payload: dict[str, Any],
    *,
    data: Path,
    dataset_index: Path,
    device: torch.device,
) -> PretrainConfig:
    stored = dict(payload.get("config", {}))
    valid = {field.name for field in fields(PretrainConfig)}
    unknown = set(stored) - valid
    if unknown:
        raise ValueError(
            f"checkpoint has unknown pretraining config keys: {sorted(unknown)}"
        )
    config = PretrainConfig(**stored)
    if not config.scientific_mode:
        raise ValueError("read-only evaluation requires a scientific checkpoint")
    if payload.get("validation_selection", {}).get("split") != EVALUATION_ROLE:
        raise ValueError("checkpoint does not bind validation to the validation role")
    if (
        payload.get("validation_selection", {}).get("strategy")
        != "manifest_validation_role_uid_hash"
    ):
        raise ValueError("checkpoint does not use scientific UID-hash validation")
    return replace(
        config,
        data=str(data),
        dataset_index=str(dataset_index),
        output_dir="",
        device=str(device),
        resume=None,
        max_events=None,
        num_workers=0,
        persistent_workers=False,
        pilot_split_repair=False,
        rescan_dataset=False,
        scientific_mode=True,
        validation_events=FIXED_VALIDATION_EVENTS,
        validation_batches=math.ceil(FIXED_VALIDATION_EVENTS / config.batch_size),
    )


def _build_model(
    payload: dict[str, Any],
    config: PretrainConfig,
    data_module: RealDataModule,
    device: torch.device,
) -> ContextualPretrainingModel:
    architecture = ModelArchitecture.from_dict(payload.get("architecture", {}))
    if config.ablation not in ALL_ABLATIONS:
        raise ValueError(f"checkpoint has unknown ablation {config.ablation!r}")
    ablation = ALL_ABLATIONS[config.ablation]
    model = ContextualPretrainingModel(
        d_model=architecture.d_model,
        hyper_dim=architecture.hyper_dim,
        curvature=architecture.curvature,
        n_heads=architecture.n_heads,
        n_context_layers=architecture.n_context_layers,
        ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        use_contextual_encoder=ablation.contextual_euclidean,
        use_physical_relations=ablation.relation_attention,
        use_hyperbolic_relations=ablation.hyperbolic_relation_attention,
        channel_memory_size=config.channel_memory_size,
        channel_pooling=config.channel_pooling,
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
        max_tangent_norm=architecture.max_tangent_norm,
        hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=data_module.normalizers["common"].mean,
            common_std=data_module.normalizers["common"].std,
            composite_mean=data_module.normalizers["composite"].mean,
            composite_std=data_module.normalizers["composite"].std,
        ).to(device)
    )
    model.eval()
    return model


def _validate_checkpoint_contract(
    payload: dict[str, Any], data_module: RealDataModule
) -> None:
    feature_spec = feature_spec_v4()
    stored_spec = payload.get("feature_specification", {})
    if stored_spec.get("feature_spec_hash") != feature_spec["feature_spec_hash"]:
        raise ValueError("checkpoint feature specification mismatch")
    if payload.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
        raise ValueError("checkpoint PID vocabulary mismatch")
    if payload.get("split_manifest_hash") != data_module.split_manifest_hash:
        raise ValueError("evaluation split manifest differs from checkpoint training")
    checkpoint_index_hash = payload.get("data_order_contract", {}).get(
        "dataset_index_hash"
    )
    evaluation_index_hash = (data_module.dataset_index or {}).get("index_hash")
    if not checkpoint_index_hash or checkpoint_index_hash != evaluation_index_hash:
        raise ValueError("evaluation dataset index differs from checkpoint training")
    checkpoint_selection_hash = payload.get("validation_selection", {}).get(
        "selection_manifest_hash"
    )
    if (
        not checkpoint_selection_hash
        or checkpoint_selection_hash != data_module.selection_manifest_hash
    ):
        raise ValueError(
            "evaluation selection manifest differs from checkpoint training"
        )
    if data_module.split_counts.get(EVALUATION_ROLE, 0) < FIXED_VALIDATION_EVENTS:
        raise ValueError("validation role contains fewer than 2,000 events")
    if data_module.split_counts.get("test", 0) != 0:
        raise ValueError("read-only validation index must exclude the sealed test role")
    schemas = set(data_module.source_schema_versions)
    checkpoint_schema = str(payload.get("preprocessing_schema_version", ""))
    if checkpoint_schema != "mixed" and schemas != {checkpoint_schema}:
        raise ValueError("evaluation schema differs from checkpoint training")


def _uid_hash_records(uids: tuple[str, ...], *, seed: int) -> dict[str, Any]:
    records = []
    for uid in uids:
        uid_hash = hashlib.sha256(uid.encode("utf-8")).hexdigest()
        rank_hash = hashlib.sha256(
            f"{FIXED_VALIDATION_VERSION}:{seed}:{uid}".encode("utf-8")
        ).hexdigest()
        records.append(
            {"event_uid_sha256": uid_hash, "selection_rank_sha256": rank_hash}
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "event_count": len(records),
        "event_uid_hash_scheme": "sha256-utf8-v1",
        "selection_rank_hash_scheme": FIXED_VALIDATION_VERSION,
        "ordered_hash_records_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "selected_uid_hashes": records,
    }


def evaluate_pretraining_checkpoint(
    *,
    checkpoint: str | Path,
    data: str | Path,
    dataset_index: str | Path,
    output: str | Path,
    device: str | torch.device,
    expected_checkpoint_sha256: str,
    expected_source_git_sha: str,
    expected_checkpoint_step: int = 3282,
) -> dict[str, Any]:
    """Evaluate exactly 2,000 validation-role events without training mutation."""

    checkpoint_path = Path(checkpoint).resolve(strict=True)
    data_path = Path(data).resolve(strict=True)
    index_path = Path(dataset_index).resolve(strict=True)
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"evaluation output already exists: {output_path}")
    if (
        checkpoint_path.parent == output_path.parent
        or checkpoint_path.parent in output_path.parents
    ):
        raise ValueError("evaluation output may not be written inside the training run")
    if any(
        parent.name == "runs" and parent.parent.name == "artifacts"
        for parent in (output_path, *output_path.parents)
    ):
        raise ValueError("evaluation output may not be written under artifacts/runs")
    current_source_sha = _git_sha()
    if current_source_sha != expected_source_git_sha:
        raise ValueError("evaluation source Git SHA differs from the job contract")
    checkpoint_sha_before = _sha256(checkpoint_path)
    if checkpoint_sha_before != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA256 differs from the job contract")

    started_at = _utc_now()
    wall_started = time.monotonic()
    payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if int(payload.get("step", -1)) != expected_checkpoint_step:
        raise ValueError(
            "checkpoint step differs from the scientifically selected step"
        )
    reason = payload.get("training_state", {}).get("checkpoint_selection_reason", {})
    if (
        reason.get("metric_name") != "validation_full_training_objective"
        or reason.get("mode") != "min"
        or reason.get("reason") != "new_principal_configured_checkpoint"
    ):
        raise ValueError(
            "checkpoint is not the configured validation-objective minimum"
        )
    _assert_finite_tensors(_iter_tensors(payload))
    config = _checkpoint_config(
        payload, data=data_path, dataset_index=index_path, device=torch.device(device)
    )
    normalizers = payload.get("normalizer_state", {})
    missing = {"track", "cluster", "common", "composite"} - set(normalizers)
    if missing:
        raise ValueError(f"checkpoint is missing normalizers: {sorted(missing)}")
    data_module = build_real_data_module(
        data_path,
        seed=config.seed,
        allow_legacy_conflated=False,
        shuffle_buffer_size=config.shuffle_buffer_size,
        normalization_state=normalizers,
        required_splits=(EVALUATION_ROLE,),
        num_workers=0,
        persistent_workers=False,
        dataset_index=index_path,
        rescan_dataset=False,
        target_policy="complete_only",
        scientific_mode=True,
    )
    _validate_checkpoint_contract(payload, data_module)
    selected, selected_uids, selection_contract = select_validation_events(
        data_module.iter_events(EVALUATION_ROLE, shuffle=False),
        limit=FIXED_VALIDATION_EVENTS,
        scientific_mode=True,
        selection_manifest_hash=data_module.selection_manifest_hash,
        seed=config.seed,
    )
    if len(selected) != FIXED_VALIDATION_EVENTS:
        raise ValueError(
            "scientific UID-hash selection did not yield exactly 2,000 events"
        )
    if len(selected_uids) != len(set(selected_uids)):
        raise ValueError("scientific validation cohort contains duplicate event UIDs")
    del selected

    evaluation_device = torch.device(device)
    model = _build_model(payload, config, data_module, evaluation_device)
    model_state_before = _state_digest(model.state_dict())
    metrics = _validate_pretraining(
        model,
        data_module,
        device=evaluation_device,
        config=config,
        selected_event_uids=list(selected_uids),
    )
    model.eval()
    if int(metrics.get("validation_events", 0)) != FIXED_VALIDATION_EVENTS:
        raise ValueError("validation did not evaluate exactly 2,000 events")
    nonfinite_metrics = [
        name
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    if nonfinite_metrics:
        raise ValueError(f"validation produced nonfinite metrics: {nonfinite_metrics}")
    model_state_after = _state_digest(model.state_dict())
    checkpoint_sha_after = _sha256(checkpoint_path)
    if model_state_before != model_state_after:
        raise RuntimeError("read-only validation mutated model parameters or buffers")
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("read-only validation mutated the checkpoint file")

    result = {
        "result_version": READ_ONLY_VALIDATION_VERSION,
        "status": "completed",
        "evaluation_role": EVALUATION_ROLE,
        "sealed_test_role_access": "forbidden_and_not_opened",
        "optimizer_created": False,
        "optimizer_steps": 0,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "unchanged": True,
            "step": int(payload["step"]),
            "git_commit": payload.get("git_commit"),
            "selection_reason": reason,
        },
        "source": {
            "git_sha": current_source_sha,
            "data": str(data_path),
            "dataset_index": str(index_path),
            "dataset_index_hash": (data_module.dataset_index or {}).get("index_hash"),
            "split_manifest_hash": data_module.split_manifest_hash,
            "selection_manifest_hash": data_module.selection_manifest_hash,
        },
        "cohort": {
            "role": EVALUATION_ROLE,
            "requested_events": FIXED_VALIDATION_EVENTS,
            "selection_contract": selection_contract,
            **_uid_hash_records(selected_uids, seed=config.seed),
        },
        "read_only_integrity": {
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_state_unchanged": True,
        },
        "timing": {
            "started_at": started_at,
            "completed_at": _utc_now(),
            "wall_seconds": time.monotonic() - wall_started,
            "validation_seconds": metrics.get("validation_seconds"),
        },
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


__all__ = [
    "EVALUATION_ROLE",
    "FIXED_VALIDATION_EVENTS",
    "READ_ONLY_VALIDATION_VERSION",
    "evaluate_pretraining_checkpoint",
]
