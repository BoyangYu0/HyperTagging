"""Explicit shared-encoder checkpoint transfer with auditable key reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class EncoderTransferReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    frozen: bool
    leaf_pid_loaded_keys: tuple[str, ...] = ()
    leaf_pid_missing_keys: tuple[str, ...] = ()
    leaf_pid_shape_mismatches: tuple[str, ...] = ()
    leaf_pid_frozen: bool = False


def load_pretrained_encoder(
    encoder: torch.nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = False,
    leaf_pid_head: torch.nn.Module | None = None,
    transfer_leaf_pid_head: bool = False,
    freeze_leaf_pid_head: bool = False,
) -> EncoderTransferReport:
    # Training checkpoints include audited metadata/RNG states and are trusted
    # local/HTCondor artifacts, not arbitrary downloads.
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    source = payload.get("encoder_state_dict")
    if not source:
        model_state = payload.get("model_state_dict", {})
        source = {
            key.removeprefix("encoder."): value
            for key, value in model_state.items()
            if key.startswith("encoder.")
        }
    target = encoder.state_dict()
    compatible = {}
    mismatch = []
    unexpected = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
        elif target[key].shape != value.shape:
            mismatch.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target[key].shape)}"
            )
        else:
            compatible[key] = value
    result = encoder.load_state_dict(compatible, strict=False)
    for parameter in encoder.parameters():
        parameter.requires_grad_(not freeze)
    leaf_loaded: list[str] = []
    leaf_missing: list[str] = []
    leaf_mismatch: list[str] = []
    if transfer_leaf_pid_head:
        if leaf_pid_head is None:
            raise ValueError("transfer_leaf_pid_head requires a destination leaf PID head")
        model_state = payload.get("model_state_dict", {})
        leaf_source = {
            key.removeprefix("leaf_pid_head."): value
            for key, value in model_state.items()
            if key.startswith("leaf_pid_head.")
        }
        target_leaf = leaf_pid_head.state_dict()
        compatible_leaf = {}
        for key, value in leaf_source.items():
            if key in target_leaf and target_leaf[key].shape == value.shape:
                compatible_leaf[key] = value
                leaf_loaded.append(key)
            elif key in target_leaf:
                leaf_mismatch.append(key)
        leaf_result = leaf_pid_head.load_state_dict(compatible_leaf, strict=False)
        leaf_missing.extend(leaf_result.missing_keys)
        for parameter in leaf_pid_head.parameters():
            parameter.requires_grad_(not freeze_leaf_pid_head)
    return EncoderTransferReport(
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(sorted(result.missing_keys)),
        unexpected_keys=tuple(sorted(set(unexpected) | set(result.unexpected_keys))),
        shape_mismatches=tuple(sorted(mismatch)),
        frozen=freeze,
        leaf_pid_loaded_keys=tuple(sorted(leaf_loaded)),
        leaf_pid_missing_keys=tuple(sorted(leaf_missing)),
        leaf_pid_shape_mismatches=tuple(sorted(leaf_mismatch)),
        leaf_pid_frozen=bool(transfer_leaf_pid_head and freeze_leaf_pid_head),
    )


def unfreeze_encoder(encoder: torch.nn.Module) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)


def optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    base_lr: float,
    encoder_lr_multiplier: float,
    leaf_pid_lr_multiplier: float = 1.0,
) -> list[dict[str, object]]:
    encoder_ids = {
        id(parameter)
        for parameter in getattr(model, "encoder").parameters()
    } if hasattr(model, "encoder") else set()
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in encoder_ids
    ]
    leaf_ids = {
        id(parameter)
        for parameter in getattr(model, "leaf_pid_head").parameters()
    } if hasattr(model, "leaf_pid_head") else set()
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids and id(parameter) not in leaf_ids
    ]
    groups: list[dict[str, object]] = []
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": base_lr * encoder_lr_multiplier,
                "name": "pretrained_encoder",
            }
        )
    if other_parameters:
        groups.append({"params": other_parameters, "lr": base_lr, "name": "reconstruction"})
    leaf_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in leaf_ids
    ]
    if leaf_parameters:
        groups.append(
            {
                "params": leaf_parameters,
                "lr": base_lr * leaf_pid_lr_multiplier,
                "name": "leaf_pid_head",
            }
        )
    return groups


__all__ = [
    "EncoderTransferReport",
    "load_pretrained_encoder",
    "optimizer_parameter_groups",
    "unfreeze_encoder",
]
