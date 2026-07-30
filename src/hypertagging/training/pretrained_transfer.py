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


def load_pretrained_encoder(
    encoder: torch.nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = False,
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
    return EncoderTransferReport(
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(sorted(result.missing_keys)),
        unexpected_keys=tuple(sorted(set(unexpected) | set(result.unexpected_keys))),
        shape_mismatches=tuple(sorted(mismatch)),
        frozen=freeze,
    )


def unfreeze_encoder(encoder: torch.nn.Module) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)


def optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    base_lr: float,
    encoder_lr_multiplier: float,
) -> list[dict[str, object]]:
    encoder_ids = {
        id(parameter)
        for parameter in getattr(model, "encoder").parameters()
    } if hasattr(model, "encoder") else set()
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in encoder_ids
    ]
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
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
    return groups


__all__ = [
    "EncoderTransferReport",
    "load_pretrained_encoder",
    "optimizer_parameter_groups",
    "unfreeze_encoder",
]
