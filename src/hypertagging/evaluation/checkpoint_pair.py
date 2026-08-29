"""Read-only compatibility checks for pretraining/reconstruction checkpoints.

The reconstruction checkpoint contains the complete inference model.  The
pretraining checkpoint is supplied separately so an evaluation receipt can
prove which frozen encoder was transferred into that model.  This module never
restores optimizer/RNG state and always maps tensors to CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from hypertagging.training.checkpointing import load_training_checkpoint


@dataclass(frozen=True)
class CheckpointPairReport:
    """Compatibility and frozen-transfer evidence for a checkpoint pair."""

    pretraining_checkpoint: str
    reconstruction_checkpoint: str
    pretraining_sha256: str
    reconstruction_sha256: str
    pretraining_step: int
    reconstruction_step: int
    feature_specification_match: bool
    model_feature_contract_match: bool
    pid_vocabulary_match: bool
    encoder_common_keys: int
    encoder_pretraining_keys: int
    encoder_reconstruction_keys: int
    encoder_shape_compatible_keys: int
    encoder_dtype_compatible_keys: int
    encoder_exact_keys: int
    encoder_exact_fraction: float
    configured_pretraining_path_match: bool
    exact_frozen_encoder_required: bool

    @property
    def compatible(self) -> bool:
        return (
            self.feature_specification_match
            and self.model_feature_contract_match
            and self.pid_vocabulary_match
            and self.encoder_pretraining_keys > 0
            and self.encoder_common_keys == self.encoder_pretraining_keys
            and self.encoder_common_keys == self.encoder_reconstruction_keys
            and self.encoder_shape_compatible_keys == self.encoder_pretraining_keys
            and self.encoder_dtype_compatible_keys == self.encoder_pretraining_keys
            and (
                not self.exact_frozen_encoder_required
                or self.encoder_exact_keys == self.encoder_pretraining_keys
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "compatible": self.compatible}


def validate_checkpoint_pair(
    pretraining_checkpoint: str | Path,
    reconstruction_checkpoint: str | Path,
    *,
    require_exact_frozen_encoder: bool = True,
) -> CheckpointPairReport:
    """Load two trusted training artifacts on CPU and validate their lineage.

    Current reconstruction runs freeze the transferred encoder for their full
    duration, so exact equality is the strongest and least ambiguous lineage
    check.  ``require_exact_frozen_encoder=False`` is available for future runs
    that intentionally fine-tune the encoder; all contract and key-coverage
    checks still apply and the exact fraction remains in the report.
    """

    pretraining_path = Path(pretraining_checkpoint).expanduser().resolve()
    reconstruction_path = Path(reconstruction_checkpoint).expanduser().resolve()
    pretraining = load_training_checkpoint(pretraining_path, map_location="cpu")
    reconstruction = load_training_checkpoint(reconstruction_path, map_location="cpu")

    pretraining_encoder = _encoder_state(pretraining, is_reconstruction=False)
    reconstruction_encoder = _encoder_state(reconstruction, is_reconstruction=True)
    common = sorted(set(pretraining_encoder) & set(reconstruction_encoder))
    shape_compatible = sum(
        int(pretraining_encoder[name].shape == reconstruction_encoder[name].shape)
        for name in common
    )
    dtype_compatible = sum(
        int(pretraining_encoder[name].dtype == reconstruction_encoder[name].dtype)
        for name in common
    )
    exact = sum(
        int(
            pretraining_encoder[name].shape == reconstruction_encoder[name].shape
            and pretraining_encoder[name].dtype
            == reconstruction_encoder[name].dtype
            and torch.equal(pretraining_encoder[name], reconstruction_encoder[name])
        )
        for name in common
    )
    configured = str(reconstruction.get("config", {}).get("pretrained_encoder", ""))
    configured_match = False
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            configured_match = configured_path.resolve() == pretraining_path
        else:
            configured_match = configured_path.name == pretraining_path.name

    pretraining_feature_hash = str(
        pretraining.get("feature_specification", {}).get("feature_spec_hash", "")
    )
    reconstruction_feature_hash = str(
        reconstruction.get("feature_specification", {}).get(
            "feature_spec_hash", ""
        )
    )
    pretraining_model_contract = str(
        pretraining.get("feature_contract", {}).get(
            "model_feature_contract_hash", ""
        )
    )
    reconstruction_model_contract = str(
        reconstruction.get("feature_contract", {}).get(
            "model_feature_contract_hash", ""
        )
    )
    pretraining_pid_vocabulary = str(
        pretraining.get("pid_vocabulary_version", "")
    )
    reconstruction_pid_vocabulary = str(
        reconstruction.get("pid_vocabulary_version", "")
    )

    report = CheckpointPairReport(
        pretraining_checkpoint=str(pretraining_path),
        reconstruction_checkpoint=str(reconstruction_path),
        pretraining_sha256=_file_sha256(pretraining_path),
        reconstruction_sha256=_file_sha256(reconstruction_path),
        pretraining_step=int(pretraining.get("step", 0)),
        reconstruction_step=int(reconstruction.get("step", 0)),
        feature_specification_match=(
            bool(pretraining_feature_hash)
            and pretraining_feature_hash == reconstruction_feature_hash
        ),
        model_feature_contract_match=(
            bool(pretraining_model_contract)
            and pretraining_model_contract == reconstruction_model_contract
        ),
        pid_vocabulary_match=(
            bool(pretraining_pid_vocabulary)
            and pretraining_pid_vocabulary == reconstruction_pid_vocabulary
        ),
        encoder_common_keys=len(common),
        encoder_pretraining_keys=len(pretraining_encoder),
        encoder_reconstruction_keys=len(reconstruction_encoder),
        encoder_shape_compatible_keys=shape_compatible,
        encoder_dtype_compatible_keys=dtype_compatible,
        encoder_exact_keys=exact,
        encoder_exact_fraction=exact / max(len(pretraining_encoder), 1),
        configured_pretraining_path_match=configured_match,
        exact_frozen_encoder_required=bool(require_exact_frozen_encoder),
    )
    failures: list[str] = []
    if not report.feature_specification_match:
        failures.append("feature specification")
    if not report.model_feature_contract_match:
        failures.append("model feature contract")
    if not report.pid_vocabulary_match:
        failures.append("PID vocabulary")
    if (
        report.encoder_common_keys != report.encoder_pretraining_keys
        or report.encoder_common_keys != report.encoder_reconstruction_keys
    ):
        failures.append("encoder key coverage")
    if report.encoder_shape_compatible_keys != report.encoder_pretraining_keys:
        failures.append("encoder tensor shapes")
    if report.encoder_dtype_compatible_keys != report.encoder_pretraining_keys:
        failures.append("encoder tensor dtypes")
    if require_exact_frozen_encoder and (
        report.encoder_exact_keys != report.encoder_pretraining_keys
    ):
        failures.append("frozen encoder tensor equality")
    if failures:
        raise ValueError("checkpoint pair mismatch: " + ", ".join(failures))
    return report


def _encoder_state(
    checkpoint: dict[str, Any], *, is_reconstruction: bool
) -> dict[str, torch.Tensor]:
    if not is_reconstruction and checkpoint.get("encoder_state_dict"):
        return dict(checkpoint["encoder_state_dict"])
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    prefix = "encoder."
    encoder = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix) and isinstance(value, torch.Tensor)
    }
    if not encoder:
        raise ValueError("checkpoint has no encoder tensors")
    return encoder


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["CheckpointPairReport", "validate_checkpoint_pair"]
