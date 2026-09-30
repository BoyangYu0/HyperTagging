"""Create an explicitly parameter-only, source-bound projection warm start."""

from pathlib import Path
import hashlib
import math
import torch

TARGETS = ("encoder.hyper_projection.weight", "encoder.hyper_projection.bias")


def create_projection_initialization(source, output, *, scale, expected_sha256):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError("Projection initialization must have a fresh output path")
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Projection initialization scale must be finite in (0, 1]")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    if before != expected_sha256:
        raise ValueError("Projection initialization source hash mismatch")
    original = torch.load(source, map_location="cpu", weights_only=False)
    state = original["model_state_dict"]
    if any(k not in state for k in TARGETS):
        raise ValueError("Projection initialization requires both weight and bias")
    if any(
        not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
        for v in state.values()
    ):
        raise ValueError("Projection initialization source tensors must be finite")
    if (
        state[TARGETS[0]].ndim != 2
        or state[TARGETS[1]].shape != state[TARGETS[0]].shape[:1]
    ):
        raise ValueError(
            "Projection initialization has invalid linear parameter shapes"
        )
    payload = {
        key: original[key]
        for key in (
            "architecture",
            "pid_vocabulary_version",
            "feature_specification",
            "preprocessing_schema_version",
            "step",
        )
    }
    payload["model_state_dict"] = {
        key: value * scale if key in TARGETS else value for key, value in state.items()
    }
    provenance = {
        "version": "phase64-projection-initialization-v1",
        "parameter_initialization_only": True,
        "source_checkpoint_sha256": before,
        "source_step": original["step"],
        "scale": float(scale),
        "rescaled_parameters": list(TARGETS),
        "optimizer_restored": False,
        "source_mutation": False,
    }
    payload["projection_initialization"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of another attempt's initialization.
    with output.open("xb") as stream:
        torch.save(payload, stream)
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("Projection initialization source changed while reading")
    return {
        **provenance,
        "derived_checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
