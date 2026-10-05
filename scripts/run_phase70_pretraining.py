"""Bind the preregistered encoder-transfer artifact without additional pretraining."""

from pathlib import Path
from scripts.run_reconstruction_phase35 import atomic_json, finite_checkpoint, sha256


def validate_pretraining_contract(contract, prereg):
    expected = prereg["pretraining_refinement"]
    if contract.get("pretraining_refinement") != expected or expected != {
        "mode": "fixed_encoder_decoder_bias_ablation",
        "checkpoint_for_reconstruction": "arm_bound_parameter_transfer",
        "additional_pretraining_steps": 0,
    }:
        raise RuntimeError(
            "Phase70 requires the shared frozen Phase64 reconditioned control encoder"
        )
    arms = prereg['arms']
    if [a['checkpoint_step'] for a in arms] != [2188, 2188] or len({a['checkpoint_sha256'] for a in arms}) != 1:
        raise RuntimeError('Phase70 requires one identical authenticated refined encoder')
    if arms[0]['checkpoint_sha256'] != "23fa26988ba0a8f97805f478689f1932d7e3e085fd65247833ff4332fe55cc8f":
        raise RuntimeError('Phase70 refined checkpoint identity changed')



def run_pretraining_refinement(contract, runtime, cohort, output: Path):
    if output.exists():
        raise RuntimeError("Refusing to overwrite Phase70 frozen encoder receipt")
    checkpoint = Path(runtime["checkpoint"])
    evidence = finite_checkpoint(checkpoint)
    digest = sha256(checkpoint)
    if (
        evidence["step"] != int(runtime["checkpoint_step"])
        or not evidence["all_checkpoint_tensors_finite"]
        or digest != runtime["checkpoint_sha256"]
    ):
        raise RuntimeError("Phase70 frozen encoder checkpoint is invalid")
    output.mkdir(parents=True)
    receipt = {
        "status": "COMPLETED",
        "mode": "frozen_checkpoint_reuse_no_training",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "source_checkpoint_sha256": digest,
        "source_unchanged": True,
        "step": int(runtime["checkpoint_step"]),
        "additional_pretraining_steps": 0,
        "checkpoint_selection": "arm_bound_parameter_transfer",
        "all_checkpoint_tensors_finite": True,
    }
    atomic_json(output / "refinement-result.json", receipt)
    return receipt
