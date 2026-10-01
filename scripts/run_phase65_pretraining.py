"""Bind a shared frozen Phase64 reconditioned encoder; Phase65 performs no pretraining."""

from pathlib import Path
from scripts.run_reconstruction_phase35 import atomic_json, finite_checkpoint, sha256


def validate_pretraining_contract(contract, prereg):
    expected = prereg["pretraining_refinement"]
    if contract.get("pretraining_refinement") != expected or expected != {
        "mode": "reuse_frozen_phase64_reconditioned",
        "step": 2188,
        "checkpoint_for_reconstruction": "fixed_final_step_2188",
        "additional_pretraining_steps": 0,
    }:
        raise RuntimeError(
            "Phase65 requires the shared frozen Phase64 reconditioned control encoder"
        )
    if (
        len(
            {
                (a["checkpoint"], a["checkpoint_sha256"], a["checkpoint_step"])
                for a in prereg["arms"]
            }
        )
        != 1
    ):
        raise RuntimeError("Phase65 arm initialization differs")


def run_pretraining_refinement(contract, runtime, cohort, output: Path):
    if output.exists():
        raise RuntimeError("Refusing to overwrite Phase65 frozen encoder receipt")
    checkpoint = Path(runtime["checkpoint"])
    evidence = finite_checkpoint(checkpoint)
    digest = sha256(checkpoint)
    if (
        evidence["step"] != 2188
        or not evidence["all_checkpoint_tensors_finite"]
        or digest != runtime["checkpoint_sha256"]
    ):
        raise RuntimeError("Phase65 frozen encoder checkpoint is invalid")
    output.mkdir(parents=True)
    receipt = {
        "status": "COMPLETED",
        "mode": "frozen_checkpoint_reuse_no_training",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "source_checkpoint_sha256": digest,
        "source_unchanged": True,
        "step": 2188,
        "additional_pretraining_steps": 0,
        "checkpoint_selection": "fixed_final_step_2188",
        "all_checkpoint_tensors_finite": True,
    }
    atomic_json(output / "refinement-result.json", receipt)
    return receipt
