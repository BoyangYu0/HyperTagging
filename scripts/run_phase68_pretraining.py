"""Bind the preregistered encoder-transfer artifact without additional pretraining."""

from pathlib import Path
import json
from scripts.run_reconstruction_phase35 import atomic_json, finite_checkpoint, sha256


def validate_pretraining_contract(contract, prereg):
    expected = prereg["pretraining_refinement"]
    if contract.get("pretraining_refinement") != expected or expected != {
        "mode": "encoder_refinement_transfer_ablation",
        "checkpoint_for_reconstruction": "arm_bound_parameter_transfer",
        "additional_pretraining_steps": 0,
    }:
        raise RuntimeError(
            "Phase68 requires the shared frozen Phase64 reconditioned control encoder"
        )
    binding = prereg['encoder_intervention']
    receipt_path = Path(__file__).resolve().parents[1] / binding['path']
    if sha256(receipt_path) != binding['sha256']:
        raise RuntimeError('Phase68 encoder ablation lineage changed')
    receipt = json.loads(receipt_path.read_text())
    if not receipt['shared_pid_and_nonencoder_state'] or not receipt['train_normalizer_preserved'] or not receipt['parameter_only_not_resumable']:
        raise RuntimeError('Phase68 shared PID and normalization controls failed')
    arms = prereg['arms']
    if arms[0]['checkpoint_sha256'] != receipt['refined_checkpoint_sha256'] or arms[1]['checkpoint_sha256'] != receipt['checkpoint_sha256']:
        raise RuntimeError('Phase68 arm checkpoint lineage differs')
    if [a['checkpoint_step'] for a in arms] != [2188, 0] or len({a['checkpoint_sha256'] for a in arms}) != 2:
        raise RuntimeError('Phase68 requires two distinct authenticated encoder states')



def run_pretraining_refinement(contract, runtime, cohort, output: Path):
    if output.exists():
        raise RuntimeError("Refusing to overwrite Phase68 frozen encoder receipt")
    checkpoint = Path(runtime["checkpoint"])
    evidence = finite_checkpoint(checkpoint)
    digest = sha256(checkpoint)
    if (
        evidence["step"] != int(runtime["checkpoint_step"])
        or not evidence["all_checkpoint_tensors_finite"]
        or digest != runtime["checkpoint_sha256"]
    ):
        raise RuntimeError("Phase68 frozen encoder checkpoint is invalid")
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
