import json
import subprocess
import sys

import torch
import torch.nn.functional as F

from hypertagging.losses.link_losses import link_cross_entropy, link_metrics, transfer_link_metrics
from hypertagging.models import CorrectedLinker, EmbLinker, StandardLinker
from hypertagging.training.dry_run import link_batch
from hypertagging.training.train_link import (
    build_link_model_input,
    link_prediction_step,
    run_link_prediction_dry_run,
)


class TinyLinkModel(torch.nn.Module):
    def forward(self, batch):
        score = batch["feature_x"][..., :2] @ batch["feature_y"][..., :2].transpose(1, 2)
        pdg_delta = (batch["pdg_x"].float().unsqueeze(-1) - batch["pdg_y"].float().unsqueeze(1)).abs()
        return score - 0.1 * pdg_delta


def test_link_model_variants_are_exposed():
    assert CorrectedLinker is StandardLinker
    assert EmbLinker.__name__ == "EmbLinker"


def test_ground_truth_link_mode_uses_ground_truth_mothers_and_cross_entropy():
    batch = link_batch("cpu")
    model = TinyLinkModel()

    result = link_prediction_step(model, batch, mode="ground_truth")
    expected_input = build_link_model_input(batch, mode="ground_truth")
    expected_logits = model(expected_input)
    expected_loss, expected_acc = link_metrics(expected_logits, batch["links"], batch["padding_mask"])

    assert result.mode == "ground_truth"
    torch.testing.assert_close(result.logits, expected_logits)
    torch.testing.assert_close(result.loss, expected_loss)
    torch.testing.assert_close(result.loss, link_cross_entropy(expected_logits, batch["links"], batch["padding_mask"]))
    torch.testing.assert_close(result.accuracy, expected_acc)
    torch.testing.assert_close(result.model_input["pdg_y"], batch["pdg_y"])
    torch.testing.assert_close(result.model_input["feature_y"], batch["feature_y"])


def test_reconstructed_mother_mode_uses_ground_truth_daughters_and_reconstructed_mothers():
    batch = link_batch("cpu")
    reconstructed = {
        "pdg": torch.tensor([[1, 1, 0], [4, 4, 5]], dtype=torch.long),
        "feature": batch["feature_y"] + 0.25,
    }
    model = TinyLinkModel()

    result = link_prediction_step(
        model,
        batch,
        mode="reconstructed_mother",
        reconstructed_mother=reconstructed,
    )

    expected_input = build_link_model_input(
        batch,
        mode="reconstructed_mother",
        reconstructed_mother=reconstructed,
    )
    expected_logits = model(expected_input)
    expected_loss = F.cross_entropy(expected_logits[batch["padding_mask"]], batch["links"][batch["padding_mask"]])

    torch.testing.assert_close(result.model_input["pdg_x"], batch["pdg_x"])
    torch.testing.assert_close(result.model_input["feature_x"], batch["feature_x"])
    torch.testing.assert_close(result.model_input["pdg_y"], reconstructed["pdg"])
    torch.testing.assert_close(result.model_input["feature_y"], reconstructed["feature"])
    torch.testing.assert_close(result.logits, expected_logits)
    torch.testing.assert_close(result.loss, expected_loss)


def test_corrected_teacher_path_uses_transfer_link_metrics():
    batch = link_batch("cpu")
    reconstructed = {
        "pdg": batch["pdg_y"],
        "feature": batch["feature_y"] + 0.1,
    }
    model = TinyLinkModel()
    teacher_logits = model(build_link_model_input(batch, mode="ground_truth")).detach()

    result = link_prediction_step(
        model,
        batch,
        mode="reconstructed_mother",
        reconstructed_mother=reconstructed,
        teacher_logits=teacher_logits,
    )
    expected_loss, expected_acc = transfer_link_metrics(result.logits, teacher_logits, batch["padding_mask"])

    torch.testing.assert_close(result.loss, expected_loss)
    torch.testing.assert_close(result.accuracy, expected_acc)


def test_link_prediction_dry_runs_for_both_training_modes():
    ground_truth = run_link_prediction_dry_run(mode="ground_truth", device="cpu")
    reconstructed = run_link_prediction_dry_run(mode="reconstructed_mother", device="cpu")

    assert ground_truth.stage == "link"
    assert ground_truth.mode == "ground_truth"
    assert reconstructed.mode == "reconstructed_mother"
    assert ground_truth.model_class == "linearLinker"
    assert reconstructed.model_class == "linearLinker"
    assert ground_truth.logits_shape == (2, 3, 3)
    assert reconstructed.logits_shape == (2, 3, 3)
    assert ground_truth.backward_ran is True
    assert reconstructed.backward_ran is True


def test_embedding_link_dry_run_shape():
    summary = run_link_prediction_dry_run(model_variant="embedding", device="cpu")

    assert summary.stage == "link"
    assert summary.model_variant == "embedding"
    assert summary.model_class == "EmbLinker"
    assert summary.logits_shape == (2, 3, 3)


def test_link_cli_supports_reconstructed_mother_mode():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_link.py",
            "--dry-run",
            "--device",
            "cpu",
            "--mode",
            "reconstructed_mother",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["stage"] == "link"
    assert summary["mode"] == "reconstructed_mother"
    assert summary["logits_shape"] == [2, 3, 3]
