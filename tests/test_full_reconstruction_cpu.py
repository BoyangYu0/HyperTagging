import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from hypertagging.evaluation.grafei_metrics import accuracy_for_level, build_evaluation_row, perfect_lca
from hypertagging.reconstruction.full_reconstruction import (
    aggregate_features_by_link,
    build_pred_lca_from_pairs,
    evaluate_event,
    prediction_step,
    recover,
    remap_links_around_empty,
)


class TinyFullGenerator(torch.nn.Module):
    def forward(self, batch):
        n_particles = batch["pdg_x"].shape[1]
        logits = torch.full((1, n_particles, 14), -10.0, dtype=torch.float32)
        feature = torch.zeros((1, n_particles, 4), dtype=torch.float32)
        pdg = [4, 5, 0] if n_particles == 3 else [13] + [0] * (n_particles - 1)
        for index, pdg_id in enumerate(pdg[:n_particles]):
            logits[0, index, pdg_id] = 10.0
        return logits, feature


class TinyFullLinker(torch.nn.Module):
    def forward(self, batch):
        n_daughters = batch["pdg_x"].shape[1]
        n_mothers = batch["pdg_y"].shape[1]
        logits = torch.zeros((1, n_daughters, n_mothers), dtype=torch.float32)
        links = [0, 0, 1] if n_daughters == 3 else [0] * n_daughters
        for daughter, mother in enumerate(links[:n_daughters]):
            logits[0, daughter, mother] = float(2 + daughter)
        return logits


class NeverRootGenerator(torch.nn.Module):
    def forward(self, batch):
        n_particles = batch["pdg_x"].shape[1]
        logits = torch.full((1, n_particles, 14), -10.0, dtype=torch.float32)
        feature = torch.zeros((1, n_particles, 4), dtype=torch.float32)
        for index in range(n_particles):
            logits[0, index, 4 + index] = 10.0
        return logits, feature


def tiny_pairs():
    pairs = np.zeros((2, 2, 3, 5), dtype=np.float32)
    pairs[-1, 0, :, 0] = [1, 2, 3]
    pairs[-1, 0, :, 1:] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    pairs[-1, 1, :, 0] = [4, 5, 0]
    pairs[-1, 1, :, 1:] = [
        [1.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    pairs[0, 1, :, 0] = [13, 0, 0]
    pairs[0, 1, :, 1:] = [
        [1.0, 1.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    return pairs


def reference_lca_from_links():
    idx_list = np.array([[0, 0], [1, 0], [2, 1]])
    out_link = np.array([0, 0])
    idx_list[..., -1] = idx_list[..., -1].clip(max=len(out_link) - 1)
    idx_list = np.column_stack((idx_list, out_link[idx_list[..., -1]][..., None]))
    return np.argmin(np.abs((idx_list[None, ...] - idx_list[:, None, :])), axis=-1)


def test_recover_and_link_remap_match_legacy_inline_formulas():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 1.0]]])
    parsed, values = recover(logits, return_value=True)
    torch.testing.assert_close(parsed, torch.tensor([[1, 0]]))
    torch.testing.assert_close(values, torch.tensor([[2.0, 3.0]]))

    out_link = torch.tensor([0, 2, 1])
    mask = torch.tensor([True, False, True])
    expected = out_link.clone()
    empty_idx = (~mask).nonzero()
    expected -= (~(empty_idx.repeat(1, len(out_link)) > out_link[:, None].repeat(1, len(empty_idx)).T)).sum(0)
    torch.testing.assert_close(remap_links_around_empty(out_link, mask), expected)


def test_prediction_step_aggregates_features_and_signal_probability_on_cpu():
    in_pdg = torch.tensor([1, 2, 3], dtype=torch.long)
    in_feat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    step = prediction_step(in_pdg, in_feat, TinyFullGenerator(), TinyFullLinker())

    torch.testing.assert_close(step.pdg, torch.tensor([4, 5]))
    torch.testing.assert_close(step.link, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(step.feature, aggregate_features_by_link(in_feat, step.link)[:2])
    torch.testing.assert_close(step.signal_probability, torch.sqrt(torch.tensor(6.0)))


def test_full_reconstruction_lca_accuracy_and_stopping_match_historical_formulas():
    result = build_pred_lca_from_pairs(tiny_pairs(), TinyFullGenerator(), TinyFullLinker())

    np.testing.assert_array_equal(result.lca, reference_lca_from_links())
    assert len(result.steps) == 2
    assert 13 in result.steps[-1].pdg
    assert result.accuracy["pdg"] == 1.2
    np.testing.assert_allclose(result.accuracy["feat"], np.array([0.2, 0.2, 0.2, 0.6]))
    torch.testing.assert_close(result.signal_probability, torch.tensor(6.0))


def test_accuracy_and_event_row_match_whole_eva_columns():
    pairs = tiny_pairs()
    truth_pdg = pairs[:, 1, :, 0][::-1]
    truth_feat = pairs[:, 1, :, 1:][::-1]
    correct, err = accuracy_for_level(
        torch.tensor([4, 5]),
        torch.tensor([[1.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
        truth_pdg,
        truth_feat,
        level=0,
        padding_size=3,
    )

    assert correct == 3
    np.testing.assert_allclose(err, np.zeros(4))
    goal_lca = reference_lca_from_links()
    row = build_evaluation_row(
        nleaves=3,
        depth=2,
        pred_lca=goal_lca,
        goal_lca=goal_lca,
        accuracy={"pdg": 1.0, "feat": np.zeros(4)},
        signal_probability=torch.tensor(6.0),
        failed=False,
    )
    assert row.as_dict()["perfect"] is True
    assert perfect_lca(goal_lca, goal_lca)


def test_evaluate_event_returns_failure_row_when_tree_cannot_stop():
    event = SimpleNamespace(pairs=tiny_pairs(), lcas=[reference_lca_from_links()])

    row = evaluate_event(event, NeverRootGenerator(), TinyFullLinker(), max_steps=1)

    assert row["failed"] is True
    assert row["perfect"] is False
    assert row["pdgAcc"] == 0.0
    np.testing.assert_allclose(row["featErr"], np.zeros(4))
    assert row["sigProb"] == 0.0


def test_evaluate_reconstruction_dry_run_cli_cpu():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_reconstruction.py",
            "--dry-run",
            "--device",
            "cpu",
        ],
        cwd=".",
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["nSteps"] == 2
    assert payload["pdgAcc"] == 1.2
    assert payload["lca"] == reference_lca_from_links().tolist()
