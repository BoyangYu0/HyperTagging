import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hypertagging.data.gpt_like import (
    collate_gpt_link,
    collate_gpt_reconstruction,
    get_level_mask,
    validate_gpt_link_batch,
    validate_gpt_reconstruction_batch,
)
from hypertagging.losses.gpt_losses import distance, radius_loss
from hypertagging.losses.link_losses import link_metrics
from hypertagging.models import EmbLinker, GPTReconstructor, MultiGPT
from hypertagging.training import run_multi_gpt_dry_run


ROOT = Path(__file__).resolve().parents[1].parent


def _legacy_module(relative_path: str, name: str):
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"legacy source file not present: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _copy_and_compare_state(new_model, old_model):
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    assert set(new_state) == set(old_state)
    for key, value in new_state.items():
        assert value.shape == old_state[key].shape
    new_model.load_state_dict(old_state)
    new_model.eval()
    old_model.eval()


def test_get_level_mask_matches_historical_formula_exactly():
    mask = get_level_mask(6, particles_per_level=2)
    expected = np.zeros((6, 6), dtype=np.float32)
    for index in range(2, 7, 2):
        expected[index - 2 : index, index:] = float("-inf")

    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(get_level_mask(6, 2), mask)


def test_gpt_reconstruction_collate_matches_historical_formula_and_contract():
    example = {
        "emb": np.arange(24, dtype=np.float32).reshape(6, 4) / 100,
        "links": np.array([0, 1, 0, 1], dtype=np.int_),
        "mass": np.array([1, 2, 3, 4, 5, 6], dtype=np.int_),
        "shape": np.array([6, 2], dtype=np.int_),
    }

    batch = collate_gpt_reconstruction([example])

    np.testing.assert_array_equal(batch["emb"].numpy()[0], example["emb"])
    np.testing.assert_array_equal(batch["target"].numpy()[0, :4], example["emb"][2:])
    np.testing.assert_array_equal(batch["src_mask"].numpy()[0], get_level_mask(6, 2))
    np.testing.assert_array_equal(batch["links"].numpy()[0, :4], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(batch["mass"].numpy()[0, :4], example["mass"][2:])
    np.testing.assert_allclose(batch["lvl_code"].numpy()[0], np.exp(-np.arange(3)).repeat(2))
    validate_gpt_reconstruction_batch(batch)


def test_gpt_link_collate_matches_historical_padding_and_contract():
    batch = collate_gpt_link(
        [
            {
                "emb_x": np.ones((2, 3), dtype=np.float32),
                "emb_y": np.ones((2, 3), dtype=np.float32) * 2,
                "links": np.array([0, 1], dtype=np.int_),
                "padding_mask": np.array([True, True], dtype=np.bool_),
            },
            {
                "emb_x": np.ones((1, 3), dtype=np.float32) * 3,
                "emb_y": np.ones((1, 3), dtype=np.float32) * 4,
                "links": np.array([0], dtype=np.int_),
                "padding_mask": np.array([True], dtype=np.bool_),
            },
        ]
    )

    assert batch["emb_x"].shape == (2, 2, 3)
    torch.testing.assert_close(batch["emb_x"][1, 1], torch.zeros(3))
    torch.testing.assert_close(batch["links"], torch.tensor([[0, 1], [0, -1]]))
    torch.testing.assert_close(batch["padding_mask"], torch.tensor([[True, True], [True, False]]))
    validate_gpt_link_batch(batch)


def test_gpt_reconstructor_cpu_forward_matches_legacy_class():
    legacy = _legacy_module("graFEI_gpt/models.py", "legacy_grafei_gpt_models")
    kwargs = dict(
        tr_width=8,
        tr_n_head=1,
        tr_n=1,
        tr_hidden_size=16,
        dim_hyper=4,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.GPTReconstructor(**kwargs)
    torch.manual_seed(2)
    new = GPTReconstructor(**kwargs)
    _copy_and_compare_state(new, old)
    batch = {
        "emb": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 100,
        "src_mask": torch.zeros((2, 4, 4), dtype=torch.float32),
        "lvl_code": torch.tensor([[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]]),
    }

    torch.testing.assert_close(new(batch), old(batch))


def test_emb_linker_cpu_forward_matches_legacy_class():
    legacy = _legacy_module("graFEI_gpt/models.py", "legacy_grafei_gpt_models_link")
    kwargs = dict(
        n_features=4,
        link_width=8,
        link_n_head=1,
        link_n_layers=1,
        link_fc=16,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.EmbLinker(**kwargs)
    torch.manual_seed(2)
    new = EmbLinker(**kwargs)
    _copy_and_compare_state(new, old)
    batch = {
        "emb_x": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 100,
        "emb_y": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4) / 100,
        "padding_mask": torch.tensor([[True, True, False], [True, True, True]]),
    }

    torch.testing.assert_close(new(batch), old(batch))


def test_multi_gpt_cpu_forward_and_loss_components_are_finite():
    model = MultiGPT(
        rec_width=8,
        rec_n_head=1,
        rec_n=1,
        rec_hidden_size=16,
        link_width=8,
        link_n_head=1,
        link_n=1,
        link_hidden_size=16,
        dim_hyper=4,
        device="cpu",
    )
    batch = {
        "emb": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 100,
        "target": torch.arange(32, 64, dtype=torch.float32).reshape(2, 4, 4) / 100,
        "src_mask": torch.zeros((2, 4, 4), dtype=torch.float32),
        "links": torch.tensor([[0, 1, -1, -1], [1, -1, 2, -1]], dtype=torch.long),
        "mass": torch.tensor([[1, 2, 0, 0], [5, 0, 10, 0]], dtype=torch.float32),
        "lvl_code": torch.tensor([[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]]),
    }

    out_rec, out_link = model(batch)
    assert out_rec.shape == (2, 4, 4)
    assert out_link.shape == (2, 4, 4)
    rec_loss = distance(out_rec, batch["target"], batch["lvl_code"].bool())
    link_loss, link_acc = link_metrics(out_link, batch["links"], batch["links"] >= 0)
    r_loss = radius_loss(out_rec, batch["mass"], batch["links"] >= 0)
    assert torch.isfinite(rec_loss)
    assert torch.isfinite(link_loss)
    assert torch.isfinite(link_acc)
    assert torch.isfinite(r_loss)

    # Historical GPT radius loss formula with the migrated output.
    mask = batch["links"] >= 0
    r_euclidean = torch.norm(out_rec[mask], dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - 1e-6))
    ref = F.l1_loss(r_poincare, 0.9 * torch.sqrt(1 - batch["mass"][mask] / 100) + 0.1)
    torch.testing.assert_close(r_loss, ref)


def test_multi_gpt_dry_run_and_cli_cpu():
    summary = run_multi_gpt_dry_run(device="cpu", backward=False)
    assert summary.stage == "gpt"
    assert summary.model_class == "MultiGPT"
    assert summary.output_shapes == ((2, 4, 4), (2, 4, 4))
    assert summary.backward_ran is False

    completed = subprocess.run(
        [sys.executable, "scripts/run_gpt_like.py", "--dry-run", "--device", "cpu", "--no-backward"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["model_class"] == "MultiGPT"
    assert payload["output_shapes"] == [[2, 4, 4], [2, 4, 4]]
    assert payload["backward_ran"] is False
