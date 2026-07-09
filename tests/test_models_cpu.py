import importlib.util
from pathlib import Path

import pytest
import torch

from hypertagging.models import (
    DNNReconstructor,
    EmbLinker,
    GPTReconstructor,
    HyperEmbedder,
    InteractingLayer,
    Reconstructor,
    linearLinker,
)


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


def _particle_batch(batch_size=2, particles=3, features=4):
    return {
        "pdg": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)[:, :particles],
        "feature": torch.arange(batch_size * particles * features, dtype=torch.float32).reshape(batch_size, particles, features) / 50,
        "padding_mask": torch.tensor([[True, True, False], [True, True, True]])[:, :particles],
    }


def _pair_batch(batch_size=2, particles=3, features=4):
    batch = _particle_batch(batch_size, particles, features)
    return {
        "pdg_x": batch["pdg"],
        "pdg_y": torch.tensor([[2, 1, 0], [4, 3, 5]], dtype=torch.long)[:, :particles],
        "feature_x": batch["feature"],
        "feature_y": batch["feature"] + 0.1,
        "padding_mask": batch["padding_mask"],
    }


def test_interacting_layer_state_and_forward_parity_with_legacy():
    legacy = _legacy_module("graFEI_reduced/models.py", "legacy_grafei_reduced_models_interacting")
    torch.manual_seed(1)
    old = legacy.InteractingLayer(embedding_size=4, head_num=1, device="cpu")
    torch.manual_seed(2)
    new = InteractingLayer(embedding_size=4, head_num=1, device="cpu")
    _copy_and_compare_state(new, old)

    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 100
    mask = torch.tensor([[True, True, False], [True, True, True]])
    torch.testing.assert_close(new(inputs, mask), old(inputs, mask))


def test_hyperembedder_cpu_forward_and_legacy_parity():
    legacy = _legacy_module("graFEI_reduced/models.py", "legacy_grafei_reduced_models_embedder")
    kwargs = dict(
        n_features=4,
        tr_width=8,
        tr_n_head=1,
        tr_n=1,
        tr_hidden_size=16,
        pdg_emb=2,
        dim_hyper=3,
        num_pdg=8,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.HyperEmbedder(**kwargs)
    torch.manual_seed(2)
    new = HyperEmbedder(**kwargs)
    _copy_and_compare_state(new, old)

    batch = _particle_batch()
    out = new(batch)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, old(batch))


def test_linear_linker_cpu_forward_and_legacy_parity():
    legacy = _legacy_module("graFEI_reduced/models.py", "legacy_grafei_reduced_models_linker")
    kwargs = dict(
        n_features=4,
        link_width=8,
        link_n_head=1,
        link_n_layers=1,
        link_fc=16,
        pdg_emb=2,
        num_pdg=8,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.linearLinker(**kwargs)
    torch.manual_seed(2)
    new = linearLinker(**kwargs)
    _copy_and_compare_state(new, old)

    batch = _pair_batch()
    out = new(batch)
    assert out.shape == (2, 3, 3)
    torch.testing.assert_close(out, old(batch))


def test_reconstructor_cpu_forward_and_legacy_parity():
    legacy = _legacy_module("graFEI_reduced/models.py", "legacy_grafei_reduced_models_reconstructor")
    kwargs = dict(
        n_features=4,
        gen_tr_width=8,
        gen_encoder_n_head=1,
        gen_encoder_n_layers=1,
        gen_encoder_fc=16,
        gen_decoder_n_head=1,
        gen_decoder_n_layers=1,
        gen_decoder_fc=16,
        pdg_emb=2,
        dim_hyper=3,
        num_pdg=8,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.Reconstructor(**kwargs)
    torch.manual_seed(2)
    new = Reconstructor(**kwargs)
    _copy_and_compare_state(new, old)

    batch = _pair_batch()
    pdg_out, feat_out = new(batch)
    assert pdg_out.shape == (2, 3, 9)
    assert feat_out.shape == (2, 3, 4)
    old_pdg, old_feat = old(batch)
    torch.testing.assert_close(pdg_out, old_pdg)
    torch.testing.assert_close(feat_out, old_feat)


def test_dnn_reconstructor_cpu_forward_and_legacy_parity():
    legacy = _legacy_module("graFEI_reduced/models.py", "legacy_grafei_reduced_models_dnn")
    kwargs = dict(
        n_features=4,
        gen_dnn_width=8,
        gen_dnn_n_layers=1,
        dim_hyper=3,
        num_pdg=5,
        device="cpu",
    )
    torch.manual_seed(1)
    old = legacy.DNNReconstructor(**kwargs)
    torch.manual_seed(2)
    new = DNNReconstructor(**kwargs)
    _copy_and_compare_state(new, old)
    batch = {
        "pid_x": torch.nn.functional.one_hot(
            torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
            num_classes=6,
        ).float(),
        "p4_x": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 50,
        "emb": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10,
    }

    pdg_out, feat_out = new(batch)
    assert pdg_out.shape == (2, 3, 6)
    assert feat_out.shape == (2, 3, 4)
    old_pdg, old_feat = old(batch)
    torch.testing.assert_close(pdg_out, old_pdg)
    torch.testing.assert_close(feat_out, old_feat)


def test_gpt_like_models_cpu_forward_shapes():
    particle = _particle_batch(features=4)
    particle_model = GPTReconstructor(
        tr_width=8,
        tr_n_head=1,
        tr_n=1,
        tr_hidden_size=16,
        dim_hyper=4,
        device="cpu",
    )
    gpt_batch = {
        "emb": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 100,
        "src_mask": torch.zeros((2, 4, 4), dtype=torch.float32),
        "lvl_code": torch.tensor([[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]]),
    }
    assert particle_model(gpt_batch).shape == (2, 4, 4)

    linker = EmbLinker(
        n_features=4,
        link_width=8,
        link_n_head=1,
        link_n_layers=1,
        link_fc=16,
        device="cpu",
    )
    link_batch = {
        "emb_x": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 100,
        "emb_y": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4) / 100,
        "padding_mask": particle["padding_mask"],
    }
    assert linker(link_batch).shape == (2, 3, 3)
