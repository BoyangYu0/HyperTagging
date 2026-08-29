from __future__ import annotations

import torch

from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair


def _save(
    path,
    *,
    encoder_value: float,
    reconstruction: bool,
    dtype: torch.dtype = torch.float32,
) -> None:
    encoder = {"weight": torch.tensor([encoder_value], dtype=dtype)}
    model = (
        {"encoder.weight": encoder["weight"], "decoder.weight": torch.ones(1)}
        if reconstruction
        else {"encoder.weight": encoder["weight"]}
    )
    torch.save(
        {
            "step": 7 if reconstruction else 3,
            "model_state_dict": model,
            "encoder_state_dict": encoder,
            "feature_specification": {"feature_spec_hash": "same"},
            "feature_contract": {"model_feature_contract_hash": "same-model"},
            "pid_vocabulary_version": "same",
            "config": {"pretrained_encoder": "pretraining.pt"},
        },
        path,
    )


def test_checkpoint_pair_reports_exact_frozen_encoder(tmp_path):
    pretraining = tmp_path / "pretraining.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    _save(pretraining, encoder_value=2.0, reconstruction=False)
    _save(reconstruction, encoder_value=2.0, reconstruction=True)

    report = validate_checkpoint_pair(pretraining, reconstruction)

    assert report.compatible
    assert report.encoder_exact_fraction == 1.0
    assert report.encoder_dtype_compatible_keys == 1
    assert report.pretraining_step == 3
    assert report.reconstruction_step == 7


def test_checkpoint_pair_rejects_changed_frozen_encoder(tmp_path):
    pretraining = tmp_path / "pretraining.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    _save(pretraining, encoder_value=2.0, reconstruction=False)
    _save(reconstruction, encoder_value=3.0, reconstruction=True)

    try:
        validate_checkpoint_pair(pretraining, reconstruction)
    except ValueError as error:
        assert "frozen encoder tensor equality" in str(error)
    else:
        raise AssertionError("changed encoder was accepted as frozen transfer")


def test_checkpoint_pair_rejects_equal_valued_encoder_with_different_dtype(tmp_path):
    pretraining = tmp_path / "pretraining.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    _save(
        pretraining,
        encoder_value=2.0,
        reconstruction=False,
        dtype=torch.float32,
    )
    _save(
        reconstruction,
        encoder_value=2.0,
        reconstruction=True,
        dtype=torch.float64,
    )

    try:
        validate_checkpoint_pair(pretraining, reconstruction)
    except ValueError as error:
        assert "encoder tensor dtypes" in str(error)
    else:
        raise AssertionError("dtype-mismatched encoder was accepted as exact")


def test_checkpoint_pair_allows_declared_finetuned_encoder(tmp_path):
    pretraining = tmp_path / "pretraining.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    _save(pretraining, encoder_value=2.0, reconstruction=False)
    _save(reconstruction, encoder_value=3.0, reconstruction=True)

    report = validate_checkpoint_pair(
        pretraining,
        reconstruction,
        require_exact_frozen_encoder=False,
    )

    assert report.compatible
    assert not report.exact_frozen_encoder_required
    assert report.encoder_exact_fraction == 0.0


def test_checkpoint_pair_rejects_mutually_missing_contract(tmp_path):
    pretraining = tmp_path / "pretraining.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    _save(pretraining, encoder_value=2.0, reconstruction=False)
    _save(reconstruction, encoder_value=2.0, reconstruction=True)
    for path in (pretraining, reconstruction):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("feature_contract")
        torch.save(payload, path)

    try:
        validate_checkpoint_pair(pretraining, reconstruction)
    except ValueError as error:
        assert "model feature contract" in str(error)
    else:
        raise AssertionError("mutually missing model contract was accepted")
