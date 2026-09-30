"""Check radial reconditioning and parameter-only transfer without source mutation."""

import hashlib
import pytest
import torch
from scripts.phase64_projection_initialization import (
    create_projection_initialization,
    TARGETS,
)
from hypertagging.training.pretrain_trainer import initialize_pretraining_parameters
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.models.hyperbolic import bound_tangent_norm


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.hyper_projection = torch.nn.Linear(3, 2)
        self.other = torch.nn.Linear(2, 2)
        self.register_buffer("normalizer", torch.tensor([5.0]))


def source(tmp_path):
    m = Model()
    p = {
        "architecture": {"test": 1},
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_specification": feature_spec_v4(),
        "preprocessing_schema_version": "direct-mdst-tree-v4",
        "model_state_dict": m.state_dict(),
        "step": 2188,
        "optimizer_state_dict": {"must_not_restore": True},
    }
    path = tmp_path / "source.pt"
    torch.save(p, path)
    return path, p, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("scale", [1.0, 0.02])
def test_initialization_changes_only_projection_and_preserves_fresh_buffers(
    tmp_path, scale
):
    path, p, digest = source(tmp_path)
    out = tmp_path / "derived.pt"
    record = create_projection_initialization(
        path, out, scale=scale, expected_sha256=digest
    )
    loaded = torch.load(out, weights_only=False)
    assert "optimizer_state_dict" not in loaded
    model = Model()
    model.normalizer.fill_(17.0)
    initialize_pretraining_parameters(model, out, architecture={"test": 1})
    for name, param in model.named_parameters():
        assert torch.equal(
            param, p["model_state_dict"][name] * (scale if name in TARGETS else 1)
        )
    assert model.normalizer.item() == 17
    assert (
        record["parameter_initialization_only"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    )
    with pytest.raises(FileExistsError):
        create_projection_initialization(path, out, scale=scale, expected_sha256=digest)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), 1.01])
def test_invalid_scale_rejected_before_output(tmp_path, scale):
    path, _, digest = source(tmp_path)
    out = tmp_path / "derived.pt"
    with pytest.raises(ValueError):
        create_projection_initialization(path, out, scale=scale, expected_sha256=digest)
    assert not out.exists()


def test_source_binding_rejected(tmp_path):
    path, _, _ = source(tmp_path)
    out = tmp_path / "derived.pt"
    with pytest.raises(ValueError, match="hash"):
        create_projection_initialization(
            path, out, scale=0.02, expected_sha256="0" * 64
        )
    assert not out.exists()


def test_rescaling_restores_radial_gradient_and_preserves_direction():
    raw = torch.tensor([[30.0, 40.0]], requires_grad=True)
    saturated = bound_tangent_norm(raw, maximum=1.5)
    grad0 = torch.autograd.grad(saturated.norm(), raw, retain_graph=True)[0]
    scaled = bound_tangent_norm(raw * 0.015, maximum=1.5)
    grad1 = torch.autograd.grad(scaled.norm(), raw)[0]
    assert grad0.norm() < 1e-6 and grad1.norm() > 0.01
    assert torch.allclose(
        torch.nn.functional.normalize(saturated, dim=-1),
        torch.nn.functional.normalize(scaled, dim=-1),
    )


@pytest.mark.parametrize("scale", [1.0, 0.02])
def test_scaled_warm_start_completes_actual_pretraining_update(tmp_path, scale):
    from dataclasses import replace
    from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
    from hypertagging.training.pretrain_trainer import (
        PretrainConfig,
        train_hyperbolic_pretraining,
    )

    data = write_notebook_fixture_v4(tmp_path / "fixture.parquet")
    c = PretrainConfig(
        data=str(data),
        output_dir=str(tmp_path / "source-run"),
        max_steps=1,
        batch_size=2,
        device="cpu",
        mixed_precision=False,
        validate_every=1,
        validation_events=2,
        validation_batches=1,
        log_every=1,
    )
    initial = train_hyperbolic_pretraining(c)
    digest = hashlib.sha256(initial.checkpoint.read_bytes()).hexdigest()
    derived = tmp_path / "parameter-only.pt"
    create_projection_initialization(
        initial.checkpoint, derived, scale=scale, expected_sha256=digest
    )
    result = train_hyperbolic_pretraining(
        replace(
            c,
            output_dir=str(tmp_path / "refinement"),
            weights_initialization_checkpoint=str(derived),
        )
    )
    assert result.steps == 1
    payload = torch.load(result.checkpoint, weights_only=False)
    assert payload["step"] == 1 and all(
        torch.isfinite(v).all() for v in payload["model_state_dict"].values()
    )
    assert hashlib.sha256(initial.checkpoint.read_bytes()).hexdigest() == digest
