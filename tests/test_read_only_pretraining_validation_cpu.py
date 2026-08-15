from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from hypertagging.evaluation import pretraining_validation as evaluation
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.training.pretrain_trainer import PretrainConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm import verify_pretraining_validation_contract as verifier  # noqa: E402


class _DataModule:
    selection_manifest_hash = "selection-hash"
    split_manifest_hash = "split-hash"
    split_counts = {"train": 35_000, "validation": 50_000, "test": 0}
    source_schema_versions = ("direct-mdst-tree-v4",)
    dataset_index = {"index_hash": "index-hash"}

    def __init__(self):
        self.events = [
            SimpleNamespace(event_uid=f"validation-event-{index}")
            for index in range(evaluation.FIXED_VALIDATION_EVENTS)
        ]

    def iter_events(self, split, *, shuffle):
        assert split == "validation"
        assert shuffle is False
        return iter(self.events)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def _payload():
    config = PretrainConfig(
        data="selection.json",
        dataset_index="index.json",
        output_dir="training-output",
        scientific_mode=True,
        seed=20260812,
        batch_size=16,
        validation_events=512,
    )
    return {
        "step": 3282,
        "git_commit": "b8579096655108943e4e22626cfbcc2cbfec6737",
        "config": asdict(config),
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "normalizer_state": {
            name: {} for name in ("track", "cluster", "common", "composite")
        },
        "feature_specification": feature_spec_v4(),
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "preprocessing_schema_version": "direct-mdst-tree-v4",
        "split_manifest_hash": "split-hash",
        "data_order_contract": {"dataset_index_hash": "index-hash"},
        "validation_selection": {
            "split": "validation",
            "strategy": "manifest_validation_role_uid_hash",
            "selection_manifest_hash": "selection-hash",
        },
        "training_state": {
            "checkpoint_selection_reason": {
                "metric_name": "validation_full_training_objective",
                "mode": "min",
                "reason": "new_principal_configured_checkpoint",
            }
        },
    }


def test_read_only_validation_binds_exact_hash_cohort_and_preserves_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "training" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"immutable trusted checkpoint fixture")
    data = tmp_path / "selection.json"
    index = tmp_path / "index.json"
    data.write_text("{}\n")
    index.write_text("{}\n")
    output = tmp_path / "evaluation" / "result.json"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    module = _DataModule()
    monkeypatch.setattr(evaluation, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        evaluation, "load_training_checkpoint", lambda *_a, **_k: _payload()
    )
    monkeypatch.setattr(evaluation, "build_real_data_module", lambda *_a, **_k: module)
    monkeypatch.setattr(evaluation, "_build_model", lambda *_a, **_k: _Model())

    def fake_validate(_model, _module, *, selected_event_uids, **_kwargs):
        assert len(selected_event_uids) == evaluation.FIXED_VALIDATION_EVENTS
        assert len(set(selected_event_uids)) == evaluation.FIXED_VALIDATION_EVENTS
        return {
            "validation_events": float(evaluation.FIXED_VALIDATION_EVENTS),
            "validation_batches": 125.0,
            "validation_full_training_objective": 1.25,
            "validation_seconds": 2.0,
        }

    monkeypatch.setattr(evaluation, "_validate_pretraining", fake_validate)
    result = evaluation.evaluate_pretraining_checkpoint(
        checkpoint=checkpoint,
        data=data,
        dataset_index=index,
        output=output,
        device="cpu",
        expected_checkpoint_sha256=checkpoint_sha,
        expected_source_git_sha="a" * 40,
    )
    assert result["optimizer_steps"] == 0
    assert result["cohort"]["event_count"] == 2000
    assert len(result["cohort"]["selected_uid_hashes"]) == 2000
    assert result["checkpoint"]["unchanged"] is True
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    assert output.is_file()
    with pytest.raises(FileExistsError):
        evaluation.evaluate_pretraining_checkpoint(
            checkpoint=checkpoint,
            data=data,
            dataset_index=index,
            output=output,
            device="cpu",
            expected_checkpoint_sha256=checkpoint_sha,
            expected_source_git_sha="a" * 40,
        )


def test_read_only_validation_rejects_train_role_fallback():
    payload = _payload()
    payload["validation_selection"]["split"] = "train"
    with pytest.raises(ValueError, match="validation role"):
        evaluation._checkpoint_config(
            payload,
            data=Path("selection.json"),
            dataset_index=Path("index.json"),
            device=torch.device("cpu"),
        )


def test_slurm_contract_is_exact_and_bootstraps_without_torch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "training" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"best checkpoint")
    data = tmp_path / "selection.json"
    index = tmp_path / "index.json"
    data.write_text("{}\n")
    index.write_text("{}\n")
    gpu_env = tmp_path / "gpu-env" / "bin"
    gpu_env.mkdir(parents=True)
    (gpu_env / "python").write_text("")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    contract = {
        "contract_version": verifier.CONTRACT_VERSION,
        "mode": "read_only_pretraining_validation",
        "evaluation_role": "validation",
        "sealed_test_role_access": "forbidden",
        "validation_events": 2000,
        "optimizer_steps": 0,
        "checkpoint_step": 3282,
        "gres": "gpu:h100nvl:1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "expected_git_sha": "a" * 40,
        "checkpoint": str(checkpoint.relative_to(tmp_path)),
        "selection_manifest": str(data.relative_to(tmp_path)),
        "dataset_index": str(index.relative_to(tmp_path)),
        "evaluation_output_base": "artifacts/evaluations/fixed-2000",
        "gpu_environment": str(gpu_env.parent),
    }
    runtime = verifier.validated_runtime_values(contract)
    assert runtime["checkpoint_step"] == "3282"
    for field, value in (
        ("evaluation_role", "train"),
        ("validation_events", 1999),
        ("optimizer_steps", 1),
        ("checkpoint_step", 4376),
        ("gres", "gpu:1"),
    ):
        with pytest.raises(RuntimeError):
            verifier.validated_runtime_values({**contract, field: value})

    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "import importlib.util,sys;"
                "spec=importlib.util.spec_from_file_location('verify_eval',"
                "'scripts/slurm/verify_pretraining_validation_contract.py');"
                "module=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module);"
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
