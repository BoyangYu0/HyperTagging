from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hypertagging.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)
from hypertagging.training.learning_rate import (
    build_warmup_cosine_scheduler,
    learning_rate_schedule_contract,
    lr_multiplier,
    step_scheduler_at_virtual_step,
)
from hypertagging.training.pretrain_trainer import _resolve_amp_dtype
from hypertagging.training.pretrain_trainer import PretrainConfig, _resolve_phase_schedule
from hypertagging.training.presentation_progress import (
    PHASE3_RESUME_PRESENTATIONS,
    PHASE3_VALIDATION_PRESENTATIONS,
    crossed_milestones,
    phase_index_for_virtual_step,
    progress_from_checkpoint,
    validate_batch_profile,
)
from hypertagging.training.device_profiles import get_device_profile
from hypertagging.training.pretraining_curriculum import DEFAULT_PRETRAINING_PHASES


def test_phase3_batch64_preserves_exact_remaining_presentations_and_milestones():
    validate_batch_profile(
        64,
        total_presentations=1_730_048,
        milestone_presentations=PHASE3_VALIDATION_PRESENTATIONS,
    )
    assert PHASE3_RESUME_PRESENTATIONS == 865_024
    assert 865_024 // 64 == 13_516
    assert crossed_milestones(
        PHASE3_RESUME_PRESENTATIONS,
        1_730_048,
        PHASE3_VALIDATION_PRESENTATIONS,
    ) == PHASE3_VALIDATION_PRESENTATIONS[4:]
    with pytest.raises(ValueError, match="divisible"):
        validate_batch_profile(
            48,
            total_presentations=1_730_048,
            milestone_presentations=PHASE3_VALIDATION_PRESENTATIONS,
        )


def test_virtual_phase_mapping_is_distinct_from_physical_optimizer_updates():
    assert phase_index_for_virtual_step(54_064) == 2
    assert phase_index_for_virtual_step(81_096) == 3


def test_new_checkpoint_virtual_phase_contract_can_resume_again():
    config = PretrainConfig(
        data="unused",
        output_dir="unused",
        presentation_total_presentations=1_730_048,
        presentation_phase_presentations=(432_512,) * 4,
        batch_size=64,
    )
    stored = {
        "training_state": {
            "curriculum_schedule_contract": {
                "version": "progressive-pretraining-phases-v1",
                "mode": "progressive",
                "unit": "presentation_virtual_step",
                "durations": [27_032] * 4,
                "total_budget": 108_128,
                "phases": [
                    {
                        "name": phase.name,
                        "view": phase.view.value,
                        "fraction": phase.fraction,
                        "objectives": list(phase.objectives),
                    }
                    for phase in DEFAULT_PRETRAINING_PHASES
                ],
            }
        }
    }
    schedule = _resolve_phase_schedule(config, stored)
    assert schedule.unit == "presentation_virtual_step"
    assert schedule.total_budget == 108_128


def test_legacy_step16_checkpoint_migrates_to_presentation_progress():
    payload = {
        "step": 54_064,
        "config": {"batch_size": 16},
        "training_state": {},
    }
    progress = progress_from_checkpoint(
        payload,
        target_batch_size=64,
        total_presentations=1_730_048,
        milestone_presentations=PHASE3_VALIDATION_PRESENTATIONS,
    )
    assert progress.presentations == 865_024
    assert progress.virtual_step == 54_064
    assert progress.optimizer_steps == 54_064


def test_scheduler_uses_virtual_step_after_batch_migration():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=5e-4)
    contract = learning_rate_schedule_contract(
        total_steps=108_128,
        warmup_fraction=0.1,
        max_warmup_steps=10_000,
        min_lr_ratio=0.1,
        base_lrs=[5e-4],
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, contract)
    step_scheduler_at_virtual_step(scheduler, 54_064)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        5e-4 * lr_multiplier(54_064, contract)
    )
    step_scheduler_at_virtual_step(scheduler, 54_068)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        5e-4 * lr_multiplier(54_068, contract)
    )


def test_checkpoint_batch_migration_preserves_optimizer_and_records_migration(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    path = save_training_checkpoint(
        tmp_path / "source.pt",
        model=model,
        optimizer=optimizer,
        step=54_064,
        data_order_contract={
            "batch_size": 16,
            "shuffle_buffer_size": 1024,
            "dataset_index_hash": "same",
        },
        training_state={"lr_schedule_contract": {"version": "test"}},
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=5e-4)
    payload = restore_training_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_data_order_contract={
            "batch_size": 64,
            "shuffle_buffer_size": 1024,
            "dataset_index_hash": "same",
        },
        allow_batch_size_migration=True,
        restore_random_states=False,
    )
    assert payload["training_state"]["checkpoint_load_migrations"][0]["kind"] == (
        "presentation_batch_size_migration_v1"
    )
    assert replacement_optimizer.state_dict()["state"] == optimizer.state_dict()["state"]


def test_precision_policies_fail_closed_without_cuda_execution():
    assert _resolve_amp_dtype(
        device=torch.device("cuda"),
        mixed_precision=True,
        amp_dtype="bfloat16",
        cuda_bf16_supported=True,
    ) is torch.bfloat16
    with pytest.raises(RuntimeError, match="not supported"):
        _resolve_amp_dtype(
            device=torch.device("cuda"),
            mixed_precision=True,
            amp_dtype="bfloat16",
            cuda_bf16_supported=False,
        )
    assert get_device_profile("h100nvl").grad_scaler_enabled is False
    assert get_device_profile("v100").amp_dtype == "float16"
    assert get_device_profile("v100").grad_scaler_enabled is True


def test_readiness_and_calibration_contracts_are_fail_closed_and_static():
    root = Path(__file__).resolve().parents[1]
    readiness = json.loads(
        (root / "artifacts/codex/ht_pretraining_1m_batch_efficiency_readiness_20260823.json").read_text()
    )
    assert readiness["calibration"]["gpu_calibration_completed"] is False
    assert readiness["submission"]["production_submission_authorized"] is False
    assert readiness["submission"]["submission_performed"] is False
    assert readiness["scientific_contract"]["scientific_slurm_submission_allowed"] is False
    harness = (root / "scripts/run_phase3_batch_efficiency_calibration.py").read_text()
    assert "HT_PHASE3_ALLOCATION_GRES" in harness
    assert "source_unchanged" in harness
    assert "objective dominance ratio exceeded fail-closed limit 20.0" in harness
    assert "sealed_test_access" in harness
    spec = importlib.util.spec_from_file_location(
        "phase3_readiness_verifier",
        root / "scripts/verify_phase3_batch_efficiency_readiness.py",
    )
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(
        root / "artifacts/codex/ht_pretraining_1m_batch_efficiency_readiness_20260823.json"
    )


def test_profile_configs_preserve_exact_validation_accounting():
    root = Path(__file__).resolve().parents[1]
    for name, amp in (("h100nvl", "bfloat16"), ("v100", "float16")):
        text = (root / f"configs/slurm/pretrain_1m_phase3_batch_efficiency_{name}_20260823.yaml").read_text()
        assert "batch_size: 64" in text
        assert "presentation_total_presentations: 1730048" in text
        assert "validation_batches: 79" in text
        assert "curriculum_phase_steps: [27032, 27032, 27032, 27032]" in text
        assert f"amp_dtype: {amp}" in text


def test_selection_requires_an_exact_receipt_aggregation():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "phase3_selector", root / "scripts/select_phase3_batch_efficiency_profile.py"
    )
    assert spec and spec.loader
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    with pytest.raises(SystemExit):
        selector.main(["--output", str(root / "artifacts/codex/unused-selection.json")])
