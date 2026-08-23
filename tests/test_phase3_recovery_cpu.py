"""Focused, read-only evidence tests for the production-1M recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from hypertagging.training.pretrain_trainer import PretrainConfig, _resolve_phase_schedule
from hypertagging.training.pretraining_curriculum import ProgressivePhaseSchedule

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CHECKPOINT_SHA256 = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
OLD_CONTRACT = ROOT / (
    "artifacts/slurm/ht-pretrain-production-1m-h100-20260821."
    "operator-authorized.job-contract.json"
)


def _finite_tensors(value: object, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                bad.append(path)
        return bad
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(_finite_tensors(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(_finite_tensors(item, f"{path}[{index}]"))
    return bad


def test_immutable_failed_lineage_and_checkpoint_are_readable_and_finite():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    assert CHECKPOINT.stat().st_size == 19_371_763
    assert hashlib.sha256(OLD_CONTRACT.read_bytes()).hexdigest() == (
        "8dfa6b2320c8992e69c68f7d570bcb0e562306b928be57c2ece0c8f8626f5a0d"
    )

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["step"] == 54064
    assert payload["training_state"]["curriculum_phase_cursor"] == {
        "completed_optimizer_steps": 54064,
        "events_completed": 865016,
        "phase_index": 1,
        "final_phase_entered": False,
    }
    assert payload["streaming_cursor"]["epoch"] == 1
    assert payload["streaming_cursor"]["batch_index"] == 1
    for section in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "normalizer_state",
        "training_state",
        "streaming_cursor",
    ):
        assert _finite_tensors(payload[section], section) == []


def test_failed_metrics_prove_validation_boundary_and_no_restart():
    metrics_path = ROOT / (
        "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
        "15933802/metrics.jsonl"
    )
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(records) == 2176
    assert max(int(record["step"]) for record in records) == 54275
    validation = next(record for record in records if record.get("step") == 54064)
    assert validation["validation_events"] == 5000.0
    assert validation["validation_batches"] == 313.0
    assert validation["validation_full_training_objective"] == pytest.approx(
        8.664798735239254
    )
    assert validation["validation_principal_loss"] == pytest.approx(
        8.494534350955448
    )
    assert validation["validation_relation_accuracy"] == pytest.approx(
        0.8384483626570565
    )
    assert validation["validation_parent_ranking_accuracy"] == pytest.approx(
        0.6717389677755368
    )
    assert validation["validation_leaf_pid_accuracy"] == pytest.approx(
        0.8452743453720507
    )
    receipt = json.loads(
        (ROOT / "artifacts/slurm/jobs/15933802/attempt-00/receipt.json").read_text()
    )
    assert receipt["slurm"]["restart_count"] == "0"
    assert receipt["wrapper"]["restart_count"] == 0
    assert receipt["wrapper"]["usr1_received"] == 0
    assert receipt["wrapper"]["termination_received"] == 0
    assert receipt["terminal_stage"] == "trainer_failed"
    assert receipt["trainer_status"] == 1


def test_recovery_config_changes_only_late_leaf_pid_taper_and_preserves_gates():
    old = yaml.safe_load(
        (ROOT / "configs/slurm/pretrain_1m_h100_20260821.yaml").read_text()
    )
    recovery = yaml.safe_load(
        (ROOT / "configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml").read_text()
    )
    assert {
        key: value for key, value in recovery.items() if key != "leaf_pid_phase_weights"
    } == {key: value for key, value in old.items() if key != "leaf_pid_phase_weights"}
    assert old["leaf_pid_phase_weights"] == [1.0, 1.0, 0.5, 0.5]
    assert recovery["leaf_pid_phase_weights"] == [1.0, 1.0, 0.4, 0.4]
    assert recovery["objective_dominance_ratio"] == 20.0
    assert recovery["pilot_objective_violation_action"] == "fail"
    assert recovery["curriculum_phase_steps"] == [27032] * 4
    assert recovery["validate_every"] == recovery["checkpoint_every"] == 13516
    assert recovery["validation_events"] == 5000
    assert recovery["validation_batches"] == 313
    assert recovery["channel_zero_positive_action"] == "fail"


def test_exact_resume_enters_phase3_at_next_optimizer_step():
    schedule = ProgressivePhaseSchedule(
        unit="optimizer_step", durations=(27032, 27032, 27032, 27032)
    )
    assert schedule.phase_index(step=54064, events=865016) == 2
    assert schedule.phase(step=54064, events=865016).name == "multilevel_channel_memory"
    assert schedule.phase(step=54063, events=865000).name == "truth_guided_distance_radius"
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    resumed = _resolve_phase_schedule(
        PretrainConfig(data="unused", output_dir="unused", max_steps=108128),
        payload,
    )
    assert resumed.phase_index(step=54064, events=865016) == 2
    assert resumed.contract() == schedule.contract()


def test_recovery_contract_constants_protect_output_isolation_and_roles():
    renderer = (ROOT / "scripts/slurm/render_phase3_recovery_job.py").read_text()
    verifier = (ROOT / "scripts/slurm/verify_phase3_recovery_contract.py").read_text()
    assert "15933802" in renderer and "15933802" in verifier
    assert "checkpoint-step-54064.pt" in renderer
    assert CHECKPOINT_SHA256 in renderer and CHECKPOINT_SHA256 in verifier
    assert "ht-pretrain-1m-phase3-recovery-20260823" in renderer
    assert "replacement_attempt_root_must_not_be" in verifier
    assert "sealed_test_role_access" in verifier
    assert "gpu:h100nvl:1" in renderer
    assert "render_one_gpu_job.py" in renderer
    assert "subprocess.run(command" in renderer
