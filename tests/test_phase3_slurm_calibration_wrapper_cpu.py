from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_calibration_module():
    path = ROOT / "scripts" / "run_phase3_batch_efficiency_calibration.py"
    spec = importlib.util.spec_from_file_location("phase3_calibration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_wrapper_is_gpu_only_and_exactly_preflights_allocation():
    wrapper = (ROOT / "scripts/slurm/run_phase3_batch_efficiency_calibration.sbatch").read_text()
    assert "preflight_gpu_environment.py" in wrapper
    assert "run_phase3_batch_efficiency_calibration.py" in wrapper
    assert "run_phase3_batch_efficiency_stability_pilot.py" in wrapper
    assert '"${gpu_environment}/bin/python"' in wrapper
    assert "--pilot-command" in wrapper
    assert "--max-steps 256" in wrapper
    assert 'HT_PHASE3_EXPECTED_GRES}' in wrapper
    assert "sbatch" not in wrapper
    assert "srun" not in wrapper
    assert "scancel" not in wrapper
    assert "--device cpu" not in wrapper


def test_checkpoint_probe_requires_loadable_finite_tensors(tmp_path):
    module = _load_calibration_module()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state": torch.ones(3)}, checkpoint)
    evidence = module._load_checkpoint_finite(checkpoint)
    assert evidence["loadable"] is True
    assert evidence["finite_tensors"] is True
    assert evidence["tensor_count"] == 1
    assert evidence["tensor_numel"] == 3


def test_submit_command_binds_exact_tuple_and_paths():
    path = ROOT / "scripts/slurm/submit_phase3_batch_efficiency_calibration.py"
    spec = importlib.util.spec_from_file_location("phase3_submit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.load_study_plan(module.DEFAULT_PLAN, root=ROOT)
    entry = module.entry_by_id(plan, "ht3-cal-v100-b64-20260823")
    command, output, error = module.build_sbatch_command(
        entry,
        plan_path=module.DEFAULT_PLAN,
        gpu_environment="/frozen/gpu-env",
        submitted_epoch=1_787_520_000,
        token="a" * 64,
    )
    assert "--gres=gpu:v100:1" in command
    assert "--partition=inter" in command
    assert "--no-requeue" in command
    assert command[-1].endswith("scripts/slurm/run_phase3_batch_efficiency_calibration.sbatch")
    assert output.name == "stdout-%j.log"
    assert error.name == "stderr-%j.log"
    export = next(value for value in command if value.startswith("--export="))
    assert "HT_PHASE3_CALIBRATION_ID=ht3-cal-v100-b64-20260823" in export
    assert "HT_PHASE3_EXPECTED_GRES=gpu:v100:1" in export
