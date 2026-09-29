import pytest

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
import hypertagging.training.reconstruction_trainer as trainer


@pytest.mark.parametrize("policy", ["masked_representable_only", "recovery_objective"])
@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), 0., 2.])
def test_weight_admission_reaches_data_only_for_valid_policy(monkeypatch, policy, weight):
    class ReachedData(Exception):
        pass

    def stop_at_data(*args, **kwargs):
        raise ReachedData

    monkeypatch.setattr(trainer, "build_real_data_module", stop_at_data)
    config = trainer.ReconstructionConfig(data="unused", output_dir="unused",
        unrepresentable_target_policy=policy, recovery_objective_weight=weight)
    valid = weight == 2. or (weight == 0. and policy == "masked_representable_only")
    with pytest.raises(ReachedData if valid else ValueError):
        trainer.train_level_reconstruction(config)


def test_masking_only_zero_weight_completes_optimizer_step(tmp_path):
    data = write_notebook_fixture_v3(tmp_path / "fixture.parquet")
    result = trainer.train_level_reconstruction(trainer.ReconstructionConfig(
        data=str(data), output_dir=str(tmp_path / "run"), max_steps=1,
        allow_legacy_conflated=True, validate_every=1, rollout_validate_every=1,
        unrepresentable_target_policy="masked_representable_only",
        recovery_objective_weight=0., scheduled_sampling_probability=1.,
        scheduled_sampling_schedule="constant",
    ))
    assert result.steps == 1
    assert result.checkpoint.is_file()


@pytest.mark.parametrize("mutation", [None, "config", "trained", "source", "scheduler", "receipt"])
def test_retry_guard_keeps_scientific_contract_and_zero_step_boundary(tmp_path, monkeypatch, mutation):
    import json
    from pathlib import Path
    from scripts.slurm import submit_reconstruction_phase63_campaign as submit
    from scripts.run_reconstruction_phase63 import canonical_contract_hash

    old_root = tmp_path / "old"
    receipt_path = old_root / "artifacts/slurm/reconstruction-phase63/jobs/16742233/attempt-00/receipt.json"
    receipt_path.parent.mkdir(parents=True)
    keys = ("arm_role", "study_id", "config", "data", "cohort", "checkpoint", "checkpoint_sha256",
            "checkpoint_step", "preregistration", "pretraining_refinement", "evaluation_contract",
            "post_training_gates", "resources", "output_root", "gpu_environment")
    old = {key: {} for key in keys}
    old.update(arm_role="masked_only", output_root="artifacts/runs/masked_only", expected_git_sha="old")
    old["contract_sha256"] = canonical_contract_hash(old)
    (receipt_path.parent / "submitted-contract.json").write_text(json.dumps(old))
    receipt = dict(slurm_job_id="16742233", arm_role="masked_only", status="failed", exit_status=1,
                   terminal_stage="trainer", sealed_test_accessed=False,
                   artifacts={"training_result": None, "full_decay_gates": None},
                   contract_sha256=old["contract_sha256"])
    receipt["receipt_sha256"] = submit.receipt_hash(receipt)
    if mutation == "receipt":
        receipt["exit_status"] = 0
    receipt_path.write_text(json.dumps(receipt))
    (old_root / "artifacts/slurm/phase63-masked_only-20260928-16742233.err").write_text(
        "ValueError: recovery_objective_weight must be finite and positive")
    trainer_path = "src/hypertagging/training/reconstruction_trainer.py"
    after = (Path(submit.ROOT) / trainer_path).read_text().strip()
    before = after.replace('or config.recovery_objective_weight < 0\n        or (config.unrepresentable_target_policy == "recovery_objective"\n            and config.recovery_objective_weight == 0)',
                           'or config.recovery_objective_weight <= 0').replace(
        'raise ValueError("recovery_objective_weight must be finite and nonnegative, and positive when recovery_objective is active")',
        'raise ValueError("recovery_objective_weight must be finite and positive")')
    def fake_run(command):
        if command[0].endswith("sacct"):
            return "16742233|COMPLETED|0:0" if mutation == "scheduler" else "16742233|FAILED|1:0"
        if command[1] == "diff":
            return trainer_path + ("\nsrc/changed.py" if mutation == "source" else "")
        return before
    monkeypatch.setattr(submit, "run", fake_run)
    current = dict(old)
    if mutation == "config":
        current["config"] = {"seed": 999}
    if mutation == "trained":
        (old_root / old["output_root"] / "16742233/training").mkdir(parents=True)
    if mutation:
        with pytest.raises(RuntimeError):
            submit.validate_failed_arm_retry(current, receipt_path)
    else:
        assert submit.validate_failed_arm_retry(current, receipt_path)["optimizer_steps_before_retry"] == 0


@pytest.mark.parametrize("arm_index", [0, 1])
def test_actual_phase63_config_passes_trainer_admission(tmp_path, monkeypatch, arm_index):
    import json
    from dataclasses import replace
    from pathlib import Path
    from scripts.run_reconstruction_phase35 import _training_config
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "configs/reconstruction/ht_reconstruction_phase63_20260928.json").read_text())
    arm = p["arms"][arm_index]
    cfg = _training_config(config={**p["common_training_contract"], **arm["overrides"]},
        runtime={"checkpoint": "unused", "selection_manifest": "unused", "dataset_index": "unused"},
        training_output=tmp_path, validation_exclusions=())
    class ReachedData(Exception):
        pass
    def stop_at_data(*args, **kwargs):
        raise ReachedData
    monkeypatch.setattr(trainer, "build_real_data_module", stop_at_data)
    with pytest.raises(ReachedData):
        trainer.train_level_reconstruction(replace(cfg, device="cpu", grad_scaler_enabled=False))
