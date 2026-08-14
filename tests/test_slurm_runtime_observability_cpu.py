from pathlib import Path
import json
import subprocess
import sys

from hypertagging.training.pretrain_trainer import _validation_progress_record
from scripts.slurm.verify_execution_receipt import verify_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_validation_progress_record_exposes_counts_elapsed_and_throughput(monkeypatch):
    monkeypatch.setattr(
        "hypertagging.training.pretrain_trainer.time.monotonic", lambda: 14.0
    )
    record = _validation_progress_record(
        started=10.0,
        batch_index=2,
        batch_count=3,
        view_index=1,
        view_count=2,
        view_name="fsp_only",
        completed_event_views=16,
        total_events=24,
    )
    assert record == {
        "event": "validation_progress",
        "elapsed_seconds": 4.0,
        "validation_batch": 2,
        "validation_batches": 3,
        "validation_view": "fsp_only",
        "validation_view_index": 1,
        "validation_views": 2,
        "validation_events": 24,
        "completed_event_views": 16,
        "event_view_throughput_per_second": 4.0,
    }


def test_periodic_gpu_monitor_writes_samples_and_peak_summary(tmp_path):
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '0, GPU-test, Tesla V100, 32768, 1234, 42, 8, 57'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    telemetry = tmp_path / "telemetry.jsonl"
    summary = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/slurm/monitor_gpu_telemetry.py",
            "--output",
            str(telemetry),
            "--summary",
            str(summary),
            "--interval-seconds",
            "0.01",
            "--max-samples",
            "2",
            "--nvidia-smi",
            str(fake_smi),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(samples) == 2
    assert all(sample["timestamp"].endswith("+00:00") for sample in samples)
    payload = json.loads(summary.read_text())
    assert payload["status"] == "completed"
    assert payload["sample_count"] == 2
    assert payload["peak_memory_used_mib"] == 1234
    assert payload["peak_gpu_utilization_percent"] == 42
    assert payload["peak_temperature_c"] == 57


def test_small_candidate_diagnostic_is_strictly_bounded():
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/slurm/pretrain_diagnostic_small_candidate.yaml").read_text()
    )
    assert config["model_preset"] == "small_candidate"
    assert config["max_steps"] == 4
    assert config["curriculum_phase_steps"] == [1, 1, 1, 1]
    assert config["validation_batches"] == 1
    assert 0 < config["validation_events"] <= 32
    assert config["checkpoint_every"] <= config["max_steps"]
    assert config["scientific_mode"] is False


def test_sbatch_starts_cleans_and_receipts_periodic_telemetry():
    sbatch = (ROOT / "scripts/slurm/train_one_gpu.sbatch").read_text()
    finalizer = (ROOT / "scripts/slurm/finalize_execution_receipt.py").read_text()
    verifier = (ROOT / "scripts/slurm/verify_execution_receipt.py").read_text()
    assert "monitor_gpu_telemetry.py" in sbatch
    assert "stop_gpu_telemetry" in sbatch
    assert "trap finalize_attempt EXIT" in sbatch
    assert "gpu-telemetry.jsonl" in sbatch + finalizer
    assert "gpu-telemetry-summary.json" in sbatch + finalizer
    assert "peak_memory_used_mib" in finalizer
    assert "healthy periodic GPU telemetry" in verifier


def test_v2_completed_receipt_hashes_telemetry_and_exposes_peaks(tmp_path):
    attempt = tmp_path / "attempt-00"
    run = tmp_path / "run"
    attempt.mkdir()
    run.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text('{}\n')
    for name, content in (
        ("stages.log", "stage=trainer_complete\n"),
        ("allocation.txt", "allocation\n"),
        ("gpu-preflight.json", '{}\n'),
        ("gpu-telemetry.jsonl", '{"memory_used_mib":2048}\n'),
    ):
        (attempt / name).write_text(content)
    (attempt / "wrapper-status.json").write_text(
        json.dumps(
            {
                "action": "trainer_exit",
                "trainer_status": 0,
                "wrapper_status": 0,
            }
        )
    )
    (attempt / "gpu-telemetry-summary.json").write_text(
        json.dumps(
            {
                "telemetry_version": "hypertagging-slurm-gpu-telemetry-v1",
                "status": "completed",
                "started_at": "2026-08-14T10:00:00+00:00",
                "completed_at": "2026-08-14T10:01:00+00:00",
                "interval_seconds": 15,
                "sample_count": 4,
                "peak_memory_used_mib": 2048,
                "peak_gpu_utilization_percent": 73,
                "peak_temperature_c": 61,
                "error": None,
            }
        )
    )
    (run / "checkpoint.pt").write_bytes(b"checkpoint")
    (run / "metrics.jsonl").write_text('{"loss":1.0}\n')
    receipt = attempt / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/slurm/finalize_execution_receipt.py",
            "--receipt",
            str(receipt),
            "--contract",
            str(contract),
            "--attempt-root",
            str(attempt),
            "--run-root",
            str(run),
            "--stage-log",
            str(attempt / "stages.log"),
            "--wrapper-status",
            str(attempt / "wrapper-status.json"),
            "--batch-exit-status",
            "0",
            "--terminal-stage",
            "trainer_complete",
            "--started-at",
            "2026-08-14T10:00:00+00:00",
            "--completed-at",
            "2026-08-14T10:01:00+00:00",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
    payload = verify_receipt(receipt, require_completed=True)
    assert payload["receipt_version"] == "hypertagging-slurm-attempt-v2"
    assert payload["gpu_telemetry"]["peak_memory_used_mib"] == 2048
    assert payload["gpu_telemetry"]["peak_gpu_utilization_percent"] == 73
