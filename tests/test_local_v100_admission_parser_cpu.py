from types import SimpleNamespace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm.v100_local_admission import (  # noqa: E402
    _run,
    evaluate_sample,
    evaluate_watchdog_sample,
    parse_gpu_row,
    run_monitored,
)


def test_v100_admission_parser_accepts_threshold_boundary_except_temperature():
    gpu = parse_gpu_row("0, GPU-1, Tesla V100, 32768, 512, 5, 5, 69\n")
    admitted, failures = evaluate_sample(
        gpu, compute_apps="", pmon="# header\n", fuser="", queue=""
    )
    assert admitted
    assert failures == []


def test_v100_admission_parser_rejects_processes_and_hot_gpu():
    gpu = parse_gpu_row("0, GPU-1, Tesla V100, 32768, 10, 0, 0, 70\n")
    admitted, failures = evaluate_sample(
        gpu,
        compute_apps="GPU-1, 1234, python, 100",
        pmon="# header\n0 1234 C python 10 0 0 0 0",
        fuser="/dev/nvidia0: 1234",
        queue="22 inter train RUNNING",
    )
    assert not admitted
    assert {
        "temperature_not_below_70_c",
        "compute_process_present",
        "pmon_process_present",
        "device_owner_present",
        "user_queue_nonempty",
    }.issubset(failures)


def test_watchdog_allows_trainer_load_but_rejects_runtime_safety_threshold():
    sample = {
        "gpu": parse_gpu_row("0, GPU-1, Tesla V100, 32768, 20000, 95, 80, 84\n"),
        "queue": "",
    }
    assert evaluate_watchdog_sample(
        sample, gpu_uuid="GPU-1", gpu_model="Tesla V100"
    ) == []
    sample["gpu"]["temperature_c"] = 85
    assert "runtime_temperature_not_below_85_c" in evaluate_watchdog_sample(
        sample, gpu_uuid="GPU-1", gpu_model="Tesla V100"
    )


@pytest.mark.parametrize("returncode", (0, 1))
def test_fuser_idle_and_owner_return_codes_are_accepted(monkeypatch, returncode):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout="", stderr=""
        ),
    )
    assert _run(("fuser", "-v", "/dev/nvidia0"), accepted_returncodes=(0, 1)) == ""


def test_fuser_unexpected_return_code_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="failure"
        ),
    )
    with pytest.raises(RuntimeError, match="telemetry command failed"):
        _run(("fuser", "-v", "/dev/nvidia0"), accepted_returncodes=(0, 1))


def test_run_subcommand_rechecks_launches_with_watchdog_and_writes_completion(
    monkeypatch, tmp_path
):
    sample = {
        "admitted": True,
        "failures": [],
        "gpu": {
            "index": "0",
            "uuid": "GPU-1",
            "name": "Tesla V100",
            "memory_total_mib": 32768,
            "memory_used_mib": 0,
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "temperature_c": 30,
        },
        "compute_apps": "",
        "pmon": "",
        "fuser": "",
        "queue": "",
    }
    receipt = {
        "receipt_sha256": "a" * 64,
        "gpu_identity": {"uuid": "GPU-1", "model": "Tesla V100"},
        "limits": {"maximum_duration_seconds": 2},
    }
    monkeypatch.setattr(
        "hypertagging.utils.gpu_safety.load_local_microtest_admission_receipt",
        lambda _path: receipt,
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.collect_sample", lambda: dict(sample)
    )
    launched = {}

    class FakeProcess:
        pid = 4321

        def __init__(self, command, *, env, start_new_session):
            launched.update(
                command=command, env=env, start_new_session=start_new_session
            )
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.subprocess.Popen", FakeProcess
    )
    completion = tmp_path / "completion.json"
    args = SimpleNamespace(
        receipt=tmp_path / "admission.json",
        completion_output=completion,
        poll_seconds=0.001,
        signal_grace=1.0,
        terminate_grace=1.0,
        trainer_command=["trainer", "--gpu-execution-mode", "local_microtest"],
    )
    assert run_monitored(args) == 0
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert launched["env"]["HYPERTAGGING_LOCAL_WATCHDOG_SENTINEL"]
    assert launched["start_new_session"] is True
    payload = json.loads(completion.read_text())
    assert payload["trainer_status"] == 0
    assert payload["receipt_sha256"]
