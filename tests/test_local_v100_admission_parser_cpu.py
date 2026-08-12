from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm.v100_local_admission import (  # noqa: E402
    _parse_pmon_compute_pids,
    _run,
    _telemetry_pids,
    collect_sample,
    evaluate_sample,
    evaluate_watchdog_sample,
    parse_gpu_row,
    run_monitored,
)


def _clear_htcondor_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("_CONDOR_") or name.startswith("CONDOR_"):
            monkeypatch.delenv(name, raising=False)


def _mock_admission_commands(
    monkeypatch,
    *,
    process_rows,
    condor_queue=None,
    user_slurm_queue="",
    node_slurm_queue="",
    node_slurm_error=None,
):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,"
            "utilization.gpu,utilization.memory,temperature.gpu",
        ):
            return "0, GPU-1, Tesla V100, 32768, 0, 0, 0, 30\n"
        if command[:2] == (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        ):
            return ""
        if command == ("nvidia-smi", "pmon", "-c", "1"):
            return "# gpu pid type sm mem enc dec command\n"
        if command[:2] == ("fuser", "-v"):
            return ""
        if command[:2] == ("/opt/slurm/bin/squeue", "-h"):
            if "-u" in command:
                return user_slurm_queue
            if "-w" in command:
                if node_slurm_error is not None:
                    raise node_slurm_error
                return node_slurm_queue
        if command == ("ps", "-eo", "pid=,comm="):
            return process_rows
        if command[:1] == ("condor_q",):
            assert condor_queue is not None
            return condor_queue
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr("scripts.slurm.v100_local_admission._run", fake_run)
    return commands


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
        pmon="# gpu pid type sm mem enc dec command\n0 1234 C 10 0 0 0 python",
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


def test_pmon_graphics_only_rows_are_ignored_consistently():
    pmon = (
        "# gpu pid type sm mem enc dec command\n"
        "0 - - - - - - -\n"
        "0 1201 G - - - - Xorg\n"
        "0 1202 G - - - - gnome-shell\n"
    )
    gpu = parse_gpu_row("0, GPU-1, Tesla V100, 32768, 125, 0, 0, 30\n")

    admitted, failures = evaluate_sample(
        gpu, compute_apps="", pmon=pmon, fuser="", queue=""
    )

    assert admitted
    assert failures == []
    assert _parse_pmon_compute_pids(pmon) == set()
    assert _telemetry_pids(
        {"compute_apps": "", "pmon": pmon, "fuser": ""}
    ) == set()


def test_pmon_compute_and_mixed_rows_are_counted():
    pmon = (
        "# gpu pid type sm mem enc dec command\n"
        "0 2201 C 10 2 - - python\n"
        "0 2202 C+G 20 3 - - mixed-worker\n"
        "0 2203 G - - - - Xorg\n"
    )
    gpu = parse_gpu_row("0, GPU-1, Tesla V100, 32768, 125, 0, 0, 30\n")

    admitted, failures = evaluate_sample(
        gpu, compute_apps="", pmon=pmon, fuser="", queue=""
    )

    assert not admitted
    assert failures == ["pmon_process_present"]
    assert _parse_pmon_compute_pids(pmon) == {2201, 2202}
    assert _telemetry_pids(
        {"compute_apps": "", "pmon": pmon, "fuser": ""}
    ) == {2201, 2202}


def test_compute_apps_and_fuser_pid_checks_are_preserved():
    assert _telemetry_pids(
        {
            "compute_apps": "GPU-1, 4401, python, 100\n",
            "pmon": "# gpu pid type sm mem enc dec command\n0 4402 G - - - - Xorg\n",
            "fuser": "/dev/nvidia0: 5501\n/dev/nvidiactl: 5502\n",
        }
    ) == {4401, 5501, 5502}


@pytest.mark.parametrize(
    "row",
    (
        "unexpected text",
        "0 3301 G - - - Xorg",
        "0 3301 unknown - - - - process",
        "0 - G - - - - Xorg",
    ),
)
def test_malformed_pmon_rows_fail_closed(row):
    pmon = f"# gpu pid type sm mem enc dec command\n{row}\n"
    gpu = parse_gpu_row("0, GPU-1, Tesla V100, 32768, 125, 0, 0, 30\n")

    admitted, failures = evaluate_sample(
        gpu, compute_apps="", pmon=pmon, fuser="", queue=""
    )

    assert not admitted
    assert failures == ["pmon_telemetry_malformed"]
    with pytest.raises(RuntimeError, match="pmon"):
        _telemetry_pids({"compute_apps": "", "pmon": pmon, "fuser": ""})


def test_watchdog_allows_trainer_load_but_rejects_runtime_safety_threshold():
    sample = {
        "gpu": parse_gpu_row("0, GPU-1, Tesla V100, 32768, 20000, 95, 80, 84\n"),
        "queue": "",
        "slurm": {
            "user_queue": {"command": [], "output": ""},
            "node_wide": {
                "command": [],
                "output": "",
                "hostname": os.uname().nodename.partition(".")[0],
            },
        },
    }
    assert evaluate_watchdog_sample(
        sample, gpu_uuid="GPU-1", gpu_model="Tesla V100"
    ) == []
    sample["gpu"]["temperature_c"] = 85
    assert "runtime_temperature_not_below_85_c" in evaluate_watchdog_sample(
        sample, gpu_uuid="GPU-1", gpu_model="Tesla V100"
    )
    sample["gpu"]["temperature_c"] = 84
    sample["htcondor"] = {
        "mode": "local_host_absence_check",
        "absence_proven": False,
    }
    assert "htcondor_absence_not_proven" in evaluate_watchdog_sample(
        sample, gpu_uuid="GPU-1", gpu_model="Tesla V100"
    )


def test_empty_node_wide_slurm_placement_accepts_and_records_exact_command(
    monkeypatch,
):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.socket.gethostname",
        lambda: "v100-node.example.org",
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.getpass.getuser", lambda: "alice"
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.shutil.which",
        lambda name: "/usr/bin/condor_q" if name == "condor_q" else None,
    )
    commands = _mock_admission_commands(
        monkeypatch,
        process_rows="",
        condor_queue="",
    )

    sample = collect_sample()

    user_command = (
        "/opt/slurm/bin/squeue",
        "-h",
        "-u",
        "alice",
        "-t",
        "RUNNING,PENDING",
        "-o",
        "%i %P %j %T",
    )
    node_command = (
        "/opt/slurm/bin/squeue",
        "-h",
        "-w",
        "v100-node",
        "-t",
        "RUNNING,COMPLETING,CONFIGURING,SUSPENDED,RESIZING",
        "-o",
        "%i %u %P %j %T %N",
    )
    assert commands.count(user_command) == 1
    assert [command for command in commands if "-w" in command] == [node_command]
    assert sample["admitted"] is True
    assert sample["slurm"] == {
        "user_queue": {"command": list(user_command), "output": ""},
        "node_wide": {
            "command": list(node_command),
            "output": "",
            "hostname": "v100-node",
        },
    }


def test_any_node_wide_slurm_placement_rejects(monkeypatch):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.socket.gethostname",
        lambda: "v100-node.example.org",
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.shutil.which",
        lambda name: "/usr/bin/condor_q" if name == "condor_q" else None,
    )
    _mock_admission_commands(
        monkeypatch,
        process_rows="",
        condor_queue="",
        node_slurm_queue=(
            "7392 bob gpu other-training RUNNING v100-node\n"
        ),
    )

    sample = collect_sample()

    assert sample["admitted"] is False
    assert "node_slurm_queue_nonempty" in sample["failures"]
    assert sample["slurm"]["node_wide"]["output"].startswith("7392 bob ")


def test_node_wide_slurm_command_error_is_fatal(monkeypatch):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.socket.gethostname",
        lambda: "v100-node.example.org",
    )
    _mock_admission_commands(
        monkeypatch,
        process_rows="",
        node_slurm_error=RuntimeError("telemetry command failed: node-wide squeue"),
    )

    with pytest.raises(RuntimeError, match="node-wide squeue"):
        collect_sample()


def test_watchdog_rejects_node_wide_slurm_placement(monkeypatch):
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.socket.gethostname",
        lambda: "v100-node.example.org",
    )
    sample = {
        "gpu": parse_gpu_row("0, GPU-1, Tesla V100, 32768, 1, 0, 0, 30\n"),
        "queue": "",
        "slurm": {
            "user_queue": {"command": [], "output": ""},
            "node_wide": {
                "command": [],
                "output": "7392 bob gpu other-training RUNNING v100-node\n",
                "hostname": "v100-node",
            },
        },
    }

    assert "node_slurm_queue_nonempty" in evaluate_watchdog_sample(
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


def test_missing_condor_q_accepts_only_with_explicit_safe_host_evidence(monkeypatch):
    _clear_htcondor_environment(monkeypatch)
    monkeypatch.setattr("scripts.slurm.v100_local_admission.shutil.which", lambda _name: None)
    _mock_admission_commands(
        monkeypatch,
        process_rows="1 systemd\n42 python\n",
    )

    sample = collect_sample()

    assert sample["admitted"] is True
    assert sample["queue"] == ""
    assert sample["htcondor"] == {
        "mode": "local_host_absence_check",
        "condor_q_available": False,
        "condor_q_path": None,
        "queue_command": None,
        "queue_output": None,
        "process_context": {
            "pid": os.getpid(),
            "environment_marker_names": [],
            "under_htcondor": False,
        },
        "local_process_scan": {
            "scope": "all_local_processes",
            "command": ["ps", "-eo", "pid=,comm="],
            "processes_scanned": 2,
            "htcondor_processes": [],
            "clear": True,
        },
        "absence_proven": True,
    }


@pytest.mark.parametrize(
    ("environment", "process_rows", "expected_failure"),
    (
        (
            {"_CONDOR_SCRATCH_DIR": "/tmp/job"},
            "1 systemd\n42 python\n",
            "htcondor_process_context_present",
        ),
        (
            {"CONDOR_CONFIG": "/etc/condor/condor_config"},
            "1 systemd\n42 python\n",
            "htcondor_process_context_present",
        ),
        (
            {},
            "1 systemd\n42 condor_starter\n",
            "local_htcondor_process_present",
        ),
    ),
)
def test_missing_condor_q_rejects_condor_evidence(
    monkeypatch, environment, process_rows, expected_failure
):
    _clear_htcondor_environment(monkeypatch)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.shutil.which", lambda _name: None
    )
    _mock_admission_commands(monkeypatch, process_rows=process_rows)

    sample = collect_sample()

    assert sample["admitted"] is False
    assert expected_failure in sample["failures"]
    assert sample["htcondor"]["absence_proven"] is False


def test_missing_condor_q_unexpected_process_telemetry_fails_closed(monkeypatch):
    _clear_htcondor_environment(monkeypatch)
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.shutil.which", lambda _name: None
    )
    _mock_admission_commands(monkeypatch, process_rows="malformed\n")

    with pytest.raises(RuntimeError, match="process telemetry shape"):
        collect_sample()


@pytest.mark.parametrize(
    ("condor_queue", "expected_admitted"),
    (("", True), ("123 0 2\n", False)),
)
def test_existing_condor_q_preserves_original_queue_behavior(
    monkeypatch, condor_queue, expected_admitted
):
    commands = _mock_admission_commands(
        monkeypatch,
        process_rows="",
        condor_queue=condor_queue,
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.shutil.which",
        lambda name: "/usr/bin/condor_q" if name == "condor_q" else None,
    )
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.getpass.getuser", lambda: "alice"
    )

    sample = collect_sample()

    assert (
        "condor_q",
        "alice",
        "-autoformat",
        "ClusterId",
        "ProcId",
        "JobStatus",
    ) in commands
    assert ("ps", "-eo", "pid=,comm=") not in commands
    assert sample["admitted"] is expected_admitted
    assert ("user_queue_nonempty" in sample["failures"]) is (not expected_admitted)
    assert sample["htcondor"]["mode"] == "condor_q"
    assert sample["htcondor"]["queue_output"] == condor_queue


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
        "slurm": {
            "user_queue": {"command": [], "output": ""},
            "node_wide": {
                "command": [],
                "output": "",
                "hostname": os.uname().nodename.partition(".")[0],
            },
        },
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
    assert payload["terminal_watchdog_failures"] == []
    assert payload["terminal_observed_pids"] == []
    assert payload["receipt_sha256"]


def test_foreign_compute_pid_triggers_bounded_abort_and_diagnostic_receipt(
    monkeypatch, tmp_path
):
    immediate = {
        "admitted": True,
        "failures": [],
        "gpu": {
            "index": "0",
            "uuid": "GPU-1",
            "name": "Tesla V100",
            "memory_total_mib": 32768,
            "memory_used_mib": 125,
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "temperature_c": 30,
        },
        "compute_apps": "",
        "pmon": "# gpu pid type sm mem enc dec command\n0 1201 G - - - - Xorg\n",
        "fuser": "",
        "queue": "",
        "slurm": {
            "user_queue": {"command": [], "output": ""},
            "node_wide": {
                "command": [],
                "output": "",
                "hostname": os.uname().nodename.partition(".")[0],
            },
        },
    }
    foreign = {
        **immediate,
        "admitted": False,
        "failures": ["pmon_process_present"],
        "pmon": (
            "# gpu pid type sm mem enc dec command\n"
            "0 1201 G - - - - Xorg\n"
            "0 9876 C 25 4 - - foreign-worker\n"
        ),
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
    samples = iter((immediate, foreign))
    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.collect_sample", lambda: next(samples)
    )

    class FakeProcess:
        pid = 4321

        def __init__(self, command, *, env, start_new_session):
            assert command == ["trainer"]
            assert env["CUDA_VISIBLE_DEVICES"] == "0"
            assert start_new_session is True

        def poll(self):
            return None

    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission.subprocess.Popen", FakeProcess
    )
    stops = []

    def fake_bounded_stop(process, *, signal_grace, terminate_grace):
        stops.append((process.pid, signal_grace, terminate_grace))
        return -10

    monkeypatch.setattr(
        "scripts.slurm.v100_local_admission._bounded_stop", fake_bounded_stop
    )
    completion = tmp_path / "completion.json"
    args = SimpleNamespace(
        receipt=tmp_path / "admission.json",
        completion_output=completion,
        poll_seconds=0.001,
        signal_grace=1.0,
        terminate_grace=2.0,
        trainer_command=["trainer"],
    )

    assert run_monitored(args) == -10
    assert stops == [(4321, 1.0, 2.0)]
    payload = json.loads(completion.read_text())
    assert payload["watchdog_reason"] == "telemetry_threshold_or_foreign_process"
    assert payload["trainer_status"] == -10
    assert payload["trainer_pid"] == 4321
    assert payload["terminal_watchdog_failures"] == ["foreign_process_present"]
    assert payload["terminal_observed_pids"] == [9876]
    assert payload["terminal_foreign_pids"] == [9876]
    assert payload["terminal_telemetry_sample"] == foreign
    canonical = {
        key: value for key, value in payload.items() if key != "receipt_sha256"
    }
    assert payload["receipt_sha256"] == hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
