#!/usr/bin/env python
"""Admit and monitor a tightly bounded node-local V100 microtest.

This script performs GPU commands and must not be run during CPU-only readiness work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any


GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,"
    "utilization.memory,temperature.gpu",
    "--format=csv,noheader,nounits",
)
COMPUTE_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)
HTCONDOR_PROCESS_QUERY = ("ps", "-eo", "pid=,comm=")
SLURM_SQUEUE = "/opt/slurm/bin/squeue"
SLURM_USER_STATES = "RUNNING,PENDING"
SLURM_NODE_STATES = "RUNNING,COMPLETING,CONFIGURING,SUSPENDED,RESIZING"


def _run(command: tuple[str, ...], *, accepted_returncodes: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    if result.returncode not in accepted_returncodes:
        raise RuntimeError(f"telemetry command failed: {command!r}: {result.stderr.strip()}")
    return result.stdout + result.stderr


def parse_gpu_row(text: str) -> dict[str, Any]:
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("exactly one GPU telemetry row is required")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 8:
        raise RuntimeError("unexpected nvidia-smi telemetry shape")
    keys = (
        "index", "uuid", "name", "memory_total_mib", "memory_used_mib",
        "gpu_utilization_percent", "memory_utilization_percent", "temperature_c",
    )
    payload = dict(zip(keys, fields, strict=True))
    for key in keys[3:]:
        payload[key] = int(payload[key])
    return payload


def _parse_pmon_compute_pids(text: str) -> set[int]:
    """Return PIDs from well-formed pmon rows whose type includes compute."""

    pids: set[int] = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=7)
        if len(fields) != 8 or not fields[0].isdigit():
            raise RuntimeError("unexpected nvidia-smi pmon telemetry shape")
        pid_text, process_type, *utilization, command = fields[1:]
        if pid_text == "-":
            if any(field != "-" for field in fields[2:]):
                raise RuntimeError("unexpected nvidia-smi pmon idle row")
            continue
        if not pid_text.isdigit() or int(pid_text) <= 0:
            raise RuntimeError("unexpected nvidia-smi pmon PID")
        if process_type not in {"C", "G", "C+G", "G+C"}:
            raise RuntimeError("unexpected nvidia-smi pmon process type")
        if any(
            value != "-" and (not value.isdigit() or not 0 <= int(value) <= 100)
            for value in utilization
        ):
            raise RuntimeError("unexpected nvidia-smi pmon utilization")
        if command == "-":
            raise RuntimeError("unexpected nvidia-smi pmon command")
        if "C" in process_type.split("+"):
            pids.add(int(pid_text))
    return pids


def evaluate_sample(
    gpu: dict[str, Any],
    *,
    compute_apps: str,
    pmon: str,
    fuser: str,
    queue: str,
    node_slurm_queue: str = "",
) -> tuple[bool, list[str]]:
    failures = []
    if "v100" not in str(gpu["name"]).lower():
        failures.append("not_v100")
    if int(gpu["memory_used_mib"]) > 512:
        failures.append("memory_used_above_512_mib")
    if int(gpu["gpu_utilization_percent"]) > 5:
        failures.append("gpu_utilization_above_5_percent")
    if int(gpu["memory_utilization_percent"]) > 5:
        failures.append("memory_utilization_above_5_percent")
    if int(gpu["temperature_c"]) >= 70:
        failures.append("temperature_not_below_70_c")
    if compute_apps.strip():
        failures.append("compute_process_present")
    try:
        pmon_compute_pids = _parse_pmon_compute_pids(pmon)
    except RuntimeError:
        failures.append("pmon_telemetry_malformed")
    else:
        if pmon_compute_pids:
            failures.append("pmon_process_present")
    if re.search(r"\b\d{2,}\b", fuser):
        failures.append("device_owner_present")
    if queue.strip():
        failures.append("user_queue_nonempty")
    if node_slurm_queue.strip():
        failures.append("node_slurm_queue_nonempty")
    return not failures, failures


def evaluate_watchdog_sample(
    sample: dict[str, Any], *, gpu_uuid: str, gpu_model: str
) -> list[str]:
    """Apply runtime safety thresholds without treating trainer activity as idle use."""

    gpu = sample["gpu"]
    failures: list[str] = []
    if gpu.get("uuid") != gpu_uuid or gpu.get("name") != gpu_model:
        failures.append("gpu_identity_changed")
    if int(gpu["temperature_c"]) >= 85:
        failures.append("runtime_temperature_not_below_85_c")
    if not 0 <= int(gpu["memory_used_mib"]) <= int(gpu["memory_total_mib"]):
        failures.append("runtime_memory_telemetry_invalid")
    for key in ("gpu_utilization_percent", "memory_utilization_percent"):
        if not 0 <= int(gpu[key]) <= 100:
            failures.append(f"runtime_{key}_invalid")
    if str(sample["queue"]).strip():
        failures.append("user_queue_nonempty")
    slurm = sample.get("slurm")
    if not isinstance(slurm, dict):
        failures.append("slurm_evidence_not_proven")
    else:
        user_queue = slurm.get("user_queue")
        node_wide = slurm.get("node_wide")
        if not isinstance(user_queue, dict) or not isinstance(node_wide, dict):
            failures.append("slurm_evidence_not_proven")
        elif (
            not isinstance(user_queue.get("output"), str)
            or not isinstance(node_wide.get("output"), str)
            or node_wide.get("hostname") != _short_hostname()
        ):
            failures.append("slurm_evidence_not_proven")
        else:
            if (
                user_queue["output"].strip()
                and "user_queue_nonempty" not in failures
            ):
                failures.append("user_queue_nonempty")
            if node_wide["output"].strip():
                failures.append("node_slurm_queue_nonempty")
    htcondor = sample.get("htcondor")
    if (
        isinstance(htcondor, dict)
        and htcondor.get("mode") == "local_host_absence_check"
        and htcondor.get("absence_proven") is not True
    ):
        failures.append("htcondor_absence_not_proven")
    return failures


def _htcondor_environment_markers(environ: dict[str, str]) -> list[str]:
    return sorted(
        name
        for name in environ
        if name.startswith("_CONDOR_") or name.startswith("CONDOR_")
    )


def _parse_process_names(text: str) -> list[tuple[int, str]]:
    processes: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].strip():
            raise RuntimeError("unexpected local process telemetry shape")
        processes.append((int(fields[0]), fields[1].strip()))
    if not processes:
        raise RuntimeError("local process telemetry returned no processes")
    return processes


def _is_htcondor_process_name(name: str) -> bool:
    normalized = Path(name).name.lower().strip("[]")
    return normalized == "condor" or normalized.startswith("condor_")


def _collect_htcondor_queue_evidence(
    user: str, *, environ: dict[str, str] | None = None
) -> tuple[str, dict[str, Any], list[str]]:
    command = ("condor_q", user, "-autoformat", "ClusterId", "ProcId", "JobStatus")
    executable = shutil.which("condor_q")
    if executable is not None:
        queue = _run(command)
        return queue, {
            "mode": "condor_q",
            "condor_q_available": True,
            "condor_q_path": executable,
            "queue_command": list(command),
            "queue_output": queue,
        }, []

    marker_names = _htcondor_environment_markers(
        dict(os.environ if environ is None else environ)
    )
    process_rows = _parse_process_names(_run(HTCONDOR_PROCESS_QUERY))
    condor_processes = [
        {"pid": pid, "name": name}
        for pid, name in process_rows
        if _is_htcondor_process_name(name)
    ]
    failures = []
    if marker_names:
        failures.append("htcondor_process_context_present")
    if condor_processes:
        failures.append("local_htcondor_process_present")
    evidence = {
        "mode": "local_host_absence_check",
        "condor_q_available": False,
        "condor_q_path": None,
        "queue_command": None,
        "queue_output": None,
        "process_context": {
            "pid": os.getpid(),
            "environment_marker_names": marker_names,
            "under_htcondor": bool(marker_names),
        },
        "local_process_scan": {
            "scope": "all_local_processes",
            "command": list(HTCONDOR_PROCESS_QUERY),
            "processes_scanned": len(process_rows),
            "htcondor_processes": condor_processes,
            "clear": not condor_processes,
        },
        "absence_proven": not failures,
    }
    return "", evidence, failures


def _short_hostname() -> str:
    hostname = socket.gethostname().strip().partition(".")[0]
    if not hostname or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", hostname):
        raise RuntimeError("could not determine a safe short hostname for Slurm telemetry")
    return hostname


def _collect_slurm_queue_evidence(user: str) -> tuple[str, str, dict[str, Any]]:
    hostname = _short_hostname()
    user_command = (
        SLURM_SQUEUE,
        "-h",
        "-u",
        user,
        "-t",
        SLURM_USER_STATES,
        "-o",
        "%i %P %j %T",
    )
    node_command = (
        SLURM_SQUEUE,
        "-h",
        "-w",
        hostname,
        "-t",
        SLURM_NODE_STATES,
        "-o",
        "%i %u %P %j %T %N",
    )
    user_queue = _run(user_command)
    node_queue = _run(node_command)
    return user_queue, node_queue, {
        "user_queue": {
            "command": list(user_command),
            "output": user_queue,
        },
        "node_wide": {
            "command": list(node_command),
            "output": node_queue,
            "hostname": hostname,
        },
    }


def collect_sample() -> dict[str, Any]:
    gpu = parse_gpu_row(_run(GPU_QUERY))
    compute = _run(COMPUTE_QUERY)
    pmon = _run(("nvidia-smi", "pmon", "-c", "1"))
    # fuser returns 1 with empty output when none of the named devices is in use.
    fuser = _run(
        ("fuser", "-v", "/dev/nvidia0", "/dev/nvidiactl"),
        accepted_returncodes=(0, 1),
    )
    user = getpass.getuser()
    slurm_queue, node_slurm_queue, slurm_evidence = (
        _collect_slurm_queue_evidence(user)
    )
    condor_queue, htcondor_evidence, condor_failures = (
        _collect_htcondor_queue_evidence(user)
    )
    queue = "\n".join(value.strip() for value in (slurm_queue, condor_queue) if value.strip())
    admitted, failures = evaluate_sample(
        gpu,
        compute_apps=compute,
        pmon=pmon,
        fuser=fuser,
        queue=queue,
        node_slurm_queue=node_slurm_queue,
    )
    failures.extend(condor_failures)
    admitted = admitted and not condor_failures
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "compute_apps": compute,
        "pmon": pmon,
        "fuser": fuser,
        "queue": queue,
        "slurm": slurm_evidence,
        "htcondor": htcondor_evidence,
        "admitted": admitted,
        "failures": failures,
    }


def _write_receipt(payload: dict[str, Any], destination: Path) -> None:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def admit(args: argparse.Namespace) -> int:
    if (
        not 1 <= args.max_steps <= 10
        or not 1 <= args.batch_size <= 2
        or not 1 <= args.duration_seconds <= 300
    ):
        raise RuntimeError("local microtest limits exceed 10 steps, batch 2, or 5 minutes")
    samples = []
    for sample_index in range(3):
        samples.append(collect_sample())
        if sample_index != 2:
            time.sleep(10)
    identities = {
        (sample["gpu"]["uuid"], sample["gpu"]["name"]) for sample in samples
    }
    admitted = all(sample["admitted"] for sample in samples) and len(identities) == 1
    receipt = {
        "receipt_version": "hypertagging-local-v100-admission-v2",
        "mode": "local_microtest",
        "status": "admitted" if admitted else "rejected",
        "hostname": socket.gethostname(),
        "gpu_identity": {
            "uuid": samples[0]["gpu"]["uuid"],
            "model": samples[0]["gpu"]["name"],
        },
        "thresholds": {
            "maximum_memory_used_mib": 512,
            "maximum_utilization_percent": 5,
            "maximum_temperature_c_exclusive": 70,
            "sample_count": 3,
            "sample_interval_seconds": 10,
        },
        "limits": {
            "maximum_steps": args.max_steps,
            "maximum_batch_size": args.batch_size,
            "maximum_duration_seconds": args.duration_seconds,
        },
        "samples": samples,
    }
    _write_receipt(receipt, args.output)
    if not admitted:
        raise RuntimeError("V100 admission rejected; receipt was written for diagnosis")
    print(args.output)
    return 0


def _non_pmon_telemetry_pids(sample: dict[str, Any]) -> set[int]:
    pids: set[int] = set()
    for line in str(sample["compute_apps"]).splitlines():
        fields = line.split(",")
        if len(fields) >= 2 and fields[1].strip().isdigit():
            pids.add(int(fields[1].strip()))
    for line in str(sample["fuser"]).splitlines():
        owner_text = line.split(":", 1)[1] if ":" in line else line
        pids.update(int(value) for value in re.findall(r"\b\d+\b", owner_text))
    return pids


def _telemetry_pids(sample: dict[str, Any]) -> set[int]:
    pids = _non_pmon_telemetry_pids(sample)
    pids.update(_parse_pmon_compute_pids(str(sample["pmon"])))
    return pids


def _write_watchdog_sentinel(
    destination: Path, *, receipt_sha256: str, duration_seconds: int
) -> None:
    payload: dict[str, Any] = {
        "sentinel_version": "hypertagging-local-watchdog-v1",
        "hostname": socket.gethostname(),
        "orchestrator_pid": os.getpid(),
        "admission_receipt_sha256": receipt_sha256,
        "expires_epoch": time.time() + duration_seconds + 30,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sentinel_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _signal_process_group(process: subprocess.Popen[Any], signum: int) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signum)


def _bounded_stop(
    process: subprocess.Popen[Any], *, signal_grace: float, terminate_grace: float
) -> int:
    _signal_process_group(process, signal.SIGUSR1)
    try:
        return process.wait(timeout=signal_grace)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)
    try:
        return process.wait(timeout=terminate_grace)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        return process.wait(timeout=5)


def run_monitored(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from hypertagging.utils.gpu_safety import load_local_microtest_admission_receipt

    if not 0 < args.poll_seconds <= 30:
        raise RuntimeError("watchdog polling must be positive and no slower than 30 seconds")
    if args.signal_grace <= 0 or args.terminate_grace <= 0:
        raise RuntimeError("watchdog shutdown grace periods must be positive")
    receipt = load_local_microtest_admission_receipt(args.receipt)
    immediate = collect_sample()
    identity = receipt["gpu_identity"]
    if (
        not immediate["admitted"]
        or immediate["gpu"]["uuid"] != identity["uuid"]
        or immediate["gpu"]["name"] != identity["model"]
    ):
        raise RuntimeError("immediate telemetry recheck no longer matches admission")
    if not re.fullmatch(r"[0-9]+", str(immediate["gpu"]["index"])):
        raise RuntimeError("admitted GPU index is not a safe CUDA device identifier")
    trainer_command = list(args.trainer_command)
    if trainer_command[:1] == ["--"]:
        trainer_command.pop(0)
    if not trainer_command:
        raise RuntimeError("run requires a trainer command after --")

    args.completion_output.parent.mkdir(parents=True, exist_ok=True)
    sentinel = args.completion_output.with_name(
        f".{args.completion_output.name}.{os.getpid()}.watchdog-sentinel.json"
    )
    if sentinel.exists():
        raise RuntimeError("refusing to overwrite an existing watchdog sentinel")
    _write_watchdog_sentinel(
        sentinel,
        receipt_sha256=str(receipt["receipt_sha256"]),
        duration_seconds=int(receipt["limits"]["maximum_duration_seconds"]),
    )
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = str(immediate["gpu"]["index"])
    child_env["HYPERTAGGING_LOCAL_WATCHDOG_SENTINEL"] = str(sentinel.resolve())
    child_env["PYTHONNOUSERSITE"] = "1"
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "LD_PRELOAD"):
        child_env.pop(key, None)
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(
        trainer_command, env=child_env, start_new_session=True
    )
    deadline = time.monotonic() + int(receipt["limits"]["maximum_duration_seconds"])
    reason = "trainer_exit"
    status: int | None = None
    samples = [immediate]
    terminal_watchdog_failures: list[str] = []
    terminal_observed_pids = sorted(_telemetry_pids(immediate))
    terminal_foreign_pids: list[int] = []
    terminal_telemetry_sample = immediate
    try:
        while time.monotonic() < deadline:
            status = process.poll()
            if status is not None:
                break
            sample = collect_sample()
            samples.append(sample)
            unexpected = evaluate_watchdog_sample(
                sample,
                gpu_uuid=str(identity["uuid"]),
                gpu_model=str(identity["model"]),
            )
            try:
                observed_pids = _telemetry_pids(sample)
            except RuntimeError:
                unexpected.append("pmon_telemetry_malformed")
                observed_pids = _non_pmon_telemetry_pids(sample)
            foreign_pids = observed_pids - {process.pid}
            terminal_watchdog_failures = list(unexpected)
            if foreign_pids:
                terminal_watchdog_failures.append("foreign_process_present")
            terminal_observed_pids = sorted(observed_pids)
            terminal_foreign_pids = sorted(foreign_pids)
            terminal_telemetry_sample = sample
            if terminal_watchdog_failures:
                reason = "telemetry_threshold_or_foreign_process"
                status = _bounded_stop(
                    process,
                    signal_grace=args.signal_grace,
                    terminate_grace=args.terminate_grace,
                )
                break
            time.sleep(args.poll_seconds)
        else:
            reason = "watchdog_deadline"
            terminal_watchdog_failures = ["watchdog_deadline"]
            status = _bounded_stop(
                process,
                signal_grace=args.signal_grace,
                terminate_grace=args.terminate_grace,
            )
        if status is None:
            status = process.wait()
    finally:
        sentinel.unlink(missing_ok=True)
    completion = {
        "receipt_version": "hypertagging-local-v100-completion-v1",
        "admission_receipt_sha256": receipt["receipt_sha256"],
        "hostname": socket.gethostname(),
        "gpu_identity": identity,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "watchdog_reason": reason,
        "trainer_status": status,
        "trainer_pid": process.pid,
        "sample_count": len(samples),
        "poll_seconds": args.poll_seconds,
        "terminal_watchdog_failures": terminal_watchdog_failures,
        "terminal_observed_pids": terminal_observed_pids,
        "terminal_foreign_pids": terminal_foreign_pids,
        "terminal_telemetry_sample": terminal_telemetry_sample,
    }
    _write_receipt(completion, args.completion_output)
    return int(status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)
    admit_parser = sub.add_parser("admit")
    admit_parser.add_argument("--output", type=Path, required=True)
    admit_parser.add_argument("--max-steps", type=int, required=True)
    admit_parser.add_argument("--batch-size", type=int, required=True)
    admit_parser.add_argument("--duration-seconds", type=int, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument("--completion-output", type=Path, required=True)
    run_parser.add_argument("--poll-seconds", type=float, default=30.0)
    run_parser.add_argument("--signal-grace", type=float, default=30.0)
    run_parser.add_argument("--terminate-grace", type=float, default=10.0)
    run_parser.add_argument("trainer_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return admit(args) if args.subcommand == "admit" else run_monitored(args)


if __name__ == "__main__":
    raise SystemExit(main())
