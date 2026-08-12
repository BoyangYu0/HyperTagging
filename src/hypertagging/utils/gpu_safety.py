"""GPU and HTCondor safety checks for HyperTagging training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
from typing import Any


@dataclass(frozen=True)
class CommandSnapshot:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def is_inside_condor() -> bool:
    return bool(
        os.environ.get("_CONDOR_SCRATCH_DIR")
        or os.environ.get("CONDOR_CLUSTER_ID")
        or os.environ.get("CONDOR_PROCESS_ID")
    )


ALLOWED_SLURM_GRES = ("gpu:h200nvl:1", "gpu:v100:1")
LOCAL_RECEIPT_MAX_AGE_SECONDS = 90
LOCAL_COMPLETION_SHUTDOWN_GRACE_SECONDS = 30


def _one_visible_device(value: str) -> bool:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return len(entries) == 1 and entries[0] not in {"NoDevFiles", "none", "-1"}


def _gpu_name_matches_gres(name: str, gres: str) -> bool:
    patterns = {
        "gpu:h200nvl:1": r"^NVIDIA H200 NVL(?: [A-Za-z0-9][A-Za-z0-9 ._-]*)?$",
        "gpu:v100:1": r"^(?:NVIDIA|Tesla) V100(?:[ -][A-Za-z0-9][A-Za-z0-9 ._-]*)?$",
    }
    pattern = patterns.get(gres)
    return pattern is not None and re.fullmatch(pattern, name.strip()) is not None


def _one_csv_entry(value: str, *, field: str) -> str:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if len(entries) != 1:
        raise RuntimeError(f"exactly one {field} entry is required")
    if entries[0] in {"NoDevFiles", "none", "-1"}:
        raise RuntimeError(f"{field} does not identify an allocated GPU")
    return entries[0]


def _parse_job_record(record: str) -> dict[str, str]:
    lines = [line.strip() for line in record.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("scontrol must return exactly one one-line job record")
    fields: dict[str, str] = {}
    for token in shlex.split(lines[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise RuntimeError(f"duplicate Slurm job-record field: {key}")
        fields[key] = value
    return fields


def _parse_tres_map(value: str, *, field: str) -> dict[str, str]:
    """Parse a Slurm comma-separated TRES field without partial matching."""

    if not value:
        raise RuntimeError(f"Slurm job record lacks {field}")
    parsed: dict[str, str] = {}
    for item in value.split(","):
        if not item or item.count("=") != 1:
            raise RuntimeError(f"Slurm {field} is not an exact TRES map")
        key, count = item.split("=", 1)
        if not key or not count or key in parsed:
            raise RuntimeError(f"Slurm {field} contains an invalid or duplicate TRES")
        parsed[key] = count
    return parsed


def _verify_exact_gpu_tres(value: str, *, field: str, expected_gpu_type: str) -> None:
    tres = _parse_tres_map(value, field=field)
    expected_typed = f"gres/gpu:{expected_gpu_type}"
    typed_gpu_keys = {key for key in tres if key.startswith("gres/gpu:")}
    if tres.get("gres/gpu") != "1" or tres.get(expected_typed) != "1":
        raise RuntimeError(
            f"Slurm {field} does not contain exact generic and typed one-GPU TRES"
        )
    if typed_gpu_keys != {expected_typed}:
        raise RuntimeError(
            f"Slurm {field} contains a mismatched or mixed typed GPU TRES"
        )


def verify_exact_typed_gres_job_record(
    record: str, *, job_id: str, expected_gres: str
) -> None:
    """Prove the exact typed one-GPU request from ``scontrol show job -o``."""

    fields = _parse_job_record(record)
    if fields.get("JobId") != job_id:
        raise RuntimeError("Slurm job record does not identify SLURM_JOB_ID exactly")
    match = re.fullmatch(r"gpu:([a-z0-9][a-z0-9_-]*):1", expected_gres)
    if match is None:
        raise RuntimeError("expected GRES is not an exact typed one-GPU request")
    expected_gpu_type = match.group(1)
    for field in ("ReqTRES", "AllocTRES"):
        _verify_exact_gpu_tres(
            fields.get(field, ""),
            field=field,
            expected_gpu_type=expected_gpu_type,
        )
    accepted_tres_per_node = {
        f"gres:gpu:{expected_gpu_type}",
        f"gres:gpu:{expected_gpu_type}:1",
    }
    if fields.get("TresPerNode") not in accepted_tres_per_node:
        raise RuntimeError(
            "Slurm TresPerNode does not prove the exact typed one-GPU request"
        )
    if "Gres" in fields and fields["Gres"] not in {"N/A", expected_gres}:
        raise RuntimeError("Slurm job record contains a mismatched or mixed GRES")


def _read_slurm_job_record(job_id: str) -> str:
    snapshot = _run(("/opt/slurm/bin/scontrol", "show", "job", "-o", job_id))
    if snapshot.returncode != 0:
        raise RuntimeError("cannot query the Slurm job allocation record")
    return snapshot.stdout


def assert_scientific_slurm_gpu_allowed(
    expected_gres: str,
    *,
    environ: dict[str, str] | None = None,
    gpu_name: str | None = None,
    job_record: str | None = None,
) -> None:
    """Fail closed unless the process has the exact one-GPU Slurm allocation."""

    if expected_gres not in ALLOWED_SLURM_GRES:
        raise RuntimeError(f"unsupported or generic Slurm GRES: {expected_gres!r}")
    env = dict(os.environ if environ is None else environ)
    job_id = env.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("scientific Slurm GPU mode requires SLURM_JOB_ID")
    _one_csv_entry(env.get("CUDA_VISIBLE_DEVICES", ""), field="CUDA_VISIBLE_DEVICES")
    _one_csv_entry(env.get("SLURM_JOB_GPUS", ""), field="SLURM_JOB_GPUS")
    if env.get("SLURM_GPUS_ON_NODE") != "1":
        raise RuntimeError("SLURM_GPUS_ON_NODE must equal exactly 1")
    verify_exact_typed_gres_job_record(
        _read_slurm_job_record(job_id) if job_record is None else job_record,
        job_id=job_id,
        expected_gres=expected_gres,
    )
    if gpu_name is None:
        snapshot = _run(
            (
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            )
        )
        if snapshot.returncode != 0:
            raise RuntimeError("cannot query allocated GPU model")
        names = [line.strip() for line in snapshot.stdout.splitlines() if line.strip()]
        if len(names) != 1:
            raise RuntimeError("nvidia-smi must report exactly one visible GPU")
        gpu_name = names[0]
    if not _gpu_name_matches_gres(gpu_name, expected_gres):
        raise RuntimeError(
            f"visible GPU model {gpu_name!r} does not match expected GRES {expected_gres!r}"
        )


def _parse_utc_timestamp(value: object, *, require_utc: bool = False) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("local GPU receipt contains an invalid timestamp") from error
    if parsed.tzinfo is None:
        raise RuntimeError("local GPU receipt timestamps must include UTC offsets")
    if require_utc and parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("local GPU completion timestamps must be UTC")
    return parsed.astimezone(timezone.utc)


def _load_hashed_receipt(path: str | Path, *, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load local GPU {kind} receipt") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"local GPU {kind} receipt must be a JSON object")
    stored = str(payload.get("receipt_sha256", ""))
    canonical = {
        key: value for key, value in payload.items() if key != "receipt_sha256"
    }
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError(f"local GPU {kind} receipt hash mismatch")
    return payload


def load_local_microtest_admission_receipt(
    path: str | Path,
    *,
    now: datetime | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Validate a node-local admission receipt without performing GPU work."""

    payload = _load_hashed_receipt(path, kind="admission")
    if payload.get("receipt_version") != "hypertagging-local-v100-admission-v2":
        raise RuntimeError("unsupported local GPU admission receipt")
    if payload.get("mode") != "local_microtest" or payload.get("status") != "admitted":
        raise RuntimeError("local GPU admission receipt is not admitted")
    actual_hostname = hostname or socket.gethostname()
    if payload.get("hostname") != actual_hostname:
        raise RuntimeError("local GPU admission receipt belongs to a different host")
    limits = payload.get("limits", {})
    if (
        not 1 <= int(limits.get("maximum_steps", 0)) <= 10
        or not 1 <= int(limits.get("maximum_batch_size", 0)) <= 2
        or not 1 <= int(limits.get("maximum_duration_seconds", 0)) <= 300
    ):
        raise RuntimeError("local GPU admission receipt exceeds microtest limits")
    samples = payload.get("samples", [])
    if len(samples) != 3 or any(
        not sample.get("admitted", False) for sample in samples
    ):
        raise RuntimeError(
            "local GPU receipt requires three admitted telemetry samples"
        )
    identity = payload.get("gpu_identity", {})
    if not identity.get("uuid") or not _gpu_name_matches_gres(
        str(identity.get("model", "")), "gpu:v100:1"
    ):
        raise RuntimeError("local GPU receipt is not bound to a V100 UUID/model")
    timestamps = [_parse_utc_timestamp(sample.get("timestamp")) for sample in samples]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != 3:
        raise RuntimeError(
            "local GPU receipt sample timestamps are not strictly ordered"
        )
    intervals = [
        (later - earlier).total_seconds()
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    if any(not 5 <= interval <= 30 for interval in intervals):
        raise RuntimeError("local GPU receipt samples do not have bounded spacing")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if (
        timestamps[-1] > current.replace(microsecond=current.microsecond)
        or (current - timestamps[-1]).total_seconds() > LOCAL_RECEIPT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("local GPU admission receipt is stale or from the future")
    for sample in samples:
        gpu = sample.get("gpu", {})
        if gpu.get("uuid") != identity["uuid"] or gpu.get("name") != identity["model"]:
            raise RuntimeError("local GPU receipt samples disagree on GPU identity")
    return payload


def load_local_microtest_completion_receipt(
    path: str | Path,
    *,
    admission_path: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate successful monitored completion and its admitted-run binding."""

    payload = _load_hashed_receipt(path, kind="completion")
    if payload.get("receipt_version") != "hypertagging-local-v100-completion-v1":
        raise RuntimeError("unsupported local GPU completion receipt")
    hostname = payload.get("hostname")
    identity = payload.get("gpu_identity")
    if not isinstance(hostname, str) or not hostname or not isinstance(identity, dict):
        raise RuntimeError("local GPU completion receipt lacks host/GPU identity")
    started = _parse_utc_timestamp(payload.get("started_at"), require_utc=True)
    completed = _parse_utc_timestamp(payload.get("completed_at"), require_utc=True)
    if completed <= started:
        raise RuntimeError("local GPU completion timestamps are not strictly ordered")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if completed > current:
        raise RuntimeError("local GPU completion receipt is from the future")
    admission = load_local_microtest_admission_receipt(
        admission_path,
        now=started,
        hostname=hostname,
    )
    if payload.get("admission_receipt_sha256") != admission["receipt_sha256"]:
        raise RuntimeError("local GPU completion is not bound to the admission receipt")
    if identity != admission["gpu_identity"]:
        raise RuntimeError("local GPU completion identity differs from admission")
    maximum_elapsed = (
        int(admission["limits"]["maximum_duration_seconds"])
        + LOCAL_COMPLETION_SHUTDOWN_GRACE_SECONDS
    )
    if (completed - started).total_seconds() > maximum_elapsed:
        raise RuntimeError(
            "local GPU completion exceeded its admitted duration and grace"
        )
    if payload.get("watchdog_reason") != "trainer_exit":
        raise RuntimeError("local GPU microtest did not complete by trainer exit")
    trainer_status = payload.get("trainer_status")
    if (
        isinstance(trainer_status, bool)
        or not isinstance(trainer_status, int)
        or trainer_status != 0
    ):
        raise RuntimeError("local GPU microtest trainer did not exit successfully")
    sample_count = payload.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise RuntimeError("local GPU completion has no monitored samples")
    poll_seconds = payload.get("poll_seconds")
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)):
        raise RuntimeError("local GPU completion has an invalid polling interval")
    if not 0 < float(poll_seconds) <= 30:
        raise RuntimeError("local GPU completion polling interval is outside bounds")
    return payload


def validate_local_watchdog_sentinel(
    path: str | Path, *, receipt_sha256: str, now_epoch: float | None = None
) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = str(payload.pop("sentinel_sha256", ""))
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("local watchdog sentinel hash mismatch")
    if payload.get("sentinel_version") != "hypertagging-local-watchdog-v1":
        raise RuntimeError("unsupported local watchdog sentinel")
    if payload.get("hostname") != socket.gethostname():
        raise RuntimeError("local watchdog sentinel belongs to a different host")
    if payload.get("admission_receipt_sha256") != receipt_sha256:
        raise RuntimeError(
            "local watchdog sentinel is not bound to the admission receipt"
        )
    if int(payload.get("orchestrator_pid", -1)) != os.getppid():
        raise RuntimeError("local microtest is not a direct watchdog child")
    if float(payload.get("expires_epoch", 0)) <= float(
        now_epoch if now_epoch is not None else datetime.now().timestamp()
    ):
        raise RuntimeError("local watchdog sentinel has expired")


def _run(command: tuple[str, ...], timeout: int = 10) -> CommandSnapshot:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        return CommandSnapshot(command, result.returncode, result.stdout, result.stderr)
    except Exception as exc:
        return CommandSnapshot(command, 127, "", str(exc))


def get_condor_q_snapshot() -> dict[str, CommandSnapshot]:
    return {"condor_q": _run(("condor_q", "-autoformat", "ClusterId", "ProcId"))}


def get_nvidia_smi_snapshot() -> dict[str, CommandSnapshot]:
    return {
        "nvidia_smi": _run(("nvidia-smi",)),
        "compute_apps": _run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            )
        ),
        "gpu_state": _run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            )
        ),
    }


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return (
        getattr(args, name, default)
        if not isinstance(args, dict)
        else args.get(name, default)
    )


def assert_local_gpu_tiny_test_allowed(
    args: Any, *, max_steps: int = 10, max_batch_size: int = 2
) -> None:
    """Allow local CUDA only for explicit tiny tests after queue/GPU checks."""

    if str(_arg(args, "device", "cpu")).split(":")[0] != "cuda":
        return
    if not _arg(args, "tiny", False):
        raise RuntimeError("Local CUDA requires --tiny.")
    if int(_arg(args, "max_steps", max_steps + 1)) > max_steps:
        raise RuntimeError(f"Local CUDA tiny tests require --max-steps <= {max_steps}.")
    if int(_arg(args, "batch_size", max_batch_size + 1)) > max_batch_size:
        raise RuntimeError(
            f"Local CUDA tiny tests require --batch-size <= {max_batch_size}."
        )
    if not _arg(args, "allow_local_tiny_gpu_test", False):
        raise RuntimeError("Local CUDA tiny tests require --allow-local-tiny-gpu-test.")
    snapshots = {**get_condor_q_snapshot(), **get_nvidia_smi_snapshot()}
    failed = [name for name, snap in snapshots.items() if snap.returncode != 0]
    if failed:
        raise RuntimeError(f"Cannot verify local GPU safety; failed commands: {failed}")
    compute_apps = snapshots["compute_apps"].stdout.strip()
    if compute_apps:
        raise RuntimeError(
            "Local GPU appears busy: nvidia-smi compute apps are present."
        )
    if snapshots["condor_q"].stdout.strip():
        raise RuntimeError(
            "HTCondor queue is non-empty; refusing ambiguous local GPU test."
        )


def assert_full_training_requires_condor(args: Any) -> None:
    """Refuse CUDA outside an explicitly admitted scheduler/local mode."""

    if str(_arg(args, "device", "cpu")).split(":")[0] != "cuda":
        return
    execution_mode = str(_arg(args, "gpu_execution_mode", "auto"))
    if execution_mode in {"scientific_slurm", "slurm_diagnostic"}:
        expected_gres = str(
            _arg(args, "expected_gres", None)
            or os.environ.get("HYPERTAGGING_EXPECTED_GRES", "")
        )
        assert_scientific_slurm_gpu_allowed(expected_gres)
        return
    if execution_mode == "local_microtest":
        if os.environ.get("SLURM_JOB_ID") or is_inside_condor():
            raise RuntimeError(
                "local_microtest mode cannot run inside a batch scheduler"
            )
        receipt = _arg(args, "local_admission_receipt", None)
        if not receipt:
            raise RuntimeError(
                "local_microtest mode requires a hashed admission receipt"
            )
        admission = load_local_microtest_admission_receipt(receipt)
        if (
            not 1 <= int(_arg(args, "max_steps", 0)) <= 10
            or not 1 <= int(_arg(args, "batch_size", 0)) <= 2
        ):
            raise RuntimeError("local_microtest exceeds the admitted step/batch limits")
        limits = admission["limits"]
        if int(_arg(args, "max_steps", 0)) > int(limits["maximum_steps"]) or int(
            _arg(args, "batch_size", 0)
        ) > int(limits["maximum_batch_size"]):
            raise RuntimeError("local_microtest exceeds its receipt-specific limits")
        if not _one_visible_device(os.environ.get("CUDA_VISIBLE_DEVICES", "")):
            raise RuntimeError("local_microtest requires exactly one visible GPU")
        sentinel = os.environ.get("HYPERTAGGING_LOCAL_WATCHDOG_SENTINEL", "")
        if not sentinel:
            raise RuntimeError("local_microtest requires the active watchdog sentinel")
        validate_local_watchdog_sentinel(
            sentinel, receipt_sha256=str(admission["receipt_sha256"])
        )
        return
    if is_inside_condor():
        return
    raise RuntimeError(
        "CUDA must run inside HTCondor or use the explicit scientific_slurm, "
        "slurm_diagnostic, or receipt-backed local_microtest mode."
    )
