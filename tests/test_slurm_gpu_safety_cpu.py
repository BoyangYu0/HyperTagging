import hashlib
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
import socket

import pytest

from hypertagging.utils.gpu_safety import (
    assert_scientific_slurm_gpu_allowed,
    assert_full_training_requires_condor,
    load_local_microtest_admission_receipt,
    load_local_microtest_completion_receipt,
)


def _slurm_env(gres: str) -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_GPUS": "0",
        "SLURM_GPUS_ON_NODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def _job_record(gres: str, *, tres_per_node_count: bool = False) -> str:
    gpu_type = gres.split(":")[1]
    tres_per_node = f"gres:gpu:{gpu_type}"
    if tres_per_node_count:
        tres_per_node += ":1"
    return (
        f"JobId=123 JobName=x Gres={gres} "
        f"ReqTRES=cpu=1,mem=8G,node=1,billing=1,gres/gpu=1,gres/gpu:{gpu_type}=1 "
        f"AllocTRES=cpu=2,mem=8G,node=1,billing=2,gres/gpu=1,gres/gpu:{gpu_type}=1 "
        f"TresPerNode={tres_per_node}\n"
    )


@pytest.mark.parametrize(
    ("gres", "name", "tres_per_node_count"),
    (
        ("gpu:h200nvl:1", "NVIDIA H200 NVL", False),
        ("gpu:v100:1", "Tesla V100-SXM2-32GB", True),
    ),
)
def test_scientific_slurm_requires_exact_site_shaped_one_gpu_record(
    gres, name, tres_per_node_count
):
    assert_scientific_slurm_gpu_allowed(
        gres,
        environ=_slurm_env(gres),
        gpu_name=name,
        job_record=_job_record(gres, tres_per_node_count=tres_per_node_count),
    )


@pytest.mark.parametrize(
    ("gres", "environment", "name"),
    (
        ("gpu:1", _slurm_env("gpu:1"), "Tesla V100"),
        ("gpu:v100:1", {"CUDA_VISIBLE_DEVICES": "0"}, "Tesla V100"),
        (
            "gpu:v100:1",
            {**_slurm_env("gpu:v100:1"), "CUDA_VISIBLE_DEVICES": "0,1"},
            "Tesla V100",
        ),
        ("gpu:v100:1", _slurm_env("gpu:v100:1"), "NVIDIA H200 NVL"),
    ),
)
def test_scientific_slurm_fails_closed(gres, environment, name):
    with pytest.raises(RuntimeError):
        assert_scientific_slurm_gpu_allowed(
            gres,
            environ=environment,
            gpu_name=name,
            job_record=_job_record("gpu:v100:1"),
        )


@pytest.mark.parametrize(
    ("req_gpu", "alloc_gpu", "tres_per_node"),
    (
        ("gres/gpu=1,gres/gpu:v100=10", "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100"),
        ("gres/gpu=10,gres/gpu:v100=1", "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100"),
        ("gres/gpu=1,gres/gpu:v100=1", "gres/gpu=1,gres/gpu:v100=10", "gres:gpu:v100"),
        ("gres/gpu=1", "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100"),
        (
            "gres/gpu=1,gres/gpu:v100=1,gres/gpu:h200nvl=1",
            "gres/gpu=1,gres/gpu:v100=1",
            "gres:gpu:v100",
        ),
        ("gres/gpu=1,gres/gpu:v100x=1", "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100"),
        (None, "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100"),
        ("gres/gpu=1,gres/gpu:v100=1", None, "gres:gpu:v100"),
        (
            "gres/gpu=1,gres/gpu:v100=1",
            "gres/gpu=1,gres/gpu:v100=1",
            "gres:gpu:v100:10",
        ),
        (
            "gres/gpu=1,gres/gpu:v100=1",
            "gres/gpu=1,gres/gpu:v100=1",
            "gres:gpu:v100,gres:gpu:h200nvl",
        ),
        ("gres/gpu=1,gres/gpu:v100=1", "gres/gpu=1,gres/gpu:v100=1", "gres:gpu:v100x"),
    ),
)
def test_scientific_slurm_rejects_nonexact_tres_maps(req_gpu, alloc_gpu, tres_per_node):
    fields = ["JobId=123", "JobName=x", "Gres=gpu:v100:1"]
    if req_gpu is not None:
        fields.append(f"ReqTRES=cpu=1,mem=8G,node=1,billing=1,{req_gpu}")
    if alloc_gpu is not None:
        fields.append(f"AllocTRES=cpu=2,mem=8G,node=1,billing=2,{alloc_gpu}")
    fields.append(f"TresPerNode={tres_per_node}")
    with pytest.raises(RuntimeError):
        assert_scientific_slurm_gpu_allowed(
            "gpu:v100:1",
            environ=_slurm_env("gpu:v100:1"),
            gpu_name="Tesla V100-SXM2-32GB",
            job_record=" ".join(fields),
        )


def _write_hashed_receipt(path, payload):
    payload = dict(payload)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    return payload


def _completion_receipts(tmp_path):
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    admission = {
        "receipt_version": "hypertagging-local-v100-admission-v2",
        "mode": "local_microtest",
        "status": "admitted",
        "hostname": socket.gethostname(),
        "gpu_identity": {"uuid": "GPU-1", "model": "Tesla V100-SXM2-32GB"},
        "limits": {
            "maximum_steps": 10,
            "maximum_batch_size": 2,
            "maximum_duration_seconds": 60,
        },
        "samples": [
            {
                "admitted": True,
                "timestamp": (started - timedelta(seconds=20 - index * 10)).isoformat(),
                "gpu": {"uuid": "GPU-1", "name": "Tesla V100-SXM2-32GB"},
            }
            for index in range(3)
        ],
    }
    admission_path = tmp_path / "admission.json"
    admission = _write_hashed_receipt(admission_path, admission)
    completion = {
        "receipt_version": "hypertagging-local-v100-completion-v1",
        "admission_receipt_sha256": admission["receipt_sha256"],
        "hostname": admission["hostname"],
        "gpu_identity": admission["gpu_identity"],
        "started_at": started.isoformat(),
        "completed_at": (started + timedelta(seconds=30)).isoformat(),
        "watchdog_reason": "trainer_exit",
        "trainer_status": 0,
        "sample_count": 2,
        "poll_seconds": 10,
    }
    completion_path = tmp_path / "completion.json"
    completion = _write_hashed_receipt(completion_path, completion)
    return admission_path, completion_path, completion


def test_completion_receipt_proves_successful_bound_monitored_run(tmp_path):
    admission_path, completion_path, completion = _completion_receipts(tmp_path)
    loaded = load_local_microtest_completion_receipt(
        completion_path,
        admission_path=admission_path,
    )
    assert loaded["receipt_sha256"] == completion["receipt_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("admission_receipt_sha256", "b" * 64),
        ("hostname", "different-host"),
        ("gpu_identity", {"uuid": "GPU-2", "model": "Tesla V100-SXM2-32GB"}),
        ("watchdog_reason", "watchdog_deadline"),
        ("watchdog_reason", "telemetry_threshold_or_foreign_process"),
        ("watchdog_reason", "trainer_failure"),
        ("trainer_status", 1),
        ("sample_count", 0),
        ("poll_seconds", 0),
        ("poll_seconds", 31),
    ),
)
def test_completion_receipt_rejects_unbound_failed_or_unmonitored_runs(
    tmp_path, field, value
):
    admission_path, completion_path, completion = _completion_receipts(tmp_path)
    completion[field] = value
    _write_hashed_receipt(
        completion_path,
        {key: item for key, item in completion.items() if key != "receipt_sha256"},
    )
    with pytest.raises(RuntimeError):
        load_local_microtest_completion_receipt(
            completion_path,
            admission_path=admission_path,
        )


def test_completion_receipt_rejects_non_utc_unordered_overlong_and_bad_hash(tmp_path):
    admission_path, completion_path, completion = _completion_receipts(tmp_path)
    base = {key: value for key, value in completion.items() if key != "receipt_sha256"}
    for mutation in (
        {"started_at": "2026-08-12T10:00:00+02:00"},
        {"completed_at": base["started_at"]},
        {
            "completed_at": (
                datetime.fromisoformat(str(base["started_at"])) + timedelta(seconds=91)
            ).isoformat()
        },
    ):
        _write_hashed_receipt(completion_path, {**base, **mutation})
        with pytest.raises(RuntimeError):
            load_local_microtest_completion_receipt(
                completion_path,
                admission_path=admission_path,
            )
    completion_path.write_text('{"receipt_sha256":"bad"}')
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_local_microtest_completion_receipt(
            completion_path,
            admission_path=admission_path,
        )


def test_local_receipt_hash_and_limits_are_enforced(tmp_path):
    now = datetime.now(timezone.utc)
    payload = {
        "receipt_version": "hypertagging-local-v100-admission-v2",
        "mode": "local_microtest",
        "status": "admitted",
        "hostname": socket.gethostname(),
        "gpu_identity": {"uuid": "GPU-1", "model": "Tesla V100-SXM2-32GB"},
        "limits": {
            "maximum_steps": 10,
            "maximum_batch_size": 2,
            "maximum_duration_seconds": 300,
        },
        "samples": [
            {
                "admitted": True,
                "timestamp": (now - timedelta(seconds=20 - index * 10)).isoformat(),
                "gpu": {"uuid": "GPU-1", "name": "Tesla V100-SXM2-32GB"},
            }
            for index in range(3)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload))
    assert (
        load_local_microtest_admission_receipt(receipt, now=now)["status"] == "admitted"
    )
    args = Namespace(
        device="cuda",
        gpu_execution_mode="local_microtest",
        local_admission_receipt=str(receipt),
        max_steps=10,
        batch_size=2,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("_CONDOR_SCRATCH_DIR", raising=False)
        monkeypatch.delenv("CONDOR_CLUSTER_ID", raising=False)
        monkeypatch.delenv("CONDOR_PROCESS_ID", raising=False)
        with pytest.raises(RuntimeError, match="watchdog sentinel"):
            assert_full_training_requires_condor(args)
    payload["limits"]["maximum_steps"] = 11
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_local_microtest_admission_receipt(receipt)
    payload["limits"]["maximum_steps"] = 0
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="limits"):
        load_local_microtest_admission_receipt(receipt)


def test_local_receipt_rejects_stale_and_wrong_host(tmp_path):
    now = datetime.now(timezone.utc)
    payload = {
        "receipt_version": "hypertagging-local-v100-admission-v2",
        "mode": "local_microtest",
        "status": "admitted",
        "hostname": socket.gethostname(),
        "gpu_identity": {"uuid": "GPU-1", "model": "Tesla V100"},
        "limits": {
            "maximum_steps": 1,
            "maximum_batch_size": 1,
            "maximum_duration_seconds": 1,
        },
        "samples": [
            {
                "admitted": True,
                "timestamp": (
                    now - timedelta(minutes=5, seconds=20 - index * 10)
                ).isoformat(),
                "gpu": {"uuid": "GPU-1", "name": "Tesla V100"},
            }
            for index in range(3)
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="stale"):
        load_local_microtest_admission_receipt(receipt, now=now)
    with pytest.raises(RuntimeError, match="different host"):
        load_local_microtest_admission_receipt(
            receipt, now=now - timedelta(minutes=5), hostname="wrong-host"
        )
