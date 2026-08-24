#!/usr/bin/env python3
"""Periodically capture one-GPU nvidia-smi telemetry and an atomic summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import threading
import time


QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sample(nvidia_smi: str) -> dict[str, object]:
    result = subprocess.run(
        (
            nvidia_smi,
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, observed {len(rows)}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(QUERY_FIELDS):
        raise RuntimeError("unexpected nvidia-smi telemetry shape")
    return {
        "timestamp": _timestamp(),
        "gpu_index": int(values[0]),
        "gpu_uuid": values[1],
        "gpu_name": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "gpu_utilization_percent": int(values[5]),
        "memory_utilization_percent": int(values[6]),
        "temperature_c": int(values[7]),
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def monitor(
    *,
    output: Path,
    summary: Path,
    interval_seconds: float,
    nvidia_smi: str,
    max_samples: int | None = None,
) -> int:
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    samples: list[dict[str, object]] = []
    error: str | None = None
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        while not stop.is_set():
            try:
                sample = _sample(nvidia_smi)
            except Exception as caught:  # preserve a terminal error record
                error = str(caught)
                stream.write(json.dumps({"timestamp": _timestamp(), "error": error}) + "\n")
                break
            samples.append(sample)
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            if max_samples is not None and len(samples) >= max_samples:
                break
            stop.wait(interval_seconds)
    payload: dict[str, object] = {
        "telemetry_version": "hypertagging-slurm-gpu-telemetry-v1",
        "status": "completed" if error is None else "failed",
        "started_at": started_at,
        "completed_at": _timestamp(),
        "interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "error": error,
        "peak_memory_used_mib": max(
            (int(sample["memory_used_mib"]) for sample in samples), default=None
        ),
        "peak_gpu_utilization_percent": max(
            (int(sample["gpu_utilization_percent"]) for sample in samples),
            default=None,
        ),
        "peak_temperature_c": max(
            (int(sample["temperature_c"]) for sample in samples), default=None
        ),
    }
    _write_atomic(summary, payload)
    return 0 if error is None and samples else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return monitor(
        output=args.output,
        summary=args.summary,
        interval_seconds=args.interval_seconds,
        nvidia_smi=args.nvidia_smi,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
