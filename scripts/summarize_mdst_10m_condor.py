#!/usr/bin/env python
"""Summarize and verify HTCondor history for the 10M production campaign."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ATTRIBUTES = (
    "ClusterId,ProcId,JobStatus,ExitCode,RemoteWallClockTime,RequestCpus,"
    "RequestMemory,NumJobStarts,HoldReason"
)


def history(cluster_id: int) -> list[dict[str, Any]]:
    command = [
        "condor_history",
        "-constraint",
        f"ClusterId == {cluster_id}",
        "-attributes",
        ATTRIBUTES,
        "-json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def summarize_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [float(job.get("RemoteWallClockTime", 0.0)) for job in jobs]
    return {
        "jobs": len(jobs),
        "job_status_counts": dict(sorted(Counter(job.get("JobStatus") for job in jobs).items())),
        "exit_code_counts": dict(sorted(Counter(job.get("ExitCode") for job in jobs).items())),
        "jobs_with_multiple_starts": sum(int(job.get("NumJobStarts", 0)) > 1 for job in jobs),
        "request_cpus": sorted({job.get("RequestCpus") for job in jobs}),
        "request_memory_mib": sorted({job.get("RequestMemory") for job in jobs}),
        "remote_wall_clock_seconds": {
            "total": sum(runtimes),
            "median": statistics.median(runtimes) if runtimes else None,
            "maximum": max(runtimes) if runtimes else None,
        },
        "hold_reasons": sorted({job.get("HoldReason") for job in jobs if job.get("HoldReason")}),
    }


def stderr_summary(log_dir: Path, cluster_id: int) -> dict[str, Any]:
    files = sorted(log_dir.glob(f"mdst-{cluster_id}.*.err"))
    non_whitespace = []
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            non_whitespace.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "preview": content[:300],
                }
            )
    return {
        "files": len(files),
        "non_whitespace_files": len(non_whitespace),
        "details": non_whitespace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--initial-preflight-cluster", type=int, default=4844425)
    parser.add_argument("--successful-preflight-cluster", type=int, default=4844426)
    parser.add_argument("--bulk-cluster", type=int, default=4844428)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.production_root.resolve()
    log_dir = root / "logs" / "condor"
    clusters = {
        "initial_preflight_rejected": args.initial_preflight_cluster,
        "successful_preflight": args.successful_preflight_cluster,
        "bulk": args.bulk_cluster,
    }
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "clusters": {},
    }
    raw: dict[str, list[dict[str, Any]]] = {}
    for role, cluster_id in clusters.items():
        jobs = history(cluster_id)
        raw[role] = jobs
        report["clusters"][role] = {
            "cluster_id": cluster_id,
            "history": summarize_jobs(jobs),
            "stderr": stderr_summary(log_dir, cluster_id),
        }
    report["initial_preflight_disposition"] = {
        "accepted": False,
        "reason": "Unix environment entries were separated with semicolons; the worker rejected its pre-processing environment before reading a task.",
        "produced_shard": False,
    }
    report["successful_campaign_checks"] = {
        "preflight_jobs": len(raw["successful_preflight"]),
        "preflight_all_exit_zero": all(job.get("ExitCode") == 0 for job in raw["successful_preflight"]),
        "bulk_jobs": len(raw["bulk"]),
        "bulk_all_completed": all(job.get("JobStatus") == 4 for job in raw["bulk"]),
        "bulk_all_exit_zero": all(job.get("ExitCode") == 0 for job in raw["bulk"]),
        "bulk_non_whitespace_stderr": report["clusters"]["bulk"]["stderr"]["non_whitespace_files"],
    }
    checks = report["successful_campaign_checks"]
    if not (
        checks["preflight_jobs"] == 1
        and checks["preflight_all_exit_zero"]
        and checks["bulk_jobs"] == 1999
        and checks["bulk_all_completed"]
        and checks["bulk_all_exit_zero"]
        and checks["bulk_non_whitespace_stderr"] == 0
    ):
        raise RuntimeError(f"Condor campaign is not cleanly complete: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
