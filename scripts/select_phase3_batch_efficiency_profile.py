#!/usr/bin/env python
"""Select exactly one production device from two sequential calibration receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SELECTION_VERSION = "ht-pretraining-1m-phase3-batch-efficiency-selection-v1"
EXPECTED_ORDER = ("h100nvl", "v100")
EXPECTED_REMAINING_PRESENTATIONS = 865_024


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    stored = payload.get("receipt_sha256")
    if stored:
        actual_payload = dict(payload)
        actual_payload.pop("receipt_sha256", None)
        if stored != _hash(actual_payload):
            raise RuntimeError(f"calibration receipt hash mismatch: {path}")
    if payload.get("artifact_version") != "ht-pretraining-1m-phase3-gpu-calibration-receipt-v1":
        raise RuntimeError(f"unsupported calibration receipt: {path}")
    if payload.get("calibration_complete") is not True:
        raise RuntimeError(f"calibration is incomplete: {path}")
    if payload.get("scientific_contract", {}).get("submission_performed") is not False:
        raise RuntimeError("calibration receipt reports an unexpected submission")
    if payload.get("checkpoint_copy", {}).get("source_unchanged") is not True:
        raise RuntimeError("calibration source checkpoint was not proven unchanged")
    return payload


def _completion_seconds(receipt: dict[str, Any]) -> float:
    pilot = receipt.get("pilot", {})
    metrics_path = pilot.get("metrics_path")
    if not metrics_path:
        raise RuntimeError("calibration receipt lacks pilot metrics")
    records = [
        json.loads(line)
        for line in Path(metrics_path).read_text().splitlines()
        if line.strip()
    ]
    usable = [
        float(row["events_per_second"])
        for row in records
        if "events_per_second" in row
        and float(row["events_per_second"]) > 0
        and all(not str(key).lower().startswith("validation") for key in row)
    ]
    if not usable:
        raise RuntimeError("calibration metrics have no finite train throughput")
    throughput = sum(usable[-min(3, len(usable)):]) / min(3, len(usable))
    return EXPECTED_REMAINING_PRESENTATIONS / throughput


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-receipt", type=Path, required=True)
    parser.add_argument("--v100-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-production", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("selection manifest exists; refusing to rewrite history")
    receipts = {
        "h100nvl": _load(args.h100_receipt),
        "v100": _load(args.v100_receipt),
    }
    expected_gres = {"h100nvl": "gpu:h100nvl:1", "v100": "gpu:v100:1"}
    candidates: dict[str, dict[str, Any]] = {}
    for name, receipt in receipts.items():
        if receipt.get("profile", {}).get("exact_gres") != expected_gres[name]:
            raise RuntimeError(f"receipt GRES mismatch for {name}")
        candidates[name] = {
            "exact_gres": expected_gres[name],
            "batch_size": receipt["profile"]["preferred_batch_size"],
            "expected_completion_seconds": _completion_seconds(receipt),
            "receipt": str(
                (args.h100_receipt if name == "h100nvl" else args.v100_receipt)
                .resolve()
            ),
        }
    selected = min(
        candidates,
        key=lambda name: (
            candidates[name]["expected_completion_seconds"],
            name != "h100nvl",
        ),
    )
    manifest: dict[str, Any] = {
        "artifact_version": SELECTION_VERSION,
        "calibration_order": list(EXPECTED_ORDER),
        "calibration_sequential_only": True,
        "candidates": candidates,
        "selected_profile": selected,
        "selection_metric": "earliest_expected_completion_seconds",
        "selection_basis": "measured_train_role_throughput_only",
        "remaining_presentations": EXPECTED_REMAINING_PRESENTATIONS,
        "production_submission_authorized": bool(args.authorize_production),
        "submission_performed": False,
        "job_count": 1,
        "one_viable_scientific_lineage": True,
        "created_at_unix": time.time(),
    }
    manifest["selection_sha256"] = _hash(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
