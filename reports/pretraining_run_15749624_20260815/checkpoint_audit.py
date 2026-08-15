#!/usr/bin/env python3
"""Produce a compact, machine-readable integrity inventory of training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory(value: object) -> dict[str, int]:
    tensors = 0
    elements = 0
    nonfinite = 0
    if isinstance(value, torch.Tensor):
        tensors = 1
        elements = value.numel()
        if value.is_floating_point() or value.is_complex():
            nonfinite = int((~torch.isfinite(value)).sum().item())
    elif isinstance(value, dict):
        for item in value.values():
            child = tensor_inventory(item)
            tensors += child["tensors"]
            elements += child["elements"]
            nonfinite += child["nonfinite"]
    elif isinstance(value, (list, tuple)):
        for item in value:
            child = tensor_inventory(item)
            tensors += child["tensors"]
            elements += child["elements"]
            nonfinite += child["nonfinite"]
    return {"tensors": tensors, "elements": elements, "nonfinite": nonfinite}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = tensor_inventory(payload.get("model_state_dict", payload))
        optimizer = tensor_inventory(payload.get("optimizer_state_dict", {}))
        scaler = payload.get("scaler_state_dict") or {}
        selection = payload.get("validation_selection") or {}
        event_uids = selection.get("event_uids") or []
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "step": payload.get("step"),
                "git_sha": payload.get("git_sha"),
                "split_manifest_hash": payload.get("split_manifest_hash"),
                "model": model,
                "optimizer": optimizer,
                "amp_scale": scaler.get("scale"),
                "amp_growth_tracker": scaler.get("_growth_tracker"),
                "validation_event_count": len(event_uids),
                "validation_selection_manifest_hash": selection.get(
                    "selection_manifest_hash"
                ),
                "checkpoint_metric": payload.get("metrics", {}).get("loss"),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
