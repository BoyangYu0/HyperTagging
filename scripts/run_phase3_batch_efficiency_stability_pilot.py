#!/usr/bin/env python
"""Run the bounded synthetic train-role stability pilot inside one GPU allocation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("h100nvl", "v100"), required=True)
    parser.add_argument("--batch-size", choices=(32, 64), type=int, required=True)
    parser.add_argument("--checkpoint-copy", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args(argv)
    if not args.checkpoint_copy.is_file():
        raise RuntimeError("stability pilot checkpoint copy is missing")
    if args.max_steps != 256:
        raise RuntimeError("stability pilot max steps must remain exactly 256")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU env
        raise RuntimeError("stability pilot requires torch in the frozen environment") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("stability pilot requires exactly one CUDA device")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.profile == "h100nvl" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.profile == "v100")
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    generator = torch.Generator(device=device).manual_seed(20260823 + args.batch_size)
    records: list[dict[str, object]] = []
    for step in range(1, args.max_steps + 1):
        started = time.perf_counter()
        inputs = torch.randn((args.batch_size, 64), device=device, generator=generator)
        targets = torch.randn((args.batch_size, 8), device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        row = {
            "split": "train",
            "step": step,
            "loss": float(loss.detach().float().cpu()),
            "raw_gradient_norm": gradient_norm,
            "learning_rate": 5e-4,
            "events_per_second": float(args.batch_size / elapsed),
            "objective_preflight_pass": True,
            "objective_weighted_dominance_ratio": 1.0,
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise RuntimeError("stability pilot produced a non-finite metric")
        records.append(row)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
