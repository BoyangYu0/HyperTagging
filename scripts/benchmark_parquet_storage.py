#!/usr/bin/env python
"""Benchmark event_json-v4 against native nested Arrow-v5 on a bounded sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.preprocessing.schema_v5 import benchmark_storage_formats
from hypertagging.training.data_module import resolve_data_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    args = parser.parse_args()
    print(
        benchmark_storage_formats(
            resolve_data_paths(args.data),
            args.output_dir,
            max_events=args.max_events,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
