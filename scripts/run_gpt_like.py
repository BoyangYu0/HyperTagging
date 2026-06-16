#!/usr/bin/env python
"""Dry-run the combined GPT-like/autoregressive stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from hypertagging.training import run_multi_gpt_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()

    summary = run_multi_gpt_dry_run(device=args.device, backward=not args.no_backward)
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
