"""Dry-run wrapper for legacy GraFEI preprocessing."""

from __future__ import annotations

import argparse

from hypertagging.data.gpt_like import prepare_gpt_like
from hypertagging.data.grafei import prepare_grafei


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk_index", type=int)
    parser.add_argument("--gpt-like", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use graFEI instead of graFEI_reduced.")
    parser.add_argument("--run", action="store_true", help="Run the legacy script instead of dry-run.")
    args = parser.parse_args()

    if args.gpt_like:
        result = prepare_gpt_like(args.chunk_index, dry_run=not args.run)
    else:
        result = prepare_grafei(args.chunk_index, reduced=not args.full, dry_run=not args.run)
    if hasattr(result, "command"):
        print(" ".join(result.command))
    else:
        print(result.stdout)


if __name__ == "__main__":
    main()
