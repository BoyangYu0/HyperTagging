"""Dry-run wrapper for legacy Toy-MC preprocessing."""

from __future__ import annotations

import argparse

from hypertagging.data.toy_mc import prepare_toy_mc, prepare_toy_mc_dataprod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--dataprod", action="store_true")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--hdf5", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run the legacy script instead of dry-run.")
    args = parser.parse_args()

    if args.dataprod:
        result = prepare_toy_mc_dataprod(args.channel_index, args.job_id, dry_run=not args.run)
    else:
        result = prepare_toy_mc(args.job_id, awkward_output=not args.hdf5, dry_run=not args.run)
    if hasattr(result, "command"):
        print(" ".join(result.command))
    else:
        print(result.stdout)


if __name__ == "__main__":
    main()
