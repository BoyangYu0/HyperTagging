#!/usr/bin/env python
"""Verify the versioned phase-3 authorization and optional fresh preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_execution_authorization_v1 import (
    AUTHORIZATION_ARTIFACT,
    BASE_CONTRACT,
    ROOT,
    verify_authorization_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / AUTHORIZATION_ARTIFACT,
    )
    parser.add_argument("--contract", type=Path, default=ROOT / BASE_CONTRACT)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--require-fresh-in-allocation-preflight", action="store_true")
    args = parser.parse_args()
    payload = verify_authorization_artifact(
        args.authorization,
        contract_path=args.contract,
        require_fresh_preflight=args.require_fresh_in_allocation_preflight,
        preflight_path=args.preflight,
    )
    print(
        json.dumps(
            {
                "authorization_verified": True,
                "authorization_artifact_sha256": payload["artifact_sha256"],
                "fresh_preflight_verified": bool(args.preflight),
                "submission_authorized": payload["submission_authorized"],
                "submission_performed": payload["submission_performed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
