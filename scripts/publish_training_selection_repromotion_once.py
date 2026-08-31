#!/usr/bin/env python3
"""Consume one exact authorization for inert metadata-package publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

from hypertagging.data.selection_repromotion_publication import (
    AUTHORITY_PARENT,
    CommandContext,
    TRUSTED_EXECUTION_UID,
    publish_authorized_once,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "configs/training_selection/repromotion/train_035k_16key_repromotion.contract.json"
)
BOUND_ENVIRONMENT_KEYS = ("PATH", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if not args.authorization.is_absolute():
        parser.error("--authorization must be absolute")
    if os.geteuid() != TRUSTED_EXECUTION_UID:
        parser.error("publication authorization trusted execution uid mismatch")
    if args.authorization.parent != AUTHORITY_PARENT:
        parser.error("publication authorization is outside the sealed authority parent")
    authority_metadata = AUTHORITY_PARENT.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(authority_metadata.st_mode)
        or authority_metadata.st_uid != TRUSTED_EXECUTION_UID
        or stat.S_IMODE(authority_metadata.st_mode) != 0o700
    ):
        parser.error("publication authority parent must be owned mode-0700 directory")
    authorization_metadata = args.authorization.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(authorization_metadata.st_mode)
        or authorization_metadata.st_uid != TRUSTED_EXECUTION_UID
        or authorization_metadata.st_nlink != 1
        or stat.S_IMODE(authorization_metadata.st_mode) != 0o444
    ):
        parser.error("publication authorization wrapper precheck failed")
    missing = [key for key in BOUND_ENVIRONMENT_KEYS if key not in os.environ]
    if missing:
        parser.error(f"bound environment is incomplete: {missing}")
    context = CommandContext(
        cwd=os.getcwd(),
        argv=tuple([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        environment={key: os.environ[key] for key in BOUND_ENVIRONMENT_KEYS},
    )
    paths = publish_authorized_once(
        args.authorization,
        repository_root=ROOT,
        contract_path=CONTRACT,
        command_context=context,
        authority_parent=AUTHORITY_PARENT,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
