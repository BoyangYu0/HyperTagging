#!/usr/bin/env python
"""Reject drift between project runtime dependencies and root uv lock metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _normalized_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise ValueError(f"cannot parse dependency name from {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def direct_dependency_errors(
    pyproject_path: Path = ROOT / "pyproject.toml",
    lock_path: Path = ROOT / "uv.lock",
) -> list[str]:
    with pyproject_path.open("rb") as stream:
        pyproject = tomllib.load(stream)
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)

    project = pyproject["project"]
    project_name = _normalized_name(str(project["name"]))
    declared = {_normalized_name(str(value)) for value in project["dependencies"]}
    roots = [
        package
        for package in lock["package"]
        if _normalized_name(str(package["name"])) == project_name
        and package.get("source", {}).get("editable") == "."
    ]
    if len(roots) != 1:
        return [f"expected one editable root package {project_name!r}, found {len(roots)}"]

    root = roots[0]
    locked = {_normalized_name(str(item["name"])) for item in root["dependencies"]}
    metadata = {
        _normalized_name(str(item["name"]))
        for item in root["metadata"]["requires-dist"]
        if "extra ==" not in str(item.get("marker", ""))
    }
    errors: list[str] = []
    for label, actual in (("root dependencies", locked), ("root requires-dist", metadata)):
        missing = sorted(declared - actual)
        stale = sorted(actual - declared)
        if missing:
            errors.append(f"{label} missing direct runtime dependencies: {', '.join(missing)}")
        if stale:
            errors.append(f"{label} has undeclared runtime dependencies: {', '.join(stale)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    args = parser.parse_args()
    errors = direct_dependency_errors(args.pyproject, args.lock)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    with args.pyproject.open("rb") as stream:
        count = len(tomllib.load(stream)["project"]["dependencies"])
    print(f"uv direct-dependency metadata PASS: {count} runtime dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
