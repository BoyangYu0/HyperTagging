"""Explicit prerequisites for tests that verify private campaign evidence."""
from pathlib import Path
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-campaign-artifacts", action="store_true",
        help="Fail rather than skip when declared campaign evidence is unavailable.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "campaign_artifacts(*paths): requires external campaign files or git:refs"
    )


def pytest_runtest_setup(item):
    root = Path(__file__).resolve().parents[1]
    missing = []
    for marker in item.iter_markers("campaign_artifacts"):
        for value in marker.args:
            if value.startswith("git:"):
                present = subprocess.run(
                    ["git", "rev-parse", "--verify", value[4:] + "^{commit}"],
                    cwd=root, capture_output=True, check=False,
                ).returncode == 0
            else:
                present = (root / value).exists()
            if not present:
                missing.append(value)
    if missing:
        reason = "External campaign evidence unavailable: " + ", ".join(missing)
        if item.config.getoption("--require-campaign-artifacts"):
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
