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


@pytest.fixture(scope="session")
def complete_documentation_status_export(tmp_path_factory):
    """Generate and fully validate the immutable real-repository export once.

    Historical download assertions do not alter repository evidence. Sharing
    this expensive projection avoids repeatedly decoding and privacy-checking
    the same complete bundles. Synthetic/mutation tests still call their own
    generator, and the documentation jobs independently regenerate all bytes.
    """
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "complete_documentation_status_fixture", root / "docs/_ext/wiki_status.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp("complete-documentation-status")
    manifest = module.generate_status(root, output)
    return output, manifest


@pytest.fixture
def copy_complete_documentation_status(complete_documentation_status_export):
    """Give each assertion its own files and mutable manifest, never shared state."""
    import copy
    import shutil

    source, manifest = complete_documentation_status_export

    def isolated_copy(output):
        shutil.copytree(source, output)
        return copy.deepcopy(manifest)

    return isolated_copy
