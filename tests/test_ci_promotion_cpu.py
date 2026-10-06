"""Promotion must use successful push checks for the exact branch and commit."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "promotion", Path(__file__).resolve().parents[1] / "scripts/promote_verified_commit.py")
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)


def runs():
    return [dict(id=i, path=path, head_sha="candidate", head_branch="feature",
                 event="push", status="completed", conclusion="success", html_url="run")
            for i, path in enumerate(promotion.REQUIRED_WORKFLOWS, 1)]


def test_exact_commit_pair_passes():
    assert len(promotion.verified_runs(runs(), "candidate", "feature")) == 2


@pytest.mark.parametrize("field,value", [
    ("head_sha", "old"), ("head_branch", "master"), ("event", "pull_request"),
    ("status", "in_progress"), ("conclusion", "failure"), ("conclusion", "cancelled"),
])
def test_wrong_identity_or_incomplete_result_refuses(field, value):
    evidence = runs()
    evidence[0][field] = value
    with pytest.raises(ValueError):
        promotion.verified_runs(evidence, "candidate", "feature")


def test_new_failure_cannot_be_hidden_by_old_success():
    evidence = runs()
    evidence.append(dict(evidence[0], id=99, conclusion="failure"))
    with pytest.raises(ValueError):
        promotion.verified_runs(evidence, "candidate", "feature")


def test_missing_workflow_refuses():
    with pytest.raises(ValueError):
        promotion.verified_runs(runs()[:1], "candidate", "feature")
