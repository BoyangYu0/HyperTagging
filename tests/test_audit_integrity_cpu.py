from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    path = ROOT / "scripts/validate_audit_integrity.py"
    spec = importlib.util.spec_from_file_location("audit_integrity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _audit_history_fixture(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "config", "user.email", "audit@example.invalid")
    (repo / "src").mkdir()
    (repo / "notebooks").mkdir()
    (repo / "docs").mkdir()
    (repo / "src/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    notebook = {
        "cells": [{"cell_type": "markdown", "id": "old", "metadata": {}, "source": ["# x"]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    (repo / "notebooks/example.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (repo / "docs/audit.md").write_text("audit\n", encoding="utf-8")
    audited = _commit(repo, "audited source")
    ledger: dict[str, object] = {
        "audited_code_sha": audited,
        "allowed_post_audit_paths": ["docs/**", "notebooks/*.ipynb"],
    }
    return repo, audited, ledger


def test_audit_archive_and_issue_ledger_integrity():
    completed = subprocess.run(
        [sys.executable, "scripts/validate_audit_integrity.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "audit integrity PASS" in completed.stdout


def test_every_ledger_issue_has_exactly_one_allowed_status():
    ledger = yaml.safe_load((ROOT / "docs/audits/issue_ledger.yaml").read_text())
    allowed = {
        "FIXED_AND_TESTED",
        "IMPLEMENTED_NOT_REAL_VERIFIED",
        "PARTIAL",
        "OPEN",
        "INTENTIONALLY_DEFERRED_SCIENCE",
        "OBSOLETE_OR_DUPLICATE",
    }
    assert ledger["items"]
    assert len({item["id"] for item in ledger["items"]}) == len(ledger["items"])
    assert all(item["current_status"] in allowed for item in ledger["items"])


def test_generated_backlog_matrix_counts_and_archive_metadata_are_authoritative():
    ledger = yaml.safe_load((ROOT / "docs/audits/issue_ledger.yaml").read_text())
    backlog = (ROOT / "docs/audits/current_backlog.md").read_text()
    matrix = (ROOT / "docs/audits/evidence_matrix.md").read_text()
    current = (ROOT / "docs/audits/current_status.md").read_text()
    unresolved = {
        "IMPLEMENTED_NOT_REAL_VERIFIED", "PARTIAL", "OPEN",
        "INTENTIONALLY_DEFERRED_SCIENCE",
    }
    for item in ledger["items"]:
        assert f"| {item['id']} |" in matrix
        if item["current_status"] in unresolved:
            assert f"### {item['id']}:" in backlog
        else:
            assert f"### {item['id']}:" not in backlog
    for status in {item["current_status"] for item in ledger["items"]}:
        count = sum(item["current_status"] == status for item in ledger["items"])
        assert f"| `{status}` | {count} |" in current
    metadata = yaml.safe_load((ROOT / "docs/audits/archive/metadata.yaml").read_text())
    assert metadata["schema_version"].endswith("v1")
    assert all(entry["worktree_state"] in {"clean", "dirty", "not_recorded"} for entry in metadata["reports"])
    assert all(len(entry["sha256"]) == 64 for entry in metadata["reports"])


def test_generated_audit_views_are_current():
    completed = subprocess.run(
        [sys.executable, "scripts/generate_audit_views.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS" in completed.stdout


def test_notebook_index_has_complete_nonduplicated_responsibilities():
    index = yaml.safe_load((ROOT / "notebooks/index.yaml").read_text())
    ledger = yaml.safe_load((ROOT / "docs/audits/issue_ledger.yaml").read_text())
    entries = index["notebooks"]
    required = {
        "group",
        "path",
        "generator",
        "fixture_or_real",
        "required_inputs",
        "machine_readable_outputs",
        "scientific_claims_allowed",
        "CI_frequency",
        "last_verified_sha",
    }
    groups = {
        "CORE_CONTRACT",
        "EXTENDED_ENGINEERING",
        "HISTORICAL_COMPATIBILITY",
        "EXTERNAL_SCIENTIFIC",
        "DIAGNOSTIC",
    }
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert all(required <= set(entry) for entry in entries)
    assert all(entry["group"] in groups for entry in entries)
    assert all((ROOT / entry["path"]).exists() for entry in entries)
    assert all((ROOT / entry["generator"]).exists() for entry in entries)
    real_only = [entry for entry in entries if entry["fixture_or_real"] == "real_only"]
    assert {entry["id"] for entry in real_only} == {
        "real_mdst_pilot",
        "trained_physics_validation",
    }
    real_status = {entry["id"]: entry["last_verified_sha"] for entry in real_only}
    pilot_sha = real_status["real_mdst_pilot"]
    assert pilot_sha == "NOT_RUN" or subprocess.run(
        ["git", "merge-base", "--is-ancestor", pilot_sha, ledger["audited_code_sha"]],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert real_status["trained_physics_validation"] == "NOT_RUN"


def test_generated_notebook_cell_ids_are_deterministic():
    module_path = ROOT / "scripts/execute_notebook_smoke_tests.py"
    spec = importlib.util.spec_from_file_location("notebook_smoke_runner", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Contract"),
            nbformat.v4.new_code_cell("answer = 42"),
        ]
    )
    second = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Contract"),
            nbformat.v4.new_code_cell("answer = 42"),
        ]
    )
    module._stabilize_notebook_cell_ids(first, "contract.ipynb")
    module._stabilize_notebook_cell_ids(second, "contract.ipynb")
    assert [cell.id for cell in first.cells] == [cell.id for cell in second.cells]
    assert len({cell.id for cell in first.cells}) == len(first.cells)


def test_validation_overview_preserves_real_only_not_run(tmp_path):
    module_path = ROOT / "scripts/execute_notebook_smoke_tests.py"
    spec = importlib.util.spec_from_file_location("notebook_overview_runner", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._write_validation_overview(tmp_path, ["dataset"], [])
    payload = json.loads((tmp_path / "validation_overview.json").read_text())
    by_id = {row["notebook_id"]: row for row in payload["rows"]}
    assert by_id["dataset"]["status"] == "PASS"
    assert by_id["real_mdst_pilot"]["status"] == "NOT_RUN"
    assert by_id["trained_physics_validation"]["status"] == "NOT_RUN"
    assert payload["visual_review_status"] == "NOT_REVIEWED"


def test_audited_code_sha_equal_to_head_passes(tmp_path):
    repo, _, ledger = _audit_history_fixture(tmp_path)
    assert _load_validator().validate_audited_code_ancestry(ledger, repo) == []


def test_audit_only_commit_after_audited_code_passes(tmp_path):
    repo, _, ledger = _audit_history_fixture(tmp_path)
    (repo / "docs/audit.md").write_text("updated audit\n", encoding="utf-8")
    _commit(repo, "audit only")
    assert _load_validator().validate_audited_code_ancestry(ledger, repo) == []


def test_notebook_cell_id_only_commit_after_audited_code_passes(tmp_path):
    repo, _, ledger = _audit_history_fixture(tmp_path)
    path = repo / "notebooks/example.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"][0]["id"] = "stable"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    _commit(repo, "normalize notebook cell id")
    assert _load_validator().validate_audited_code_ancestry(ledger, repo) == []


def test_source_commit_after_audited_code_fails(tmp_path):
    repo, _, ledger = _audit_history_fixture(tmp_path)
    (repo / "src/model.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(repo, "source change")
    errors = _load_validator().validate_audited_code_ancestry(ledger, repo)
    assert errors == ["post-audit path is outside the allowlist: src/model.py"]


def test_missing_or_nonancestor_audited_code_sha_fails(tmp_path):
    repo, _, ledger = _audit_history_fixture(tmp_path)
    missing = dict(ledger, audited_code_sha="f" * 40)
    assert "does not identify a commit" in _load_validator().validate_audited_code_ancestry(
        missing, repo
    )[0]
    _git(repo, "checkout", "--orphan", "unrelated")
    (repo / "src/model.py").write_text("VALUE = 3\n", encoding="utf-8")
    unrelated = _commit(repo, "unrelated history")
    nonancestor = dict(ledger, audited_code_sha=unrelated)
    _git(repo, "checkout", "master")
    assert "not an ancestor" in _load_validator().validate_audited_code_ancestry(
        nonancestor, repo
    )[0]
