from pathlib import Path
import importlib.util
import subprocess
import sys

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]


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


def test_notebook_index_has_complete_nonduplicated_responsibilities():
    index = yaml.safe_load((ROOT / "notebooks/index.yaml").read_text())
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
    assert all(entry["last_verified_sha"] == "NOT_RUN" for entry in real_only)


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
