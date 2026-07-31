from pathlib import Path
import subprocess
import sys

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
