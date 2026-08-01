#!/usr/bin/env python
"""Validate the active audit index and machine-readable issue ledger."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / "docs" / "audits"
ARCHIVE_ROOT = AUDIT_ROOT / "archive"
README = AUDIT_ROOT / "README.md"
LEDGER = AUDIT_ROOT / "issue_ledger.yaml"
NOTEBOOK_INDEX = REPO_ROOT / "notebooks" / "index.yaml"
ALLOWED_STATUSES = {
    "FIXED_AND_TESTED",
    "IMPLEMENTED_NOT_REAL_VERIFIED",
    "PARTIAL",
    "OPEN",
    "INTENTIONALLY_DEFERRED_SCIENCE",
    "OBSOLETE_OR_DUPLICATE",
}
REQUIRED_FIELDS = {
    "id",
    "title",
    "originating_audit",
    "first_reported_sha",
    "current_status",
    "current_evidence_files",
    "current_tests",
    "notebook_evidence",
    "external_evidence_required",
    "last_verified_sha",
    "notes",
}


def _markdown_targets(text: str) -> list[str]:
    return re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text)


def validate() -> list[str]:
    errors: list[str] = []
    readme_text = README.read_text(encoding="utf-8")
    listed_archives = {
        target
        for target in _markdown_targets(readme_text)
        if target.startswith("archive/")
    }
    actual_archives = {
        path.relative_to(AUDIT_ROOT).as_posix() for path in ARCHIVE_ROOT.glob("*.md")
    }
    if listed_archives != actual_archives:
        errors.append(
            "archive index mismatch: "
            f"missing={sorted(actual_archives - listed_archives)}, "
            f"stale={sorted(listed_archives - actual_archives)}"
        )

    for target in _markdown_targets(readme_text):
        if "://" in target:
            continue
        if not (AUDIT_ROOT / target).exists():
            errors.append(f"invalid audit/supersession link: {target}")

    current_documents = [
        path
        for path in AUDIT_ROOT.glob("*current*status*.md")
        if path.is_file()
    ]
    if current_documents != [AUDIT_ROOT / "current_status.md"]:
        errors.append(
            "expected exactly one current-status document, found "
            + ", ".join(path.name for path in current_documents)
        )

    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    audited_head = str(ledger.get("audited_head", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", audited_head):
        errors.append(f"invalid audited_head: {audited_head!r}")
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if audited_head != actual_head:
        errors.append(
            f"ledger audited_head {audited_head!r} does not match Git HEAD {actual_head!r}"
        )
    identifiers: set[str] = set()
    for index, item in enumerate(ledger.get("items", [])):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            errors.append(f"ledger item {index} misses fields: {sorted(missing)}")
            continue
        identifier = str(item["id"])
        if identifier in identifiers:
            errors.append(f"duplicate ledger id: {identifier}")
        identifiers.add(identifier)
        if item["current_status"] not in ALLOWED_STATUSES:
            errors.append(f"{identifier} has invalid status {item['current_status']!r}")
        for field in (
            "originating_audit",
            "current_evidence_files",
            "current_tests",
            "notebook_evidence",
        ):
            values = item[field] if isinstance(item[field], list) else [item[field]]
            for value in values:
                if not (REPO_ROOT / str(value)).exists():
                    errors.append(f"{identifier} has missing {field} path: {value}")

    active_documents = [
        REPO_ROOT / "docs" / "audit_index.md",
        README,
        AUDIT_ROOT / "current_status.md",
    ]
    old_current_pattern = re.compile(
        r"(?:current|audited)\s+(?:HEAD|SHA|revision)\s*[:=` ]+"
        r"(?!" + re.escape(audited_head) + r")[0-9a-f]{7,40}",
        re.IGNORECASE,
    )
    for path in active_documents:
        text = path.read_text(encoding="utf-8")
        if old_current_pattern.search(text):
            errors.append(f"active audit describes an older SHA as current: {path}")
    current_text = (AUDIT_ROOT / "current_status.md").read_text(encoding="utf-8")
    if audited_head not in current_text:
        errors.append("current_status.md does not name the ledger audited_head")
    if "sole authoritative current audit report" not in current_text:
        errors.append("current_status.md lacks the single-current-document declaration")

    notebook_index = yaml.safe_load(NOTEBOOK_INDEX.read_text(encoding="utf-8"))
    for entry in notebook_index.get("notebooks", []):
        expected = (
            "NOT_RUN"
            if entry.get("fixture_or_real") == "real_only"
            else audited_head
        )
        if str(entry.get("last_verified_sha")) != expected:
            errors.append(
                f"notebook {entry.get('id')} last_verified_sha must be {expected}"
            )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    print(
        f"audit integrity PASS: {len(list(ARCHIVE_ROOT.glob('*.md')))} archives, "
        f"{len(ledger['items'])} ledger items, one current status"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
