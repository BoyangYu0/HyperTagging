#!/usr/bin/env python
"""Generate concise active audit views from the machine-readable ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import re
from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "docs" / "audits" / "issue_ledger.yaml"
BACKLOG = REPO_ROOT / "docs" / "audits" / "current_backlog.md"
ARCHIVE_ROOT = REPO_ROOT / "docs" / "audits" / "archive"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "manifest.yaml"
BACKLOG_STATUSES = {
    "IMPLEMENTED_NOT_REAL_VERIFIED",
    "PARTIAL",
    "OPEN",
    "INTENTIONALLY_DEFERRED_SCIENCE",
}


def render_backlog(ledger: dict[str, object]) -> str:
    items = list(ledger.get("items", []))
    counts = Counter(str(item["current_status"]) for item in items)
    lines = [
        "# Current audit backlog",
        "",
        "This file is generated from `issue_ledger.yaml` by",
        "`scripts/generate_audit_views.py`; do not edit status counts here.",
        "",
        "## Ledger status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in sorted(counts):
        lines.append(f"| `{status}` | {counts[status]} |")
    lines.extend(["", "## Unresolved or externally bounded items", ""])
    for item in items:
        status = str(item["current_status"])
        if status not in BACKLOG_STATUSES:
            continue
        requirements = item.get("external_evidence_required", [])
        evidence = ", ".join(str(value) for value in requirements) or "none"
        lines.extend(
            [
                f"### {item['id']}: {item['title']}",
                "",
                f"- Status: `{status}`",
                f"- External evidence: {evidence}",
                f"- Notes: {item['notes']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Issue-evidence matrix",
            "",
            "Generated directly from the ledger; paths are repository-relative.",
            "",
            "| Issue | Status | Source evidence | Focused tests | Notebook evidence | External evidence | Last verified code SHA |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in items:
        def joined(field: str) -> str:
            values = item.get(field, [])
            if not isinstance(values, list):
                values = [values]
            return "<br>".join(str(value).replace("|", "\\|") for value in values) or "none"

        lines.append(
            "| {id} | `{status}` | {sources} | {tests} | {notebooks} | "
            "{external} | `{sha}` |".format(
                id=item["id"],
                status=item["current_status"],
                sources=joined("current_evidence_files"),
                tests=joined("current_tests"),
                notebooks=joined("notebook_evidence"),
                external=joined("external_evidence_required"),
                sha=item["last_verified_sha"],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _resolve_short_sha(value: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{value}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else value


def _archive_type(name: str) -> str:
    lowered = name.lower()
    if "definition" in lowered or "retained_tree" in lowered:
        return "definition"
    if "completion" in lowered or "readiness_report" in lowered:
        return "completion"
    if "gap" in lowered or "revision_audit" in lowered or "review" in lowered:
        return "gap"
    return "verification"


def render_archive_manifest() -> str:
    reports = []
    for path in sorted(ARCHIVE_ROOT.glob("*.md")):
        match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<sha>[0-9a-f]{7,40})_", path.name)
        text = path.read_text(encoding="utf-8")
        dirty = bool(re.search(r"\bdirty\b|uncommitted", text, re.IGNORECASE))
        clean = bool(re.search(r"\bclean\b", text, re.IGNORECASE))
        worktree = "dirty" if dirty else "clean" if clean else "not_recorded"
        reports.append(
            {
                "date": match.group("date") if match else "unknown",
                "audited_sha": _resolve_short_sha(match.group("sha")) if match else "unknown",
                "audit_type": _archive_type(path.name),
                "historical_filename": path.name,
                "superseded_by": "../current_status.md",
                "evidence_scope": "immutable historical report contents",
                "worktree_state": worktree,
            }
        )
    return yaml.safe_dump(
        {"schema_version": "hypertagging-audit-archive-manifest-v1", "reports": reports},
        sort_keys=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    expected = render_backlog(ledger)
    expected_manifest = render_archive_manifest()
    if args.check:
        stale = []
        if not BACKLOG.exists() or BACKLOG.read_text(encoding="utf-8") != expected:
            stale.append(BACKLOG)
        if not ARCHIVE_MANIFEST.exists() or ARCHIVE_MANIFEST.read_text(encoding="utf-8") != expected_manifest:
            stale.append(ARCHIVE_MANIFEST)
        if stale:
            for path in stale:
                print(f"STALE: {path.relative_to(REPO_ROOT)}")
            return 1
        print("audit generated views PASS")
        return 0
    BACKLOG.write_text(expected, encoding="utf-8")
    ARCHIVE_MANIFEST.write_text(expected_manifest, encoding="utf-8")
    print(BACKLOG)
    print(ARCHIVE_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
