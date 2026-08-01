#!/usr/bin/env python
"""Generate concise active audit views from the machine-readable ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "docs" / "audits" / "issue_ledger.yaml"
BACKLOG = REPO_ROOT / "docs" / "audits" / "current_backlog.md"
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
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    expected = render_backlog(ledger)
    if args.check:
        if not BACKLOG.exists() or BACKLOG.read_text(encoding="utf-8") != expected:
            print(f"STALE: {BACKLOG.relative_to(REPO_ROOT)}")
            return 1
        print("audit generated views PASS")
        return 0
    BACKLOG.write_text(expected, encoding="utf-8")
    print(BACKLOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
