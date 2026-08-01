#!/usr/bin/env python
"""Generate all active audit views from structured audit authorities."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / "docs" / "audits"
LEDGER = AUDIT_ROOT / "issue_ledger.yaml"
BACKLOG = AUDIT_ROOT / "current_backlog.md"
EVIDENCE_MATRIX = AUDIT_ROOT / "evidence_matrix.md"
README = AUDIT_ROOT / "README.md"
CURRENT_STATUS = AUDIT_ROOT / "current_status.md"
VERIFICATION_RUNS = AUDIT_ROOT / "verification_runs.yaml"
ARCHIVE_ROOT = AUDIT_ROOT / "archive"
ARCHIVE_METADATA = ARCHIVE_ROOT / "metadata.yaml"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "manifest.yaml"
ARCHIVE_INDEX = ARCHIVE_ROOT / "index.md"
BACKLOG_STATUSES = {
    "IMPLEMENTED_NOT_REAL_VERIFIED",
    "PARTIAL",
    "OPEN",
    "INTENTIONALLY_DEFERRED_SCIENCE",
}
SUMMARY_START = "<!-- GENERATED_STATUS_SUMMARY_START -->"
SUMMARY_END = "<!-- GENERATED_STATUS_SUMMARY_END -->"


def _values(item: dict[str, object], field: str) -> list[str]:
    value = item.get(field, [])
    if not isinstance(value, list):
        value = [value]
    return [str(entry) for entry in value]


def _joined(item: dict[str, object], field: str) -> str:
    return "<br>".join(value.replace("|", "\\|") for value in _values(item, field)) or "none"


def render_backlog(ledger: dict[str, object]) -> str:
    unresolved = [
        item for item in ledger.get("items", [])
        if str(item["current_status"]) in BACKLOG_STATUSES
    ]
    counts = Counter(str(item["current_status"]) for item in unresolved)
    lines = [
        "# Current audit backlog",
        "",
        "This unresolved-only view is generated from `issue_ledger.yaml`; do not edit it",
        "by hand. The complete source/test/notebook mapping is in",
        "[`evidence_matrix.md`](evidence_matrix.md).",
        "",
        "## Unresolved status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in sorted(counts):
        lines.append(f"| `{status}` | {counts[status]} |")
    lines.extend(["", "## Items", ""])
    for item in unresolved:
        external = ", ".join(_values(item, "external_evidence_required")) or "none"
        lines.extend(
            [
                f"### {item['id']}: {item['title']}",
                "",
                f"- Status: `{item['current_status']}`",
                f"- Next external evidence: {external}",
                f"- Current boundary: {item['notes']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_evidence_matrix(ledger: dict[str, object]) -> str:
    lines = [
        "# Audit issue-evidence matrix",
        "",
        "Generated from `issue_ledger.yaml`; the ledger remains authoritative.",
        "",
        "| Issue | Status | Source evidence | Focused tests | Notebook evidence | External evidence | Last verified source SHA |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in ledger.get("items", []):
        lines.append(
            f"| {item['id']} | `{item['current_status']}` | "
            f"{_joined(item, 'current_evidence_files')} | {_joined(item, 'current_tests')} | "
            f"{_joined(item, 'notebook_evidence')} | {_joined(item, 'external_evidence_required')} | "
            f"`{item['last_verified_sha']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _reports(metadata: dict[str, object]) -> list[dict[str, object]]:
    return sorted(
        list(metadata.get("reports", [])),
        key=lambda entry: (str(entry["date"]), str(entry["historical_filename"])),
    )


def render_archive_manifest(metadata: dict[str, object]) -> str:
    fields = (
        "date", "audited_sha", "audit_type", "historical_filename",
        "superseded_by", "evidence_scope", "worktree_state", "sha256",
    )
    reports = [{field: entry[field] for field in fields} for entry in _reports(metadata)]
    return yaml.safe_dump(
        {"schema_version": "hypertagging-audit-archive-manifest-v2", "reports": reports},
        sort_keys=False,
    )


def _grouped_reports(metadata: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in _reports(metadata):
        grouped[str(entry["audited_sha"])].append(entry)
    return grouped


def render_archive_index(metadata: dict[str, object]) -> str:
    lines = [
        "# Immutable audit archive",
        "",
        "This index is generated from explicit `metadata.yaml`. Report contents and",
        "recorded digests are immutable evidence; historical claims are not current truth.",
        "",
    ]
    for sha, entries in _grouped_reports(metadata).items():
        lines.extend([f"## `{sha}`", ""])
        for entry in entries:
            label = str(entry["historical_filename"])
            lines.append(
                f"- {entry['date']} · {entry['audit_type']} · {entry['worktree_state']} · "
                f"[{label}]({label})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_readme(metadata: dict[str, object]) -> str:
    lines = [
        "# Repository audit history",
        "",
        "[`current_status.md`](current_status.md) is the sole authoritative current audit",
        "report. [`issue_ledger.yaml`](issue_ledger.yaml) is the machine-readable authority;",
        "[`current_backlog.md`](current_backlog.md) and",
        "[`evidence_matrix.md`](evidence_matrix.md) are generated views. Structured complete",
        "runs are recorded in [`verification_runs.yaml`](verification_runs.yaml).",
        "",
        "Every report under `archive/` is an immutable historical evidence snapshot. Its",
        "numerical results, SHA claims, and worktree descriptions apply only to the state",
        "inspected at the time. Archive inventory and worktree state come only from explicit",
        "[`archive/metadata.yaml`](archive/metadata.yaml), never prose inference.",
        "",
        "## Historical snapshots",
        "",
    ]
    for sha, entries in _grouped_reports(metadata).items():
        lines.extend([f"### `{sha}`", ""])
        for entry in entries:
            label = str(entry["historical_filename"])
            lines.append(
                f"- {entry['date']} · {entry['audit_type']} · "
                f"[historical report](archive/{label}) · superseded by "
                f"[current status]({str(entry['superseded_by']).removeprefix('../')})"
            )
        lines.append("")
    lines.extend(
        [
            "## Evidence boundary",
            "",
            "The current non-self-referential source boundary and exact post-boundary",
            "allowlist are declared in `issue_ledger.yaml`. The validator requires that the",
            "source commit be an ancestor of HEAD and rejects every later non-audit path.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_current_status_summary(
    current: str,
    ledger: dict[str, object],
    verification_runs: dict[str, object],
) -> str:
    counts = Counter(str(item["current_status"]) for item in ledger.get("items", []))
    runs = list(verification_runs.get("runs", []))
    latest = runs[-1] if runs else {}
    pytest_result = latest.get("pytest", {})
    block = [
        SUMMARY_START,
        "## Generated authoritative summary",
        "",
        f"- Audited source SHA: `{ledger['audited_code_sha']}`",
        "- Audit metadata HEAD: resolved dynamically with `git rev-parse HEAD`",
        (
            "- Canonical complete CPU pytest result: "
            f"{pytest_result.get('passed', 'NOT_RUN')} passed, "
            f"{pytest_result.get('skipped', 'NOT_RUN')} skipped, "
            f"{pytest_result.get('warnings', 'NOT_RUN')} warnings"
        ),
        f"- Human visual review: `{latest.get('human_visual_review_status', 'NOT_REVIEWED')}`",
        "",
        "| Ledger status | Count |",
        "|---|---:|",
    ]
    block.extend(f"| `{status}` | {counts[status]} |" for status in sorted(counts))
    block.extend([SUMMARY_END, ""])
    rendered = "\n".join(block)
    if SUMMARY_START in current and SUMMARY_END in current:
        prefix, remainder = current.split(SUMMARY_START, 1)
        _, suffix = remainder.split(SUMMARY_END, 1)
        return prefix.rstrip() + "\n\n" + rendered + suffix.lstrip("\n")
    title_end = current.find("\n\n")
    if title_end < 0:
        return current.rstrip() + "\n\n" + rendered
    return current[: title_end + 2] + rendered + current[title_end + 2 :]


def generated_outputs(
    ledger: dict[str, object],
    metadata: dict[str, object],
    verification_runs: dict[str, object],
) -> dict[Path, str]:
    outputs = {
        BACKLOG: render_backlog(ledger),
        EVIDENCE_MATRIX: render_evidence_matrix(ledger),
        ARCHIVE_MANIFEST: render_archive_manifest(metadata),
        ARCHIVE_INDEX: render_archive_index(metadata),
        README: render_readme(metadata),
    }
    outputs[CURRENT_STATUS] = render_current_status_summary(
        CURRENT_STATUS.read_text(encoding="utf-8"), ledger, verification_runs
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    metadata = yaml.safe_load(ARCHIVE_METADATA.read_text(encoding="utf-8"))
    verification_runs = yaml.safe_load(VERIFICATION_RUNS.read_text(encoding="utf-8"))
    outputs = generated_outputs(ledger, metadata, verification_runs)
    if args.check:
        stale = [path for path, expected in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != expected]
        if stale:
            for path in stale:
                print(f"STALE: {path.relative_to(REPO_ROOT)}")
            return 1
        print(f"audit generated views PASS: {len(outputs)} files")
        return 0
    for path, expected in outputs.items():
        path.write_text(expected, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
