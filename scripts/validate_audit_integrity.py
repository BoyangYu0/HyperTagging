#!/usr/bin/env python
"""Validate the active audit index and machine-readable issue ledger."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / "docs" / "audits"
ARCHIVE_ROOT = AUDIT_ROOT / "archive"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "manifest.yaml"
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


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=check, capture_output=True, text=True
    )


def _notebook_diff_is_cell_ids_only(
    repo_root: Path, audited_code_sha: str, path: str
) -> bool:
    try:
        old = json.loads(_git(repo_root, "show", f"{audited_code_sha}:{path}").stdout)
        new = json.loads(_git(repo_root, "show", f"HEAD:{path}").stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return False
    for notebook in (old, new):
        for cell in notebook.get("cells", []):
            cell.pop("id", None)
    return old == new


def validate_audited_code_ancestry(
    ledger: dict[str, object], repo_root: Path
) -> list[str]:
    """Validate the non-self-referential audited-code commit boundary."""

    audited_code_sha = str(ledger.get("audited_code_sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", audited_code_sha):
        return [f"invalid audited_code_sha: {audited_code_sha!r}"]
    if _git(
        repo_root, "cat-file", "-e", f"{audited_code_sha}^{{commit}}", check=False
    ).returncode:
        return [f"audited_code_sha does not identify a commit: {audited_code_sha}"]
    if _git(
        repo_root, "merge-base", "--is-ancestor", audited_code_sha, "HEAD", check=False
    ).returncode:
        return [f"audited_code_sha is not an ancestor of HEAD: {audited_code_sha}"]

    allowed = ledger.get("allowed_post_audit_paths", [])
    if not isinstance(allowed, list) or not all(isinstance(value, str) for value in allowed):
        return ["allowed_post_audit_paths must be a list of path patterns"]
    errors: list[str] = []
    changed = _git(
        repo_root, "diff", "--name-only", f"{audited_code_sha}..HEAD"
    ).stdout.splitlines()
    for path in changed:
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed):
            errors.append(f"post-audit path is outside the allowlist: {path}")
            continue
        if path.startswith("notebooks/") and path.endswith(".ipynb"):
            if not _notebook_diff_is_cell_ids_only(repo_root, audited_code_sha, path):
                errors.append("post-audit notebook change is not cell-ID-only: " + path)
    return errors


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

    if not ARCHIVE_MANIFEST.exists():
        errors.append("archive manifest is missing")
    else:
        manifest = yaml.safe_load(ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
        manifest_files = {
            f"archive/{entry.get('historical_filename')}"
            for entry in manifest.get("reports", [])
        }
        if manifest_files != actual_archives:
            errors.append(
                "archive manifest mismatch: "
                f"missing={sorted(actual_archives - manifest_files)}, "
                f"stale={sorted(manifest_files - actual_archives)}"
            )
        for entry in manifest.get("reports", []):
            if entry.get("audit_type") not in {
                "gap", "verification", "completion", "definition"
            }:
                errors.append(
                    "archive manifest has invalid audit type: "
                    + str(entry.get("historical_filename"))
                )
            target = ARCHIVE_ROOT / str(entry.get("superseded_by", ""))
            if not target.resolve().exists():
                errors.append(
                    "archive manifest has invalid superseded-by link: "
                    + str(entry.get("historical_filename"))
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
    active_files = {
        path.name
        for path in AUDIT_ROOT.iterdir()
        if path.is_file() and path.suffix in {".md", ".yaml", ".yml"}
    }
    expected_active_files = {
        "README.md", "current_status.md", "current_backlog.md", "issue_ledger.yaml"
    }
    if active_files != expected_active_files:
        errors.append(
            "active audit views mismatch: "
            f"unexpected={sorted(active_files - expected_active_files)}, "
            f"missing={sorted(expected_active_files - active_files)}"
        )
    work_root = AUDIT_ROOT / "work"
    if work_root.exists() and any(work_root.iterdir()):
        errors.append("temporary audit work reports must be archived before final validation")

    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    audited_code_sha = str(ledger.get("audited_code_sha", ""))
    errors.extend(validate_audited_code_ancestry(ledger, REPO_ROOT))
    for field in (
        "audit_generated_at",
        "audit_scope",
        "audit_tool_version",
        "verification_commands",
        "verification_result_summary",
    ):
        if field not in ledger:
            errors.append(f"ledger misses audit metadata field: {field}")
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
    actual_head = _git(REPO_ROOT, "rev-parse", "HEAD").stdout.strip()
    old_current_pattern = re.compile(
        r"(?:current|starting)\s+HEAD\s*[:=` ]+"
        r"(?!" + re.escape(actual_head) + r")[0-9a-f]{7,40}",
        re.IGNORECASE,
    )
    for path in active_documents:
        text = path.read_text(encoding="utf-8")
        if old_current_pattern.search(text):
            errors.append(f"active audit describes an older SHA as current: {path}")
    current_text = (AUDIT_ROOT / "current_status.md").read_text(encoding="utf-8")
    if audited_code_sha not in current_text:
        errors.append("current_status.md does not name the ledger audited_code_sha")
    if "git rev-parse HEAD" not in current_text:
        errors.append(
            "current_status.md must identify metadata HEAD dynamically with git rev-parse HEAD"
        )
    if "sole authoritative current audit report" not in current_text:
        errors.append("current_status.md lacks the single-current-document declaration")

    notebook_index = yaml.safe_load(NOTEBOOK_INDEX.read_text(encoding="utf-8"))
    if notebook_index.get("visual_review_status") not in {
        "NOT_REVIEWED", "PASS", "FAIL"
    }:
        errors.append("notebook visual_review_status is invalid")
    for entry in notebook_index.get("notebooks", []):
        verified = str(entry.get("last_verified_sha"))
        if entry.get("fixture_or_real") == "real_only":
            valid = verified == "NOT_RUN" or (
                bool(re.fullmatch(r"[0-9a-f]{40}", verified))
                and not _git(
                    REPO_ROOT, "merge-base", "--is-ancestor", verified, "HEAD", check=False
                ).returncode
            )
            if not valid:
                errors.append(
                    f"real notebook {entry.get('id')} has invalid last_verified_sha"
                )
            continue
        if verified != audited_code_sha:
            errors.append(
                f"notebook {entry.get('id')} last_verified_sha must be audited_code_sha {audited_code_sha}"
            )
    generated_views = subprocess.run(
        [sys.executable, "scripts/generate_audit_views.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if generated_views.returncode:
        errors.append("generated audit views are stale")
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
