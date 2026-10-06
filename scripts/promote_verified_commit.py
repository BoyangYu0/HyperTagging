#!/usr/bin/env python3
"""Check exact-commit branch CI before an optional fast-forward to master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "BoyangYu0/HyperTagging"
REQUIRED_WORKFLOWS = (".github/workflows/cpu-tests.yml", ".github/workflows/docs.yml")


def git(*args: str) -> str:
    """Run Git without a shell in this checkout, preserving command failures."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verified_runs(runs: list[dict], sha: str, branch: str) -> list[dict]:
    """Require the newest push run of each workflow at the exact source identity."""
    selected = []
    for path in REQUIRED_WORKFLOWS:
        matching = [run for run in runs if run.get("path") == path
                    and run.get("head_sha") == sha
                    and run.get("head_branch") == branch
                    and run.get("event") == "push"]
        if not matching:
            raise ValueError(f"No push validation for {path} at the candidate commit")
        latest = max(matching, key=lambda run: (run["id"], run.get("run_attempt", 1)))
        if latest.get("status") != "completed" or latest.get("conclusion") != "success":
            raise ValueError(f"Latest {path} run has not succeeded: {latest.get('html_url')}")
        selected.append(latest)
    return selected


def main(argv=None) -> int:
    """Inspect promotion eligibility; mutate master only with explicit --push."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Fast-forward remote master after all checks pass")
    args = parser.parse_args(argv)
    try:
        remote = git("remote", "get-url", "--push", "origin")
        if remote not in (f"git@github.com:{REPOSITORY}.git", f"https://github.com/{REPOSITORY}.git"):
            raise ValueError("Origin push URL does not match the reviewed repository")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Commit all tracked changes and audit lineage before promotion")
        branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
        if branch == "master":
            raise ValueError("Run from the validated development branch")
        sha = git("rev-parse", "HEAD")
        query = urllib.parse.urlencode({"head_sha": sha, "branch": branch, "event": "push", "per_page": 100})
        request = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}/actions/runs?{query}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "HyperTagging-CI-promotion"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            runs = json.load(response)["workflow_runs"]
        selected = verified_runs(runs, sha, branch)
        # Recheck refs after the API read. Push the checked SHA, never a moving HEAD.
        refs = dict(line.split()[::-1] for line in git(
            "ls-remote", "origin", f"refs/heads/{branch}", "refs/heads/master"
        ).splitlines())
        if refs.get(f"refs/heads/{branch}") != sha:
            raise ValueError("Remote development branch differs from the checked commit")
        master = refs.get("refs/heads/master")
        if not master:
            raise ValueError("Remote master is missing; refusing to create it")
        subprocess.run(["git", "merge-base", "--is-ancestor", master, sha], cwd=ROOT, check=True)
        print(json.dumps({"source_sha": sha, "branch": branch,
                          "validated_runs": [run["html_url"] for run in selected],
                          "promotion": "push" if args.push else "eligible"}, indent=2))
        if args.push:
            subprocess.run(["git", "push", "origin", f"{sha}:refs/heads/master"], cwd=ROOT, check=True)
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Promotion refused: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
