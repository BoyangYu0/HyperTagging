#!/usr/bin/env python3
"""Build and validate the offline Sphinx wiki in a new, isolated output directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def _output_path(root: Path, requested: Path) -> Path:
    """Validate a fresh build destination without following filesystem aliases."""
    requested = requested.absolute()
    for path in (requested, *requested.parents):
        if path.is_symlink():
            raise ValueError("output must not contain a symlink")
        if path.exists() and not path.is_dir():
            raise ValueError("output and its parents must be directories")
    output = requested.resolve()
    if output == root or output in root.parents:
        raise ValueError("output must not overlap repository sources or evidence")
    allowed = (root / "artifacts", root / "outputs", root / "docs" / "_build")
    if output.is_relative_to(root) and not any(output.is_relative_to(path) for path in allowed):
        raise ValueError("inside the checkout, output must be under artifacts/, outputs/, or docs/_build/")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty; existing files are preserved")
    return output


def main(argv=None):
    """Stage sources and run warning-fatal HTML/text Sphinx without project imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty task-specific output directory")
    parser.add_argument("--builder", choices=("html", "text"), default="html")
    parser.add_argument("--layout", choices=("standalone", "basf2"), default="standalone")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        output = _output_path(root, args.output)
        # copytree normally dereferences source symlinks; refuse them before any
        # output mutation so a source document cannot copy unrelated payloads.
        wiki_source = root / "docs" / "wiki"
        for path in (wiki_source, *wiki_source.parents, *wiki_source.rglob("*")):
            if path.is_symlink():
                raise ValueError("documentation source must not be a symlink")
        for path in (root / "docs/index.rst", root / "doc/index-hypertagging.rst",
                     root / "docs/conf.py", wiki_source / "conf.py", root / "docs/_ext"):
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("documentation entry/configuration must not be a symlink")
        for path in (wiki_source / "_static").rglob("*"):
            if path.relative_to(wiki_source / "_static").as_posix() != "wiki.css" or not path.is_file():
                raise ValueError("unexpected documentation static asset; only wiki.css is allowed")
        if (wiki_source / ".nojekyll").read_bytes():
            raise ValueError("Pages marker must be empty")
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    content = source / ("docs/wiki" if args.layout == "basf2" else "wiki")
    content.mkdir(parents=True)
    # Stage only authored RST guides and local presentation assets. Arbitrary
    # repository Markdown, scripts, reports and configuration never enter it.
    for path in wiki_source.glob("*.rst"):
        shutil.copyfile(path, content / path.name)
    (content / "_static").mkdir()
    shutil.copyfile(wiki_source / "_static/wiki.css", content / "_static/wiki.css")
    shutil.copyfile(wiki_source / ".nojekyll", content / ".nojekyll")
    shutil.copyfile(root / "docs" / "index.rst", content.parent / "index.rst")
    if args.layout == "basf2":
        (source / "doc").mkdir()
        shutil.copyfile(root / "doc" / "index-hypertagging.rst", source / "doc" / "index-hypertagging.rst")
        (source / "index.rst").write_text(
            "basf2 package discovery validation\n==================================\n\n"
            ".. toctree::\n\n   doc/index-hypertagging\n", encoding="utf-8")
    command = [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-E", "-a",
               "-b", args.builder, "-c", str(root / "docs"),
               "-D", "wiki_content_path=" + content.relative_to(source).as_posix(),
               "-d", str(output / "doctrees"), str(source), str(output / args.builder)]
    result = subprocess.run(command, cwd=root)
    if result.returncode:
        return result.returncode
    checks = [sys.executable, str(root / "scripts" / "validate_docs.py"),
              "--generated", str(content / "_generated")]
    if args.builder == "html":
        checks += ["--html", str(output / "html")]
    else:
        checks += ["--artifact", str(output / args.builder)]
    return subprocess.run(checks, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
