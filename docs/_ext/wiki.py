"""Sphinx entry point for import-free, offline documentation generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

from wiki_api import generate_api
from wiki_catalog import generate_catalog
from wiki_repository import generate_repository
from wiki_status import generate_status
from wiki_privacy import _canonical_body, sensitive_literals, validate_artifact, violations


_UNSAFE_AUTHORED_RST = re.compile(
    r"(?im)^\s*\.\.\s+(?:\|[^|]+\|\s+)?"
    r"(?:raw|include|literalinclude|image|figure|meta|role)::|"
    r":download:|^\s*:file\s*:",
)


def _check_path(path: Path, *, directory: bool = False) -> None:
    """Reject aliases and obstructing parents before publishing any files."""
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("Refusing symlink in generated output path")
        if parent.exists() and not parent.is_dir():
            raise ValueError("Generated output parent is not a directory")
    if path.is_symlink():
        raise ValueError("Refusing symlink generated output")
    if path.exists():
        if directory:
            if not path.is_dir():
                raise ValueError("Generated output is not a directory")
        elif not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Generated output is not a singly linked regular file")


def _owned_path(output: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name.startswith("/") or any(
        part in {"", ".", ".."} for part in name.split("/")
    ):
        raise ValueError("Unsafe generated-file manifest path")
    path = output / name
    if not path.resolve().is_relative_to(output):
        raise ValueError("Unsafe generated-file manifest path")
    _check_path(path)
    return path


def _check_previous(output: Path, previous: dict) -> None:
    for name, digest in previous.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid generated-file manifest digest")
        path = _owned_path(output, name)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Generated file has manual changes; preserving it")


def _check_owned_sources(output: Path, previous: dict) -> None:
    for page in output.rglob("*.rst"):
        if page.relative_to(output).as_posix() not in previous:
            raise ValueError("Unowned source page would enter the publication; preserving it")


def _manifest_text(manifests: dict) -> str:
    """Index the complete status download without duplicating its growing metric rows."""
    manifests = dict(manifests)
    if "status" in manifests:
        payload = json.dumps(manifests["status"], separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
        manifests["status"] = {"download": "status/status.json", "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                               "bytes": len(payload.encode()), "format": "complete-status-json-v1"}
    return json.dumps(manifests, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"


def generate(root: Path, output: Path) -> dict:
    """Generate in a temporary tree, then update only unchanged owned output files."""
    root, requested = root.resolve(), output.absolute()
    _check_path(requested, directory=True)
    output = requested.resolve()
    if output == root or output in root.parents:
        raise ValueError("Generated output must not contain repository sources")
    marker = output / ".wiki-generated.json"
    _check_path(marker)
    previous = {}
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise ValueError("Refusing nonempty unowned generated directory")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise ValueError("Generated-file manifest must be a mapping")
        _check_previous(output, previous)
        _check_owned_sources(output, previous)
    with tempfile.TemporaryDirectory(prefix="hypertagging-wiki-generate-") as temporary:
        stage = Path(temporary)
        manifests = {
            "api": generate_api(root, stage / "api"),
            "catalog": generate_catalog(root, stage / "catalog"),
            "status": generate_status(root, stage / "status"),
            "repository": generate_repository(root, stage / "repository"),
        }
        (stage / "manifest.json").write_text(_manifest_text(manifests), encoding="utf-8")
        validate_artifact(stage, root, generated_projection=True)
        payloads = {p.relative_to(stage).as_posix(): p.read_bytes() for p in stage.rglob("*") if p.is_file()}
        # Preflight the complete write/delete set, including new paths. An
        # unowned directory symlink must never redirect a newly generated file.
        _check_path(requested, directory=True)
        _check_path(marker)
        _check_previous(output, previous)
        _check_owned_sources(output, previous)
        for name in payloads:
            target = _owned_path(output, name)
            if target.exists() and name not in previous:
                raise ValueError("Refusing to overwrite unowned file")
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != payload:
                target.write_bytes(payload)
        for name in previous.keys() - payloads.keys():
            (output / name).unlink(missing_ok=True)
        digests = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
        marker.write_text(json.dumps(digests, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifests


def on_builder_inited(app):
    root = Path(__file__).resolve().parents[2]
    for asset in (root / "docs/wiki/_static").rglob("*"):
        if asset.name != "wiki.css" or not asset.is_file() or asset.is_symlink():
            raise ValueError("Unexpected documentation static asset")
    if (root / "docs/wiki/.nojekyll").read_bytes():
        raise ValueError("Pages marker must be empty")
    content = Path(app.config.wiki_content_path)
    if content.is_absolute() or ".." in content.parts:
        raise ValueError("wiki_content_path must be relative to the Sphinx source root")
    source_root = Path(app.srcdir).resolve()
    content_root = (source_root / content).resolve()
    if not content_root.is_relative_to(source_root):
        raise ValueError("wiki_content_path escapes the Sphinx source root")
    authored = set(content_root.rglob("*.rst")) if content_root.is_dir() else set()
    authored = {
        path for path in authored
        if "_generated" not in path.relative_to(content_root).parts
    }
    for relative in (
        "index.rst", "docs/index.rst", "doc/index-hypertagging.rst",
    ):
        candidate = source_root / relative
        if candidate.is_file():
            authored.add(candidate)
    deny = {
        canonical
        for value in sensitive_literals(root)
        if (canonical := _canonical_body(value))
    }
    for path in sorted(authored):
        if path.is_symlink() or not path.resolve().is_relative_to(source_root):
            raise ValueError("Authored documentation source is not confined")
        payload = path.read_text(encoding="utf-8")
        canonical = _canonical_body(payload)
        if (
            _UNSAFE_AUTHORED_RST.search(payload)
            or violations(payload)
            or any(value in canonical for value in deny)
        ):
            raise ValueError("Authored documentation source failed publication policy")
    app._wiki_authored_sources_validated = True
    generate(root, Path(app.srcdir) / content / "_generated")
    if "sphinx.ext.viewcode" in app.extensions:
        # Supplying an empty analyzed source prevents viewcode's fallback from
        # importing modules or publishing package bodies. AST provenance remains.
        app.connect("viewcode-find-source", lambda _app, _module: ("", {}))


def suppress_source_pages(app, *unused):
    """Discard viewcode's empty source cache before it can create source pages."""
    app.env._viewcode_modules = {}
    return []


def on_build_finished(app, exception):
    if exception is not None or app.builder.name not in {"html", "text"}:
        return
    output = Path(app.outdir)
    # Sphinx creates this empty directory even with html_copy_source=False.
    # Remove only the empty builder-created directory, never existing contents.
    empty_sources = output / "_sources"
    if empty_sources.is_dir() and not empty_sources.is_symlink() and not any(empty_sources.iterdir()):
        empty_sources.rmdir()
    validate_artifact(
        output,
        Path(__file__).resolve().parents[2],
        renderer_output=bool(
            getattr(app, "_wiki_authored_sources_validated", False)
        ),
    )


def setup(app):
    app.add_config_value("wiki_content_path", "", "env")
    app.connect("builder-inited", on_builder_inited)
    app.connect("env-updated", suppress_source_pages)
    app.connect("html-collect-pages", suppress_source_pages, priority=100)
    app.connect("build-finished", on_build_finished)
    return {"version": "2", "parallel_read_safe": True, "parallel_write_safe": True}
