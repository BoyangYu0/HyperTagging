"""Compact whole-Git inventory with opaque identities and no source publication."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


_EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "_generated", "_build"}
_GROUPS = frozenset({
    ".github", "configs", "docs", "environment", "examples", "notebooks",
    "reports", "reconstruction", "schemas", "scripts", "src", "tests",
})
_DATA_MODEL_SUFFIXES = frozenset({
    ".pt", ".pth", ".ckpt", ".parquet", ".root", ".pkl", ".pickle",
    ".h5", ".hdf5", ".npz", ".npy", ".onnx", ".arrow", ".feather",
    ".safetensors", ".bin", ".db", ".sqlite", ".sqlite3",
})
_TEXT_KINDS = {
    ".py": "python", ".md": "documentation", ".rst": "documentation",
    ".txt": "text", ".sh": "shell", ".sbatch": "shell",
    ".yaml": "configuration", ".yml": "configuration",
    ".toml": "configuration", ".json": "structured_text",
    ".jsonl": "structured_text", ".csv": "structured_text",
    ".ini": "configuration", ".cfg": "configuration", ".lock": "configuration",
    ".condor": "configuration", ".sub": "configuration",
    ".template": "configuration", ".tmpl": "configuration",
    ".ipynb": "notebook", ".html": "web_text", ".css": "web_text",
    ".js": "web_text", ".xml": "structured_text", ".svg": "web_text",
    ".tex": "documentation", ".log": "text",
}
_TEXT_NAMES = frozenset({
    ".gitignore", ".gitattributes", ".editorconfig", ".nojekyll",
    "Makefile", "LICENSE", "Dockerfile", "CMakeLists.txt",
})


def repository_files(root: Path) -> list[Path]:
    """List tracked/nonignored untracked files locally, never following symlinks.

    This internal helper returns paths for generators and privacy validation.
    Its return value must never be serialized into the published inventory.
    Exported source trees without Git metadata use bounded directory discovery.
    """
    root = root.resolve()
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    except (OSError, subprocess.CalledProcessError):
        names = sorted({path.relative_to(root).as_posix() for directory in
                        sorted(_GROUPS) for path in (root / directory).rglob("*")
                        if path.is_file()} | {path.name for path in root.iterdir() if path.is_file()})
    files = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or set(relative.parts) & _EXCLUDED_PARTS:
            continue
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            continue
        if path.is_file() and path.resolve().is_relative_to(root):
            files.append(path)
    return files


def _kind(path: Path) -> tuple[str, bool]:
    """Return a fixed publication category and whether content hashing is allowed."""
    suffix = path.suffix.lower()
    if suffix in _DATA_MODEL_SUFFIXES:
        return "data_or_model", False
    if suffix in _TEXT_KINDS:
        return _TEXT_KINDS[suffix], True
    if path.name in _TEXT_NAMES:
        return "text", True
    return "binary_or_other", False


def _content_hash(path: Path) -> str:
    """Create a domain-separated public fingerprint with bounded memory."""
    digest = hashlib.sha256()
    digest.update(b"hypertagging-docs-public-content-v1\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def generate_repository(root: Path, output: Path) -> dict:
    """Publish two compact files: opaque per-file inventory and group summary.

    Repository/configuration/report/script/package source bodies, names and
    paths never enter either output. Text inputs carry domain-separated public
    fingerprints; binary,
    model and data artifacts are represented by size only and are never read.
    Detailed signatures/docstrings belong only to the dedicated package API and
    script/example generators. Existing source mirrors are refused, not reused.
    """
    root = root.resolve()
    output = output.absolute()
    if output == root or output.resolve() in root.parents:
        raise ValueError("repository inventory output must not contain its inputs")
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("repository inventory output must not follow symlinks")
    if output.exists():
        allowed = {"index.rst", "manifest.json"}
        if not output.is_dir() or any(path.name not in allowed or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in output.iterdir()):
            raise ValueError("repository inventory refuses nonempty unrelated output or source mirrors")
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in repository_files(root):
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 and relative.parts[0] in _GROUPS else "root" if len(relative.parts) == 1 else "other"
        kind, readable = _kind(path)
        entries.append({
            "path_id": hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest(),
            "group": group, "kind": kind, "bytes": path.stat().st_size,
            "sha256": _content_hash(path) if readable else None,
            "content_status": "hashed_text" if readable else "stat_only_not_read",
        })
    entries.sort(key=lambda item: item["path_id"])
    groups = dict(sorted(Counter(item["group"] for item in entries).items()))
    kinds = dict(sorted(Counter(item["kind"] for item in entries).items()))
    manifest = {
        "schema_version": 3,
        "publication": "opaque file identities, fixed categories, size and domain-separated fingerprints only",
        "files": entries, "total_files": len(entries),
        "python_modules": kinds.get("python", 0),
        "fingerprinted_files": sum(item["sha256"] is not None for item in entries),
        "stat_only_files": sum(item["sha256"] is None for item in entries),
        "groups": groups, "kinds": kinds,
    }
    lines = [
        "Repository inventory", "====================", "",
        "The inventory covers tracked and nonignored untracked files in this source",
        "snapshot. It publishes opaque identities, fixed group/kind categories, byte",
        "counts and domain-separated content fingerprints. It contains no repository source bodies, file",
        "names, raw paths, docstrings or individual source pages. Detailed static",
        "reference is limited to the authoritative package API and scripts/examples.", "",
        "Each opaque identity is the SHA-256 of the relative UTF-8 Git path. Text",
        "content fingerprints use SHA-256 with a documentation-specific domain",
        "separator, without importing or executing it. Model, data, binary",
        "and unknown file types are recorded by size only; their payloads are never",
        "read. Symlinks and ignored runtime/build artifacts are excluded. Source",
        "exports without Git metadata use the standard source directories only.", "",
        ":download:`Machine-readable compact inventory <manifest.json>`", "",
        f"This snapshot contains **{manifest['total_files']} files**, including",
        f"**{manifest['python_modules']} Python modules** across the whole tree.",
        f"**{manifest['fingerprinted_files']} text files** have public fingerprints;",
        f"**{manifest['stat_only_files']} data/model/binary files** have stat-only records.", "",
        "Group distribution", "------------------", "",
        "Groups come from a fixed source-directory allowlist. Other directories",
        "share the generic ``other`` category, and root files share ``root``.", "",
        ".. list-table:: Files by group", "   :header-rows: 1", "",
        "   * - Group", "     - Files",
    ]
    for group, count in groups.items():
        lines.extend([f"   * - ``{group}``", f"     - {count}"])
    lines.extend(["", "File-kind distribution", "----------------------", "",
                  ".. list-table:: Files by fixed kind", "   :header-rows: 1", "",
                  "   * - Kind", "     - Files"])
    for kind, count in kinds.items():
        lines.extend([f"   * - ``{kind}``", f"     - {count}"])
    lines.append("")
    _write(output / "index.rst", "\n".join(lines))
    _write(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
