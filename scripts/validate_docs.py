#!/usr/bin/env python3
"""Validate wiki coverage, reproducibility, local HTML and Pages security offline."""
from __future__ import annotations

import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))


class Links(HTMLParser):
    """Collect local references and runtime assets without opening any URL."""
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.assets = set(), [], []
        self.script_sources, self.active_errors = [], []
        self.form_actions = []
        self.names = []
        self._script = None

    @staticmethod
    def _active_scheme(value: str) -> bool:
        value = re.sub(r"[\x00-\x20]+", "", unquote(value)).lower()
        return re.search(
            r"(?<![a-z0-9_+.-])(?:data|javascript):", value,
        ) is not None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.names.append(tag)
        self.names.extend(name.lower() for name, _ in attrs)
        names = [name.lower() for name, _ in attrs]
        attributes = {name.lower(): value for name, value in attrs}
        from wiki_privacy import (
            _RUNTIME_URL_ATTRIBUTES, _remote_runtime_reference,
        )
        if len(names) != len(set(names)):
            self.active_errors.append("Duplicate HTML attribute")
        if tag in {
            "iframe", "object", "embed", "applet", "style", "base", "svg",
        }:
            self.active_errors.append("Active embedded content")
        for name, value in attrs:
            lowered_name = name.lower()
            if (
                lowered_name == "attributionsrc"
                or value and (
                    lowered_name.startswith("on")
                or lowered_name in {"srcset", "imagesrcset"}
                or lowered_name in {
                    "formaction", "formmethod", "ping",
                }
                or tag in {"image", "use"}
                and lowered_name in {"href", "xlink:href"}
                or (tag, lowered_name) in _RUNTIME_URL_ATTRIBUTES
                and _remote_runtime_reference(value)
                or self._active_scheme(value)
                )
            ):
                self.active_errors.append("Active HTML attribute")
            if value and lowered_name == "style":
                from wiki_privacy import _unsafe_css
                if "/*" in value or "*/" in value or _unsafe_css(value):
                    self.active_errors.append("Active inline CSS")
        if tag == "input" and str(attributes.get("type", "")).lower() == "image":
            self.active_errors.append("Active image input")
        if tag == "form":
            action = str(attributes.get("action", ""))
            from wiki_privacy import _runtime_url_variants
            action_variants = _runtime_url_variants(action)
            if (
                str(attributes.get("method", "get")).lower() != "get"
                or any(
                    urlsplit(variant).scheme or urlsplit(variant).netloc
                    or urlsplit(variant).path.startswith("/")
                    for variant in action_variants
                )
                or self._active_scheme(action)
            ):
                self.active_errors.append("Active form submission")
            else:
                self.form_actions.append(action)
                self.links.append(action)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("name"):
            self.ids.add(attributes["name"])
        if attributes.get("href"):
            self.links.append(attributes["href"])
        if tag in {
            "script", "img", "iframe", "source", "video", "audio", "embed",
            "track", "input", "frame",
        }:
            if attributes.get("src"):
                self.assets.append(attributes["src"])
                self.links.append(attributes["src"])
        if tag == "video" and attributes.get("poster"):
            self.assets.append(attributes["poster"])
            self.links.append(attributes["poster"])
        if tag == "object" and attributes.get("data"):
            self.assets.append(attributes["data"])
            self.links.append(attributes["data"])
        if tag in {
            "body", "table", "colgroup", "col", "thead", "tbody", "tfoot",
            "tr", "td", "th",
        } and attributes.get("background"):
            self.assets.append(attributes["background"])
            self.links.append(attributes["background"])
        if tag == "html" and attributes.get("manifest"):
            self.assets.append(attributes["manifest"])
            self.links.append(attributes["manifest"])
        if tag in {"image", "use"}:
            for name in ("href", "xlink:href"):
                if attributes.get(name):
                    self.assets.append(attributes[name])
                    if name == "xlink:href":
                        self.links.append(attributes[name])
        if tag == "link" and attributes.get("href"):
            self.assets.append(attributes["href"])
        if tag == "script":
            if self._script is not None:
                self.active_errors.append("Nested script element")
            source = attributes.get("src")
            if source:
                self.script_sources.append(source)
            self._script = {"src": source, "body": []}
        if tag == "meta" and "http-equiv" in attributes:
            self.active_errors.append("Active metadata refresh")

    def handle_data(self, data):
        if self._script is not None:
            self._script["body"].append(data)

    def handle_endtag(self, tag):
        self.names.append(tag.lower())
        if tag.lower() != "script":
            return
        if self._script is None:
            self.active_errors.append("Unmatched script element")
            return
        from wiki_privacy import _SAFE_INLINE_SCRIPTS
        body = "".join(self._script["body"]).strip()
        source = self._script["src"]
        if (source and body) or (not source and body not in _SAFE_INLINE_SCRIPTS):
            self.active_errors.append("Untrusted inline script")
        self._script = None

    def close(self):
        super().close()
        if self._script is not None:
            self.active_errors.append("Unclosed script element")


def validate_html(directory: Path) -> dict:
    """Check every local file/fragment and reject remote runtime resources."""
    directory = directory.resolve()
    parsers = {}
    errors = []
    for path in sorted(directory.rglob("*.html")):
        parser = Links()
        payload = path.read_text(encoding="utf-8")
        parser.feed(payload)
        parser.close()
        from wiki_privacy import _raw_markup_channels
        raw_names, _, _, raw_markup_complete = _raw_markup_channels(payload)
        raw_names = [name.lower() for name in raw_names]
        if not raw_markup_complete:
            parser.active_errors.append("Unparseable incomplete HTML markup")
        if raw_names != parser.names:
            parser.active_errors.append("Unparseable HTML name channel")
        errors.extend(parser.active_errors)
        parsers[path] = parser
    if not parsers or not (directory / "index.html").is_file():
        raise ValueError("No HTML site/index found")
    checked = 0
    for path, parser in parsers.items():
        from wiki_privacy import _runtime_url_variants
        for action in parser.form_actions:
            for variant in _runtime_url_variants(action):
                action_path = urlsplit(variant).path
                target = (
                    (path.parent / action_path).resolve()
                    if action_path else path
                )
                if not target.is_relative_to(directory):
                    errors.append("Form action escapes publication")
                    break
        for asset in parser.assets:
            for variant in _runtime_url_variants(asset):
                parsed_asset = urlsplit(variant)
                if parsed_asset.scheme or parsed_asset.netloc:
                    errors.append("Remote runtime asset")
                    break
                if parsed_asset.path.startswith("/"):
                    errors.append("Root-relative runtime asset")
                    break
                target = (
                    (path.parent / parsed_asset.path).resolve()
                    if parsed_asset.path else path
                )
                if not target.is_relative_to(directory):
                    errors.append("Runtime asset escapes publication")
                    break
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme.lower() in {"data", "javascript"}:
                errors.append("Active link scheme")
                continue
            if parsed.scheme or parsed.netloc:
                continue
            checked += 1
            if parsed.path.startswith("/"):
                errors.append("Root-relative URL breaks project Pages")
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(directory) or not target.is_file():
                errors.append("Missing local target")
            elif parsed.fragment and target in parsers and unquote(parsed.fragment) not in parsers[target].ids:
                errors.append("Missing local fragment")
        from wiki_privacy import _trusted_vendor_hashes
        trusted_scripts = _trusted_vendor_hashes()
        for source in parser.script_sources:
            parsed = urlsplit(source)
            target = (path.parent / unquote(parsed.path)).resolve()
            if (
                parsed.scheme or parsed.netloc or parsed.path.startswith("/")
                or not target.is_relative_to(directory) or not target.is_file()
            ):
                errors.append("Untrusted external script")
                continue
            if target.name == "searchindex.js":
                continue
            digest = _hash_file(target)
            if digest not in trusted_scripts.get(target.name, set()):
                errors.append("Untrusted external script")
    from wiki_privacy import _unsafe_css
    for path in directory.rglob("*.css"):
        css = path.read_text(encoding="utf-8")
        if re.search(r"(?:url\(\s*['\"]?|@import\s+['\"])(?:https?:)?//", css):
            errors.append("Remote CSS dependency")
        if re.search(r"(?:url\(\s*['\"]?|@import\s+['\"])(?:data|javascript):", css, re.I):
            errors.append("Active CSS dependency")
        if _unsafe_css(css):
            errors.append("Untrusted active CSS")
    if any(path.is_file() and path.suffix.lower() == ".svg"
           for path in directory.rglob("*")):
        errors.append("Untrusted active SVG")
    for name in ("objects.inv", "searchindex.js", ".nojekyll"):
        if not (directory / name).is_file():
            errors.append(f"Missing Sphinx/Pages artifact: {name}")
    if errors:
        raise ValueError("\n".join(errors[:50]) + f"\nTotal HTML errors: {len(errors)}")
    return {"html_pages": len(parsers), "local_links_checked": checked, "remote_assets": 0}


def _require(condition, message: str) -> None:
    """Security checks remain active under optimized Python."""
    if not condition:
        raise ValueError(message)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _definition_locations(tree, module: str, source: str):
    """Independently inspect every function and class in all lexical scopes."""
    definitions, locations = {}, {}

    def visit(node, prefix, scope, local=False):
        if isinstance(node, ast.ClassDef):
            name = prefix + "." + node.name
            key = (source, name, node.lineno)
            internal = local or any(part.startswith("_") for part in name.split(".")[1:])
            definitions[key] = (
                node.end_lineno,
                "internal_unstable" if internal else "public_stable",
                local,
            )
            locations[key] = node.end_lineno
            for child in node.body:
                visit(child, name, "class", local)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + "." + node.name
            key = (source, name, node.lineno)
            internal = local or any(part.startswith("_") for part in name.split(".")[1:])
            definitions[key] = (
                node.end_lineno,
                "internal_unstable" if internal else "public_stable",
                local,
            )
            locations[key] = node.end_lineno
            for child in node.body:
                visit(child, name, "function", True)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and scope in {"module", "class"}:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    locations[(source, prefix + "." + target.id, node.lineno)] = node.end_lineno
        for child in ast.iter_child_nodes(node):
            visit(child, prefix, scope, local)

    visit(tree, module, "module")
    return definitions, locations


def validate_coverage(generated: Path, html: Path | None = None) -> dict:
    """Compare every AST definition occurrence, compact file identity and catalog."""
    from collections import Counter
    from wiki_repository import _GROUPS, _content_hash, _kind, repository_files

    generated = generated.resolve()
    api = json.loads((generated / "api" / "inventory.json").read_text())
    _require(api.get("schema_version") == 2, "API inventory requires schema 2")
    modules = {module["source"]: module for module in api["modules"]}
    records = api["symbols"]
    _require(len(modules) == len(api["modules"]) == api["module_count"],
             "Duplicate API modules or inconsistent module count")
    _require(len(records) == api["symbol_count"], "Inconsistent API symbol count")
    _require(len({item["occurrence_id"] for item in records}) == len(records),
             "Duplicate API definition occurrences")
    expected_paths = {path.relative_to(ROOT).as_posix()
                      for path in (ROOT / "src" / "hypertagging").rglob("*.py")}
    _require(set(modules) == expected_paths, "API module coverage differs from package sources")
    _require(not api["unresolved_exports"] and not api["dynamic_exports"],
             "Unresolved/dynamic API exports require an explicit generator update")
    expected_definitions, locations, listed_names = {}, {}, []
    for source, module in modules.items():
        path = ROOT / source
        _require(path.resolve().is_relative_to(ROOT.resolve()), "API source escapes repository")
        data = path.read_bytes()
        relative = Path(source).relative_to("src").with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        expected_name = ".".join(parts)
        _require(module["name"] == expected_name, "API module identity differs from its source")
        _require(module["sha256"] == hashlib.sha256(data).hexdigest(), "Stale API source hash")
        _require(module["source_lines"] == len(data.splitlines()), "Stale API source line count")
        _require("source_download" not in module, "Raw API source downloads are forbidden")
        _require(module["page"] == expected_name + ".rst", "Unsafe API page identity")
        _require((generated / "api" / module["page"]).is_file(), "Missing API page")
        tree = ast.parse(data, filename=source)
        expected, found_locations = _definition_locations(tree, expected_name, source)
        expected_definitions.update(expected)
        locations.update(found_locations)
        definition_ids = {f"{key[0]}:{key[2]}:{key[1]}" for key in expected}
        _require(set(module["definitions"]) == definition_ids
                 and len(module["definitions"]) == len(definition_ids),
                 "Missing API definitions in module inventory")
        _require(all(name.startswith(expected_name + ".") for name in module["symbols"]),
                 "API symbol listed under wrong module")
        listed_names.extend(module["symbols"])
    _require(Counter(listed_names) == Counter(item["name"] for item in records),
             "API symbols differ from module inventories")
    actual_definitions = {}
    for symbol in records:
        name, canonical, source = symbol["name"], symbol["canonical"], symbol["source"]
        key = (source, canonical, symbol["line"])
        _require(key in locations and symbol["end_line"] == locations[key],
                 "API symbol provenance does not match source definition")
        _require(symbol["sha256"] == modules[source]["sha256"], "Stale API definition hash")
        _require(symbol["alias"] == (name != canonical), "Inconsistent API alias identity")
        _require(symbol["definition_id"] == f"{source}:{symbol['line']}:{canonical}",
                 "Inconsistent API definition identity")
        _require(symbol["occurrence_id"] == f"{name}@{source}:{symbol['line']}",
                 "Inconsistent API occurrence identity")
        _require(symbol["surface"] in {"public_stable", "internal_unstable"},
                 "API surface classification is missing")
        _require(isinstance(symbol["signature"], str), "API signature is not static text")
        _require(symbol["definition"] == (key in expected_definitions),
                 "API definition kind differs from the AST")
        if symbol["definition"] and not symbol["alias"]:
            _require(key not in actual_definitions, "Duplicate canonical API definition")
            actual_definitions[key] = symbol
            _require(symbol["surface"] == expected_definitions[key][1],
                     "API internal/public surface differs from lexical scope")
            if expected_definitions[key][2]:
                _require(not symbol["indexed"],
                         "Function-local definitions must remain inert in the Python object inventory")
    _require(set(actual_definitions) == set(expected_definitions),
             "Missing API definitions: every lexical function and class occurrence is required")
    _require(api["definition_count"] == len(expected_definitions), "Inconsistent API definition count")
    _require(api["documented_symbol_count"] == sum(bool(item["documented"]) for item in records),
             "Inconsistent documented API symbol count")
    _require(api["surface_counts"] == {surface: sum(item["surface"] == surface for item in records)
                                      for surface in ("public_stable", "internal_unstable")},
             "Inconsistent API surface counts")

    repository = json.loads((generated / "repository" / "manifest.json").read_text())
    _require(repository.get("schema_version") == 3, "Repository inventory requires opaque schema 3")
    entries = repository["files"]
    file_map = {hashlib.sha256(path.relative_to(ROOT).as_posix().encode("utf-8")).hexdigest(): path
                for path in repository_files(ROOT)}
    _require(len(entries) == len({item["path_id"] for item in entries}) == len(file_map),
             "Duplicate or missing repository inventory files")
    _require({item["path_id"] for item in entries} == set(file_map), "Whole-Git inventory incomplete")
    allowed_fields = {"path_id", "group", "kind", "bytes", "sha256", "content_status"}
    for entry in entries:
        _require(set(entry) == allowed_fields, "Repository inventory contains unapproved fields")
        path = file_map[entry["path_id"]]
        relative = path.relative_to(ROOT)
        group = (relative.parts[0] if len(relative.parts) > 1 and relative.parts[0] in _GROUPS
                 else "root" if len(relative.parts) == 1 else "other")
        kind, readable = _kind(path)
        _require(entry["group"] == group and entry["kind"] == kind,
                 "Repository inventory category differs from source")
        _require(entry["bytes"] == path.stat().st_size, "Stale repository size evidence")
        _require(entry["sha256"] == (_content_hash(path) if readable else None),
                 "Stale repository content fingerprint")
        _require(entry["content_status"] == ("hashed_text" if readable else "stat_only_not_read"),
                 "Repository content-access policy differs")
    groups = dict(Counter(item["group"] for item in entries))
    kinds = dict(Counter(item["kind"] for item in entries))
    _require(repository["groups"] == groups and repository["kinds"] == kinds,
             "Repository summary distributions differ")
    _require(repository["total_files"] == len(entries)
             and repository["python_modules"] == kinds.get("python", 0)
             and repository["fingerprinted_files"] == sum(item["sha256"] is not None for item in entries)
             and repository["stat_only_files"] == sum(item["sha256"] is None for item in entries),
             "Repository summary counts differ")

    catalog = json.loads((generated / "catalog" / "manifest.json").read_text())
    _require(catalog.get("schema_version") == 2, "Catalog requires redacted schema 2")
    catalog_paths = {entry["path"] for entry in catalog["entries"]}
    expected_catalog = {path.relative_to(ROOT).as_posix() for group in ("scripts", "examples")
                        for path in (ROOT / group).rglob("*")
                        if path.suffix in {".py", ".sh", ".sbatch"}
                        and "__pycache__" not in path.parts and path.is_file()}
    _require(catalog_paths == expected_catalog
             and len(catalog["entries"]) == len(catalog_paths) == catalog["total"],
             "Script/example catalog incomplete")
    catalog_definition_count = 0
    for entry in catalog["entries"]:
        path = ROOT / entry["path"]
        _require(path.resolve().is_relative_to(ROOT.resolve()), "Catalog source escapes repository")
        _require(entry["sha256"] == _hash_file(path), "Stale catalog source hash")
        _require(entry["docname"] == "entries/" + entry["path"]
                 and (generated / "catalog" / (entry["docname"] + ".rst")).is_file(),
                 "Missing or unsafe catalog page")
        if path.suffix != ".py":
            _require(not entry.get("definitions"),
                     "Non-Python catalog entries cannot declare Python definitions")
            continue
        # Traverse independently of wiki_catalog. The explicit stack preserves
        # lexical ownership without adding a helper definition to the catalogued
        # validator script itself.
        tree = ast.parse(path.read_bytes(), filename=entry["path"])
        expected_catalog_definitions = {}
        stack = [(tree, "", "module", False)]
        while stack:
            node, prefix, scope, function_local = stack.pop()
            if isinstance(node, ast.ClassDef):
                name = prefix + "." + node.name if prefix else node.name
                key = (name, node.lineno)
                _require(key not in expected_catalog_definitions,
                         "Duplicate lexical catalog definition identity")
                internal = function_local or any(
                    part.startswith("_") for part in name.split(".")
                )
                expected_catalog_definitions[key] = (
                    node.end_lineno, "class",
                    "internal/unstable" if internal else "public",
                )
                stack.extend(
                    (child, name, "class", function_local)
                    for child in reversed(node.body)
                )
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + "." + node.name if prefix else node.name
                key = (name, node.lineno)
                _require(key not in expected_catalog_definitions,
                         "Duplicate lexical catalog definition identity")
                internal = function_local or any(
                    part.startswith("_") for part in name.split(".")
                )
                expected_catalog_definitions[key] = (
                    node.end_lineno, "method" if scope == "class" else "function",
                    "internal/unstable" if internal else "public",
                )
                stack.extend(
                    (child, name, "function", True)
                    for child in reversed(node.body)
                )
                continue
            if not isinstance(node, ast.Lambda):
                stack.extend(
                    (child, prefix, scope, function_local)
                    for child in reversed(list(ast.iter_child_nodes(node)))
                )
        definition_records = entry.get("definitions", [])
        _require(isinstance(definition_records, list),
                 "Catalog definitions must be an ordered list")
        actual_catalog_definitions = {}
        definition_fields = {
            "name", "kind", "signature", "docstring", "line", "end_line", "surface",
        }
        for definition in definition_records:
            _require(set(definition) == definition_fields,
                     "Catalog definition contains unapproved fields")
            key = (definition["name"], definition["line"])
            _require(key not in actual_catalog_definitions,
                     "Duplicate catalog definition occurrence")
            _require(key in expected_catalog_definitions,
                     "Catalog definition has no matching lexical AST occurrence")
            expected_end, expected_kind, expected_surface = expected_catalog_definitions[key]
            _require(
                definition["end_line"] == expected_end
                and definition["kind"] == expected_kind
                and definition["surface"] == expected_surface,
                "Catalog definition provenance or lexical classification differs",
            )
            _require(
                isinstance(definition["signature"], str)
                and isinstance(definition["docstring"], str),
                "Catalog definition projection must contain static text",
            )
            actual_catalog_definitions[key] = definition
        _require(
            set(actual_catalog_definitions) == set(expected_catalog_definitions),
            "Missing catalog definitions: every lexical function and class occurrence is required",
        )
        catalog_definition_count += len(expected_catalog_definitions)
    _require(catalog.get("definition_count") == catalog_definition_count,
             "Inconsistent independently verified catalog definition count")
    result = {"package_modules": len(modules), "api_symbols": len(records),
              "api_definitions": len(expected_definitions), "repository_files": len(entries),
              "repository_python_modules": kinds.get("python", 0),
              "catalog_entries": len(catalog_paths),
              "catalog_definitions": catalog_definition_count}
    if html is not None:
        from sphinx.util.inventory import InventoryFile
        with (html / "objects.inv").open("rb") as stream:
            objects = InventoryFile.load(stream, "", lambda base, location: location)
        inventory_names = {name for kind, values in objects.items() if kind.startswith("py:") for name in values}
        indexed = {symbol["name"] for symbol in records if symbol["indexed"]}
        _require(indexed.issubset(inventory_names), "Sphinx symbol inventory coverage differs")
        _require({module["name"] for module in modules.values()}.issubset(objects.get("py:module", {})),
                 "Sphinx module inventory coverage differs")
        result["sphinx_indexed_symbols"] = len(indexed)
        result["sphinx_indexed_modules"] = len(modules)
    return result


ACTION_PINS = {
    "checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "configure-pages": "45bfe0192ca1faeb007ade9deae92b16b8254a0d",
    "upload-pages-artifact": "fc324d3547104276b827a68afc52ff2a11cc49c9",
    "deploy-pages": "368f82528645a54fb793d4d04e342629a3f51346",
}
TRUSTED_MASTER_GATE = ("(github.event_name == 'push' || github.event_name == 'workflow_dispatch') && "
                       "github.ref == 'refs/heads/master'")
BASH_PIPEFAIL = "bash --noprofile --norc -e -o pipefail {0}"


def validate_workflow() -> dict:
    """Enforce immutable action pins and privilege boundaries without assertions."""
    import yaml

    path = ROOT / ".github" / "workflows" / "docs.yml"
    raw = path.read_text()
    workflow = yaml.safe_load(raw)
    _require(isinstance(workflow, dict), "Workflow must be a YAML mapping")
    triggers = workflow.get("on", workflow.get(True))
    _require(isinstance(triggers, dict)
             and set(triggers) == {"push", "pull_request", "workflow_dispatch"},
             "Docs validation must run on PR, master push and manual dispatch only")
    _require(triggers["push"] == {"branches": ["master"]}, "Push validation must target master")
    _require(triggers["pull_request"] in (None, {}) and triggers["workflow_dispatch"] in (None, {}),
             "PR/manual validation must not be silently restricted")
    _require(workflow.get("permissions") == {}, "Top-level permissions must be empty")
    _require(workflow.get("defaults", {}).get("run", {}).get("shell") == BASH_PIPEFAIL,
             "Workflow requires explicit Bash with pipefail")
    _require(workflow.get("concurrency", {}).get("cancel-in-progress") is False,
             "Publication concurrency must preserve running validations")
    _require(set(workflow.get("jobs", {})) == {"build", "deploy"}, "Unexpected workflow jobs")
    build, deploy = workflow["jobs"]["build"], workflow["jobs"]["deploy"]
    _require(build.get("permissions") == {"contents": "read"}, "Build permissions must be contents:read only")
    _require(deploy.get("permissions") == {"pages": "write", "id-token": "write"},
             "Deployment permissions must be pages:write and id-token:write only")
    _require("if" not in build, "PR documentation validation must not be gated away")
    _require(deploy.get("needs") == "build", "Deployment must depend on successful build")
    _require(deploy.get("environment", {}).get("name") == "github-pages",
             "Deployment must use the github-pages environment")
    _require(deploy.get("concurrency") == {"group": "github-pages", "cancel-in-progress": False},
             "Deployment must serialize the Pages environment")
    gate = " ".join(str(deploy.get("if", "")).split())
    _require(gate == TRUSTED_MASTER_GATE, "Deployment requires trusted push/manual master gate")
    _require(not re.search(r"\bsecrets\s*(?:\.|\[)", raw, re.I)
             and "pull_request_target" not in raw and "workflow_run" not in raw,
             "Docs workflow must not consume secrets or privileged untrusted events")
    expected_actions = {"actions/" + name + "@" + digest for name, digest in ACTION_PINS.items()}
    observed_actions = []
    for job in (build, deploy):
        _require(isinstance(job.get("timeout-minutes"), int) and 0 < job["timeout-minutes"] <= 30,
                 "Workflow jobs require bounded timeouts")
        _require(job.get("runs-on") == "ubuntu-24.04", "Workflow runner must be the reviewed hosted image")
        _require(not job.get("continue-on-error"), "Workflow jobs must fail closed")
        _require("uses" not in job, "Reusable jobs are outside the reviewed privilege boundary")
        if "defaults" in job:
            _require(job["defaults"].get("run", {}).get("shell") == BASH_PIPEFAIL,
                     "Job shell overrides must preserve Bash pipefail")
        _require(isinstance(job.get("steps"), list) and job["steps"], "Workflow job needs reviewed steps")
        for step in job["steps"]:
            _require(not step.get("continue-on-error"), "Workflow steps must fail closed")
            _require(not ("uses" in step and "run" in step), "A step cannot both run code and use an action")
            if "shell" in step:
                _require(step["shell"] == BASH_PIPEFAIL, "Step shell overrides must preserve Bash pipefail")
            if "uses" in step:
                _require(step["uses"] in expected_actions, "Unexpected or unverified immutable action pin")
                observed_actions.append(step["uses"])
    _require(set(observed_actions) == expected_actions and len(observed_actions) == len(expected_actions),
             "Every reviewed action must occur exactly once")
    build_actions = [step["uses"] for step in build["steps"] if "uses" in step]
    _require(build_actions == ["actions/checkout@" + ACTION_PINS["checkout"],
                               "actions/setup-python@" + ACTION_PINS["setup-python"],
                               "actions/upload-pages-artifact@" + ACTION_PINS["upload-pages-artifact"]],
             "Build action order or ownership differs")
    checkout = build["steps"][0]
    _require(checkout.get("uses") == build_actions[0]
             and checkout.get("with") == {"fetch-depth": 0, "persist-credentials": False},
             "Checkout must not persist credentials or override source identity")
    upload = build["steps"][-1]
    _require(upload.get("uses") == build_actions[-1], "Pages upload must be the final build step")
    _require(" ".join(str(upload.get("if", "")).split()) == TRUSTED_MASTER_GATE,
             "Artifact upload requires trusted push/manual master gate")
    _require(upload.get("with") == {
        "path": "${{ runner.temp }}/hypertagging-docs/html",
        "retention-days": 1,
        "include-hidden-files": True,
    },
             "Pages upload must contain validated HTML only")
    commands = [step["run"] for step in build["steps"] if "run" in step]
    _require(any("python -m pytest" in command and "tests/test_docs_" in command for command in commands),
             "Documentation regression checks are missing")
    _require(any("scripts/build_docs.py" in command for command in commands),
             "Strict documentation build is missing")
    _require(any("--builder text" in command and "--layout basf2" in command for command in commands),
             "Text and basf2 discovery validation are missing")
    _require(all("if" not in step for step in build["steps"] if "run" in step),
             "Required validation steps must not be conditionally skipped")
    _require(any("scripts/validate_docs.py" in command
                 and all(flag in command for flag in ("--html", "--generated", "--check-generation", "--workflow"))
                 for command in commands), "Independent coverage/privacy/workflow validation is missing")
    _require(len(deploy["steps"]) == 2 and all("run" not in step for step in deploy["steps"]),
             "Deployment must execute no repository shell code")
    configure, publish = deploy["steps"]
    _require(configure.get("uses") == "actions/configure-pages@" + ACTION_PINS["configure-pages"]
             and configure.get("with") == {"enablement": False},
             "Pages configuration must not enable repository settings")
    _require(publish.get("uses") == "actions/deploy-pages@" + ACTION_PINS["deploy-pages"]
             and publish.get("id") == "deployment" and "with" not in publish,
             "Deployment must publish the default validated Pages artifact")
    return {"workflow": ".github/workflows/docs.yml", "permission_boundary": "PASS",
            "action_pins": len(ACTION_PINS), "deployment_branch": "master",
            "shell_pipefail": True}


def validate_generation(generated: Path) -> dict:
    """Compare every generated byte against a fresh build from the same source."""
    from wiki import generate
    with tempfile.TemporaryDirectory(prefix="hypertagging-docs-repro-") as temporary:
        regenerated = Path(temporary) / "generated"
        generate(ROOT, regenerated)
        def hashes(base):
            return {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in base.rglob("*") if p.is_file()}
        if hashes(generated) != hashes(regenerated):
            raise ValueError("Generated output changed: sources, provenance or generation are stale")
        return {"deterministic_generated_files": len(hashes(regenerated))}


def main(argv=None):
    """Run requested independent checks and print machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="Scan a built HTML/text or generated projection artifact for privacy")
    parser.add_argument("--html", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--check-generation", action="store_true")
    parser.add_argument("--workflow", action="store_true")
    args = parser.parse_args(argv)
    if args.check_generation and not args.generated:
        parser.error("--check-generation requires --generated")
    if not (args.artifact or args.html or args.generated or args.workflow):
        parser.error("select at least one check")
    from wiki_privacy import validate_artifact
    result = {}
    if args.artifact:
        result["artifact_privacy"] = validate_artifact(args.artifact, ROOT)
    if args.html:
        result["html_privacy"] = validate_artifact(
            args.html, ROOT, renderer_output=True,
        )
        result["html"] = validate_html(args.html)
    if args.generated:
        result["generated_privacy"] = validate_artifact(
            args.generated, ROOT, generated_projection=True,
        )
        result["coverage"] = validate_coverage(args.generated, html=args.html)
    if args.check_generation:
        result["generation"] = validate_generation(args.generated)
    if args.workflow:
        result["workflow"] = validate_workflow()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
