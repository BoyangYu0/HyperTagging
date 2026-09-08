"""Generate an import-free, deterministic Sphinx Python API reference.

Only Python's syntax tree is inspected. Project modules, decorators, defaults,
annotations, ``__getattr__`` hooks, and optional dependencies are never executed.
Every lexical function and class definition is included, including private,
undocumented, overloaded, conditional and function-local declarations. Package
imports and explicit same-name imports are supplemental reexports. Only redacted
signatures, literal docstrings, relative source locations and hashes are
published; source bodies, constant values and source downloads are never emitted.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from wiki_privacy import redact, safe_signature


SCHEMA_VERSION = 2
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS = (ast.ClassDef, *_FUNCTIONS)
# ``match_case`` was added with structural pattern matching in Python 3.10.
# Current basf2 uses a newer Python, but keeping the traversal conditional lets
# historical basf2 documentation environments inspect pre-match source trees.
_CONTROL_BODY_NODES = (ast.ExceptHandler,) + (
    (ast.match_case,) if hasattr(ast, "match_case") else ()
)
_UNKNOWN = object()


def _static_value(node: ast.AST | None, names: dict[str, Any]) -> Any:
    """Read bounded literal export expressions without eval or Python calls."""
    if node is None:
        return _UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_static_value(item, names) for item in node.elts]
        if any(value is _UNKNOWN for value in values):
            return _UNKNOWN
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            decoded = _static_value(value, names)
            if decoded is _UNKNOWN:
                return _UNKNOWN
            if key is None:
                if not isinstance(decoded, dict):
                    return _UNKNOWN
                result.update(decoded)
            else:
                decoded_key = _static_value(key, names)
                if not isinstance(decoded_key, (str, int, float, tuple)):
                    return _UNKNOWN
                try:
                    result[decoded_key] = decoded
                except TypeError:
                    return _UNKNOWN
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_value(node.left, names), _static_value(node.right, names)
        if type(left) is type(right) and isinstance(left, (str, tuple, list)):
            return left + right
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        values = _static_value(generator.iter, names)
        if (
            isinstance(generator.target, ast.Name)
            and not generator.ifs
            and not generator.is_async
            and isinstance(values, (tuple, list))
            and len(values) <= 10_000
        ):
            result = {}
            for value in values:
                scope = {**names, generator.target.id: value}
                key = _static_value(node.key, scope)
                item = _static_value(node.value, scope)
                if not isinstance(key, str) or item is _UNKNOWN:
                    return _UNKNOWN
                result[key] = item
            return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"sorted", "list", "tuple"}
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _static_value(node.args[0], names)
        if isinstance(value, (dict, list, tuple)):
            items = list(value)
            if node.func.id == "sorted":
                if not all(isinstance(item, str) for item in items):
                    return _UNKNOWN
                return sorted(items)
            return tuple(items) if node.func.id == "tuple" else items
    return _UNKNOWN


def _statements(body: list[ast.stmt]):
    """Traverse one lexical scope's control flow, never enter a child definition."""
    for node in body:
        yield node
        if isinstance(node, _DEFINITIONS):
            continue
        # This also covers async blocks, match cases, exception groups and
        # conditional alternatives without flattening a child lexical scope.
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for child in value:
                if isinstance(child, ast.stmt):
                    yield from _statements([child])
                elif isinstance(child, _CONTROL_BODY_NODES):
                    yield from _statements(child.body)


def _lexical_definitions(node: ast.AST) -> list[ast.AST]:
    """Return direct child definitions, including conditional alternatives."""
    return [
        child for child in _statements(getattr(node, "body", []))
        if isinstance(child, _DEFINITIONS)
    ]


def _expression(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""


def _import_base(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = module.split(".") if is_package else module.split(".")[:-1]
    if node.level > len(parts):
        return ""
    parts = parts[: len(parts) - node.level + 1]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def _reference(node: ast.AST, module: dict[str, Any]) -> str | None:
    if isinstance(node, ast.Name):
        return module["imports"].get(node.id, module["name"] + "." + node.id)
    if isinstance(node, ast.Attribute):
        base = _reference(node.value, module)
        return base + "." + node.attr if base else None
    return None


def _parse_module(path: Path, source_root: Path, repo_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    # tokenize.open would also support PEP 263 encodings; ast.parse accepts bytes
    # directly and therefore honors the Python source encoding declaration.
    tree = ast.parse(raw, filename=str(path.relative_to(repo_root)))
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    module: dict[str, Any] = {
        "name": ".".join(parts),
        "path": path,
        "source": path.relative_to(repo_root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lines": len(raw.splitlines()),
        "docstring": ast.get_docstring(tree) or "",
        "is_package": is_package,
        "definitions": {},
        "definition_occurrences": [],
        "assignments": {},
        "imports": {},
        "reexports": set(),
        "star_imports": [],
        "values": {},
        "exports": None,
        "export_expression": None,
        "lazy_exports": {},
    }
    statements = list(_statements(tree.body))
    for position, node in enumerate(statements):
        if isinstance(node, _DEFINITIONS):
            module["definition_occurrences"].append(node)
            previous = module["definitions"].get(node.name)
            # A conditional alternative or overload must not erase documentation.
            if previous is None or ast.get_docstring(node) or not ast.get_docstring(previous):
                module["definitions"][node.name] = node
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(module["name"], is_package, node)
            for imported in node.names:
                if imported.name == "*":
                    module["star_imports"].append(base)
                    continue
                name = imported.asname or imported.name
                module["imports"][name] = base + "." + imported.name
                if is_package or imported.asname == imported.name:
                    module["reexports"].add(name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                name = imported.asname or imported.name.split(".")[0]
                module["imports"][name] = imported.name if imported.asname else name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _static_value(node.value, module["values"])
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if value is not _UNKNOWN:
                    module["values"][name] = value
                docstring = ""
                if position + 1 < len(statements):
                    following = statements[position + 1]
                    if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                        if isinstance(following.value.value, str):
                            docstring = following.value.value
                module["assignments"][name] = (node, docstring)
                if name == "__all__":
                    module["export_expression"] = _expression(node.value)
                    module["exports"] = (
                        list(value) if isinstance(value, (list, tuple))
                        and all(isinstance(item, str) for item in value) else None
                    )
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.op, ast.Add):
                name = node.target.id
                left = module["values"].get(name, _UNKNOWN)
                right = _static_value(node.value, module["values"])
                if type(left) is type(right) and isinstance(left, (list, tuple)):
                    module["values"][name] = left + right
                    if name == "__all__":
                        module["exports"] = list(left + right)
                elif name == "__all__":
                    module["exports"] = None
                if name == "__all__":
                    module["export_expression"] = _expression(node)
        elif (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "__all__"
        ):
            call = node.value
            previous = module["exports"]
            value = _static_value(call.args[0], module["values"]) if len(call.args) == 1 else _UNKNOWN
            if previous is not None and not call.keywords and call.func.attr == "append" and isinstance(value, str):
                module["exports"] = [*previous, value]
            elif previous is not None and not call.keywords and call.func.attr == "extend" and isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
                module["exports"] = [*previous, *value]
            else:
                module["exports"] = None
            if module["exports"] is None:
                module["values"].pop("__all__", None)
            else:
                module["values"]["__all__"] = module["exports"]
            module["export_expression"] = _expression(call)
    # This repository's import-cycle-safe facade uses a literal name -> module
    # map. Resolve that declaration, never invoke its __getattr__ implementation.
    lazy = module["values"].get("_EXPORT_MODULE", {})
    if isinstance(lazy, dict):
        module["lazy_exports"] = {
            name: destination + "." + name
            for name, destination in lazy.items()
            if isinstance(name, str) and isinstance(destination, str)
        }
    return module


def _resolve(
    qualified: str, modules: dict[str, dict[str, Any]], seen: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], ast.AST, str, str] | None:
    if qualified in seen:
        return None
    seen = seen | {qualified}
    module_name, _, name = qualified.rpartition(".")
    module = modules.get(module_name)
    if module is None:
        return None
    if name in module["definitions"]:
        return module, module["definitions"][name], qualified, ""
    if name in module["assignments"]:
        node, docstring = module["assignments"][name]
        reference = _reference(node.value, module) if node.value is not None else None
        resolved = _resolve(reference, modules, seen) if reference else None
        return resolved or (module, node, qualified, docstring)
    destination = module["imports"].get(name) or module["lazy_exports"].get(name)
    if destination:
        return _resolve(destination, modules, seen)
    for imported_module in module["star_imports"]:
        imported = modules.get(imported_module)
        if imported and name in _public_names(imported):
            resolved = _resolve(imported_module + "." + name, modules, seen)
            if resolved:
                return resolved
    return None


def _public_names(module: dict[str, Any]) -> set[str]:
    names = {name for name in module["definitions"] if not name.startswith("_")}
    names.update(name for name in module["reexports"] if not name.startswith("_"))
    for name, (node, docstring) in module["assignments"].items():
        if not name.startswith("_") and (docstring or isinstance(node.value, (ast.Name, ast.Attribute))):
            names.add(name)
    # Export lists are data and can contain arbitrary strings. Only valid Python
    # identifiers identify an API object; never publish invalid literal payloads.
    names.update(name for name in (module["exports"] or []) if isinstance(name, str) and name.isidentifier())
    return names


def _annotation(node: ast.AST, module: dict[str, Any]) -> ast.AST:
    """Qualify known project annotations statically to avoid facade ambiguity."""
    names = module.get("annotation_names", {})
    class QualifiedAnnotation(ast.NodeTransformer):
        def visit_Name(self, value):
            if value.id in names:
                return ast.copy_location(ast.parse(names[value.id], mode="eval").body, value)
            return value

        def visit_Constant(self, value):
            if isinstance(value.value, str):
                try:
                    parsed = ast.parse(value.value, mode="eval").body
                except SyntaxError:
                    return value
                # Only strings describing a type expression are annotations.
                # Avoid recursively reparsing strings inside Literal values.
                parsed = QualifiedNames().visit(parsed)
                return ast.copy_location(ast.Constant(value=ast.unparse(parsed)), value)
            return value

    class QualifiedNames(QualifiedAnnotation):
        def visit_Constant(self, value):
            return value

    return QualifiedAnnotation().visit(copy.deepcopy(node))


def _signature(node: ast.AST, name: str, *, drop_receiver: bool = False, module: dict[str, Any] | None = None) -> str:
    if not isinstance(node, _FUNCTIONS):
        return name
    # ast.unparse preserves positional-only, keyword-only, variadic arguments,
    # defaults and annotations without evaluating any of them.
    args = copy.deepcopy(node.args)
    if module:
        for argument in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if argument is not None and argument.annotation is not None:
                argument.annotation = _annotation(argument.annotation, module)
    class RedactedDefault(ast.NodeTransformer):
        def visit_Constant(self, value):
            if isinstance(value.value, (str, bytes)):
                return ast.copy_location(ast.Constant(value="[redacted]"), value)
            return value
    transformer = RedactedDefault()
    args.defaults = [transformer.visit(value) for value in args.defaults]
    args.kw_defaults = [transformer.visit(value) if value is not None else None for value in args.kw_defaults]
    if drop_receiver and (args.posonlyargs or args.args):
        args = ast.arguments(
            posonlyargs=list(args.posonlyargs), args=list(args.args),
            vararg=args.vararg, kwonlyargs=list(args.kwonlyargs),
            kw_defaults=list(args.kw_defaults), kwarg=args.kwarg,
            defaults=list(args.defaults),
        )
        positional = args.posonlyargs or args.args
        if positional[0].arg in {"self", "cls"}:
            original_count = len(args.posonlyargs) + len(args.args)
            positional.pop(0)
            if len(args.defaults) == original_count:
                args.defaults.pop(0)
    result = name + "(" + ast.unparse(args) + ")"
    if node.returns is not None:
        result += " -> " + _expression(_annotation(node.returns, module) if module else node.returns)
    return safe_signature(result)


def _literal(text: str, indent: str = "") -> list[str]:
    if not text:
        return []
    return [indent + ".. code-block:: text", ""] + [
        indent + "   " + line if line else "" for line in redact(text).splitlines()
    ] + [""]


def _member_record(
    name: str, source_module: dict[str, Any], node: ast.AST,
    canonical: str, kind: str, signature: str, docstring: str, surface: str,
) -> dict[str, Any]:
    return {
        "name": name, "kind": kind, "canonical": canonical,
        "source": source_module["source"], "line": node.lineno,
        "end_line": node.end_lineno, "signature": signature,
        "documented": bool(docstring), "alias": name != canonical,
        "sha256": source_module["sha256"], "surface": surface,
        "definition": isinstance(node, _DEFINITIONS),
        "definition_id": f"{source_module['source']}:{node.lineno}:{canonical}",
        "occurrence_id": f"{name}@{source_module['source']}:{node.lineno}",
    }


def _render_symbol(
    name: str, resolved: tuple[dict[str, Any], ast.AST, str, str],
    records: list[dict[str, Any]], *, parent_name: str | None = None,
    noindex: bool = False,
    internal: bool = False,
    member: bool = False,
    function_local: bool = False,
) -> list[str]:
    module, node, canonical, assignment_doc = resolved
    local_name = name.rsplit(".", 1)[-1]
    display_name = parent_name + "." + local_name if parent_name else local_name
    internal = internal or function_local or local_name.startswith("_") or any(
        part.startswith("_") for part in module["name"].split(".")[1:]
    )
    surface = "internal_unstable" if internal else "public_stable"
    docstring = ast.get_docstring(node) or "" if isinstance(node, _DEFINITIONS) else assignment_doc
    options = []
    if isinstance(node, ast.ClassDef):
        kind, directive = "class", "class"
        constructor = next((item for item in node.body if isinstance(item, _FUNCTIONS) and item.name == "__init__"), None)
        signature = _signature(constructor, display_name, drop_receiver=True, module=module) if constructor else display_name
        # Python class signatures have no return annotation.
        if constructor and constructor.returns is not None:
            signature = signature.rsplit(" -> ", 1)[0]
    elif isinstance(node, _FUNCTIONS):
        kind, directive = ("method", "method") if member else ("function", "function")
        decorators = {_expression(item).split("(", 1)[0] for item in node.decorator_list}
        if member and (
            "property" in decorators or "cached_property" in decorators
            or "functools.cached_property" in decorators
        ):
            kind, directive = "property", "attribute"
            signature = display_name
            if node.returns is not None:
                options.append(":type: " + safe_signature(_expression(_annotation(node.returns, module))))
        else:
            signature = _signature(
                node, display_name,
                drop_receiver=member and "staticmethod" not in decorators,
                module=module,
            )
            if member and "staticmethod" in decorators:
                options.append(":staticmethod:")
            elif member and "classmethod" in decorators:
                options.append(":classmethod:")
        if isinstance(node, ast.AsyncFunctionDef):
            options.append(":async:")
    else:
        kind, directive = ("attribute", "attribute") if member else ("data", "data")
        signature = display_name
        if isinstance(node, ast.AnnAssign):
            options.append(":type: " + safe_signature(_expression(_annotation(node.annotation, module))))
    # Conditional definitions, overloads and property accessors retain separate
    # records and anchors. Only one occurrence owns a Python domain target.
    duplicate_name = any(record["name"] == name for record in records)
    # Function-local objects are not importable Python API targets. Keep their
    # Python-domain presentation inert while retaining a unique explicit anchor.
    noindex = noindex or duplicate_name or function_local
    record = _member_record(name, module, node, canonical, kind, signature, docstring, surface)
    record["indexed"] = not noindex
    records.append(record)
    if noindex:
        options.append(":noindex:")
    anchor = "api-definition-" + hashlib.sha256(record["occurrence_id"].encode()).hexdigest()[:20]
    record["anchor"] = anchor
    lines = [f".. _{anchor}:", "", f".. py:{directive}:: {signature}"]
    lines.extend("   " + option for option in options)
    lines += ["", f"   Source: ``{module['source']}:{node.lineno}``.", ""]
    lines += ["   Surface: " + ("internal / unstable." if internal else "public / stable naming surface."), ""]
    if name != canonical:
        lines += [f"   Reexport of :py:obj:`{canonical}`.", ""]
    if function_local:
        lines += ["   This function-local definition has a source-line anchor but "
                  "is not published as an importable Python object.", ""]
    elif noindex:
        lines += ["   This occurrence has its own source-line anchor; the shared "
                  "Python name is indexed once.", ""]
    if isinstance(node, ast.ClassDef) and node.bases:
        lines += ["   Declared bases: " + ", ".join("``" + safe_signature(_expression(base)) + "``" for base in node.bases) + ".", ""]
    if docstring:
        lines.extend(_literal(docstring, "   "))
    else:
        lines += ["   No source docstring is declared.", ""]
    if isinstance(node, ast.ClassDef):
        members = []
        body = list(_statements(node.body))
        for index, child in enumerate(body):
            if isinstance(child, _DEFINITIONS):
                members.append((child.name, child, ""))
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                if not child.target.id.startswith("_"):
                    doc = ""
                    if index + 1 < len(body) and isinstance(body[index + 1], ast.Expr):
                        value = body[index + 1].value
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            doc = value.value
                    members.append((child.target.id, child, doc))
        current_surface = None
        for member_name, child, child_doc in sorted(members, key=lambda item: (internal or item[0].startswith("_"), item[0], item[1].lineno)):
            member_internal = internal or member_name.startswith("_")
            if member_internal != current_surface:
                heading = "Internal / unstable members" if member_internal else "Public / stable members"
                lines += ["**" + heading + "**", ""]
                current_surface = member_internal
            child_name = name + "." + member_name
            child_canonical = canonical + "." + member_name
            lines += _render_symbol(
                child_name, (module, child, child_canonical, child_doc), records,
                parent_name=display_name, internal=member_internal, member=True,
                function_local=function_local,
            )
    elif isinstance(node, _FUNCTIONS):
        lexical_definitions = sorted(
            _lexical_definitions(node), key=lambda item: (item.name, item.lineno)
        )
        if lexical_definitions:
            lines += ["**Function-local definitions (internal / unstable)**", "",
                      "These lexical definitions are documented statically and are "
                      "not importable API objects.", ""]
        for child in lexical_definitions:
            lines += _render_symbol(
                name + "." + child.name,
                (module, child, canonical + "." + child.name, ""), records,
                parent_name=display_name, internal=True, member=False,
                function_local=True,
            )
    return lines


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def generate_api(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write every module page plus a machine-readable coverage inventory.

    ``output_dir`` must be a dedicated generated directory. Existing unrelated
    files are never deleted. The returned inventory contains no absolute paths,
    timestamps, runtime imports, or environment-dependent values. Unsupported
    dynamic exports remain visible as unresolved entries rather than being
    silently omitted or executed.
    """
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    source_root = repo_root / "src"
    package_root = source_root / "hypertagging"
    if not package_root.is_dir():
        raise ValueError(f"Missing Python package: {package_root}")
    if output_dir == repo_root or output_dir.is_relative_to(source_root):
        raise ValueError("API output must be a dedicated directory outside src")
    modules = {}
    for path in sorted(package_root.rglob("*.py")):
        if path.is_symlink() and not path.resolve().is_relative_to(repo_root):
            raise ValueError(f"Python source escapes the repository: {path}")
        module = _parse_module(path, source_root, repo_root)
        modules[module["name"]] = module
    for module in modules.values():
        annotations = {}
        for local_name in set(module["definitions"]) | set(module["imports"]) | set(module["lazy_exports"]):
            qualified = module["name"] + "." + local_name
            resolved = _resolve(qualified, modules)
            if resolved is not None:
                annotations[local_name] = resolved[2]
            elif module["imports"].get(local_name, "").startswith("hypertagging."):
                annotations[local_name] = module["imports"][local_name]
        module["annotation_names"] = annotations
    records: list[dict[str, Any]] = []
    module_records = []
    unresolved = []
    dynamic_exports = []
    for module_name, module in sorted(modules.items()):
        filename = module_name + ".rst"
        page = output_dir / filename
        lines = [
            module_name, "=" * len(module_name), "",
            ".. Generated by docs/_ext/wiki_api.py; edit the Python source.", "",
            f".. py:module:: {module_name}", "",
            f"Source provenance: ``{module['source']}`` ({module['lines']} lines).", "",
            f"Source SHA-256: ``{module['sha256']}``.", "",
        ]
        lines += _literal(module["docstring"])
        names = _public_names(module)
        for base in module["star_imports"]:
            if base in modules:
                names.update(_public_names(modules[base]))
        if module["export_expression"] is not None and module["exports"] is None:
            dynamic_exports.append({"module": module_name, "sha256": module["sha256"]})
            lines += ["The export list is dynamic and cannot be resolved safely from source.", ""]
        start = len(records)
        local_nodes = module["definition_occurrences"]
        local_names = {node.name for node in local_nodes}
        for internal, title in ((False, "Public / stable definitions"), (True, "Internal / unstable definitions")):
            selected = [node for node in local_nodes if (
                node.name.startswith("_") or any(part.startswith("_") for part in module_name.split(".")[1:])
            ) == internal]
            if not selected:
                continue
            lines += [title, "-" * len(title), ""]
            for node in sorted(selected, key=lambda item: (item.name, item.lineno)):
                qualified = module_name + "." + node.name
                lines += _render_symbol(
                    qualified, (module, node, qualified, ""), records,
                    noindex=qualified in modules, internal=internal,
                )
        supplemental = sorted(names - local_names)
        if supplemental:
            lines += ["Exports and declared data", "-------------------------", ""]
        for name in supplemental:
            qualified = module_name + "." + name
            resolved = _resolve(qualified, modules)
            if resolved is None:
                destination = module["imports"].get(name) or module["lazy_exports"].get(name)
                # External imports without an explicit __all__ entry are not
                # project-owned reexports (e.g. pathlib.Path in a facade).
                if destination and not destination.startswith("hypertagging.") and name not in (module["exports"] or []):
                    continue
                destination = redact(destination) if destination else None
                unresolved.append({"name": qualified, "target": destination})
                lines += [f".. py:data:: {name}", "", "   Export declared in source; its definition cannot be resolved statically.", ""]
                if destination:
                    lines += [f"   Declared import target: ``{destination}``.", ""]
            else:
                lines += _render_symbol(qualified, resolved, records, noindex=qualified in modules)
        if not names and not local_nodes:
            lines += ["This module declares no Python functions or classes.", ""]
        module_records.append({
            "name": module_name, "source": module["source"], "sha256": module["sha256"],
            "source_lines": module["lines"], "page": filename,
            "documented": bool(module["docstring"]),
            "exports": [item for item in module["exports"] if isinstance(item, str) and item.isidentifier()]
                       if module["exports"] is not None else None,
            "invalid_export_count": sum(not isinstance(item, str) or not item.isidentifier()
                                        for item in (module["exports"] or [])),
            "symbols": [record["name"] for record in records[start:]],
            "definitions": [record["definition_id"] for record in records[start:]
                            if record["definition"] and not record["alias"]],
        })
        _write_if_changed(page, "\n".join(lines))
    inventory = {
        "schema_version": SCHEMA_VERSION, "generator": "docs/_ext/wiki_api.py",
        "policy": "AST only; every lexical function and class definition; public/internal surfaces; redacted literal docstrings; no source bodies or downloads",
        "module_count": len(module_records), "symbol_count": len(records),
        "documented_symbol_count": sum(record["documented"] for record in records),
        "definition_count": sum(record["definition"] and not record["alias"] for record in records),
        "surface_counts": {surface: sum(record["surface"] == surface for record in records)
                           for surface in ("public_stable", "internal_unstable")},
        "modules": module_records, "symbols": records,
        "unresolved_exports": unresolved, "dynamic_exports": dynamic_exports,
    }
    _write_if_changed(output_dir / "inventory.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    lines = [
        "Python API reference", "====================", "",
        f"Every one of the {len(module_records)} Python modules under ``src/hypertagging`` "
        "is indexed below, including compatibility and deployment modules.", "",
        "The generator reads source syntax only. It never imports basf2, PyTorch, "
        "CUDA, optional dependencies, or project modules. Signatures, declared "
        "class fields, relative line provenance, and docstrings come from the checked-out "
        "source. Sensitive values are redacted. Mixed-format docstrings are shown "
        "literally without interpreting directives or requiring external roles. "
        "Source bodies and raw source downloads are excluded.", "",
        "Every lexical function and class definition is documented, including "
        "underscore-prefixed internals, undocumented dunder methods, overloads, "
        "conditional alternatives, property setters, nested functions and local classes. "
        "Each definition has a separate source-line identity in the inventory.", "",
        "Public / stable names and internal / unstable names are separated. "
        "These labels describe the naming surface; they do not promise a versioned "
        "compatibility guarantee. Underscore names and all function-local definitions "
        "are internal even when explicitly exported. Function-local entries keep unique "
        "anchors but are not added to the importable Python object inventory. Explicit exports and package "
        "reexports are supplemental; inherited and decorator-generated definitions "
        "are not invented.", "",
        ":download:`Download the coverage and provenance inventory <inventory.json>`.", "",
        ".. toctree::", "   :maxdepth: 1", "",
    ]
    lines.extend("   " + module["name"] for module in module_records)
    lines.append("")
    _write_if_changed(output_dir / "index.rst", "\n".join(lines))
    return inventory
