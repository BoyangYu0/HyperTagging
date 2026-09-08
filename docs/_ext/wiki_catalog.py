"""Build an offline script catalogue from source text without importing scripts.

This module uses only the standard library.  CLI declarations are evidence from
the AST, not a promise that arbitrary scripts support a harmless ``--help``.
Commands with fixture guarantees are deliberately curated against the examples
and CLI CPU tests; new files are inventoried conservatively until reviewed.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from wiki_privacy import redact, safe_signature


_SUFFIXES = {".py", ".sh", ".sbatch"}
_CPU_PREFIX = 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2'
_EXAMPLES = {
    f"examples/{name}_minimal/run_example.py"
    for name in ("toy_mc", "grafei", "gpt_like")
}
_DRY_RUNS = {
    f"scripts/{name}.py": "--dry-run --device cpu"
    for name in (
        "train_embedding", "train_link", "train_reconstruction", "train_gpt_like",
        "evaluate_reconstruction", "run_gpt_like",
    )
}
_DRY_RUNS.update({
    f"scripts/{name}.py": "--dry-run --tiny --device cpu --max-steps 2 --batch-size 2"
    for name in (
        "train_hyperbolic_pretrain", "train_level_reconstruction",
        "run_level_reconstruction",
    )
})
_DRY_RUNS["scripts/evaluate_level_reconstruction.py"] = "--dry-run --tiny --device cpu"
_OPERATOR_SCRIPTS = {
    "publish_training_selection_repromotion_once.py",
    "run_phase3_batch_efficiency_calibration.py",
    "run_phase3_batch_efficiency_stability_pilot.py",
    "run_reconstruction_fullscale.py",
    "run_reconstruction_transfer_probe.py",
    "run_basf2_full_decay.py",
}
# Only reviewed entry points advertise --help. An arbitrary future script may
# perform work before argument parsing, so discovering argparse is insufficient.
_HELP_REVIEWED = {
    "benchmark_parquet_storage.py", "build_dataset_index.py",
    "build_training_selection.py", "check_uv_lock_direct_dependencies.py",
    "diagnose_reconstruction_query_activation.py", "evaluate_full_decay.py",
    "evaluate_hyperbolic_pretrain.py", "execute_notebook_smoke_tests.py",
    "export_full_decay_onnx.py", "generate_audit_views.py",
    "mdst_batch_production.py", "prepare_grafei.py", "prepare_toy_mc.py",
    "preprocess_mdst.py", "report_reconstruction_capacity.py",
    "validate_reconstruction_checkpoint_pair.py", "validate_training_provenance.py",
    "verify_preprocessing.py", "render_condor_job.py",
    "create_dataset_inspection_notebook.py", "create_exact_tree_geometry_notebook.py",
    "create_first_level_ambiguity_notebook.py", "create_hyperbolic_inspection_notebook.py",
    "create_leaf_input_pid_notebook.py", "create_leaf_pid_composite_notebook.py",
    "create_mdst_10k_production_notebooks.py", "create_mdst_10m_validation_notebook.py",
    "create_mdst_10m_validation_report.py", "create_parquet_gpt_inspection_notebook.py",
    "create_preprocessing_qa_notebook.py", "create_preprocessing_visualization_notebook.py",
    "create_production_manifest_notebook.py", "create_query_capacity_notebook.py",
    "create_real_mdst_pilot_notebook.py", "create_reconstruction_inspection_notebook.py",
    "create_rollout_search_calibration_notebook.py", "create_runtime_scaling_notebook.py",
    "create_streaming_dataset_notebook.py", "create_trained_physics_validation_notebook.py",
    "create_training_pipeline_notebook.py",
}
_BATCH_SUMMARIES = {
    "evaluate_pretraining_checkpoint_study.sbatch": "Run a source-bound checkpoint comparison on the fixed pretraining validation cohort inside Slurm.",
    "evaluate_pretraining_validation.sbatch": "Run read-only pretraining validation with allocation checks, GPU telemetry and a terminal receipt.",
    "run_phase3_batch_efficiency_calibration.sbatch": "Run one bounded phase-3 calibration tuple after the tracked submitter supplies its contract and allocation environment.",
    "run_reconstruction_fullscale.sbatch": "Execute contract-bound reconstruction calibration or production with telemetry and bounded restart support.",
    "run_reconstruction_transfer_probe.sbatch": "Execute a source-bound frozen-encoder reconstruction transfer probe and record its terminal evidence.",
    "train_one_gpu.sbatch": "Train one contract-bound pretraining job in an authenticated single-GPU Slurm allocation with bounded requeue.",
}
_SPECIAL = {
    "scripts/activate_env.sh": (
        "environment helper",
        "source scripts/activate_env.sh project",
        "An existing project .venv; source this helper in Bash from the repository root.",
        "Changes the current shell environment and working directory; direct execution is refused.",
    ),
    "scripts/condor/check_condor_env.py": (
        "environment diagnostic",
        "python scripts/condor/check_condor_env.py",
        "Project dependencies; condor_q and nvidia-smi for populated diagnostics.",
        "Runs scheduler/GPU inspection commands and prints their results; no jobs are submitted. It has no --help parser.",
    ),
    "scripts/condor/submit_hyperbolic_pretrain.sh": (
        "HTCondor renderer",
        "DATA_MANIFEST=INPUT_MANIFEST bash scripts/condor/submit_hyperbolic_pretrain.sh --dry-run",
        "Project environment, repository configuration and DATA_MANIFEST; the resulting job needs the GPU environment and data.",
        "Renders a job description; --dry-run prints it. Submission is a separate operator action.",
    ),
    "scripts/condor/submit_level_reconstruction.sh": (
        "HTCondor renderer",
        "DATA_MANIFEST=INPUT_MANIFEST PRETRAINED_ENCODER=TRUSTED_ENCODER bash scripts/condor/submit_level_reconstruction.sh --dry-run",
        "Project environment, DATA_MANIFEST and PRETRAINED_ENCODER; the resulting job needs the GPU environment and data.",
        "Renders a job description; --dry-run prints it. Submission is a separate operator action.",
    ),
    "scripts/condor/submit_preprocess_mdst.sh": (
        "HTCondor renderer",
        "bash scripts/condor/submit_preprocess_mdst.sh --dry-run",
        "Python and repository configuration; inspect the hard-coded input/output paths before rendering a real job.",
        "Prints a submit description in --dry-run mode. The rendered worker preprocesses mDST and enables overwrite; review its paths.",
    ),
    "scripts/condor/submit_mdst_production_10m.sh": (
        "production launcher",
        "bash scripts/condor/submit_mdst_production_10m.sh --dry-run",
        "Site-specific environment paths, mDST inputs, approved campaign readiness report, manifest and production volume.",
        "Even --dry-run can plan/write a missing manifest and create directories. --submit submits HTCondor jobs; --worker executes preprocessing. This is not a fixture smoke command.",
    ),
    "scripts/slurm/run_with_bounded_requeue.sh": (
        "Slurm execution wrapper", None,
        "A guarded Slurm allocation, restart limit, status destination and trainer command supplied by a tracked job wrapper.",
        "Runs the supplied trainer, forwards signals, writes status and may requeue the current Slurm job. It has no --help mode.",
    ),
    "scripts/validate_audit_integrity.py": (
        "read-only validator", "python scripts/validate_audit_integrity.py",
        "Python, PyYAML, Git history and the checked-in audit authorities.",
        "Reads audit metadata and Git history and prints validation findings; no --help parser is defined.",
    ),
}


def _literal_block(value: str, language: str = "text") -> list[str]:
    return [f".. code-block:: {language}", "", *[f"   {line}" if line else "" for line in redact(value).splitlines()], ""]


def _escape(value: str) -> str:
    return re.sub(r"([\\`*_|])", r"\\\1", redact(value))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
            operands = [node.test.left, *node.test.comparators]
            if any(isinstance(item, ast.Name) and item.id == "__name__" for item in operands) and any(
                isinstance(item, ast.Constant) and item.value == "__main__" for item in operands
            ):
                return True
    return False


def _definition_signature(node: ast.AST, name: str) -> str:
    """Return a qualified signature with every default and string literal hidden."""
    if isinstance(node, ast.ClassDef):
        return safe_signature(name)
    arguments = copy.deepcopy(node.args)

    class RedactedAnnotation(ast.NodeTransformer):
        def visit_Constant(self, value):
            if isinstance(value.value, (str, bytes)):
                return ast.copy_location(ast.Constant(value="[redacted]"), value)
            return value

    transformer = RedactedAnnotation()
    for argument in [
        *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
        arguments.vararg, arguments.kwarg,
    ]:
        if argument is not None and argument.annotation is not None:
            argument.annotation = transformer.visit(argument.annotation)
    arguments.defaults = [ast.Constant(Ellipsis) for _ in arguments.defaults]
    arguments.kw_defaults = [
        ast.Constant(Ellipsis) if item is not None else None
        for item in arguments.kw_defaults
    ]
    signature = f"{name}({ast.unparse(arguments)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(transformer.visit(copy.deepcopy(node.returns)))}"
    return safe_signature(signature)


def _definitions(tree: ast.Module) -> list[dict]:
    """Project every lexical definition without bodies, defaults or imports."""
    definitions = []

    def visit(node, prefix="", scope="module", function_local=False):
        if isinstance(node, ast.ClassDef):
            name = f"{prefix}.{node.name}" if prefix else node.name
            definitions.append({
                "name": redact(name), "kind": "class",
                "signature": _definition_signature(node, name),
                "docstring": redact(ast.get_docstring(node) or ""),
                "line": node.lineno, "end_line": node.end_lineno,
                "surface": "internal/unstable" if function_local or any(
                    part.startswith("_") for part in name.split(".")
                ) else "public",
            })
            for child in node.body:
                visit(child, name, "class", function_local)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            definitions.append({
                "name": redact(name),
                "kind": "method" if scope == "class" else "function",
                "signature": _definition_signature(node, name),
                "docstring": redact(ast.get_docstring(node) or ""),
                "line": node.lineno, "end_line": node.end_lineno,
                "surface": "internal/unstable" if function_local or any(
                    part.startswith("_") for part in name.split(".")
                ) else "public",
            })
            for child in node.body:
                visit(child, name, "function", True)
            return
        if isinstance(node, ast.Lambda):
            return
        for child in ast.iter_child_nodes(node):
            visit(child, prefix, scope, function_local)

    visit(tree)
    return sorted(definitions, key=lambda item: (item["line"], item["name"]))


def _python_metadata(source: str, relative: str) -> dict:
    """Read a narrow AST projection, excluding CLI help/defaults and source bodies."""
    tree = ast.parse(source, filename=relative)
    arguments = []
    imports = set()
    effects = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"add_argument", "add_parser"}:
                arguments.append({
                    "line": node.lineno,
                    "kind": node.func.attr,
                    "names": [redact(value.value) for value in node.args if isinstance(value, ast.Constant) and isinstance(value.value, str) and re.fullmatch(r"--?[A-Za-z][A-Za-z0-9_-]*|[A-Za-z][A-Za-z0-9_-]*", value.value)],
                })
            if name.rsplit(".", 1)[-1] in {
                "write", "write_text", "write_bytes", "mkdir", "replace", "unlink",
                "save", "load", "run", "Popen", "copy", "copy2", "rmtree",
            }:
                effects.append({"line": node.lineno, "operation": name})
    return {
        "docstring": redact(ast.get_docstring(tree) or ""),
        "definitions": _definitions(tree),
        "main_guard": _has_main_guard(tree),
        "arguments": sorted(arguments, key=lambda item: item["line"]),
        "imports": sorted(redact(name) for name in imports),
        "io_evidence": sorted(effects, key=lambda item: (item["line"], item["operation"])),
    }


def _description(metadata: dict, relative: str) -> str:
    if metadata.get("docstring"):
        return " ".join(metadata["docstring"].split("\n\n", 1)[0].split())
    if Path(relative).name in _BATCH_SUMMARIES:
        return _BATCH_SUMMARIES[Path(relative).name]
    if relative in _SPECIAL:
        return _SPECIAL[relative][3]
    return f"Static inventory entry for {relative}; no Python module docstring is present."


def _usage(relative: str, metadata: dict) -> dict:
    path = Path(relative)
    if relative in _SPECIAL:
        kind, command, prerequisites, effects = _SPECIAL[relative]
        return dict(kind=kind, command=command, command_kind="documented invocation", prerequisites=prerequisites, side_effects=effects, fixture=False)
    if relative in _EXAMPLES:
        return dict(
            kind="CPU fixture example", command=f"{_CPU_PREFIX} python {relative}",
            command_kind="runnable CPU example", fixture=True,
            prerequisites="Installed project dependencies (including CPU-capable PyTorch and NumPy), or PYTHONPATH=src with those dependencies installed. No datasets, checkpoints, basf2 or CUDA are needed.",
            side_effects="Runs tiny synthetic calculations in memory and prints a JSON summary. Legacy preprocessing commands are printed only; input-root paths in the output are not opened.",
        )
    if relative in _DRY_RUNS:
        return dict(
            kind="CPU fixture / training CLI", command=f"{_CPU_PREFIX} python {relative} {_DRY_RUNS[relative]}",
            command_kind="runnable CPU dry-run", fixture=True,
            prerequisites="Installed project dependencies. The shown command uses synthetic CPU fixtures and needs no data, checkpoints or basf2.",
            side_effects="The shown dry-run runs bounded in-memory inference/training and prints diagnostics. Real training modes, where supported, read datasets and write checkpoints/logs; use their guarded production workflow.",
        )
    if path.suffix == ".sbatch":
        return dict(kind="Slurm batch entry point", command=None, command_kind="operator workflow only", fixture=False,
                    prerequisites="An authorized contract, matching immutable source and data identities, the locked GPU environment and an allocated Slurm GPU. Use the paired tracked renderer.",
                    side_effects="Executes allocation-bound training or evaluation and writes telemetry, logs and receipts; training wrappers may use bounded requeue. Direct execution and --help are not documentation checks.")
    if path.suffix == ".sh":
        return dict(kind="shell helper / operator script", command=None, command_kind="source inspection required", fixture=False,
                    prerequisites="Bash and the commands/environment named in the source. This uncatalogued shell interface has no verified help or fixture mode.",
                    side_effects="Shell scripts may change files, execute commands or submit jobs. Read the source before invoking this entry.")
    if not metadata.get("main_guard") and not metadata.get("arguments"):
        return dict(kind="imported helper module", command=None, command_kind="not a standalone command", fixture=False,
                    prerequisites="Used by its importing Python modules; direct imports are listed below. Consult the source for the caller contract.",
                    side_effects="No standalone CLI is declared. Helper functions can read artifacts or mutate caller state when invoked; generation never imports them.")
    special_environment = path.name in _OPERATOR_SCRIPTS or "slurm" in path.parts
    kind = "operator CLI" if special_environment else "Python CLI"
    prerequisites = "Python plus the direct imports listed below; hypertagging imports require project dependencies. Supply the input files, configuration and output paths declared by the CLI."
    effects = "Behavior depends on the selected arguments. Static option names and operation locations below are review pointers; consult the local source for complete file/output and execution controls."
    if path.name.startswith("create_"):
        kind = "notebook/report generator"
        prerequisites = "Python and the direct imports below (usually the notebook extra / nbformat). Real-campaign generators additionally require their declared campaign inputs."
        effects = "Writes generated notebooks or reports; default destinations may be tracked files. Select a new output path when experimenting. Notebook generation does not itself execute notebook cells."
    elif path.name.startswith("render_") or relative == "scripts/condor/render_condor_job.py":
        kind = "contract/job renderer"
        effects = "Reads configuration/provenance and writes the requested contract or job files. Renderers do not submit jobs; some inspect scheduler state to validate contracts."
    elif path.name.startswith(("verify_", "validate_", "check_")):
        kind = "validator"
        effects = "Validates its specified inputs and reports failures. Options for outputs, receipts and shell output can write files; inspect source before selecting those options."
    elif path.name.startswith("finalize_"):
        kind = "artifact finalizer"
        effects = "Reads an existing run/campaign and writes final receipts, dataset metadata or checksums. Its inputs and outputs must match the owning workflow."
    if "slurm" in path.parts:
        prerequisites += " Execution modes can require a matching site GPU/Slurm environment, immutable contracts and operator authorization."
    if path.name == "preprocess_mdst.py":
        prerequisites += " Real preprocessing requires the documented basf2 release, raw mDST inputs and writable data-volume output."
        effects = "Real execution creates schema-v4 Parquet and sidecar output; --dry-run only describes the producer invocation. --overwrite is an explicit destructive option."
    if path.name == "export_full_decay_onnx.py":
        prerequisites += " Export and --dry-run inspection both require a trusted reconstruction checkpoint; export needs PyTorch and ONNX. basf2 is needed only to run the bundle in basf2."
        effects = "Writes manifest-bound ONNX graphs to a new/empty output directory; --dry-run inspects checkpoint compatibility. --force changes overwrite behavior."
    if path.name == "run_basf2_full_decay.py":
        prerequisites = "Compatible basf2/CVMFS release, ONNX Runtime, a manifest-bound exported bundle, input uDST and existing ParticleLists."
        effects = "Runs reconstruction inside basf2. No output file is written by default; --output-udst creates a new file and refuses an existing destination."
    if path.name == "publish_training_selection_repromotion_once.py":
        effects = "Consumes a specific operator authorization and publishes metadata. This command is not a read-only inspection tool."
    if path.name == "submit_phase3_batch_efficiency_calibration.py":
        effects = "Calls sbatch and writes submission artifacts after contract checks. Running the real invocation submits a GPU job."
    if path.name == "generate_audit_views.py":
        effects = "Regenerates active audit Markdown/YAML views by default. --check compares without writing."
    has_parser = bool(metadata.get("arguments"))
    reviewed_help = path.name in _HELP_REVIEWED
    command = f"python {relative} --help" if has_parser and reviewed_help and not special_environment else None
    return dict(kind=kind, command=command, command_kind="CLI help after installing prerequisites" if command else "operator workflow / source inspection", fixture=False, prerequisites=prerequisites, side_effects=effects)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8") if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def _redacted_projection(value):
    """Sanitize every published string, including curated prose and commands."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redacted_projection(item) for item in value]
    if isinstance(value, dict):
        return {key: _redacted_projection(item) for key, item in value.items()}
    return value


def generate_catalog(repo_root: Path, output_dir: Path) -> dict:
    """Generate portable RST pages and a JSON inventory for all repository scripts.

    ``output_dir`` belongs to the docs build. Only AST signatures/docstrings,
    line provenance, operation/option names, hashes and curated invocation
    guidance are published. Source bodies, comments, argument defaults and
    argparse help strings are excluded. No source is imported, no command is
    run and no modification time enters the output. Symlinks that would expose
    files outside the repository are rejected.
    """
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == repo_root or any(output_dir.is_relative_to(repo_root / name) for name in ("scripts", "examples")):
        raise ValueError("catalogue output must not overlap script/example source directories")
    entries = []
    paths = sorted(path for directory in ("examples", "scripts") for path in (repo_root / directory).rglob("*") if path.is_file() and path.suffix in _SUFFIXES and "__pycache__" not in path.parts)
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        if not path.resolve().is_relative_to(repo_root):
            raise ValueError("script source escapes the repository")
        if redact(relative) != relative or not re.fullmatch(r"(?:scripts|examples)/[A-Za-z0-9_./-]+", relative):
            raise ValueError("script source name is unsuitable for public provenance")
        data = path.read_bytes()
        source = data.decode("utf-8")
        metadata = _python_metadata(source, relative) if path.suffix == ".py" else {}
        entry = _redacted_projection({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "line": 1, "end_line": len(source.splitlines()), "summary": _description(metadata, relative), **metadata, **_usage(relative, metadata)})
        # Preserve the extension in the page name: x.py and x.sh are distinct.
        page = output_dir / "entries" / f"{relative}.rst"
        entry["docname"] = page.relative_to(output_dir).with_suffix("").as_posix()
        lines = [relative, "=" * len(relative), "", *_literal_block(entry["summary"]), f"**Kind:** {_escape(entry['kind'])}", "", f"Relative source provenance: ``{relative}``, lines {entry['line']}-{entry['end_line']}.", "", f"Source SHA-256: ``{entry['sha256']}``", "", "Invocation", "----------", "", entry["command_kind"] + ".", ""]
        if entry["command"]:
            lines.extend(_literal_block(entry["command"], "bash"))
        else:
            lines.extend(["Use the owning workflow or importing module described by the source. No generic runnable or help command is claimed for this entry.", ""])
        lines.extend(["Prerequisites", "-------------", "", entry["prerequisites"], "", "Side effects", "------------", "", entry["side_effects"], ""])
        if entry.get("docstring"):
            lines.extend(["Module docstring (static, redacted)", "-" * len("Module docstring (static, redacted)"), ""])
            lines.extend(_literal_block(entry["docstring"]))
        if entry.get("imports"):
            lines.extend(["Direct imports (static; includes conditional imports)", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""])
            lines.extend(_literal_block("\n".join(entry["imports"])))
        if entry.get("arguments"):
            lines.extend(["Static CLI option inventory", "---------------------------", "", "Option and subcommand names are parsed without execution. Defaults, help strings and declaration bodies are not published. Conditional options may be unavailable at runtime; consult the local source for the complete interface.", ""])
            for argument in entry["arguments"]:
                names = ", ".join(argument["names"]) or "dynamic names omitted"
                lines.extend([f"Source line {argument['line']} ({argument['kind']}):", ""])
                lines.extend(_literal_block(names))
        if entry.get("io_evidence"):
            lines.extend(["I/O call locations", "------------------", "", "Static call names are review pointers, not an exhaustive effect analysis; they include calls in inactive branches and helper functions.", ""])
            lines.extend(_literal_block("\n".join(f"{item['line']}: {item['operation']}" for item in entry["io_evidence"])))
        if entry.get("definitions"):
            lines.extend(["Static definitions", "------------------", "", "Every lexical function and class occurrence is extracted without importing the script. Dotted names preserve lexical ownership and distinguish class methods from functions nested inside callables. Underscore-prefixed and function-local definitions belong to the internal/unstable surface. Signatures omit defaults and string annotation values; docstrings are redacted literal text. Public names are not a promise of a stable script API.", ""])
            for definition in entry["definitions"]:
                lines.extend([f"**{_escape(definition['name'])}** ({definition['kind']}; {definition['surface']}), source lines {definition['line']}-{definition['end_line']}.", ""])
                lines.extend(_literal_block(definition["signature"]))
                if definition["docstring"]:
                    lines.extend(_literal_block(definition["docstring"]))
        _write(page, "\n".join(lines).rstrip() + "\n")
        entries.append(entry)
    counts = dict(sorted(Counter(entry["kind"] for entry in entries).items()))
    manifest = {"schema_version": 2, "publication": "redacted static projection; no source bodies", "entries": entries, "counts": counts, "total": len(entries), "definition_count": sum(len(entry.get("definitions", [])) for entry in entries)}
    lines = ["Examples and script catalogue", "=============================", "", "Every Python, Bash and Slurm script under ``examples/`` and ``scripts/`` is indexed from the current source tree, including imported helpers. Every lexical Python function and class occurrence is included with its dotted lexical name; function-local definitions are marked internal/unstable. Generation parses source text and never imports or executes the scripts. Relative line provenance and SHA-256 hashes bind redacted static projections to the build inputs. Source bodies, shell comments, string annotation values and argument help/defaults are not published.", "", ":download:`Machine-readable catalogue <manifest.json>`", "", f"The catalogue contains {manifest['total']} entries and {manifest['definition_count']} Python definitions. Manifest fields include relative path, line range, source hash, static definitions, option names, operation locations, category and curated invocation guidance.", "", "Run commands from the repository root after installing project dependencies (see the setup guide). The documentation environment alone intentionally does not include the training stack. The CPU commands below need no basf2, CUDA, data or checkpoints. Synthetic fixture success is software evidence only; it does not establish physics performance.", "", "Runnable CPU examples", "---------------------", ""]
    for entry in entries:
        if entry["path"] in _EXAMPLES:
            lines.extend([f":doc:`{entry['path']} <{entry['docname']}>`", ""])
            lines.extend(_literal_block(entry["command"], "bash"))
    lines.extend(["Bounded CPU dry-runs", "--------------------", ""])
    for entry in entries:
        if entry["path"] in _DRY_RUNS:
            lines.extend(_literal_block(entry["command"], "bash"))
    lines.extend(["Complete inventory", "------------------", "", "Job renderers, job submission, execution wrappers and imported helpers have different effects. In particular, a production launcher's dry-run can still create planning artifacts. Each entry identifies prerequisites and side effects; ``--help`` is only shown for declared Python CLI parsers outside allocation/operator entry points.", ""])
    for kind in counts:
        lines.extend([f"{kind} ({counts[kind]})", "~" * len(f"{kind} ({counts[kind]})"), "", ".. toctree::", "   :maxdepth: 1", ""])
        lines.extend(f"   {entry['docname']}" for entry in entries if entry["kind"] == kind)
        lines.append("")
    _write(output_dir / "index.rst", "\n".join(lines).rstrip() + "\n")
    _write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
