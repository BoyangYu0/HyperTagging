#!/usr/bin/env python
"""Generate and execute revised notebooks on deterministic CPU fixtures."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_INDEX_PATH = REPO_ROOT / "notebooks" / "index.yaml"


def load_notebook_index() -> list[dict[str, object]]:
    payload = yaml.safe_load(NOTEBOOK_INDEX_PATH.read_text(encoding="utf-8"))
    entries = payload.get("notebooks", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{NOTEBOOK_INDEX_PATH} has no notebook entries")
    identifiers = [str(entry["id"]) for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("notebook index contains duplicate ids")
    return entries


NOTEBOOK_INDEX = load_notebook_index()
NOTEBOOK_INDEX_BY_ID = {str(entry["id"]): entry for entry in NOTEBOOK_INDEX}
NOTEBOOKS = {
    str(entry["id"]): entry for entry in NOTEBOOK_INDEX if entry.get("default_smoke") is True
}
DIAGNOSTICS = {
    str(entry["id"]): entry for entry in NOTEBOOK_INDEX if entry.get("group") == "DIAGNOSTIC"
}


def _stabilize_notebook_cell_ids(notebook: nbformat.NotebookNode, name: str) -> None:
    """Assign content-derived cell IDs so regeneration is byte-stable.

    ``nbformat.v4.new_*_cell`` creates random IDs.  Without this normalization,
    a smoke run dirties every generated source notebook even when no cell
    content changed, which makes audit diffs and review evidence unreliable.
    The cell position is included so repeated boilerplate cells remain unique.
    """

    for index, cell in enumerate(notebook.cells):
        identity = "\0".join(
            (name, str(index), str(cell.cell_type), str(cell.get("source", "")))
        )
        cell["id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _normalize_generated_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    _stabilize_notebook_cell_ids(notebook, path.name)
    nbformat.write(notebook, path)


def _generate_notebook(entry: dict[str, object], destination: Path) -> Path:
    generator = REPO_ROOT / str(entry["generator"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(generator), "--output", str(destination)],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    _normalize_generated_notebook(destination)
    return destination


def check_generated_notebooks(work_root: Path) -> list[str]:
    """Return tracked notebook paths whose normalized generated source is stale."""

    stale: list[str] = []
    generated_root = work_root / "generated"
    def compare(entry: dict[str, object]) -> str | None:
        tracked = REPO_ROOT / str(entry["path"])
        generated = _generate_notebook(entry, generated_root / tracked.name)
        tracked_notebook = nbformat.read(tracked, as_version=4)
        _stabilize_notebook_cell_ids(tracked_notebook, tracked.name)
        generated_notebook = nbformat.read(generated, as_version=4)
        _stabilize_notebook_cell_ids(generated_notebook, generated.name)
        if nbformat.writes(tracked_notebook) != nbformat.writes(generated_notebook):
            return str(entry["path"])
        return None

    with ThreadPoolExecutor(max_workers=min(4, len(NOTEBOOK_INDEX))) as executor:
        stale.extend(path for path in executor.map(compare, NOTEBOOK_INDEX) if path)
    return stale


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stamp_json_report(path: Path, provenance: dict[str, object]) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError(f"machine-readable report must be a JSON object: {path}")
    else:
        payload = {}
    payload.update(provenance)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_one(
    name: str,
    entry: dict[str, object],
    *,
    work_root: Path,
    timeout: int,
) -> Path:
    tracked_source = REPO_ROOT / str(entry["path"])
    source = _generate_notebook(entry, work_root / "generated" / tracked_source.name)
    expected_sections = tuple(str(section) for section in entry.get("expected_sections", []))
    notebook_text = source.read_text(encoding="utf-8")
    for section in expected_sections:
        if section.lower() not in notebook_text.lower():
            raise AssertionError(f"{source.name} misses expected section text: {section}")
    executed_path = work_root / tracked_source.name
    shutil.copy2(source, executed_path)
    notebook = nbformat.read(executed_path, as_version=4)
    figure_dir = work_root / f"{name}_figures"
    old_environment = os.environ.copy()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HYPERTAGGING_FIGURE_DIR": str(figure_dir),
            "HYPERTAGGING_QA_JSON": str(work_root / "preprocessing_qa.json"),
            "HYPERTAGGING_NOTEBOOK_SEED": "20260730",
        }
    )
    os.environ.pop("HYPERTAGGING_PARQUET", None)
    os.environ.pop("HYPERTAGGING_PREPROCESS_OUTPUT", None)
    os.environ.pop("HYPERTAGGING_CHECKPOINT", None)
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    nbformat.write(notebook, executed_path)
    error_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if error_outputs:
        raise AssertionError(f"{source.name} produced notebook errors: {error_outputs}")
    if not list(figure_dir.glob("*.png")):
        raise AssertionError(f"{source.name} did not produce expected figures")
    required_artifacts = {
        "leaf_composite": ("leaf_composite_contract.json",),
        "streaming": (
            "streaming_report.json",
            "dataset_index.json",
            "storage_benchmark/storage_benchmark.json",
        ),
        "leaf_pid": ("leaf_pid_token_range.csv", "leaf_input_leakage_check.json"),
        "capacity": ("capacity_report.json", "sparse_loss_table.csv"),
        "training": ("training_pipeline_pass.json", "reconstruction/checkpoint.pt"),
        "hyperbolic": ("curriculum_runtime_report.json",),
        "exact_geometry": ("exact_geometry_scale_summary.json",),
        "rollout_search": ("rollout_search_calibration_summary.json",),
        "runtime_scaling": ("runtime_scaling_summary.json",),
        "reconstruction": ("scheduled_context_report.json",),
        "qa": (),
        "manifest": ("production_manifest_report.json",),
    }.get(name, ())
    for relative in required_artifacts:
        if not (figure_dir / relative).exists():
            raise AssertionError(f"{source.name} missed required artifact {relative}")
    if name == "leaf_composite":
        report = json.loads(
            (figure_dir / "leaf_composite_contract.json").read_text(encoding="utf-8")
        )
        if not report.get("truth_clean_composite_input") or not report.get(
            "raw_track_unknown_input_pass"
        ) or not report.get("runtime_dynamic_normalization_pass") or not report.get(
            "teacher_composite_type_source_pass"
        ) or not report.get("target_metadata_invariance"):
            raise AssertionError(f"leaf/composite contract failed: {report}")
        required_pid_diagnostics = {
            "soft_hard_energy_difference",
            "soft_hard_mother_mass_difference",
            "pid_entropy",
            "soft_hard_relation_bias_change",
            "soft_hard_pointer_logit_change",
        }
        if not required_pid_diagnostics <= report.keys():
            raise AssertionError(f"leaf/composite PID diagnostics missing: {report}")
    if name == "streaming":
        report = json.loads(
            (figure_dir / "streaming_report.json").read_text(encoding="utf-8")
        )
        if not report.get("manifest_output_file_resolved") or not report.get(
            "source_leakage_pass"
        ) or not report.get("disjoint_worker_units_pass") or not report.get(
            "cursor_resume_pass"
        ) or not report.get("dataset_index_pass"):
            raise AssertionError(f"streaming report failed: {report}")
    if name == "hyperbolic":
        report = json.loads(
            (figure_dir / "curriculum_runtime_report.json").read_text(encoding="utf-8")
        )
        if not report.get("level_causal_pass") or not report.get(
            "actual_corruption_labels_pass"
        ) or not report.get("hard_negatives_are_explicit_tree_relations") or not report.get(
            "runtime_two_pass_pid_semantics"
        ) or not report.get("invalid_corruptions_excluded_from_positive_structure"):
            raise AssertionError(f"curriculum runtime report failed: {report}")
    if name == "reconstruction":
        report = json.loads(
            (figure_dir / "scheduled_context_report.json").read_text(encoding="utf-8")
        )
        if report.get("teacher_event_double_counted") or report.get(
            "sampled_context"
        ) != "predicted" or not report.get(
            "fallback_teacher_on_unrepresentable"
        ) or not report.get("schedule_not_at_endpoint") or not report.get(
            "constraint_policy_round_trip"
        ):
            raise AssertionError(f"scheduled context report failed: {report}")
        if not report.get("duplicate_metrics_by_level") or not report.get(
            "bounded_set_packing_rollout"
        ) or not report.get("evaluation_slices"):
            raise AssertionError(f"rollout ambiguity diagnostics missing: {report}")
    if name == "exact_geometry":
        report = json.loads((figure_dir / "exact_geometry_scale_summary.json").read_text())
        if report.get("direct_leaf_root_edges") != 1 or not report.get(
            "eligible_different_b_positions"
        ) or not all(row.get("finite_z") and row.get("finite_gradients") for row in report.get("presets", [])):
            raise AssertionError(f"exact geometry/scale contract failed: {report}")
    if name == "rollout_search":
        report = json.loads((figure_dir / "rollout_search_calibration_summary.json").read_text())
        if not report.get("pid_modes") or not report.get("resolver", {}).get("beam") or not report.get("reliability"):
            raise AssertionError(f"rollout search/calibration contract failed: {report}")
    if name == "runtime_scaling":
        report = json.loads((figure_dir / "runtime_scaling_summary.json").read_text())
        if report.get("throughput_claim") or len(report.get("measurements", [])) != 4:
            raise AssertionError(f"runtime scaling contract failed: {report}")
    if name == "capacity":
        report = json.loads(
            (figure_dir / "capacity_report.json").read_text(encoding="utf-8")
        )
        rows = report.get("production_baseline_by_level", [])
        if report.get("schema_default") != "direct-mdst-tree-v4" or not rows:
            raise AssertionError(f"capacity report misses v4/every-level contract: {report}")
        if any(row.get("overflow_count") for row in rows):
            raise AssertionError(f"fixture capacity overflow: {report}")
    if name == "qa":
        qa_path = work_root / "preprocessing_qa.json"
        if not qa_path.exists():
            raise AssertionError("QA notebook did not write its JSON summary")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if not qa.get("p4_closure_pass"):
            raise AssertionError(f"QA JSON did not pass p4 closure: {qa}")

    provenance = {
        "git_sha": _git_sha(),
        "schema_version": "direct-mdst-tree-v4",
        "fixture_or_real": str(entry["fixture_or_real"]),
        "data_path_or_fixture_name": ",".join(str(value) for value in entry["required_inputs"]),
        "checkpoint_path_or_none": "none",
        "seed": int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730")),
        "pass_fail_status": "PASS",
    }
    for report_name in entry.get("machine_readable_outputs", []):
        report_path = figure_dir / str(report_name)
        if not report_path.exists() and (work_root / str(report_name)).exists():
            report_path = work_root / str(report_name)
        _stamp_json_report(report_path, provenance)
    return executed_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-output", type=Path, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the runner-derived group count and names without executing them.",
    )
    parser.add_argument(
        "--check-generated",
        action="store_true",
        help="Regenerate in a temporary directory and compare normalized source bytes.",
    )
    parser.add_argument(
        "--ci-frequency",
        choices=tuple(sorted({str(entry["CI_frequency"]) for entry in NOTEBOOK_INDEX})),
        help="Select default notebooks from the authoritative index by CI frequency.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(NOTEBOOKS),
        help="Execute only the named notebook group; repeat to select several.",
    )
    parser.add_argument(
        "--diagnostic-first-level-ambiguity",
        action="store_true",
        help="Also execute the separate, non-CI first-level ambiguity diagnostic.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Skip the stable suite; requires a diagnostic flag.",
    )
    args = parser.parse_args(argv)
    if args.diagnostic_only and not args.diagnostic_first_level_ambiguity:
        parser.error("--diagnostic-only requires a diagnostic notebook flag")
    if args.list:
        print(f"{len(NOTEBOOKS)} deterministic CPU notebook groups")
        for name in NOTEBOOKS:
            print(name)
        return 0
    if args.check_generated:
        with tempfile.TemporaryDirectory(prefix="hypertagging-generated-check-") as directory:
            stale = check_generated_notebooks(Path(directory))
        if stale:
            for path in stale:
                print(f"STALE: {path}", file=sys.stderr)
            return 1
        print(f"generated notebook consistency PASS: {len(NOTEBOOK_INDEX)} notebooks")
        return 0
    if args.keep_output:
        work_root = args.keep_output.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="hypertagging-notebooks-")
        work_root = Path(temporary.name)
    executed = []
    selected = set() if args.diagnostic_only else set(args.only or NOTEBOOKS)
    if args.ci_frequency:
        selected = {
            name for name, entry in NOTEBOOKS.items()
            if entry.get("CI_frequency") == args.ci_frequency
        }
    for name, entry in NOTEBOOKS.items():
        if name not in selected:
            continue
        executed.append(
            execute_one(
                name,
                entry,
                work_root=work_root,
                timeout=args.timeout,
            )
        )
    if args.diagnostic_first_level_ambiguity:
        entry = DIAGNOSTICS["first_level_ambiguity"]
        executed.append(
            execute_one(
                "first_level_ambiguity",
                entry,
                work_root=work_root,
                timeout=args.timeout,
            )
        )
    print(f"Executed {len(executed)} notebooks on CPU fixtures")
    for path in executed:
        print(path)
    figure_paths = sorted(work_root.glob("*_figures/*.png"))
    visual_rows = [
        {
            "notebook": figure.parent.name.removesuffix("_figures"),
            "figure_path": str(figure.relative_to(work_root)),
            "section": figure.stem.replace("_", " "),
            "caption": figure.stem.replace("_", " "),
            "review_status": "NOT_REVIEWED",
            "reviewer": "",
            "comments": "",
        }
        for figure in figure_paths
    ]
    html = [
        "<html><body><h1>Notebook figures</h1>",
        "<p>Visual review: NOT_REVIEWED</p>",
    ]
    for row in visual_rows:
        html.append(
            f'<h2>{row["notebook"]}: {row["caption"]}</h2>'
            f'<p>Section: {row["section"]}; status: {row["review_status"]}; '
            f'reviewer: {row["reviewer"] or "(unassigned)"}; '
            f'comments: {row["comments"] or "(none)"}</p>'
            f'<img src="{row["figure_path"]}" style="max-width:900px">'
        )
    html.append("</body></html>")
    (work_root / "visual_review_index.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    (work_root / "visual_review_index.json").write_text(
        json.dumps(
            {
                "visual_review_status": "NOT_REVIEWED",
                "figures": visual_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    executed_ids = [name for name in NOTEBOOKS if name in selected]
    if args.diagnostic_first_level_ambiguity:
        executed_ids.append("first_level_ambiguity")
    _write_validation_overview(work_root, executed_ids, visual_rows)
    (work_root / "notebook_execution_summary.json").write_text(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "schema_version": "direct-mdst-tree-v4",
                "fixture_or_real": "fixture",
                "data_path_or_fixture_name": "notebooks/index.yaml deterministic fixtures",
                "checkpoint_path_or_none": "none",
                "seed": 20260730,
                "pass_fail_status": "PASS",
                "configured_group_count": len(NOTEBOOKS),
                "executed_group_count": len(executed),
                "executed_groups": [name for name in NOTEBOOKS if name in selected]
                + (
                    ["first_level_ambiguity"]
                    if args.diagnostic_first_level_ambiguity
                    else []
                ),
                "fixture_only": True,
                "visual_review_status": "NOT_REVIEWED",
                "figure_count": len(figure_paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if temporary is not None:
        temporary.cleanup()
    return 0


AUDIT_LEDGER = REPO_ROOT / "docs" / "audits" / "issue_ledger.yaml"


def _covered_ledger_items(notebook_path: str) -> list[str]:
    """Return ledger IDs that explicitly cite this notebook as evidence."""

    if not AUDIT_LEDGER.exists():
        return []
    ledger = yaml.safe_load(AUDIT_LEDGER.read_text(encoding="utf-8"))
    return [
        str(item["id"])
        for item in ledger.get("items", [])
        if notebook_path in item.get("notebook_evidence", [])
    ]


def _write_validation_overview(
    work_root: Path,
    executed_ids: list[str],
    visual_rows: list[dict[str, str]],
) -> None:
    """Aggregate registry contracts and run artifacts without promoting NOT RUN."""

    rows: list[dict[str, object]] = []
    executed = set(executed_ids)
    for notebook_id, entry in NOTEBOOK_INDEX_BY_ID.items():
        ran = notebook_id in executed
        figure_dir = work_root / f"{notebook_id}_figures"
        artifacts = [
            str((figure_dir / str(name)).relative_to(work_root))
            for name in entry.get("machine_readable_outputs", [])
            if (figure_dir / str(name)).exists()
        ]
        rows.append(
            {
                "notebook_id": notebook_id,
                "group": entry["group"],
                "source_sha": _git_sha() if ran else str(entry.get("last_verified_sha", "NOT_RUN")),
                "fixture_or_real": entry["fixture_or_real"],
                "dataset": ", ".join(str(value) for value in entry.get("required_inputs", [])),
                "checkpoint": "required" if "trained_checkpoint" in entry.get("required_inputs", []) else "none",
                "schema_and_feature_contract": "direct-mdst-tree-v4 / registry-declared inputs",
                "status": "PASS" if ran else "NOT_RUN",
                "scientific_claims_allowed": entry.get("scientific_claims_allowed", []),
                "required_human_action": "human figure review" if ran else "supply required real inputs and execute",
                "ledger_items_covered": _covered_ledger_items(str(entry["path"])),
                "figures": [row["figure_path"] for row in visual_rows if row["notebook"] == notebook_id],
                "machine_readable_artifacts": artifacts,
            }
        )
    payload = {
        "git_sha": _git_sha(),
        "visual_review_status": "NOT_REVIEWED",
        "rows": rows,
    }
    (work_root / "validation_overview.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Validation overview",
        "",
        "Machine execution and human visual review are separate. Visual review: `NOT_REVIEWED`.",
        "",
        "| Notebook | Group | Source | Input | Status | Allowed claim | Ledger items | Artifacts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        links = "<br>".join(f"[{Path(path).name}]({path})" for path in row["machine_readable_artifacts"]) or "none"
        claims = ", ".join(row["scientific_claims_allowed"])
        issues = ", ".join(row["ledger_items_covered"]) or "none"
        md.append(
            f"| {row['notebook_id']} | {row['group']} | `{row['source_sha']}` | "
            f"{row['fixture_or_real']} | `{row['status']}` | {claims} | {issues} | {links} |"
        )
    (work_root / "validation_overview.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html_rows = "".join(
        "<tr>" + "".join(
            f"<td>{row[key]}</td>" for key in
            ("notebook_id", "group", "source_sha", "fixture_or_real", "status", "required_human_action")
        ) + "</tr>"
        for row in rows
    )
    (work_root / "validation_overview.html").write_text(
        "<html><body><h1>Validation overview</h1><p>Human visual review: NOT_REVIEWED</p>"
        "<table><tr><th>Notebook</th><th>Group</th><th>Source</th><th>Input</th>"
        "<th>Status</th><th>Required human action</th></tr>" + html_rows + "</table></body></html>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
