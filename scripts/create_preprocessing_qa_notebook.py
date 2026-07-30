#!/usr/bin/env python
"""Generate a compact schema-v1/v2 preprocessing QA report notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "preprocessing_qa_report.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Preprocessing QA report

                ## Goal

                Produce a compact, machine-readable QA summary for v1 or v2 without
                modifying the source parquet. Diagnostic MC values are never treated as
                reconstructed mother targets.
                """
            ),
            md("## Setup and data"),
            code(
                """
                from pathlib import Path
                import json, os, sys
                import matplotlib.pyplot as plt
                import numpy as np

                REPO_ROOT = Path.cwd()
                if not (REPO_ROOT / "src").exists(): REPO_ROOT = Path("..").resolve()
                sys.path.insert(0, str(REPO_ROOT / "src"))
                from hypertagging.data.notebook_fixtures import write_notebook_fixture
                from hypertagging.preprocessing.schema_v2 import load_payload_v2

                requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
                FIXTURE_MODE = not bool(requested)
                INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v2.parquet")
                if FIXTURE_MODE: write_notebook_fixture(INPUT_PATH)
                if not INPUT_PATH.exists(): raise FileNotFoundError(INPUT_PATH)
                FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/qa"))
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                QA_JSON = Path(os.environ.get("HYPERTAGGING_QA_JSON", str(FIGURE_DIR / "preprocessing_qa.json")))
                payload = load_payload_v2(INPUT_PATH)
                events = payload["events"]
                print("TINY FIXTURE QA — NOT REAL DATA" if FIXTURE_MODE else "REAL-DATA SAMPLE QA")
                """
            ),
            md("## Results"),
            code(
                """
                level_violations, closure_residuals, invalid_values = [], [], 0
                for event in events:
                    by_id = {int(node["node_id"]): node for node in event["nodes"]}
                    for mother in event["nodes"]:
                        common = [mother[name] for name in ("px", "py", "pz", "energy", "mass", "charge")]
                        invalid_values += int(not np.isfinite(common).all())
                        daughters = [by_id[int(child)] for child in mother["daughter_ids"]]
                        if not daughters: continue
                        expected_level = 1 + max(daughter["level"] for daughter in daughters)
                        if mother["level"] != expected_level:
                            level_violations.append([event["event_uid"], mother["node_id"], mother["level"], expected_level])
                        closure_residuals.append([
                            mother[name] - sum(daughter[name] for daughter in daughters)
                            for name in ("energy", "px", "py", "pz")
                        ])
                closure = np.asarray(closure_residuals, dtype=float)
                maximum_closure = float(np.abs(closure).max()) if closure.size else 0.0
                event_uids = [event["event_uid"] for event in events]
                report = {
                    "mode": "fixture" if FIXTURE_MODE else "real_sample",
                    "schema_version": payload["schema_version"],
                    "events": len(events),
                    "unique_event_uids": len(set(event_uids)),
                    "duplicate_event_uids": len(event_uids) - len(set(event_uids)),
                    "nodes": sum(len(event["nodes"]) for event in events),
                    "level_violations": level_violations,
                    "invalid_common_value_nodes": invalid_values,
                    "maximum_absolute_p4_closure_residual": maximum_closure,
                    "p4_closure_pass": maximum_closure < 1e-8,
                }
                QA_JSON.parent.mkdir(parents=True, exist_ok=True)
                QA_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                print(json.dumps(report, indent=2))
                if closure.size:
                    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                    for axis, values, label in zip(axes.flat, closure.T, ["delta E", "delta px", "delta py", "delta pz"]):
                        axis.hist(values, bins=25); axis.set_title(label)
                    fig.tight_layout(); fig.savefig(FIGURE_DIR / "qa_p4_closure.png"); plt.show()
                assert not level_violations
                assert invalid_values == 0
                assert maximum_closure < 1e-8
                print("Machine-readable summary:", QA_JSON)
                """
            ),
            md(
                """
                ## Takeaways

                A passing report verifies finite common values, exact retained-level
                recurrence, unique event IDs, and daughter-summed composite p4 closure.
                It does not validate full-training physics performance.
                """
            ),
        ]
    )
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
