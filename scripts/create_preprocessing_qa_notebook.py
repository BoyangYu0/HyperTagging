#!/usr/bin/env python
"""Generate a compact schema-v4 preprocessing QA report notebook."""

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
                import hashlib, json, os, sys
                from collections import Counter
                import matplotlib.pyplot as plt
                import numpy as np

                REPO_ROOT = Path.cwd()
                if not (REPO_ROOT / "src").exists(): REPO_ROOT = Path("..").resolve()
                sys.path.insert(0, str(REPO_ROOT / "src"))
                from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
                from hypertagging.preprocessing.schema_v4 import load_payload_v4

                requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
                FIXTURE_MODE = not bool(requested)
                INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v4.parquet")
                if FIXTURE_MODE: write_notebook_fixture_v4(INPUT_PATH)
                if not INPUT_PATH.exists(): raise FileNotFoundError(INPUT_PATH)
                FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/qa"))
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                QA_JSON = Path(os.environ.get("HYPERTAGGING_QA_JSON", str(FIGURE_DIR / "preprocessing_qa.json")))
                payload = load_payload_v4(INPUT_PATH)
                events = payload["events"]
                sidecar_path=INPUT_PATH.with_suffix(INPUT_PATH.suffix+".metadata.json")
                marker_path=INPUT_PATH.with_suffix(INPUT_PATH.suffix+".complete")
                sidecar=json.loads(sidecar_path.read_text()); marker=json.loads(marker_path.read_text())
                sha256=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
                marker_hashes_pass=marker.get("parquet_sha256")==sha256(INPUT_PATH) and marker.get("sidecar_sha256")==sha256(sidecar_path)
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
                all_nodes = [node for event in events for node in event["nodes"]]
                token_range_pass = all(
                    0 <= int(node["input_pid_token"]) < 41
                    and 0 <= int(node["pid_target_token"]) < 41
                    for node in all_nodes
                )
                no_truth_leakage_contract_pass = all(
                    node["leaf_kinematics_mode"] != "raw_track_predicted_pid"
                    or (
                        int(node["input_pid_token"]) == 0
                        and node["energy_source"].startswith("canonical_")
                    )
                    for node in all_nodes
                )
                node_kind_pass = all(
                    node["node_kind"] in {"unknown","track","ecl_cluster","klm_cluster","composite","other"}
                    for node in all_nodes
                )
                partial_rate = float(np.mean([node["partial_missing_daughters"] for node in all_nodes]))
                query_counts = [
                    sum(node["level"] == level for node in event["nodes"])
                    for event in events
                    for level in set(node["level"] for node in event["nodes"] if node["level"] > 0)
                ]
                cardinalities = [len(node["daughter_ids"]) for node in all_nodes if node["daughter_ids"]]
                pid_vocabulary=sorted(set(int(node["input_pid_token"]) for node in all_nodes)|set(int(node["pid_target_token"]) for node in all_nodes))
                node_kind_distribution=Counter(str(node["node_kind"]) for node in all_nodes)
                availability_distribution=Counter(f"{block}:{name}:{bool(value)}" for node in all_nodes for block,key in (("track","track_availability"),("ecl","cluster_availability"),("klm","klm_availability")) for name,value in node.get(key,{}).items())
                klm_nodes=[node for node in all_nodes if node.get("node_kind")=="klm_cluster"]
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
                    "no_truth_leakage_pass": no_truth_leakage_contract_pass,
                    "token_range_pass": token_range_pass,
                    "node_kind_consistency_pass": node_kind_pass,
                    "pid_likelihood_availability_explicit": all(
                        "pid_likelihood_availability" in node for node in all_nodes
                    ),
                    "partial_decay_rate": partial_rate,
                    "query_overflow_rate_at_capacity_8": float(np.mean([value > 8 for value in query_counts])) if query_counts else 0.0,
                    "cardinality_overflow_rate_at_capacity_6": float(np.mean([value > 6 for value in cardinalities])) if cardinalities else 0.0,
                    "schema_config_consistency_pass": payload["schema_version"] == "direct-mdst-tree-v4",
                    "completion_marker_hashes_pass":marker_hashes_pass,
                    "campaign_provenance_status":"PASS" if sidecar.get("campaign_id") and all(sidecar.get(key)==marker.get(key) for key in ("campaign_id","source_git_commit","task_record_hash")) else ("NOT_APPLICABLE_FIXTURE" if FIXTURE_MODE else "FAIL"),
                    "campaign_id":sidecar.get("campaign_id"),"source_git_commit":sidecar.get("source_git_commit"),"task_record_hash":sidecar.get("task_record_hash"),
                    "complete_available_pid_vocabulary":pid_vocabulary,
                    "node_kind_distribution":dict(node_kind_distribution),
                    "availability_distribution":dict(availability_distribution),
                    "level_distribution":dict(Counter(int(node["level"]) for node in all_nodes)),
                    "node_multiplicity_by_event":[len(event["nodes"]) for event in events],
                    "klm_diagnostics":{"retained_nodes":len(klm_nodes),"associated_with_ecl":sum(bool(node.get("associated_reco_id")) for node in klm_nodes),"feature_availability":dict(Counter(f"{name}:{bool(value)}" for node in klm_nodes for name,value in node.get("klm_availability",{}).items()))},
                    "query_cardinality_capacity":{"maximum_queries":max(query_counts,default=0),"maximum_daughter_cardinality":max(cardinalities,default=0)},
                    "composite_pid_input_truth_separated_pass": all(
                        "daughter_input_pid_histogram" in node
                        and "daughter_truth_pid_histogram" in node
                        and "daughter_pid_histogram" not in node
                        for node in all_nodes
                    ),
                    "recursive_completeness_available": all(
                        "recursive_reconstructable_complete" in node
                        for node in all_nodes
                    ),
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
                assert token_range_pass and no_truth_leakage_contract_pass and node_kind_pass
                assert marker_hashes_pass
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
