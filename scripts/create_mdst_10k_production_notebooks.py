#!/usr/bin/env python
"""Generate real-only notebooks for the 10K middle-scale mDST campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "notebooks" / "inspect_mdst_10k_campaign.ipynb"
DEFAULT_OUTPUTS = ROOT / "notebooks" / "inspect_mdst_10k_outputs.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def _metadata(notebook):
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    return notebook


def build_status_notebook():
    return _metadata(
        nbf.v4.new_notebook(
            cells=[
                md(
                    """
                    # 10K middle-scale mDST campaign status

                    Real-data-only operator notebook for manifest, Condor-log,
                    provenance, and full-shard validation. It never substitutes a
                    fixture. Override `HYPERTAGGING_10K_CAMPAIGN_ROOT` to inspect a
                    relocated campaign.
                    """
                ),
                md("## tl;dr"),
                code(
                    """
                    from collections import Counter
                    import json, os, sys
                    from pathlib import Path
                    import matplotlib.pyplot as plt
                    import pandas as pd

                    CAMPAIGN_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_CAMPAIGN_ROOT",
                        "/data/dust/user/boyangyu/hypertagging/production_10k_middle_20260804",
                    ))
                    MANIFEST = Path(os.environ.get(
                        "HYPERTAGGING_10K_MANIFEST",
                        CAMPAIGN_ROOT / "manifests" / "mdst_10k_random_seed_20260804_v3.jsonl",
                    ))
                    SOURCE_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_SOURCE_ROOT",
                        CAMPAIGN_ROOT / "source" / "d02e823e0cbc",
                    ))
                    REPORT_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_REPORT_ROOT", CAMPAIGN_ROOT / "inspection"
                    ))
                    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
                    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]
                    from scripts import mdst_batch_production as production

                    if not MANIFEST.is_file():
                        raise FileNotFoundError(f"Real campaign manifest is required: {MANIFEST}")
                    records = production.read_manifest(MANIFEST)
                    status = production.production_status(MANIFEST)
                    require_complete = os.environ.get("HYPERTAGGING_REQUIRE_COMPLETE", "1") == "1"
                    tldr = {
                        "campaign_id": status["campaign_id"],
                        "planned_events": sum(int(row["planned_events"]) for row in records),
                        "tasks": len(records),
                        "complete_valid": status["complete"],
                        "missing_or_invalid": status["missing_or_invalid"],
                        "source_commit": records[0]["source_git_commit"],
                    }
                    print(json.dumps(tldr, indent=2))
                    if require_complete and status["missing_or_invalid"]:
                        raise RuntimeError(f"Campaign is not complete: {status['classifications']}")
                    """
                ),
                md(
                    """
                    ## Context & Methods

                    ### Key Assumptions

                    The JSONL manifest is authoritative. Validity means the
                    Parquet file, metadata sidecar, and completion marker all
                    satisfy the manifest-bound schema-v4 provenance contract.
                    Condor logs are diagnostic evidence, not a substitute for
                    shard validation.
                    """
                ),
                code(
                    """
                    summary_path = MANIFEST.with_suffix(".summary.json")
                    sample_path = MANIFEST.parent / "mdst_10k_sampled_inputs_v3.json"
                    summary = json.loads(summary_path.read_text())
                    sample = json.loads(sample_path.read_text())
                    context = {
                        "manifest": str(MANIFEST),
                        "input_root": summary["input_root"],
                        "output_root": summary["output_root"],
                        "selection_strategy": sample["selection_strategy"],
                        "random_seed": sample["random_seed"],
                        "selected_input_files": len(sample["selected_input_files_in_manifest_order"]),
                        "events_per_task": summary["events_per_task"],
                        "source_state": summary["source_state"],
                    }
                    display(pd.DataFrame(context.items(), columns=["parameter", "value"]))
                    """
                ),
                md("## Data"),
                code(
                    """
                    frame = pd.DataFrame(records)
                    configured_input_root = Path(summary["input_root"])
                    configured_output_root = Path(summary["output_root"])
                    frame["input_exists"] = frame.input_file.map(lambda value: Path(value).is_file())
                    frame["input_under_configured_root"] = frame.input_file.map(
                        lambda value: Path(value).is_relative_to(configured_input_root)
                    )
                    frame["output_under_configured_root"] = frame.output_file.map(
                        lambda value: Path(value).is_relative_to(configured_output_root)
                    )
                    assert frame.planned_events.sum() == 10_000
                    assert frame.task_id.is_unique
                    assert frame.input_exists.all()
                    assert frame.input_under_configured_root.all()
                    assert frame.output_under_configured_root.all()
                    display(frame[[
                        "task_id", "physics_category", "input_file", "entry_sequence",
                        "planned_events", "output_file",
                    ]])
                    """
                ),
                md("## Results"),
                code(
                    """
                    task_status = pd.DataFrame(status["task_status"])
                    display(task_status)
                    planned = frame.groupby("physics_category").planned_events.sum()
                    completed_ids = set(task_status.loc[
                        task_status.classification.eq("COMPLETE_VALID"), "task_id"
                    ])
                    completed = frame[frame.task_id.isin(completed_ids)].groupby(
                        "physics_category"
                    ).planned_events.sum().reindex(planned.index, fill_value=0)
                    pd.DataFrame({"planned": planned, "complete_valid": completed}).plot.bar(
                        title="Campaign events by physics category"
                    )
                    plt.ylabel("events")
                    plt.tight_layout()
                    plt.savefig(REPORT_ROOT / "campaign_category_completion.png", dpi=140)
                    plt.show()
                    """
                ),
                code(
                    """
                    result_rows = []
                    for row in records:
                        result_path = Path(row["output_file"]).with_suffix(".parquet.result.json")
                        if result_path.is_file():
                            result_rows.append(json.loads(result_path.read_text()))
                    results = pd.DataFrame(result_rows)
                    if not results.empty:
                        display(results[[
                            "task_id", "physics_category", "events", "elapsed_seconds",
                            "events_per_second", "peak_resident_memory_kib", "output_bytes",
                        ]].sort_values("task_id"))
                        results.events_per_second.plot.hist(bins=12, title="Worker throughput")
                        plt.xlabel("events / second")
                        plt.tight_layout()
                        plt.savefig(REPORT_ROOT / "worker_throughput.png", dpi=140)
                        plt.show()
                    """
                ),
                code(
                    """
                    log_root = CAMPAIGN_ROOT / "logs" / "condor"
                    err_files = sorted(log_root.glob("*.err"))
                    out_files = sorted(log_root.glob("*.out"))
                    nonempty_errors = [path for path in err_files if path.stat().st_size]
                    log_summary = {
                        "log_root": str(log_root),
                        "stdout_files": len(out_files),
                        "stderr_files": len(err_files),
                        "nonempty_stderr_files": len(nonempty_errors),
                        "event_log_files": len(list(log_root.glob("*.log"))),
                    }
                    print(json.dumps(log_summary, indent=2))
                    for path in nonempty_errors[:10]:
                        print(f"--- {path.name} (last 20 lines) ---")
                        print("\\n".join(path.read_text(errors="replace").splitlines()[-20:]))
                    """
                ),
                md("## Takeaways"),
                code(
                    """
                    global_validation = None
                    if status["missing_or_invalid"] == 0:
                        global_validation = production.validate_production_manifest(MANIFEST)
                    report = {
                        "artifact": "real_10k_campaign_status",
                        "fixture_or_real": "real",
                        "manifest": str(MANIFEST),
                        "summary": tldr,
                        "input_selection": sample,
                        "status": status,
                        "condor_logs": log_summary,
                        "worker_result_count": len(result_rows),
                        "global_validation": global_validation,
                        "pass_fail_status": (
                            "PASS" if global_validation and global_validation["validated_events"] == 10_000
                            else "INCOMPLETE"
                        ),
                    }
                    report_path = REPORT_ROOT / "campaign_status_report.json"
                    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
                    print(json.dumps({
                        "report": str(report_path),
                        "pass_fail_status": report["pass_fail_status"],
                        "validated_events": None if global_validation is None else global_validation["validated_events"],
                        "unique_event_uids": None if global_validation is None else global_validation["unique_event_uids"],
                    }, indent=2))
                    if require_complete:
                        assert report["pass_fail_status"] == "PASS"
                    """
                ),
            ]
        )
    )


def build_outputs_notebook():
    return _metadata(
        nbf.v4.new_notebook(
            cells=[
                md(
                    """
                    # Inspect produced 10K mDST Parquet shards

                    Bounded, real-data-only inspection of representative valid
                    shards from every physics category in the 10K campaign. The
                    campaign-status notebook owns exhaustive global validation;
                    this notebook owns readable event/node/content inspection.
                    """
                ),
                md("## tl;dr"),
                code(
                    """
                    from collections import Counter
                    from itertools import islice
                    import json, os, sys
                    from pathlib import Path
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import pandas as pd

                    CAMPAIGN_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_CAMPAIGN_ROOT",
                        "/data/dust/user/boyangyu/hypertagging/production_10k_middle_20260804",
                    ))
                    MANIFEST = Path(os.environ.get(
                        "HYPERTAGGING_10K_MANIFEST",
                        CAMPAIGN_ROOT / "manifests" / "mdst_10k_random_seed_20260804_v3.jsonl",
                    ))
                    SOURCE_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_SOURCE_ROOT",
                        CAMPAIGN_ROOT / "source" / "d02e823e0cbc",
                    ))
                    REPORT_ROOT = Path(os.environ.get(
                        "HYPERTAGGING_10K_REPORT_ROOT", CAMPAIGN_ROOT / "inspection"
                    ))
                    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
                    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]
                    from scripts import mdst_batch_production as production
                    from hypertagging.preprocessing.schema_v4 import iter_event_records_v4

                    records = production.read_manifest(MANIFEST)
                    status = production.production_status(MANIFEST)
                    if status["missing_or_invalid"]:
                        raise RuntimeError(f"Inspect only a complete valid campaign: {status['classifications']}")
                    rows_by_category = {}
                    for row in records:
                        rows_by_category.setdefault(row["physics_category"], row)
                    max_events_per_category = int(os.environ.get(
                        "HYPERTAGGING_INSPECTION_EVENTS_PER_CATEGORY", "50"
                    ))
                    print(json.dumps({
                        "campaign_id": status["campaign_id"],
                        "valid_shards": status["complete"],
                        "representative_categories": sorted(rows_by_category),
                        "bounded_events_per_category": max_events_per_category,
                    }, indent=2))
                    """
                ),
                md(
                    """
                    ## Context & Methods

                    ### Key Assumptions

                    One manifest-bound, `COMPLETE_VALID` shard is selected from
                    each category. Up to 50 events per selected shard are loaded;
                    therefore distribution plots are diagnostic samples, not
                    production-wide physics estimates.
                    """
                ),
                code(
                    """
                    selected_rows = [rows_by_category[key] for key in sorted(rows_by_category)]
                    validation_rows = []
                    for row in selected_rows:
                        result = production.classify_shard(
                            Path(row["output_file"]), **production._validation_kwargs(row)
                        )
                        validation_rows.append({
                            "task_id": row["task_id"],
                            "category": row["physics_category"],
                            "output_file": row["output_file"],
                            "classification": result["classification"],
                        })
                    validation_frame = pd.DataFrame(validation_rows)
                    assert validation_frame.classification.eq("COMPLETE_VALID").all()
                    display(validation_frame)
                    """
                ),
                md("## Data"),
                code(
                    """
                    sampled_events = []
                    for row in selected_rows:
                        path = Path(row["output_file"])
                        for event in islice(iter_event_records_v4(path), max_events_per_category):
                            event = dict(event)
                            event["_inspection_category"] = row["physics_category"]
                            event["_inspection_shard"] = str(path)
                            sampled_events.append(event)
                    if not sampled_events:
                        raise RuntimeError("No events were loaded from valid shards")
                    event_rows = []
                    node_rows = []
                    for event in sampled_events:
                        nodes = event["nodes"]
                        event_rows.append({
                            "event_uid": event["event_uid"],
                            "category": event["_inspection_category"],
                            "nodes": len(nodes),
                            "maximum_depth": max((int(node.get("level", 0)) for node in nodes), default=0),
                            "active_nodes": sum(bool(node.get("active", False)) for node in nodes),
                        })
                        for node in nodes:
                            node_rows.append({
                                "event_uid": event["event_uid"],
                                "category": event["_inspection_category"],
                                "node_kind": node.get("node_kind", "unknown"),
                                "level": int(node.get("level", -1)),
                                "pdg": int(node.get("pdg", 0)),
                                "pid_target_token": int(node.get("pid_target_token", node.get("token", -1))),
                                "is_leaf": not bool(node.get("daughter_ids")),
                                "active": bool(node.get("active", False)),
                            })
                    events_frame = pd.DataFrame(event_rows)
                    nodes_frame = pd.DataFrame(node_rows)
                    assert events_frame.event_uid.is_unique
                    display(events_frame.groupby("category").agg(
                        events=("event_uid", "size"),
                        mean_nodes=("nodes", "mean"),
                        max_nodes=("nodes", "max"),
                        max_depth=("maximum_depth", "max"),
                    ))
                    display(nodes_frame.head(20))
                    """
                ),
                md("## Results"),
                code(
                    """
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    for category, group in events_frame.groupby("category"):
                        axes[0].hist(group.nodes, bins=15, alpha=.45, label=category)
                    axes[0].set(title="Nodes per sampled event", xlabel="nodes", ylabel="events")
                    axes[0].legend(fontsize=7)
                    kind_counts = nodes_frame.node_kind.value_counts().sort_values()
                    kind_counts.plot.barh(ax=axes[1], title="Sampled node kinds")
                    axes[1].set_xlabel("nodes")
                    plt.tight_layout()
                    plt.savefig(REPORT_ROOT / "sampled_event_node_distributions.png", dpi=140)
                    plt.show()
                    """
                ),
                code(
                    """
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    nodes_frame.level.value_counts().sort_index().plot.bar(
                        ax=axes[0], title="Node levels"
                    )
                    leaf_pid = nodes_frame.loc[nodes_frame.is_leaf, "pid_target_token"].value_counts().head(20)
                    leaf_pid.plot.bar(ax=axes[1], title="Top leaf PID target tokens")
                    axes[0].set_xlabel("level")
                    axes[1].set_xlabel("token")
                    plt.tight_layout()
                    plt.savefig(REPORT_ROOT / "sampled_levels_and_pid.png", dpi=140)
                    plt.show()
                    """
                ),
                code(
                    """
                    example = sampled_events[0]
                    example_nodes = {int(node["node_id"]): node for node in example["nodes"]}
                    edge_rows = []
                    for parent_id, parent in example_nodes.items():
                        for child_id in parent.get("daughter_ids", []):
                            child = example_nodes.get(int(child_id), {})
                            edge_rows.append({
                                "parent_id": parent_id,
                                "parent_pdg": parent.get("pdg"),
                                "child_id": int(child_id),
                                "child_pdg": child.get("pdg"),
                                "child_kind": child.get("node_kind"),
                            })
                    print({
                        "example_event_uid": example["event_uid"],
                        "category": example["_inspection_category"],
                        "nodes": len(example_nodes),
                        "edges": len(edge_rows),
                    })
                    display(pd.DataFrame(edge_rows).head(40))
                    """
                ),
                md("## Takeaways"),
                code(
                    """
                    category_summary = events_frame.groupby("category").agg(
                        events=("event_uid", "size"),
                        mean_nodes=("nodes", "mean"),
                        maximum_nodes=("nodes", "max"),
                        maximum_depth=("maximum_depth", "max"),
                    ).reset_index().to_dict(orient="records")
                    report = {
                        "artifact": "real_10k_bounded_output_inspection",
                        "fixture_or_real": "real",
                        "manifest": str(MANIFEST),
                        "campaign_id": status["campaign_id"],
                        "validated_representative_shards": validation_rows,
                        "sampled_events": len(sampled_events),
                        "sampled_nodes": len(nodes_frame),
                        "unique_sampled_event_uids": int(events_frame.event_uid.nunique()),
                        "category_summary": category_summary,
                        "node_kind_distribution": {
                            str(key): int(value) for key, value in nodes_frame.node_kind.value_counts().items()
                        },
                        "pass_fail_status": "PASS",
                        "sampling_caveat": "bounded representative content inspection; see campaign_status_report.json for exhaustive validation",
                    }
                    report_path = REPORT_ROOT / "shard_inspection_report.json"
                    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
                    print(json.dumps({
                        "report": str(report_path),
                        "pass_fail_status": report["pass_fail_status"],
                        "sampled_events": report["sampled_events"],
                        "sampled_nodes": report["sampled_nodes"],
                    }, indent=2))
                    """
                ),
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--outputs-output", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()
    for path, notebook in (
        (args.status_output, build_status_notebook()),
        (args.outputs_output, build_outputs_notebook()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
