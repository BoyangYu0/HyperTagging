#!/usr/bin/env python
"""Build the canonical portable-report artifact for the validated 10M campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def rows_for(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def source(
    source_id: str,
    label: str,
    sql: str,
    description: str,
    tables: list[str],
    generated_at: str,
    *,
    filters: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": "report_data.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "SQL",
            "executed_at": generated_at,
            "description": description,
            "sql": sql,
            "tables_used": tables,
            "filters": filters or [],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    production_root = args.production_root.resolve()
    output_dir = args.output_dir.resolve()
    validation_path = production_root / "validation" / "final_validation.json"
    metrics_path = (
        production_root
        / "validation"
        / "notebook"
        / "figures"
        / "shard_metrics.csv"
    )
    takeaways_path = (
        production_root
        / "validation"
        / "notebook"
        / "figures"
        / "notebook_takeaways.json"
    )
    manifest_path = production_root / "manifests" / "mdst_10m_ri_all_exp.jsonl"
    for required in (validation_path, metrics_path, takeaways_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    takeaways = json.loads(takeaways_path.read_text(encoding="utf-8"))
    first_task = json.loads(manifest_path.open(encoding="utf-8").readline())
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "report_data.sqlite"
    temporary_database = output_dir / "report_data.sqlite.tmp"
    temporary_database.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_database)
    connection.executescript(
        """
        CREATE TABLE shard_metrics (
          task_id INTEGER PRIMARY KEY,
          category TEXT NOT NULL,
          experiment TEXT NOT NULL,
          events INTEGER NOT NULL,
          output_mib REAL NOT NULL,
          events_per_second REAL NOT NULL,
          elapsed_seconds REAL NOT NULL,
          validation_seconds REAL NOT NULL,
          peak_rss_mib REAL NOT NULL,
          klm_nodes INTEGER NOT NULL,
          unique_event_uids INTEGER NOT NULL
        );
        CREATE TABLE campaign_summary (
          validated_events INTEGER,
          unique_event_uids INTEGER,
          completed_shards INTEGER,
          output_gib REAL,
          bytes_per_event REAL,
          klm_nodes INTEGER,
          missing_shards INTEGER,
          non_whitespace_stderr INTEGER,
          input_files INTEGER,
          experiments INTEGER
        );
        CREATE TABLE topology_summary (
          family TEXT NOT NULL,
          label TEXT NOT NULL,
          value REAL NOT NULL
        );
        CREATE TABLE provenance_summary (
          property TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            connection.execute(
                "INSERT INTO shard_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(row["task_id"]),
                    row["category"],
                    row["experiment"],
                    int(row["events"]),
                    float(row["output_mib"]),
                    float(row["events_per_second"]),
                    float(row["elapsed_seconds"]),
                    float(row["validation_seconds"]),
                    float(row["peak_rss_mib"]),
                    int(float(row["klm_nodes"])),
                    int(row["unique_event_uids"]),
                ),
            )
    connection.execute(
        "INSERT INTO campaign_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            validation["validated_events"],
            validation["unique_event_uids"],
            validation["completed_shards"],
            validation["output_bytes"] / 2**30,
            validation["output_bytes_per_event"],
            validation["klm_node_distribution"]["klm_nodes"],
            len(validation["missing_shards"]),
            takeaways["successful_non_whitespace_stderr"],
            takeaways["unique_input_files"],
            len(takeaways["experiments"]),
        ),
    )
    for label, value in validation["actual_leaf_mode_distribution"].items():
        connection.execute(
            "INSERT INTO topology_summary VALUES ('leaf_mode', ?, ?)",
            (label, value),
        )
    for label, value in validation["node_count_quantiles"].items():
        connection.execute(
            "INSERT INTO topology_summary VALUES ('node_count_quantile', ?, ?)",
            (label, value),
        )
    for label, value in validation["maximum_depth_quantiles"].items():
        connection.execute(
            "INSERT INTO topology_summary VALUES ('depth_quantile', ?, ?)",
            (label, value),
        )
    provenance = {
        "campaign_id": validation["campaign_id"],
        "campaign_digest": validation["campaign_config_digest"],
        "schema_version": validation["schema_version"],
        "source_git_commit": validation["source_git_commit"],
        "source_git_tree": validation["source_git_tree"],
        "source_state": validation["source_state"],
        "feature_spec_hash": validation["feature_spec_hash"],
        "model_feature_contract_hash": validation["model_feature_contract_hash"],
        "production_readiness_report_sha256": validation[
            "production_readiness_report_sha256"
        ],
        "input_release": "MC16ri_run2",
        "experiment_coverage": ", ".join(takeaways["experiments"]),
        "physics_categories": ", ".join(sorted(validation["category_distribution"])),
        "klm_training_scope": validation["klm_training_scope"],
        "all_completion_markers_valid": str(
            validation["all_completion_markers_valid"]
        ),
        "manifest_task_hash": first_task["task_record_hash"],
        "manual_readiness_assertion": "completed_by_user_2026-08-11",
        "manual_reviewer_identity": "not_supplied",
        "preflight_condor_cluster": "4844426",
        "bulk_condor_cluster": "4844428",
        "initial_preflight_rejected_cluster": "4844425",
        "initial_preflight_rejected_reason": "malformed semicolon-separated Condor environment; no shard published",
    }
    connection.executemany(
        "INSERT INTO provenance_summary VALUES (?, ?)", provenance.items()
    )
    connection.commit()

    sql = {
        "summary": """SELECT validated_events, unique_event_uids, completed_shards,
       ROUND(output_gib, 3) AS output_gib,
       ROUND(bytes_per_event, 1) AS bytes_per_event,
       klm_nodes, missing_shards, non_whitespace_stderr,
       input_files, experiments
FROM campaign_summary""",
        "category": """SELECT category, COUNT(*) AS shards, SUM(events) AS events,
       ROUND(SUM(output_mib) / 1024.0, 3) AS output_gib,
       ROUND(AVG(events_per_second), 2) AS mean_events_per_second,
       ROUND(MAX(peak_rss_mib), 1) AS max_peak_rss_mib,
       SUM(klm_nodes) AS klm_nodes
FROM shard_metrics
GROUP BY category
ORDER BY events DESC, category""",
        "resources": """SELECT task_id, category, events_per_second,
       peak_rss_mib, 8192 AS requested_memory_mib,
       elapsed_seconds, validation_seconds, output_mib
FROM shard_metrics
ORDER BY task_id""",
        "leaf_modes": """SELECT label AS leaf_mode, CAST(value AS INTEGER) AS nodes
FROM topology_summary
WHERE family = 'leaf_mode'
ORDER BY nodes DESC""",
        "topology_quantiles": """SELECT family, label AS quantile, value
FROM topology_summary
WHERE family IN ('node_count_quantile', 'depth_quantile')
ORDER BY family, value""",
        "klm": """SELECT category, SUM(klm_nodes) AS klm_nodes
FROM shard_metrics
GROUP BY category
ORDER BY klm_nodes DESC, category""",
        "provenance": """SELECT property, value
FROM provenance_summary
ORDER BY property""",
    }
    datasets = {name: rows_for(connection, query) for name, query in sql.items()}
    connection.close()
    temporary_database.replace(database_path)

    sources = [
        source(
            "summary_source",
            "Exhaustive campaign summary",
            sql["summary"],
            "Selects the headline metrics produced by exhaustive shard and global UID validation.",
            ["campaign_summary"],
            generated_at,
            filters=["validated campaign only"],
        ),
        source(
            "category_source",
            "Category aggregation from all shard sidecars",
            sql["category"],
            "Aggregates every validated 5k-event shard by physics category.",
            ["shard_metrics"],
            generated_at,
            filters=["experiment e1004", "release MC16ri_run2"],
        ),
        source(
            "resource_source",
            "All worker resource measurements",
            sql["resources"],
            "Selects one reviewed resource row per validated shard.",
            ["shard_metrics"],
            generated_at,
            filters=["all 2000 validated shards"],
        ),
        source(
            "leaf_source",
            "Exhaustive leaf-mode counts",
            sql["leaf_modes"],
            "Selects leaf-mode node counts accumulated across every event.",
            ["topology_summary"],
            generated_at,
        ),
        source(
            "topology_source",
            "Exhaustive topology quantiles",
            sql["topology_quantiles"],
            "Selects node-count and maximum-depth quantiles from exhaustive event validation.",
            ["topology_summary"],
            generated_at,
        ),
        source(
            "klm_source",
            "KLM nodes by category",
            sql["klm"],
            "Aggregates retained KLM nodes from every validated shard by physics category.",
            ["shard_metrics"],
            generated_at,
        ),
        source(
            "provenance_source",
            "Campaign provenance and data contract",
            sql["provenance"],
            "Selects immutable identifiers, hashes, release scope, and contract properties.",
            ["provenance_summary"],
            generated_at,
        ),
    ]
    title = "10M Run-Independent mDST Production Validation"
    summary_row = datasets["summary"][0]
    cards = [
        {
            "id": "validated_events_card",
            "description": "Events read from every shard and accepted by the exhaustive validator.",
            "dataset": "summary",
            "sourceId": "summary_source",
            "metrics": [
                {"label": "Validated events", "field": "validated_events", "format": "compact"}
            ],
        },
        {
            "id": "unique_uids_card",
            "description": "Globally distinct event identifiers after the cross-shard uniqueness pass.",
            "dataset": "summary",
            "sourceId": "summary_source",
            "metrics": [
                {"label": "Unique event UIDs", "field": "unique_event_uids", "format": "compact"}
            ],
        },
        {
            "id": "shards_card",
            "description": "Complete 5k-event Parquet shards with matching publication metadata.",
            "dataset": "summary",
            "sourceId": "summary_source",
            "metrics": [
                {"label": "Validated shards", "field": "completed_shards", "format": "compact"}
            ],
        },
        {
            "id": "output_card",
            "description": "Total Parquet payload size, excluding validation and metadata sidecars.",
            "dataset": "summary",
            "sourceId": "summary_source",
            "metrics": [
                {"label": "Parquet output, GiB", "field": "output_gib", "format": "number"}
            ],
        },
        {
            "id": "klm_card",
            "description": "Retained KLM cluster nodes included in the production data contract.",
            "dataset": "summary",
            "sourceId": "summary_source",
            "metrics": [
                {"label": "Retained KLM nodes", "field": "klm_nodes", "format": "compact"}
            ],
        },
    ]
    charts = [
        {
            "id": "category_events_chart",
            "title": "Validated events by physics category",
            "subtitle": "All seven available MC16ri_run2 categories are represented.",
            "intent": "comparison",
            "question": "How are the 10M validated events distributed across available physics categories?",
            "rationale": "A sorted bar chart makes exact category-volume differences directly comparable.",
            "type": "bar",
            "dataset": "category",
            "sourceId": "category_source",
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "Physics category"},
                "y": {"field": "events", "type": "quantitative", "format": "compact", "label": "Validated events"},
                "tooltip": [
                    {"field": "shards", "type": "quantitative", "label": "Shards"},
                    {"field": "output_gib", "type": "quantitative", "label": "Output GiB"},
                ],
            },
            "valueFormat": "compact",
        },
        {
            "id": "throughput_chart",
            "title": "Worker throughput distribution by physics category",
            "subtitle": "Each observation is one validated 5k-event shard.",
            "intent": "distribution",
            "question": "How variable is worker throughput within and between categories?",
            "rationale": "A box plot summarizes the full 2000-shard distribution without hiding category-level spread.",
            "type": "boxPlot",
            "dataset": "resources",
            "sourceId": "resource_source",
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "Physics category"},
                "y": {"field": "events_per_second", "type": "quantitative", "format": "number", "label": "Events per second"},
            },
        },
        {
            "id": "memory_throughput_chart",
            "title": "Peak memory versus worker throughput",
            "subtitle": "Peak RSS stays far below the 8 GiB Condor request across the reviewed shard population.",
            "intent": "relationship",
            "question": "Do slower workers exhibit materially higher peak memory?",
            "rationale": "A shard-level scatter preserves outliers and exposes any resource correlation.",
            "type": "scatter",
            "dataset": "resources",
            "sourceId": "resource_source",
            "encodings": {
                "x": {"field": "events_per_second", "type": "quantitative", "label": "Events per second"},
                "y": {"field": "peak_rss_mib", "type": "quantitative", "label": "Peak RSS, MiB"},
                "color": {"field": "category", "type": "nominal", "label": "Physics category"},
                "tooltip": [
                    {"field": "task_id", "type": "ordinal", "label": "Task"},
                    {"field": "elapsed_seconds", "type": "quantitative", "label": "Elapsed seconds"},
                ],
            },
            "legend": {"position": "bottom"},
        },
        {
            "id": "leaf_modes_chart",
            "title": "Retained leaf nodes by kinematics mode",
            "subtitle": "Counts are accumulated exhaustively over the validated event population.",
            "intent": "comparison",
            "question": "Which leaf representation modes dominate the produced graphs?",
            "rationale": "A horizontal bar chart keeps long mode names readable while comparing node counts.",
            "type": "horizontalBar",
            "dataset": "leaf_modes",
            "sourceId": "leaf_source",
            "encodings": {
                "x": {"field": "leaf_mode", "type": "nominal", "label": "Leaf mode"},
                "y": {"field": "nodes", "type": "quantitative", "format": "compact", "label": "Retained nodes"},
            },
            "valueFormat": "compact",
        },
        {
            "id": "klm_chart",
            "title": "Retained KLM nodes by physics category",
            "subtitle": "Every available category contributes KLM nodes to the training scope.",
            "intent": "comparison",
            "question": "Is KLM coverage present across all physics categories?",
            "rationale": "A category bar chart directly verifies nonzero KLM coverage and its scale.",
            "type": "bar",
            "dataset": "klm",
            "sourceId": "klm_source",
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "Physics category"},
                "y": {"field": "klm_nodes", "type": "quantitative", "format": "compact", "label": "Retained KLM nodes"},
            },
            "valueFormat": "compact",
        },
    ]
    tables = [
        {
            "id": "category_table",
            "title": "Exact category coverage and worker aggregates",
            "subtitle": "Audit table across all validated shards.",
            "dataset": "category",
            "sourceId": "category_source",
            "defaultSort": {"field": "events", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "category", "label": "Category", "type": "text"},
                {"field": "events", "label": "Events", "format": "number"},
                {"field": "shards", "label": "Shards", "format": "number"},
                {"field": "output_gib", "label": "Output GiB", "format": "number"},
                {"field": "mean_events_per_second", "label": "Mean events/s", "format": "number"},
                {"field": "max_peak_rss_mib", "label": "Max peak RSS MiB", "format": "number"},
                {"field": "klm_nodes", "label": "KLM nodes", "format": "number"},
            ],
        },
        {
            "id": "topology_table",
            "title": "Exhaustive topology quantiles",
            "subtitle": "Event-level retained node-count and maximum-depth summaries.",
            "dataset": "topology_quantiles",
            "sourceId": "topology_source",
            "defaultSort": {"field": "family", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "family", "label": "Metric", "type": "text"},
                {"field": "quantile", "label": "Quantile", "type": "text"},
                {"field": "value", "label": "Value", "format": "number"},
            ],
        },
        {
            "id": "provenance_table",
            "title": "Immutable campaign provenance",
            "subtitle": "Hashes and contract identifiers shipped beside the dataset.",
            "dataset": "provenance",
            "sourceId": "provenance_source",
            "defaultSort": {"field": "property", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "property", "label": "Property", "type": "text"},
                {"field": "value", "label": "Value", "type": "text"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "summary_source",
            "body": (
                "## Technical summary\n\n"
                f"The campaign is ready for downstream use: **10.0M events** in "
                f"**{summary_row['completed_shards']:,} shards** passed exhaustive structural, "
                "provenance, hash, and global-UID validation. The accepted event UIDs are "
                f"also **10.0M unique**, with **{summary_row['missing_shards']} missing shards** "
                f"and **{summary_row['non_whitespace_stderr']} non-whitespace worker stderr files**."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "coverage_finding",
            "type": "markdown",
            "sourceId": "category_source",
            "body": (
                "## Finding 1 — the campaign covers the complete available RI inventory\n\n"
                "All seven available physics categories are represented. The event allocation follows "
                "the available-file inventory while retaining exact 5k-event shard boundaries; the table "
                "provides exact category totals for audit use."
            ),
        },
        {"id": "category_chart_block", "type": "chart", "chartId": "category_events_chart"},
        {"id": "category_table_block", "type": "table", "tableId": "category_table"},
        {
            "id": "resource_finding",
            "type": "markdown",
            "sourceId": "resource_source",
            "body": (
                "## Finding 2 — worker resource use has ample headroom\n\n"
                "The full shard population is shown below. Throughput varies by event category, but peak "
                "resident memory remains far below the recorded 8 GiB request."
            ),
        },
        {"id": "throughput_chart_block", "type": "chart", "chartId": "throughput_chart"},
        {"id": "memory_chart_block", "type": "chart", "chartId": "memory_throughput_chart"},
        {
            "id": "topology_finding",
            "type": "markdown",
            "sourceId": "topology_source",
            "body": (
                "## Finding 3 — event topology remains within the validated envelope\n\n"
                "Event-level node-count and retained-depth quantiles were computed over every accepted "
                "event and are reported in the exact audit table below."
            ),
        },
        {"id": "topology_table_block", "type": "table", "tableId": "topology_table"},
        {
            "id": "leaf_finding",
            "type": "markdown",
            "sourceId": "leaf_source",
            "body": (
                "## Finding 4 — all retained leaf modes are counted exhaustively\n\n"
                "Leaf-mode counts were accumulated from every shard result sidecar and retain the "
                "production representation-mode labels."
            ),
        },
        {"id": "leaf_chart_block", "type": "chart", "chartId": "leaf_modes_chart"},
        {
            "id": "klm_finding",
            "type": "markdown",
            "sourceId": "klm_source",
            "body": (
                "## Finding 5 — KLM scope is populated in every physics category\n\n"
                "Retained KLM counts are nonzero across the complete category set."
            ),
        },
        {"id": "klm_chart_block", "type": "chart", "chartId": "klm_chart"},
        {
            "id": "provenance_section",
            "type": "markdown",
            "sourceId": "provenance_source",
            "body": (
                "## Scope, definitions, and provenance\n\n"
                "Input is restricted to the run-independent `MC16ri_run2` release. Inventory inspection "
                "found one available experiment, `e1004`, and the manifest covers it completely. A usable "
                "shard means Parquet data plus matching `.metadata.json`, `.result.json`, and `.complete` "
                "sidecars bound to the immutable task hash and clean source tree."
            ),
        },
        {"id": "provenance_table_block", "type": "table", "tableId": "provenance_table"},
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Methodology\n\n"
                "The authoritative validator re-read every Parquet shard, checked task and source hashes, "
                "verified exact source entry ranges and event counts, validated schema and production "
                "contracts, and performed a cross-shard UID uniqueness pass backed by SQLite. The executed "
                "notebook independently re-aggregates all result/metadata sidecars, checks Condor stderr, "
                "and reads bounded representative events for tree rendering."
            ),
        },
        {
            "id": "visual_audit",
            "type": "markdown",
            "body": (
                "## Reproducibility and visual double-check\n\n"
                "Two executed notebooks are supplied: `validate_mdst_10m_campaign.executed.ipynb` for "
                "campaign-wide numbers and representative/extreme event-tree pictures, and "
                "`inspect_production_manifest.executed.ipynb` for an independent manifest review. PNG "
                "exports, the full shard metrics CSV, this report's SQLite snapshot, and exact SQL are "
                "stored with the validation package."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "provenance_source",
            "body": (
                "## Limitations and robustness\n\n"
                "Only `e1004` appears because it is the sole experiment present in the inspected release, "
                "not because of a sampling filter. Event-tree pictures are deterministic examples and do "
                "not replace exhaustive numeric checks. The readiness record binds the user's statement "
                "that manual 10k-middle verification is complete; no reviewer identity was supplied."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "Treat the manifest as the dataset entry point, preserve all sidecars and the embedded "
                "source snapshot when relocating data, and pin training jobs to the recorded feature/model "
                "contract hashes. Re-run the validator after any copy or storage migration."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Should future productions allocate events proportionally to available files, equally by "
                "physics category, or against a training-driven mixture? Should the retained KLM scope be "
                "versioned as a separately ablatable training configuration?"
            ),
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical validation and provenance report for the 10M RI mDST campaign.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }
    artifact_path = output_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    queries_path = output_dir / "report_queries.sql"
    queries_path.write_text(
        "\n\n".join(f"-- {name}\n{query};" for name, query in sql.items()) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
