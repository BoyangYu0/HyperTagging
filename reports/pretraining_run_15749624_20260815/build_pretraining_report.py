#!/usr/bin/env python3
"""Build the canonical portable report artifact from verified run evidence."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_REL = "reports/pretraining_run_15749624_20260815/evidence.json"
ARTIFACT_REL = "reports/pretraining_run_15749624_20260815/artifact.json"


def dataset_sql(dataset: str) -> str:
    return (
        "SELECT value AS row_json\n"
        f"FROM json_each(:artifact_json, '$.snapshot.datasets.{dataset}')\n"
        "ORDER BY CAST(key AS INTEGER)"
    )


def encoding(
    field: str,
    kind: str,
    label: str,
    *,
    unit: str | None = None,
    fmt: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "field": field,
        "type": kind,
        "aggregate": "none",
        "label": label,
    }
    if unit is not None:
        value["unit"] = unit
    if fmt is not None:
        value["format"] = fmt
    return value


def multi(fields: list[str], label: str, *, unit: str, fmt: str) -> dict[str, Any]:
    return {
        "fields": fields,
        "type": "quantitative",
        "aggregate": "none",
        "label": label,
        "unit": unit,
        "format": fmt,
    }


def dataset_source(dataset: str, description: str) -> dict[str, Any]:
    return {
        "id": f"dataset-{dataset}",
        "label": f"Verified evidence dataset: {dataset}",
        "path": ARTIFACT_REL,
        "query": {
            "engine": "sqlite3",
            "language": "SQL",
            "dialect": "SQLite JSON1",
            "description": description + " The packaged rows are SQL-verified against the artifact snapshot.",
            "filters": [
                "Slurm job 15749624 only",
                "fixed validation-role UID cohort for sample inference",
                "sealed-test payload excluded",
            ],
            "metric_definitions": [
                "optimizer step = one successfully logged training update",
                "boundary fraction = active embeddings beyond the configured near-boundary threshold",
                "PCA = one joint two-component projection of 32D origin-tangent embeddings",
            ],
            "parameters": {
                "artifact_json": "UTF-8 JSON text for the complete canonical artifact.json document."
            },
            "sql": dataset_sql(dataset),
            "tables_used": [ARTIFACT_REL, REPORT_REL],
        },
    }


def common_chart(
    *,
    chart_id: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    question: str,
    rationale: str,
    intent: str,
    context: dict[str, str],
    encodings: dict[str, Any],
    x_title: str,
    y_title: str,
    unit: str,
    value_format: str,
    max_rows: int,
    settings: dict[str, Any],
    palette: str,
    reference_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": f"dataset-{dataset}",
        "intent": intent,
        "question": question,
        "rationale": rationale,
        "comparisonContext": context,
        "encodings": encodings,
        "xAxisTitle": x_title,
        "yAxisTitle": y_title,
        "palette": {"kind": "categorical", "name": palette},
        "valueFormat": value_format,
        "unit": unit,
        "layout": "full",
        "maxRows": max_rows,
        "settings": settings,
        "surface": {
            "surface": "explorer",
            "compact": False,
            "showControls": False,
            "viewMode": "both",
        },
    }
    if reference_lines:
        value["referenceLines"] = reference_lines
    return value


def build_datasets(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    training = evidence["training"]
    embeddings = evidence["realValidationEmbeddings"]
    architecture = evidence["architecture"]
    config = evidence["trainingConfig"]

    checkpoints = []
    for row in evidence["checkpoints"]:
        checkpoints.append(
            {
                "file": row["file"],
                "step": row["step"],
                "sizeMiB": row["bytes"] / 1024**2,
                "sha256Prefix": row["sha256"][:16],
                "modelElements": row["modelInventory"]["elements"],
                "modelNonfinite": row["modelInventory"]["nonfinite"],
                "optimizerNonfinite": row["optimizerInventory"]["nonfinite"],
                "ampScale": row["ampScale"],
                "validationEvents": row["validationEventCount"],
            }
        )

    validation_table = []
    for row in training["validationCurve"]:
        validation_table.append(
            {
                "step": row["step"],
                "principalLoss": row["principalLoss"],
                "fspLoss": row["fspLoss"],
                "relationAccuracy": row["relationAccuracy"],
                "parentRankingAccuracy": row["parentRankingAccuracy"],
                "channelRetrievalAccuracy": row["channelRetrievalAccuracy"],
                "effectiveRank": row["effectiveRank"],
                "boundaryFraction": row["boundaryFraction"],
            }
        )

    phase_steps = config["curriculum_phase_steps"]
    completed = training["lastStep"]
    executed_phase1 = min(completed, phase_steps[0])
    curriculum_progress = [
        {
            "run": "job 15749624",
            "executedPhase1": executed_phase1,
            "unexecutedPhase1": phase_steps[0] - executed_phase1,
            "unexecutedPhase2": phase_steps[1],
            "unexecutedPhase3": phase_steps[2],
            "unexecutedPhase4": phase_steps[3],
        }
    ]

    embedding_compare = []
    embedding_best_level = []
    first_event_short = embeddings["representativeEvents"][0]["eventShort"]
    representative_embedding = []
    node_count = int(embeddings["sampleNodeCount"])
    visual_limit_per_checkpoint = min(900, node_count)
    if visual_limit_per_checkpoint == 1:
        selected_visual_positions = {0}
    else:
        selected_visual_positions = {
            round(index * (node_count - 1) / (visual_limit_per_checkpoint - 1))
            for index in range(visual_limit_per_checkpoint)
        }
    checkpoint_positions: dict[str, int] = {}
    for point in embeddings["points"]:
        checkpoint_position = checkpoint_positions.get(point["checkpoint"], 0)
        checkpoint_positions[point["checkpoint"]] = checkpoint_position + 1
        selected_for_visual = checkpoint_position in selected_visual_positions
        base = {
            "pc1": point["pc1"],
            "pc2": point["pc2"],
            "eventShort": point["eventShort"],
            "nodeKind": point["nodeKind"],
            "level": point["level"],
            "truthPdg": point["truthPdg"],
            "hyperbolicRadius": point["radius"],
        }
        if selected_for_visual and point["checkpoint"] == "step3000":
            embedding_compare.append(base)
        if point["checkpoint"] == "best_step500":
            if selected_for_visual:
                embedding_best_level.append(base)
            if point["eventShort"] == first_event_short:
                representative_embedding.append(
                    {
                        **base,
                        "nodeLabel": (
                            f"{point['truthPdg']} · L{point['level']}"
                            if int(point["level"]) > 0 else ""
                        ),
                    }
                )

    radius_pivot: dict[int, dict[str, Any]] = {}
    for row in embeddings["radiusByLevel"]:
        target = radius_pivot.setdefault(row["level"], {"level": row["level"]})
        prefix = "bestStep500" if row["checkpoint"] == "best_step500" else "step3000"
        target[f"{prefix}Median"] = row["radiusMedian"]
        target[f"{prefix}P10"] = row["radiusP10"]
        target[f"{prefix}P90"] = row["radiusP90"]
        target[f"{prefix}Nodes"] = row["nodes"]

    job_rows = [
        {"field": "Slurm outcome", "value": evidence["job"]["state"], "evidenceStatus": "observed"},
        {"field": "Exit code", "value": evidence["job"]["exitCode"], "evidenceStatus": "observed"},
        {"field": "Node / GRES", "value": f"{evidence['job']['node']} / {evidence['job']['gres']}", "evidenceStatus": "observed"},
        {"field": "Elapsed", "value": evidence["job"]["elapsed"], "evidenceStatus": "observed"},
        {"field": "Logged optimizer steps", "value": training["lastStep"], "evidenceStatus": "observed"},
        {"field": "Planned optimizer steps", "value": config["max_steps"], "evidenceStatus": "configured"},
        {"field": "Completion", "value": f"{100*training['lastStep']/config['max_steps']:.1f}%", "evidenceStatus": "derived"},
        {"field": "Failure", "value": evidence["job"]["failureClass"], "evidenceStatus": "observed"},
        {"field": "Restarts", "value": evidence["job"]["restartCount"], "evidenceStatus": "observed"},
        {"field": "Sealed test accessed", "value": False, "evidenceStatus": "observed boundary"},
    ]

    model_rows = [
        {"field": "preset", "value": architecture["preset"]},
        {"field": "serialized model-state elements", "value": checkpoints[0]["modelElements"]},
        {"field": "d_model", "value": architecture["d_model"]},
        {"field": "hyperbolic dimensions", "value": architecture["hyper_dim"]},
        {"field": "attention heads", "value": architecture["n_heads"]},
        {"field": "context layers", "value": architecture["n_context_layers"]},
        {"field": "feed-forward width", "value": architecture["ffn_dim"]},
        {"field": "dropout", "value": architecture["dropout"]},
        {"field": "curvature", "value": architecture["curvature"]},
        {"field": "hyperbolic level encoding", "value": architecture["hyperbolic_level_encoding"]},
        {"field": "tangent scale", "value": architecture["tangent_scale_mode"]},
    ]

    input_output = [
        {"stage": "Input publication", "shapeOrType": "schema-v4 decay events", "meaning": "Tracks, ECL/KLM clusters, composite candidates, four-momenta, charge, availability masks and reduced PID tokens."},
        {"stage": "Batch contract", "shapeOrType": "B × N × feature blocks", "meaning": "Variable node sets are padded; node masks and relation masks retain the valid event topology."},
        {"stage": "Typed adapters", "shapeOrType": f"B × N × {architecture['d_model']}", "meaning": "Separate track, cluster, KLM and composite frontends enter one shared latent space."},
        {"stage": "Context encoder", "shapeOrType": f"{architecture['n_context_layers']} layers, {architecture['n_heads']} heads", "meaning": "Permutation-aware self-attention consumes physical relation biases and explicit visibility masks."},
        {"stage": "Euclidean outputs", "shapeOrType": f"B × N × {architecture['d_model']}", "meaning": "Tree, reconstruction and channel projections feed relation, PID and contrastive objectives."},
        {"stage": "Hyperbolic output", "shapeOrType": f"B × N × {architecture['hyper_dim']}", "meaning": "Tangent projection followed by exp-map into the curvature-1 Poincaré ball."},
        {"stage": "Pretraining heads", "shapeOrType": "relations + leaf PID + corruption + correctness", "meaning": "Supervision covers topology, parent/distance/radius geometry, channel structure and anti-collapse."},
        {"stage": "Downstream use", "shapeOrType": "encoder initialization", "meaning": "A reconstruction decoder would consume transferred encoder weights; this failed run is not promoted for final reconstruction."},
    ]

    feature_rows = []
    for block, names in evidence["featureBlocks"].items():
        feature_rows.append(
            {
                "block": block,
                "count": len(names),
                "examples": ", ".join(str(value) for value in names[:8]) + (" …" if len(names) > 8 else ""),
            }
        )

    return {
        "job_contract": job_rows,
        "model_contract": model_rows,
        "input_output_contract": input_output,
        "feature_blocks": feature_rows,
        "curriculum_progress": curriculum_progress,
        "training_curve": training["curve25StepBins"],
        "validation_curve": training["validationCurve"],
        "validation_table": validation_table,
        "boundary_crossings": training["boundaryCrossings"],
        "checkpoint_inventory": checkpoints,
        "embedding_compare": embedding_compare,
        "embedding_best_level": embedding_best_level,
        "representative_embedding": representative_embedding,
        "embedding_summary": embeddings["embeddingSummary"],
        "radius_by_level": list(radius_pivot.values()),
        "representative_events": embeddings["representativeEvents"],
        "telemetry_curve": evidence["telemetry"]["curve30SampleBins"],
    }


def table(table_id: str, title: str, subtitle: str, dataset: str, columns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": f"dataset-{dataset}",
        "layout": "full",
        "density": "spacious",
        "columns": columns,
    }


def build_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    data = build_datasets(evidence)
    training = evidence["training"]
    config = evidence["trainingConfig"]
    emb = evidence["realValidationEmbeddings"]
    best_summary = next(row for row in emb["embeddingSummary"] if row["checkpoint"] == "best_step500")
    late_summary = next(row for row in emb["embeddingSummary"] if row["checkpoint"] == "step3000")
    val500 = training["validationCurve"][0]
    val3000 = training["validationCurve"][-1]
    telemetry = evidence["telemetry"]["summary"]
    checkpoint_count = len(evidence["checkpoints"])
    finite_checkpoints = sum(
        row["modelInventory"]["nonfinite"] == 0 and row["optimizerInventory"]["nonfinite"] == 0
        for row in evidence["checkpoints"]
    )
    levels = sorted({int(point["level"]) for point in emb["points"]})

    zero_y = [{"axis": "y", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "solid"}]
    charts = [
        common_chart(
            chart_id="curriculum-progress", title="The run stopped inside curriculum phase 1",
            subtitle=f"Observed {training['lastStep']:,} of {config['max_steps']:,} planned optimizer steps",
            chart_type="stackedBar", dataset="curriculum_progress", intent="composition",
            question="How much of the registered four-phase schedule actually ran?",
            rationale="A single stacked bar separates executed work from each unexecuted phase without treating the failed run as complete.",
            context={"baseline":"17,500-step registered plan","denominator":"planned optimizer steps","grain":"curriculum phase and execution state","normalization":"none","semanticFamily":"run completion","unit":"optimizer steps"},
            encodings={"x":encoding("run","nominal","Run"),"y":multi(["executedPhase1","unexecutedPhase1","unexecutedPhase2","unexecutedPhase3","unexecutedPhase4"],"Optimizer steps",unit="optimizer steps",fmt="number")},
            x_title="Run", y_title="Optimizer steps", unit="optimizer steps", value_format="number",
            max_rows=1, settings={"groupMode":"stacked","sort":"none","showValues":True}, palette="blue-neutral-purple-orange-pink-curriculum",
        ),
        common_chart(
            chart_id="training-loss", title="Logged training loss declined but remained noisy",
            subtitle="25-step means; phase-1 FSP topology/anti-collapse batches only",
            chart_type="line", dataset="training_curve", intent="trend",
            question="How did the phase-1 optimization objective evolve before failure?",
            rationale="Binned means retain the full 3,342-step trajectory while reducing batch-level noise; validation is shown separately.",
            context={"baseline":"step 1","denominator":"25 logged optimizer steps per bin except the final bin","grain":"optimizer-step bin","normalization":"arithmetic mean","semanticFamily":"training objective","unit":"loss"},
            encodings={"x":encoding("step","quantitative","Optimizer step",unit="steps",fmt="number"),"y":encoding("lossMean","quantitative","Mean training loss",unit="loss",fmt="number"),"tooltip":[encoding("lossMin","quantitative","Bin minimum",unit="loss",fmt="number"),encoding("lossMax","quantitative","Bin maximum",unit="loss",fmt="number")]},
            x_title="Optimizer step", y_title="Mean phase-1 loss", unit="loss", value_format="number",
            max_rows=len(data["training_curve"]), settings={"sort":"none","showPoints":"hover","showValues":False}, palette="blue-observed", reference_lines=zero_y,
        ),
        common_chart(
            chart_id="boundary-saturation", title="Poincaré boundary saturation emerged early and became dominant",
            subtitle="25-step mean boundary fraction across logged training batches",
            chart_type="line", dataset="training_curve", intent="trend",
            question="When did embeddings approach the numerical boundary of the Poincaré ball?",
            rationale="Boundary fraction is the most direct logged precursor of the terminal non-finite gradient norm.",
            context={"baseline":"zero boundary occupancy","denominator":"active embeddings in each training batch","grain":"optimizer-step bin","normalization":"mean fraction","semanticFamily":"geometry stability","unit":"share"},
            encodings={"x":encoding("step","quantitative","Optimizer step",unit="steps",fmt="number"),"y":encoding("boundaryFractionMean","quantitative","Boundary fraction",unit="share",fmt="percent"),"tooltip":[encoding("effectiveRankMean","quantitative","Effective rank",unit="dimensions",fmt="number"),encoding("gradientProjectionHyperMean","quantitative","Hyper-projection gradient",unit="norm",fmt="number")]},
            x_title="Optimizer step", y_title="Mean near-boundary share", unit="share", value_format="percent",
            max_rows=len(data["training_curve"]), settings={"sort":"none","showPoints":"hover","showValues":False}, palette="orange-warning", reference_lines=zero_y,
        ),
        common_chart(
            chart_id="validation-view-loss", title="FSP validation improved while broader views worsened",
            subtitle="Same fixed 2,000-event validation cohort at six checkpoints",
            chart_type="line", dataset="validation_curve", intent="comparison",
            question="Did improvements generalize beyond the phase-1 FSP view?",
            rationale="Four named validation views reveal objective-specific progress and the widening failure to generalize to multilevel/corrupted inputs.",
            context={"baseline":"step-500 validation","denominator":"2,000 fixed validation events × four named views","grain":"validation checkpoint","normalization":"repository principal-loss aggregation","semanticFamily":"validation generalization","unit":"loss"},
            encodings={"x":encoding("step","ordinal","Checkpoint step"),"y":multi(["fspLoss","truthGuidedLoss","multilevelLoss","corruptedLoss"],"Principal loss",unit="loss",fmt="number"),"tooltip":[encoding("boundaryFraction","quantitative","Boundary fraction",unit="share",fmt="percent"),encoding("effectiveRank","quantitative","Effective rank",unit="dimensions",fmt="number")]},
            x_title="Checkpoint step", y_title="Validation principal loss", unit="loss", value_format="number",
            max_rows=6, settings={"sort":"none","showPoints":"always","showValues":False}, palette="blue-purple-orange-pink-validation", reference_lines=zero_y,
        ),
        common_chart(
            chart_id="embedding-checkpoint-comparison", title="Step-3000 validation embeddings in the joint PCA basis",
            subtitle=f"Even-stride display sample; numeric summaries use all {emb['sampleNodeCount']:,} nodes from {emb['sampleEventCount']} validation events",
            chart_type="scatter", dataset="embedding_compare", intent="distribution",
            question="Where do the late, boundary-saturated representations occupy the common projection?",
            rationale="The late checkpoint is shown on the same jointly fitted coordinates used by the following step-500 plot, while keeping one valid scatter measure per chart.",
            context={"baseline":"best.pt at step 500","denominator":f"{emb['sampleNodeCount']} identical validation nodes per checkpoint","grain":"event node × checkpoint","normalization":"joint mean-centered PCA in 32D origin-tangent space","semanticFamily":"learned embedding geometry","unit":"PCA coordinate"},
            encodings={"x":encoding("pc1","quantitative","Tangent PCA 1",unit="PCA coordinate",fmt="number"),"y":encoding("pc2","quantitative","Tangent PCA 2",unit="PCA coordinate",fmt="number"),"tooltip":[encoding("eventShort","text","Event hash"),encoding("level","ordinal","Truth level"),encoding("nodeKind","text","Node kind"),encoding("truthPdg","nominal","Truth PDG"),encoding("hyperbolicRadius","quantitative","Hyperbolic radius",unit="distance",fmt="number")]},
            x_title="Joint tangent-space PCA 1", y_title="Joint tangent-space PCA 2", unit="PCA coordinate", value_format="number",
            max_rows=len(data["embedding_compare"]), settings={"sort":"none","showValues":False}, palette="blue-orange-checkpoints",
        ),
        common_chart(
            chart_id="embedding-best-by-level", title="Selected step-500 validation embeddings in the joint PCA basis",
            subtitle="Same validation cohort and common projection; truth level remains available in tooltips",
            chart_type="scatter", dataset="embedding_best_level", intent="distribution",
            question="What hierarchical organization is visible in the safer available checkpoint?",
            rationale="The safer checkpoint is displayed separately on identical projected coordinates; tooltips preserve truth level, node kind and PDG identity.",
            context={"baseline":"origin-tangent mean","denominator":f"{emb['sampleNodeCount']} nodes from {emb['sampleEventCount']} validation events","grain":"real validation node","normalization":"joint checkpoint PCA basis","semanticFamily":"hierarchical representation","unit":"PCA coordinate"},
            encodings={"x":encoding("pc1","quantitative","Tangent PCA 1",unit="PCA coordinate",fmt="number"),"y":encoding("pc2","quantitative","Tangent PCA 2",unit="PCA coordinate",fmt="number"),"tooltip":[encoding("eventShort","text","Event hash"),encoding("level","ordinal","Truth level"),encoding("nodeKind","text","Node kind"),encoding("truthPdg","nominal","Truth PDG"),encoding("hyperbolicRadius","quantitative","Hyperbolic radius",unit="distance",fmt="number")]},
            x_title="Joint tangent-space PCA 1", y_title="Joint tangent-space PCA 2", unit="PCA coordinate", value_format="number",
            max_rows=len(data["embedding_best_level"]), settings={"sort":"none","showValues":False}, palette="blue-purple-orange-pink-green-yellow-levels",
        ),
        common_chart(
            chart_id="representative-decay-embedding", title=f"One real validation decay sample in embedding space ({data['representative_events'][0]['eventShort']})",
            subtitle=f"{data['representative_events'][0]['sourceCategory']} event; {data['representative_events'][0]['nodes']} nodes, {data['representative_events'][0]['maxLevel'] + 1} populated levels",
            chart_type="scatter", dataset="representative_embedding", intent="distribution",
            question="Where do the nodes of one actual decay event land under the step-500 encoder?",
            rationale="A single event connects the aggregate cloud to physical identities; composite nodes are labeled by truth PDG and retained-tree level.",
            context={"baseline":"event-local node set","denominator":"all retained nodes in one fixed validation event","grain":"event node","normalization":"same joint PCA basis as the cohort charts","semanticFamily":"real decay example","unit":"PCA coordinate"},
            encodings={"x":encoding("pc1","quantitative","Tangent PCA 1",unit="PCA coordinate",fmt="number"),"y":encoding("pc2","quantitative","Tangent PCA 2",unit="PCA coordinate",fmt="number"),"label":encoding("nodeLabel","text","Composite PDG · level"),"tooltip":[encoding("level","ordinal","Truth level"),encoding("nodeKind","text","Node kind"),encoding("truthPdg","nominal","Truth PDG"),encoding("hyperbolicRadius","quantitative","Hyperbolic radius",unit="distance",fmt="number")]},
            x_title="Joint tangent-space PCA 1", y_title="Joint tangent-space PCA 2", unit="PCA coordinate", value_format="number",
            max_rows=len(data["representative_embedding"]), settings={"sort":"none","showValues":False}, palette="blue-purple-orange-pink-green-yellow-levels",
        ),
        common_chart(
            chart_id="radius-by-level", title="Late weights lose a useful radius–level profile",
            subtitle="Median true 32D hyperbolic radius by retained-tree level; PCA is not used here",
            chart_type="line", dataset="radius_by_level", intent="comparison",
            question="Does hyperbolic radius track hierarchy in the sampled real decays?",
            rationale="Original-space radii avoid projection distortion and directly compare the selected and late checkpoints by truth level.",
            context={"baseline":"best.pt step 500","denominator":"real validation nodes within each retained-tree level","grain":"checkpoint × truth level","normalization":"median Poincaré distance from origin","semanticFamily":"radius hierarchy","unit":"hyperbolic distance"},
            encodings={"x":encoding("level","ordinal","Truth level"),"y":multi(["bestStep500Median","step3000Median"],"Median radius",unit="hyperbolic distance",fmt="number"),"tooltip":[encoding("bestStep500Nodes","quantitative","Step-500 nodes",unit="nodes",fmt="number"),encoding("step3000Nodes","quantitative","Step-3000 nodes",unit="nodes",fmt="number")]},
            x_title="Retained-tree reconstruction level", y_title="Median hyperbolic radius", unit="hyperbolic distance", value_format="number",
            max_rows=len(data["radius_by_level"]), settings={"sort":"none","showPoints":"always","showValues":False}, palette="blue-orange-checkpoints", reference_lines=zero_y,
        ),
        common_chart(
            chart_id="gpu-utilization", title="H100 utilization remained low for this small candidate run",
            subtitle=f"30-sample (~7.5 minute) means; peak {telemetry['peak_gpu_utilization_percent']}% and {telemetry['peak_memory_used_mib']} MiB",
            chart_type="line", dataset="telemetry_curve", intent="trend",
            question="How much of the allocated H100 was used over the run?",
            rationale="One utilization unit on the primary axis avoids mixing percent, memory and temperature; exact resource peaks remain in the text.",
            context={"baseline":"zero utilization","denominator":"one H100 NVL","grain":"30 consecutive 15-second samples","normalization":"arithmetic mean","semanticFamily":"resource use","unit":"percent"},
            encodings={"x":encoding("elapsedMinutes","quantitative","Elapsed time",unit="minutes",fmt="number"),"y":encoding("gpuUtilizationMean","quantitative","Mean GPU utilization",unit="percent",fmt="number"),"tooltip":[encoding("gpuUtilizationPeak","quantitative","Bin peak utilization",unit="percent",fmt="number"),encoding("memoryUsedPeakMiB","quantitative","Bin peak memory",unit="MiB",fmt="number"),encoding("temperatureMeanC","quantitative","Mean temperature",unit="°C",fmt="number")]},
            x_title="Elapsed minutes", y_title="Mean GPU utilization (%)", unit="percent", value_format="number",
            max_rows=len(data["telemetry_curve"]), settings={"sort":"none","showPoints":"hover","showValues":False}, palette="blue-observed", reference_lines=zero_y,
        ),
    ]

    tables = [
        table("job-contract-table", "Observed Slurm outcome", "Authoritative accounting and receipt evidence", "job_contract", [
            {"field":"field","label":"Field","type":"text"},{"field":"value","label":"Value","type":"text","role":"value"},{"field":"evidenceStatus","label":"Evidence status","type":"text"},
        ]),
        table("input-output-table", "Pretraining input/output contract", "What enters the encoder, what it emits, and how outputs are intended to be used", "input_output_contract", [
            {"field":"stage","label":"Pipeline stage","type":"text"},{"field":"shapeOrType","label":"Shape / type","type":"text"},{"field":"meaning","label":"Meaning","type":"text"},
        ]),
        table("feature-block-table", "Observed feature and vocabulary blocks", "Names are read from the checkpoint-compatible repository schema", "feature_blocks", [
            {"field":"block","label":"Input block","type":"text"},{"field":"count","label":"Width / entries","type":"number"},{"field":"examples","label":"Examples","type":"text"},
        ]),
        table("model-contract-table", "small_candidate model contract", "Serialized architecture attached to the selected checkpoint", "model_contract", [
            {"field":"field","label":"Field","type":"text"},{"field":"value","label":"Value","type":"text","role":"value"},
        ]),
        table("validation-table", "Fixed-cohort validation checkpoints", "Six evaluations × 2,000 validation events; no sealed test", "validation_table", [
            {"field":"step","label":"Step","type":"number"},{"field":"principalLoss","label":"Principal loss","type":"number","format":"0.000"},{"field":"fspLoss","label":"FSP loss","type":"number","format":"0.000"},{"field":"relationAccuracy","label":"Relation accuracy","type":"number","format":"0.000"},{"field":"effectiveRank","label":"Effective rank","type":"number","format":"0.00"},{"field":"boundaryFraction","label":"Boundary fraction","type":"number","format":"0.0%"},
        ]),
        table("checkpoint-table", "Saved weight inventory", f"{checkpoint_count} CPU-loadable files; hashes are abbreviated for display", "checkpoint_inventory", [
            {"field":"file","label":"Checkpoint","type":"text"},{"field":"step","label":"Step","type":"number"},{"field":"sizeMiB","label":"MiB","type":"number","format":"0.00"},{"field":"modelNonfinite","label":"Non-finite model values","type":"number"},{"field":"optimizerNonfinite","label":"Non-finite optimizer values","type":"number"},{"field":"ampScale","label":"AMP scale","type":"number"},{"field":"sha256Prefix","label":"SHA-256 prefix","type":"text"},
        ]),
        table("embedding-summary-table", "Real-sample embedding summary", "Same validation nodes under step-500 and step-3000 weights", "embedding_summary", [
            {"field":"checkpoint","label":"Checkpoint","type":"text"},{"field":"step","label":"Step","type":"number"},{"field":"nodes","label":"Nodes","type":"number"},{"field":"radiusMean","label":"Mean radius","type":"number","format":"0.000"},{"field":"radiusMedian","label":"Median radius","type":"number","format":"0.000"},{"field":"radiusP90","label":"p90 radius","type":"number","format":"0.000"},{"field":"boundaryFractionRadiusGt10","label":"Radius > 10","type":"number","format":"0.0%"},
        ]),
        table("representative-events-table", "Real validation decay examples", "Three fixed-cohort events from distinct production categories", "representative_events", [
            {"field":"eventShort","label":"Event hash","type":"text"},{"field":"sourceCategory","label":"Source category","type":"text"},{"field":"nodes","label":"Nodes","type":"number"},{"field":"leaves","label":"Leaves","type":"number"},{"field":"composites","label":"Composites","type":"number"},{"field":"maxLevel","label":"Max level","type":"number"},{"field":"b1FullTruthChannelId","label":"B1 channel ID","type":"number"},{"field":"b2FullTruthChannelId","label":"B2 channel ID","type":"number"},
        ]),
    ]

    blocks: list[dict[str, Any]] = []
    def md(block_id: str, body: str) -> None:
        blocks.append({"id":block_id,"type":"markdown","layout":"full","body":body})
    def chart_block(chart_id: str) -> None:
        blocks.append({"id":f"block-{chart_id}","type":"chart","chartId":chart_id,"layout":"full"})
    def table_block(table_id: str) -> None:
        blocks.append({"id":f"block-{table_id}","type":"table","tableId":table_id,"layout":"full"})

    md("title", "# HyperTagging pretraining run 15749624: verified partial-run report")
    md("summary", f"""## Technical summary — the job produced readable checkpoints but did not complete pretraining

**Outcome.** Slurm job `15749624` ran on one H100 NVL for 5:23:09 and failed after **{training['lastStep']:,} / {config['max_steps']:,} planned optimizer steps ({100*training['lastStep']/config['max_steps']:.1f}%)**. The next update was rejected because the clipped total gradient norm was non-finite.

**What is trustworthy.** The JSONL contains {training['trainingRows']:,} contiguous training rows and six fixed-cohort validation rows; every logged scalar is finite. All {checkpoint_count} saved checkpoints load on CPU, and {finite_checkpoints}/{checkpoint_count} have zero non-finite values in both model and optimizer tensors.

**What is not established.** The run never left the first `fsp_topology_anticollapse` phase. It therefore does not establish multilevel, channel, corruption/hard-negative, downstream reconstruction, or sealed-test performance. The most defensible diagnostic snapshot is `best.pt` at step 500, not the later step-3000 state.

**Data boundary.** Real visualizations below use {emb['sampleEventCount']} events from the checkpoint's fixed **validation** UID cohort. Sealed-test data were never requested or opened.""")
    table_block("job-contract-table")
    chart_block("curriculum-progress")
    md("training-interpretation", f"""## Training behavior — apparent objective progress was accompanied by geometric instability

The phase-1 batch loss fell from {training['curve25StepBins'][0]['lossMean']:.3f} in the first bin to {training['curve25StepBins'][-1]['lossMean']:.3f} in the final partial bin. That improvement is narrow: it covers only FSP topology, anti-collapse and leaf-PID training. Logged gradients remained finite through step {training['lastStep']}, so the terminal fault occurred during backpropagation for the next batch, before another optimizer-step record could be written.""")
    chart_block("training-loss")
    md("boundary-interpretation", f"""The dominant precursor is boundary saturation. The boundary fraction first crossed 50% at step {next(row['firstStep'] for row in training['boundaryCrossings'] if row['threshold']==0.5)} and 99% at step {next(row['firstStep'] for row in training['boundaryCrossings'] if row['threshold']==0.99)}. Over the last 100 logged steps it averaged {training['tail100']['boundaryFractionMean']:.1%}; the final batch was {training['lastRow']['boundary_fraction']:.1%}. In a Poincaré ball this drives distance and derivative terms into a numerically stiff regime, making the later non-finite gradient norm unsurprising rather than isolated.""")
    chart_block("boundary-saturation")
    md("validation-interpretation", f"""## Validation — the trained view improved while broader views degraded

The fixed 2,000-event cohort shows a split result. FSP principal loss improved from {val500['fspLoss']:.3f} at step 500 to {val3000['fspLoss']:.3f} at step 3000, but the configured principal validation loss worsened from {val500['principalLoss']:.3f} to {val3000['principalLoss']:.3f}. Truth-guided, multilevel and corrupted views all became substantially worse. Validation boundary fraction simultaneously rose from {val500['boundaryFraction']:.1%} to {val3000['boundaryFraction']:.1%}. This is not a converged transferable representation.""")
    chart_block("validation-view-loss")
    table_block("validation-table")
    md("io-intro", """## Inputs and outputs — typed detector objects enter a shared Euclidean/hyperbolic encoder

The model consumes variable-size node sets rather than token sequences. Each event includes reconstructed four-momenta, charge, availability-aware detector blocks, reduced PID tokens, node kind, reconstruction level and physical relation features. Track, ECL, KLM and composite adapters map these heterogeneous inputs into a shared 128-dimensional space; relation-aware attention contextualizes the set; dedicated projections emit tree, reconstruction and channel representations; a 32-dimensional tangent projection is mapped into the curvature-1 Poincaré ball.

During pretraining, topology and auxiliary heads provide self-/truth-supervision. During later reconstruction, the intended product is transferred encoder initialization—not a ready-made decay tree. Since this run failed in phase 1, its weights must be treated as diagnostic initialization only.""")
    table_block("input-output-table")
    table_block("feature-block-table")
    table_block("model-contract-table")
    md("embedding-method", f"""## Real decay embeddings — joint tangent PCA exposes the failure mode

The report reloaded `best.pt` (step 500) and `checkpoint-step-3000.pt`, rebuilt exact training normalizers, restored the checkpoint's validation UID selection, and ran the same {emb['sampleNodeCount']:,} nodes from {emb['sampleEventCount']} real validation decays through both encoders. Hyperbolic points were mapped to the tangent space at the origin and projected with one PCA basis fitted jointly across both checkpoints. PC1+PC2 explain {(emb['pcaExplainedVariance']['pc1']+emb['pcaExplainedVariance']['pc2']):.1%} of joint tangent variance; the pictures are therefore diagnostic projections, while radius plots use the original 32D geometry.""")
    chart_block("embedding-checkpoint-comparison")
    md("embedding-compare-interpretation", f"""The step-500 checkpoint has median radius {best_summary['radiusMedian']:.3f} and {best_summary['boundaryFractionRadiusGt10']:.1%} of sampled nodes above radius 10. By step 3000 the median is {late_summary['radiusMedian']:.3f} and {late_summary['boundaryFractionRadiusGt10']:.1%} exceed radius 10. The large scale expansion is consistent with the logged boundary fraction and explains why `best.pt` is safer for any diagnostic transfer test.""")
    chart_block("embedding-best-by-level")
    md("embedding-level-interpretation", "Compared on the common axes, the step-500 cloud is compact and interior while the late cloud spans a much larger PC1 range. Truth level, node kind and PDG remain available per point in the interactive tooltip. Topology is relational, and a two-dimensional Euclidean projection cannot preserve all Poincaré distances; this supports inspection, not a clustering-accuracy claim.")
    chart_block("representative-decay-embedding")
    md("sample-interpretation", "The sample is an actual validation decay, not a synthetic fixture. Composite-node labels show truth PDG and retained-tree level; leaves remain available through tooltips to avoid overprinting. The same joint PCA basis is used so locations can be read against the cohort plot. Parent–daughter validity remains evaluated by original tree relations and 32D distances, not by apparent 2D proximity.")
    table_block("representative-events-table")
    chart_block("radius-by-level")
    md("radius-interpretation", "The original-space radius profile is the more faithful hierarchy check. The late checkpoint pushes every populated level into a high-radius regime, reducing useful radial headroom and magnifying numerical sensitivity near the ball boundary.")
    table_block("embedding-summary-table")
    md("weights", f"""## Weight verification — readable and finite does not mean scientifically promotable

All {checkpoint_count} files deserialize with `torch.load(..., map_location='cpu')`. Each contains {data['checkpoint_inventory'][0]['modelElements']:,} model-state elements, a complete optimizer state, the same split-manifest hash, and the same 2,000-event validation-selection hash. No model or optimizer tensor contains NaN or infinity. `best.pt` has SHA-256 `{evidence['checkpoints'][0]['sha256']}` and records step {evidence['checkpoints'][0]['step']}.

The report does not call any file a final pretrained model. Specialized “best” files select different diagnostics, several are aliases of the same saved step, and no snapshot covers phases 2–4.""")
    table_block("checkpoint-table")
    md("resources", f"""## Resource behavior — correct H100 allocation, low utilization

Telemetry contains {telemetry['sample_count']:,} samples at {telemetry['interval_seconds']:.0f}-second cadence. Peak utilization was {telemetry['peak_gpu_utilization_percent']}%, peak allocated memory use {telemetry['peak_memory_used_mib']} MiB, and peak temperature {telemetry['peak_temperature_c']} °C. The small model used only about {100*telemetry['peak_memory_used_mib']/95830:.1f}% of H100 memory at peak; the training/validation pipeline, not device capacity, dominated elapsed time.""")
    chart_block("gpu-utilization")
    md("limitations", """## Limitations and robustness

- This is a forensic report on one failed seed and one 35k selection, not an ablation or convergence study.
- The real-sample figures use 64 events sampled in saved fixed-validation order from a 2,000-event validation cohort. They are reproducible but not the entire validation distribution.
- PCA is a visualization aid. Scientific geometry claims must use original 32D Poincaré distances, radius statistics, relation metrics and parent ranking.
- Step 500 is “best” only under the configured principal validation rule among saved checkpoints; it is not validated for downstream free reconstruction.
- No sealed-test payload or metric was accessed. Final held-out physics performance remains unknown.
- Low H100 utilization is observational; precise input/validation profiling was not captured in this job.""")
    md("next-steps", """## Recommended corrective experiment

1. Preserve this failed run and all hashes as the baseline attempt; do not overwrite it.
2. Add a pre-clip raw-gradient finite diagnostic that identifies the first offending parameter/objective, plus per-objective finite guards around hyperbolic distance/radius terms.
3. Prevent runaway radii with a preregistered geometry intervention: lower/bounded tangent scale or a stricter interior projection margin, followed by a short matched diagnostic—not an automatic silent clamp.
4. Resume only from the step-500 checkpoint or restart from seed after the corrective patch; do not resume from step 3000 because boundary saturation is already severe.
5. Require a small H100 diagnostic to cross phase boundaries with finite gradients, boundary fraction below the registered ceiling, and healthy validation across all four named views.
6. Only then rerun the full 17,500-step 35k campaign and compare the registered geometry ablations. Downstream reconstruction and sealed-test evaluation remain later gates.""")
    md("questions", """## Further questions

- Which objective first creates non-finite gradients when computed separately on the failing-region batches?
- Does a bounded tangent scale preserve FSP relation gains while keeping multilevel validation radii interior?
- Why is fixed-cohort validation much more expensive than GPU compute, and can data loading/view reuse be profiled without changing scientific semantics?
- Do checkpoint-selection tracks need an explicit geometry-safety gate so a high effective rank cannot select a boundary-saturated state?""")
    md("provenance", f"""## Provenance

Repository commit: `{evidence['provenance']['gitCommit']}`. Slurm job: `15749624`. Metrics SHA-256: `{training['metricsSha256']}`. Split-manifest SHA-256: `{evidence['provenance']['splitManifestHash']}`. Fixed-validation selection manifest: `{evidence['provenance']['validationSelectionManifestHash']}`. Generated from `{REPORT_REL}`; sealed test accessed: **false**.""")

    sources = [
        {
            "id":"verified-evidence",
            "label":"Canonical verified evidence snapshot",
            "path":REPORT_REL,
            "query":{
                "engine":"repository","language":"python",
                "description":"Deterministic extraction from Slurm accounting, receipt, metrics, checkpoints, schema code and fixed validation-role inference.",
                "filters":["job 15749624 only","validation role only for real-sample inference","sealed test excluded"],
                "metric_definitions":["observed = read or recomputed from authoritative artifacts","derived = arithmetic or projection computed from observed evidence"],
                "tables_used":[
                    "artifacts/slurm/jobs/15749624/attempt-00/receipt.json",
                    "artifacts/runs/pretrain-035k-small-candidate-science-1ae7d74/20260812/15749624/metrics.jsonl",
                    "artifacts/runs/pretrain-035k-small-candidate-science-1ae7d74/20260812/15749624/*.pt",
                    "src/hypertagging/models/heterogeneous.py",
                    "src/hypertagging/training/pretrain_trainer.py",
                ],
            },
        }
    ] + [dataset_source(name, f"Report-ready rows derived unchanged or deterministically from evidence field {name}.") for name in data]

    generated = datetime.now().astimezone().isoformat()
    return {
        "manifest": {
            "version": 1,
            "title": "HyperTagging pretraining run 15749624: verified partial-run report",
            "description": "Forensic verification of a failed H100 pretraining attempt, checkpoint integrity and real validation-sample embedding geometry.",
            "generatedAt": generated,
            "surface": "report",
            "sources": sources,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "partial",
            "generatedAt": generated,
            "accessIssues": [],
            "datasets": data,
        },
        "sources": sources,
        "surface": "report",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    artifact = build_artifact(evidence)
    compact_artifact = json.dumps(artifact, sort_keys=True)
    connection = sqlite3.connect(":memory:")
    try:
        for dataset, rows in artifact["snapshot"]["datasets"].items():
            verified = connection.execute(
                dataset_sql(dataset), {"artifact_json": compact_artifact}
            ).fetchall()
            if len(verified) != len(rows):
                raise ValueError(f"SQL verification failed for dataset {dataset}")
    finally:
        connection.close()
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(args.output),"bytes":args.output.stat().st_size,"charts":len(artifact['manifest']['charts']),"tables":len(artifact['manifest']['tables']),"blocks":len(artifact['manifest']['blocks'])}, sort_keys=True))


if __name__ == "__main__":
    main()
