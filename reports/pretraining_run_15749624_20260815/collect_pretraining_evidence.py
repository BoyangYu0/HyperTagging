#!/usr/bin/env python3
"""Collect bounded, test-sealed evidence for Slurm pretraining job 15749624."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.models.hyperbolic import logmap0, radius
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KINDS
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.schema_v4 import KLM_FEATURE_NAMES
from hypertagging.training.data_module import build_real_data_module
from hypertagging.training.fixed_validation import select_validation_events
from hypertagging.training.model_config import ModelArchitecture
from hypertagging.training.pretrain_trainer import (
    ContextualPretrainingModel,
    _add_topology_labels,
)
from hypertagging.training.pretraining_curriculum import (
    PretrainingStage,
    build_curriculum_batch,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def tensor_inventory(value: object) -> dict[str, int]:
    result = {"tensors": 0, "elements": 0, "nonfinite": 0}
    if isinstance(value, torch.Tensor):
        result["tensors"] = 1
        result["elements"] = value.numel()
        if value.is_floating_point() or value.is_complex():
            result["nonfinite"] = int((~torch.isfinite(value)).sum().item())
    elif isinstance(value, dict):
        for child in value.values():
            item = tensor_inventory(child)
            for key in result:
                result[key] += item[key]
    elif isinstance(value, (list, tuple)):
        for child in value:
            item = tensor_inventory(child)
            for key in result:
                result[key] += item[key]
    return result


def audit_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selection = payload.get("validation_selection") or {}
        scaler = payload.get("scaler_state_dict") or {}
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "step": int(payload.get("step", 0)),
                "gitCommit": payload.get("git_commit", "unknown"),
                "splitManifestHash": payload.get("split_manifest_hash", ""),
                "modelInventory": tensor_inventory(payload.get("model_state_dict", payload)),
                "optimizerInventory": tensor_inventory(payload.get("optimizer_state_dict", {})),
                "ampScale": float(scaler.get("scale", 0.0)),
                "validationEventCount": len(selection.get("event_uids") or []),
                "validationSelectionManifestHash": selection.get(
                    "selection_manifest_hash", ""
                ),
                "checkpointMetric": float((payload.get("metrics") or {}).get("loss", 0.0)),
            }
        )
    return rows


def compact_training_evidence(metrics_path: Path) -> dict[str, Any]:
    rows = json_lines(metrics_path)
    training = [row for row in rows if row.get("split") != "validation"]
    validation = [row for row in rows if row.get("split") == "validation"]
    steps = [int(row["step"]) for row in training]
    scalar_nonfinite = []
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    scalar_nonfinite.append({"row": row_index + 1, "key": key})

    curve = []
    bin_size = 25
    for start in range(0, len(training), bin_size):
        part = training[start : start + bin_size]
        curve.append(
            {
                "step": int(part[-1]["step"]),
                "lossMean": mean([float(row["loss"]) for row in part]),
                "lossMin": min(float(row["loss"]) for row in part),
                "lossMax": max(float(row["loss"]) for row in part),
                "boundaryFractionMean": mean(
                    [float(row["boundary_fraction"]) for row in part]
                ),
                "effectiveRankMean": mean([float(row["effective_rank"]) for row in part]),
                "learningRateMean": mean([float(row["learning_rate"]) for row in part]),
                "gradientProjectionHyperMean": mean(
                    [float(row["gradient_projection_hyper_projection"]) for row in part]
                ),
            }
        )

    validation_curve = []
    for row in validation:
        validation_curve.append(
            {
                "step": int(row["step"]),
                "principalLoss": float(row["validation_principal_loss"]),
                "fspLoss": float(row["validation_fsp_topology_anticollapse_principal_loss"]),
                "truthGuidedLoss": float(row["validation_truth_guided_distance_radius_principal_loss"]),
                "multilevelLoss": float(row["validation_multilevel_channel_memory_principal_loss"]),
                "corruptedLoss": float(row["validation_corrupted_composites_hard_negatives_principal_loss"]),
                "relationAccuracy": float(row["validation_relation_accuracy"]),
                "parentRankingAccuracy": float(row["validation_parent_ranking_accuracy"]),
                "channelRetrievalAccuracy": float(row["validation_channel_retrieval_accuracy"]),
                "effectiveRank": float(row["validation_effective_rank"]),
                "boundaryFraction": float(row["validation_boundary_fraction"]),
                "radiusLevelMonotonicity": float(row["validation_radius_level_monotonicity"]),
            }
        )

    boundary_crossings = []
    for threshold in (0.01, 0.1, 0.5, 0.9, 0.99):
        match = next(
            (row for row in training if float(row["boundary_fraction"]) >= threshold),
            None,
        )
        boundary_crossings.append(
            {
                "threshold": threshold,
                "firstStep": int(match["step"]) if match is not None else None,
            }
        )

    tail = training[-100:]
    return {
        "metricsSha256": sha256(metrics_path),
        "jsonlRows": len(rows),
        "trainingRows": len(training),
        "validationRows": len(validation),
        "firstStep": min(steps),
        "lastStep": max(steps),
        "stepsContiguous": steps == list(range(1, max(steps) + 1)),
        "duplicateStepCount": len(steps) - len(set(steps)),
        "scalarNonfiniteCount": len(scalar_nonfinite),
        "scalarNonfiniteExamples": scalar_nonfinite[:10],
        "phaseCounts": dict(Counter(str(row["curriculum_phase"]) for row in training)),
        "lastRow": {
            key: training[-1][key]
            for key in (
                "step", "loss", "learning_rate", "effective_rank", "boundary_fraction",
                "covariance_off_diagonal_norm", "mean_dimension_std", "min_dimension_std",
                "mean_negative_relation_distance", "mean_positive_relation_distance",
                "gradient_projection_hyper_projection",
            )
        },
        "tail100": {
            "stepStart": int(tail[0]["step"]),
            "stepEnd": int(tail[-1]["step"]),
            "lossMean": mean([float(row["loss"]) for row in tail]),
            "lossMin": min(float(row["loss"]) for row in tail),
            "lossMax": max(float(row["loss"]) for row in tail),
            "boundaryFractionMean": mean([float(row["boundary_fraction"]) for row in tail]),
            "boundaryFractionMin": min(float(row["boundary_fraction"]) for row in tail),
            "effectiveRankMean": mean([float(row["effective_rank"]) for row in tail]),
            "gradientProjectionHyperMean": mean(
                [float(row["gradient_projection_hyper_projection"]) for row in tail]
            ),
            "gradientProjectionHyperMax": max(
                float(row["gradient_projection_hyper_projection"]) for row in tail
            ),
        },
        "boundaryCrossings": boundary_crossings,
        "curve25StepBins": curve,
        "validationCurve": validation_curve,
    }


def build_model(payload: dict[str, Any], data_module: Any) -> ContextualPretrainingModel:
    architecture = ModelArchitecture.from_dict(payload["architecture"])
    config = payload["config"]
    ablation = ALL_ABLATIONS[str(config.get("ablation", "full_revised"))]
    model = ContextualPretrainingModel(
        d_model=architecture.d_model,
        hyper_dim=architecture.hyper_dim,
        curvature=architecture.curvature,
        n_heads=architecture.n_heads,
        n_context_layers=architecture.n_context_layers,
        ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        use_contextual_encoder=ablation.contextual_euclidean,
        use_physical_relations=ablation.relation_attention,
        use_hyperbolic_relations=ablation.hyperbolic_relation_attention,
        channel_memory_size=int(config.get("channel_memory_size", 0)),
        channel_pooling=str(config.get("channel_pooling", "mean_all")),
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
        hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=data_module.normalizers["common"].mean,
            common_std=data_module.normalizers["common"].std,
            composite_mean=data_module.normalizers["composite"].mean,
            composite_std=data_module.normalizers["composite"].std,
        )
    )
    return model.eval()


def collect_real_embeddings(
    run_dir: Path,
    *,
    sample_events: int,
) -> dict[str, Any]:
    checkpoints = {
        "best_step500": run_dir / "best.pt",
        "step3000": run_dir / "checkpoint-step-3000.pt",
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in checkpoints.items()
    }
    base_payload = payloads["best_step500"]
    config = base_payload["config"]
    data_module = build_real_data_module(
        config["data"],
        dataset_index=config["dataset_index"],
        max_events=config.get("max_events"),
        seed=int(config["seed"]),
        pilot_split_repair=bool(config.get("pilot_split_repair", False)),
        allow_legacy_conflated=bool(config.get("allow_legacy_conflated", False)),
        shuffle_buffer_size=int(config.get("shuffle_buffer_size", 1024)),
        num_workers=0,
        normalization_state=base_payload["normalizer_state"],
        required_splits=("validation",),
        target_policy=str(config.get("target_policy") or "complete_only"),
        scientific_mode=True,
    )
    restored_uids = tuple(base_payload["validation_selection"]["event_uids"])
    events, fixed_uids, selection_contract = select_validation_events(
        data_module.iter_events("validation", shuffle=False),
        limit=len(restored_uids),
        scientific_mode=True,
        selection_manifest_hash=data_module.selection_manifest_hash,
        seed=int(config["seed"]),
        restored_event_uids=restored_uids,
    )
    events = events[:sample_events]
    fixed_uids = fixed_uids[:sample_events]
    metadata = []
    for event in events:
        for node_position in range(len(event.level_ids)):
            parent = int(event.parent_ids[node_position])
            truth_token = int(event.truth_pid_labels[node_position])
            kind_id = int(event.node_kind_ids[node_position])
            metadata.append(
                {
                    "eventUid": event.event_uid,
                    "eventShort": hashlib.sha256(event.event_uid.encode()).hexdigest()[:8],
                    "sourceCategory": event.source_category,
                    "nodePosition": node_position,
                    "nodeId": int(event.node_ids[node_position]),
                    "parentPosition": parent,
                    "level": int(event.level_ids[node_position]),
                    "truthRootDistance": int(event.truth_root_distance[node_position]),
                    "nodeKind": NODE_KINDS[kind_id] if 0 <= kind_id < len(NODE_KINDS) else str(kind_id),
                    "truthPidToken": truth_token,
                    "truthPdg": int(PDG_TOKENS[truth_token]),
                    "bSide": int(event.b_side[node_position]),
                }
            )

    tangent_by_checkpoint: dict[str, np.ndarray] = {}
    radius_by_checkpoint: dict[str, np.ndarray] = {}
    for checkpoint_name, payload in payloads.items():
        model = build_model(payload, data_module)
        tangent_parts = []
        radius_parts = []
        for start in range(0, len(events), 8):
            batch = data_module.normalize_batch(
                collate_heterogeneous_events(events[start : start + 8])
            )
            _add_topology_labels(batch)
            curriculum = build_curriculum_batch(
                batch,
                PretrainingStage.TRUTH_GUIDED_MULTILEVEL,
                seed=int(config["seed"]) + start,
                truth_guided_structural_relation_inputs=bool(
                    config.get("truth_guided_structural_relation_inputs", False)
                ),
            )
            with torch.no_grad():
                encoded, _, output_batch = model.encode_runtime(
                    curriculum.batch,
                    attention_mask=curriculum.batch["curriculum_attention_mask"],
                )
            mask = output_batch["node_mask"].bool()
            tangent_parts.append(logmap0(encoded.hyperbolic_embeddings)[mask].cpu().numpy())
            radius_parts.append(radius(encoded.hyperbolic_embeddings)[mask].cpu().numpy())
        tangent_by_checkpoint[checkpoint_name] = np.concatenate(tangent_parts, axis=0)
        radius_by_checkpoint[checkpoint_name] = np.concatenate(radius_parts, axis=0)
        if len(tangent_by_checkpoint[checkpoint_name]) != len(metadata):
            raise ValueError("embedding rows do not match the fixed validation node grain")

    all_tangent = np.concatenate(list(tangent_by_checkpoint.values()), axis=0)
    centered = all_tangent - all_tangent.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    total_variance = float((singular_values**2).sum())
    explained = [float(value**2 / total_variance) for value in singular_values[:2]]

    points = []
    radius_levels: dict[tuple[str, int], list[float]] = defaultdict(list)
    summary = []
    for checkpoint_name in checkpoints:
        projected = (tangent_by_checkpoint[checkpoint_name] - all_tangent.mean(axis=0)) @ components.T
        radii = radius_by_checkpoint[checkpoint_name]
        for index, meta in enumerate(metadata):
            points.append(
                {
                    **meta,
                    "checkpoint": checkpoint_name,
                    "checkpointStep": int(payloads[checkpoint_name]["step"]),
                    "pc1": float(projected[index, 0]),
                    "pc2": float(projected[index, 1]),
                    "radius": float(radii[index]),
                    "boundary": bool(radii[index] > 10.0),
                }
            )
            radius_levels[(checkpoint_name, meta["level"])].append(float(radii[index]))
        summary.append(
            {
                "checkpoint": checkpoint_name,
                "step": int(payloads[checkpoint_name]["step"]),
                "nodes": len(radii),
                "radiusMean": mean(radii.tolist()),
                "radiusMedian": percentile(radii.tolist(), 0.5),
                "radiusP90": percentile(radii.tolist(), 0.9),
                "boundaryFractionRadiusGt10": float(np.mean(radii > 10.0)),
                "tangentNormMean": float(
                    np.linalg.norm(tangent_by_checkpoint[checkpoint_name], axis=1).mean()
                ),
            }
        )
    radius_by_level = [
        {
            "checkpoint": checkpoint,
            "level": level,
            "nodes": len(values),
            "radiusMedian": percentile(values, 0.5),
            "radiusP10": percentile(values, 0.1),
            "radiusP90": percentile(values, 0.9),
        }
        for (checkpoint, level), values in sorted(radius_levels.items())
    ]

    representative_events = []
    used_categories: set[str] = set()
    candidates = sorted(events, key=lambda event: (-int(event.level_ids.max()), len(event.level_ids)))
    for event in candidates:
        if event.source_category in used_categories:
            continue
        representative_events.append(
            {
                "eventUid": event.event_uid,
                "eventShort": hashlib.sha256(event.event_uid.encode()).hexdigest()[:8],
                "sourceCategory": event.source_category,
                "nodes": len(event.level_ids),
                "leaves": int((event.level_ids == 0).sum()),
                "composites": int((event.level_ids > 0).sum()),
                "maxLevel": int(event.level_ids.max()),
                "b1FullTruthChannelId": int(event.b1_full_truth_channel_id),
                "b2FullTruthChannelId": int(event.b2_full_truth_channel_id),
            }
        )
        used_categories.add(event.source_category)
        if len(representative_events) == 3:
            break

    node_counts = [len(event.level_ids) for event in events]
    return {
        "split": "validation",
        "sealedTestAccessed": False,
        "selectionContract": selection_contract,
        "selectionManifestHash": data_module.selection_manifest_hash,
        "fixedValidationCohortSize": len(restored_uids),
        "sampleEventCount": len(events),
        "sampleNodeCount": len(metadata),
        "sampleUidOrderHash": hashlib.sha256("\n".join(fixed_uids).encode()).hexdigest(),
        "sampleNodeCountMean": mean(node_counts),
        "sampleNodeCountP90": percentile(node_counts, 0.9),
        "sourceCategoryCounts": dict(Counter(event.source_category for event in events)),
        "pcaExplainedVariance": {"pc1": explained[0], "pc2": explained[1]},
        "embeddingSummary": summary,
        "radiusByLevel": radius_by_level,
        "points": points,
        "representativeEvents": representative_events,
    }


def telemetry_evidence(attempt_dir: Path) -> dict[str, Any]:
    summary = json.loads((attempt_dir / "gpu-telemetry-summary.json").read_text())
    rows = json_lines(attempt_dir / "gpu-telemetry.jsonl")
    first = datetime.fromisoformat(rows[0]["timestamp"])
    curve = []
    for start in range(0, len(rows), 30):
        part = rows[start : start + 30]
        timestamp = datetime.fromisoformat(part[-1]["timestamp"])
        curve.append(
            {
                "elapsedMinutes": (timestamp - first).total_seconds() / 60.0,
                "gpuUtilizationMean": mean(
                    [float(row["gpu_utilization_percent"]) for row in part]
                ),
                "gpuUtilizationPeak": max(
                    float(row["gpu_utilization_percent"]) for row in part
                ),
                "memoryUsedMeanMiB": mean([float(row["memory_used_mib"]) for row in part]),
                "memoryUsedPeakMiB": max(float(row["memory_used_mib"]) for row in part),
                "temperatureMeanC": mean([float(row["temperature_c"]) for row in part]),
            }
        )
    return {"summary": summary, "curve30SampleBins": curve}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-events", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    best = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    receipt = json.loads((args.attempt_dir / "receipt.json").read_text())
    stderr_path = args.repo / "artifacts/slurm/pretrain-035k-small-candidate-science-1ae7d74-15749624.err"
    error_tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-24:]
    evidence = {
        "generatedAt": datetime.now().astimezone().isoformat(),
        "reportStatus": "partial_pretraining_failed",
        "provenance": {
            "jobId": "15749624",
            "gitCommit": best.get("git_commit", "unknown"),
            "runDirectory": str(args.run_dir),
            "attemptDirectory": str(args.attempt_dir),
            "selectionManifest": best["config"]["data"],
            "datasetIndex": best["config"]["dataset_index"],
            "splitManifestHash": best["split_manifest_hash"],
            "validationSelectionManifestHash": best["validation_selection"][
                "selection_manifest_hash"
            ],
            "sealedTestAccessed": False,
        },
        "job": {
            "state": "FAILED",
            "exitCode": "1:0",
            "elapsed": "05:23:09",
            "start": "2026-08-14T20:18:23+02:00",
            "end": "2026-08-15T01:41:32+02:00",
            "node": "usm-cl-nv01",
            "gres": "gpu:h100nvl:1",
            "cpus": 8,
            "memory": "64G",
            "restartCount": int(receipt["slurm"]["restart_count"]),
            "terminalStage": receipt["terminal_stage"],
            "failureClass": "non_finite_gradient_norm_before_optimizer_step_3343",
            "failureMessage": next(
                (line.strip() for line in error_tail if "total norm" in line),
                "RuntimeError: non-finite gradient norm",
            ),
            "errorTail": error_tail,
        },
        "architecture": best["architecture"],
        "trainingConfig": {
            key: best["config"].get(key)
            for key in (
                "model_preset", "ablation", "seed", "max_steps", "batch_size",
                "learning_rate", "weight_decay", "mixed_precision", "amp_init_scale",
                "validate_every", "validation_batches", "validation_events",
                "curriculum_mode", "curriculum_phase_steps", "scientific_mode",
                "channel_memory_size", "channel_pooling",
            )
        },
        "featureBlocks": {
            "common": list(V3_COMMON_FEATURE_NAMES),
            "track": list(V3_TRACK_FEATURE_NAMES),
            "eclCluster": list(V3_CLUSTER_FEATURE_NAMES),
            "klmCluster": list(KLM_FEATURE_NAMES),
            "compositeStored": list(V3_COMPOSITE_FEATURE_NAMES),
            "nodeKinds": list(NODE_KINDS),
            "reducedPidVocabulary": [int(value) for value in PDG_TOKENS],
        },
        "training": compact_training_evidence(args.run_dir / "metrics.jsonl"),
        "checkpoints": audit_checkpoints(args.run_dir),
        "telemetry": telemetry_evidence(args.attempt_dir),
        "realValidationEmbeddings": collect_real_embeddings(
            args.run_dir, sample_events=args.sample_events
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "trainingSteps": evidence["training"]["trainingRows"],
        "checkpointCount": len(evidence["checkpoints"]),
        "embeddingEvents": evidence["realValidationEmbeddings"]["sampleEventCount"],
        "embeddingNodes": evidence["realValidationEmbeddings"]["sampleNodeCount"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
