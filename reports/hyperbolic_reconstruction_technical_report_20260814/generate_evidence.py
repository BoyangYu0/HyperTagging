#!/usr/bin/env python3
"""Generate bounded, deterministic evidence for the 2026-08-14 technical report.

This program reads only an explicit allowlist of tracked repository metadata and
source files.  It never opens Parquet payloads, selection-role payloads, model
checkpoints, sealed-test results, or scheduler/GPU interfaces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import subprocess
from typing import Any, Iterable, Mapping

import yaml


REPORT_DATE = "2026-08-14"
GENERATED_AT = "2026-08-14T00:00:00Z"
REPORT_TITLE = "Hyperbolic Reconstruction: Data, Model, Training and Inference Readiness"
EVIDENCE_VERSION = 1

DEFAULT_METADATA_ROOTS = {
    "artifacts": "artifacts",
    "configs": "configs",
    "docs": "docs",
    "src": "src",
}

# Exact allowlist.  Metadata-root overrides may relocate these logical roots,
# but may not expand the set of readable files.
REQUIRED_INPUTS: dict[str, tuple[str, str]] = {
    "inventory": (
        "configs",
        "training_selection/production_1m_20260812/inventory.json",
    ),
    "selection_summary": (
        "configs",
        "training_selection/production_1m_20260812/summary.json",
    ),
    "training_readiness": (
        "configs",
        "training_selection/production_1m_20260812/training_readiness.json",
    ),
    "capacity_small_candidate": (
        "artifacts",
        "experiment_readiness/production_1m_20260812/train_035k/"
        "capacity.small_candidate.complete_only.json",
    ),
    "diagnostic_receipt": (
        "artifacts",
        "slurm/jobs/15745941/attempt-00/receipt.json",
    ),
    "diagnostic_telemetry": (
        "artifacts",
        "slurm/jobs/15745941/attempt-00/gpu-telemetry.jsonl",
    ),
    "diagnostic_telemetry_summary": (
        "artifacts",
        "slurm/jobs/15745941/attempt-00/gpu-telemetry-summary.json",
    ),
    "diagnostic_metrics": (
        "artifacts",
        "runs/pretrain-035k-small-candidate-diag4-0c5e054/20260812/15745941/"
        "metrics.jsonl",
    ),
    "no_submit_contract": (
        "artifacts",
        "experiment_readiness/production_1m_20260812/slurm/"
        "scientific-v100-small-candidate-final-no-submit.job-contract.json",
    ),
    "pretrain_scientific_config": (
        "configs",
        "slurm/pretrain_035k_scientific.yaml",
    ),
    "pretrain_diagnostic_config": (
        "configs",
        "slurm/pretrain_diagnostic_small_candidate.yaml",
    ),
    "reconstruction_config": ("configs", "level_reconstruction.yaml"),
    "small_candidate_config": ("configs", "model_presets/small_candidate.yaml"),
    "training_plan": ("docs", "training_execution_plan_20260812.md"),
    "current_status": ("docs", "audits/current_status.md"),
    "architecture_document": (
        "docs",
        "hyperbolic_level_autoregressive_reconstruction.md",
    ),
    "model_config_source": ("src", "hypertagging/training/model_config.py"),
    "model_source": ("src", "hypertagging/models/level_autoregressive.py"),
    "ablation_source": ("src", "hypertagging/models/ablation.py"),
    "pretrain_source": ("src", "hypertagging/training/pretrain_trainer.py"),
}

CURRICULUM_CONFIGS = tuple(
    ("configs", f"ablations/pretrain_stage{index}_{suffix}.yaml")
    for index, suffix in (
        (1, "topology_parent_anticollapse"),
        (2, "distance_radius"),
        (3, "channel"),
        (4, "candidate_hard_negative"),
    )
)

ABLATION_CONFIG_PATHS = (
    "ablations/flat_baseline.yaml",
    "ablations/heterogeneous_only.yaml",
    "ablations/contextual_euclidean.yaml",
    "ablations/contextual_hyperbolic_parent_lca.yaml",
    "ablations/plus_cross_event_channel.yaml",
    "ablations/plus_hyperbolic_relation_attention.yaml",
    "ablations/full_revised.yaml",
    "ablations/level_encoding_none.yaml",
    "ablations/level_encoding_euclidean.yaml",
    "ablations/level_encoding_bounded_tangent.yaml",
    "ablations/radius_generation_height.yaml",
    "ablations/radius_exact_root_depth.yaml",
    "ablations/radius_weak_or_learned.yaml",
    "ablations/learned_bounded_tangent_scale.yaml",
    "ablations/lower_radius_tangent_scale.yaml",
    "ablations/plus_scheduled_sampling.yaml",
)

ALLOWED_SUFFIXES = {".json", ".jsonl", ".yaml", ".md", ".py"}

REPORT_EVIDENCE_PATH = (
    "reports/hyperbolic_reconstruction_technical_report_20260814/evidence.json"
)
SAFE_DATASET_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_CURRICULUM_SOURCE_IDS = tuple(
    f"curriculum_stage_{index}" for index in range(1, len(CURRICULUM_CONFIGS) + 1)
)
_ABLATION_CONFIG_SOURCE_IDS = tuple(
    f"ablation_config_{Path(relative).stem}" for relative in ABLATION_CONFIG_PATHS
)

DATASET_SUPPORT_SOURCE_IDS: dict[str, tuple[str, ...]] = {
    "production_category_composition": ("inventory", "selection_summary"),
    "structural_percentiles": ("capacity_small_candidate", "training_readiness"),
    "subset_ladder": ("selection_summary", "training_readiness"),
    "model_parameter_scale": (
        "model_config_source",
        "model_source",
        "pretrain_source",
        "small_candidate_config",
    ),
    "model_module_composition": (
        "model_config_source",
        "model_source",
        "pretrain_source",
        "small_candidate_config",
    ),
    "curriculum_plan": (
        "pretrain_scientific_config",
        "training_plan",
        *_CURRICULUM_SOURCE_IDS,
    ),
    "staged_training_budgets": ("training_plan",),
    "inference_capacity_controls": (
        "capacity_small_candidate",
        "reconstruction_config",
    ),
    "diagnostic_telemetry": (
        "diagnostic_receipt",
        "diagnostic_telemetry",
        "diagnostic_telemetry_summary",
    ),
    "ablation_coverage": (
        "ablation_source",
        "training_plan",
        *_ABLATION_CONFIG_SOURCE_IDS,
    ),
    "readiness_gates": (
        "training_readiness",
        "no_submit_contract",
        "current_status",
        "training_plan",
    ),
    "data_contract_rows": (
        "inventory",
        "selection_summary",
        "capacity_small_candidate",
        "training_readiness",
    ),
    "model_contract_rows": (
        "model_config_source",
        "model_source",
        "pretrain_source",
        "small_candidate_config",
    ),
    "diagnostic_contract_rows": (
        "diagnostic_receipt",
        "diagnostic_telemetry_summary",
        "diagnostic_metrics",
        "pretrain_diagnostic_config",
    ),
    "inference_contract_rows": (
        "architecture_document",
        "reconstruction_config",
        "model_source",
        "capacity_small_candidate",
    ),
    "no_submit_contract_rows": ("no_submit_contract",),
    "ablation_arm_rows": (
        "ablation_source",
        "training_plan",
        *_ABLATION_CONFIG_SOURCE_IDS,
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository root: {path}") from exc
    value = relative.as_posix()
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe path: {value}")
    return value


def discover_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"could not find repository root from {start}")


def parse_metadata_root_overrides(values: Iterable[str]) -> dict[str, str]:
    roots = dict(DEFAULT_METADATA_ROOTS)
    for value in values:
        key, separator, path = value.partition("=")
        require(bool(separator), f"metadata root override must be KEY=PATH: {value!r}")
        require(key in roots, f"unknown metadata root key: {key!r}")
        require(bool(path), f"empty metadata root override for {key!r}")
        roots[key] = path
    return roots


class EvidenceReader:
    """Read an exact tracked-file allowlist through safe configurable roots."""

    def __init__(self, repo_root: Path, metadata_roots: Mapping[str, str]) -> None:
        self.repo_root = repo_root.resolve()
        self.roots: dict[str, Path] = {}
        for key, raw in metadata_roots.items():
            path = Path(raw)
            resolved = (self.repo_root / path).resolve() if not path.is_absolute() else path.resolve()
            _repo_relative(resolved, self.repo_root)
            require(resolved.is_dir(), f"metadata root does not exist: {key}={resolved}")
            self.roots[key] = resolved
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.repo_root,
            check=True,
            stdout=subprocess.PIPE,
        )
        self.tracked = {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}
        self.accessed: dict[str, dict[str, str]] = {}

    def resolve(self, source_id: str, root_key: str, relative: str) -> Path:
        require(root_key in self.roots, f"unknown metadata root: {root_key}")
        require(source_id not in self.accessed, f"source read twice: {source_id}")
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe input: {relative}")
        require(pure.suffix in ALLOWED_SUFFIXES, f"disallowed input suffix: {relative}")
        require("sealed" not in relative.lower(), f"sealed-test path is forbidden: {relative}")
        path = (self.roots[root_key] / Path(relative)).resolve()
        try:
            path.relative_to(self.roots[root_key])
        except ValueError as exc:
            raise RuntimeError(f"input escapes metadata root: {relative}") from exc
        require(path.is_file(), f"required evidence is missing: {path}")
        repo_relative = _repo_relative(path, self.repo_root)
        return path

    def read_bytes(self, source_id: str, root_key: str, relative: str) -> bytes:
        path = self.resolve(source_id, root_key, relative)
        data = path.read_bytes()
        self.accessed[source_id] = {
            "id": source_id,
            "path": _repo_relative(path, self.repo_root),
            "provenance": (
                "tracked_repo"
                if _repo_relative(path, self.repo_root) in self.tracked
                else "allowed_reduced_metadata"
            ),
            "sha256": _sha256(data),
        }
        return data

    def read_text(self, source_id: str, root_key: str, relative: str) -> str:
        return self.read_bytes(source_id, root_key, relative).decode("utf-8")

    def read_json(self, source_id: str, root_key: str, relative: str) -> Any:
        return json.loads(self.read_text(source_id, root_key, relative))

    def read_jsonl(self, source_id: str, root_key: str, relative: str) -> list[dict[str, Any]]:
        text = self.read_text(source_id, root_key, relative)
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        require(all(isinstance(row, dict) for row in rows), f"non-object JSONL row: {source_id}")
        return rows

    def read_yaml(self, source_id: str, root_key: str, relative: str) -> dict[str, Any]:
        value = yaml.safe_load(self.read_text(source_id, root_key, relative))
        require(isinstance(value, dict), f"YAML root is not an object: {source_id}")
        return value


def _iso_seconds(start: str, end: str) -> int:
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return int((end_dt - start_dt).total_seconds())


def _parse_markdown_table(text: str, start_heading: str, end_heading: str) -> list[dict[str, str]]:
    require(start_heading in text and end_heading in text, f"missing table bounds: {start_heading}")
    section = text.split(start_heading, 1)[1].split(end_heading, 1)[0]
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    require(len(lines) >= 3, f"no markdown table under {start_heading}")
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells, strict=True)))
    require(rows, f"empty markdown table under {start_heading}")
    return rows


def _extract_seen_event_budgets(plan_text: str) -> list[dict[str, Any]]:
    sections = (
        ("pretraining", "### Pretraining", "### Reconstruction"),
        ("reconstruction", "### Reconstruction", "## Prioritized ablations"),
    )
    result: list[dict[str, Any]] = []
    for track, start, end in sections:
        for row in _parse_markdown_table(plan_text, start, end):
            budget = row.get("Budget", "")
            match = re.search(r"([\d,]+)k seen events", budget)
            if not match:
                continue
            stage_label = row["Stage"]
            lower = stage_label.lower()
            if "screening" in lower:
                stage = "screening"
            elif lower == "candidate":
                stage = "candidate"
            elif "confirmation" in lower:
                stage = "confirmation"
            else:
                continue
            result.append(
                {
                    "track": track,
                    "stage": stage,
                    "stageLabel": stage_label.replace("`", ""),
                    "seenEvents": int(match.group(1).replace(",", "")) * 1000,
                    "budgetText": budget.replace("`", ""),
                    "evidenceStatus": "planned",
                }
            )
    require(len(result) == 6, f"expected six staged event budgets, found {len(result)}")
    return result


def _extract_test_counts(status_text: str) -> dict[str, int]:
    match = re.search(
        r"Canonical complete CPU pytest result:\s*(\d+) passed,\s*(\d+) skipped,\s*(\d+) warnings",
        status_text,
    )
    require(match is not None, "canonical CPU pytest counts are missing")
    passed, skipped, warnings = (int(value) for value in match.groups())
    return {"passed": passed, "skipped": skipped, "warnings": warnings}


def _count_trainable_by_top_module(model: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[name.split(".", 1)[0]] += int(parameter.numel())
    require(bool(counts), "instantiated model has no trainable parameters")
    return dict(sorted(counts.items()))


def _verify_module_location(module_name: str, repo_root: Path) -> None:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    require(module_file is not None, f"module has no source path: {module_name}")
    relative = _repo_relative(Path(module_file), repo_root)
    require(relative.startswith("src/hypertagging/"), f"module imported outside repository: {relative}")


def _build_model_datasets(
    repo_root: Path,
    small_candidate_yaml: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for module_name in (
        "hypertagging.training.model_config",
        "hypertagging.training.pretrain_trainer",
        "hypertagging.models.ablation",
    ):
        _verify_module_location(module_name, repo_root)

    from hypertagging.models.ablation import build_ablation_model
    from hypertagging.preprocessing.pid_filter import PDG_TOKENS
    from hypertagging.training.model_config import MODEL_PRESETS
    from hypertagging.training.pretrain_trainer import ContextualPretrainingModel

    require("small_candidate" in MODEL_PRESETS, "small_candidate preset is missing from code")
    small_architecture = MODEL_PRESETS["small_candidate"]
    yaml_keys = {
        "d_model": "d_model",
        "hyper_dim": "hyper_dim",
        "n_heads": "n_heads",
        "n_context_layers": "n_context_layers",
        "ffn_dim": "ffn_dim",
        "dropout": "dropout",
        "curvature": "curvature",
        "n_queries": "n_queries",
        "max_cardinality": "max_cardinality",
        "tangent_variance_target": "tangent_variance_target",
        "hyper_projection_init_scale": "hyper_projection_init_scale",
        "tangent_scale_mode": "tangent_scale_mode",
        "hyperbolic_level_encoding": "hyperbolic_level_encoding",
    }
    for yaml_key, attribute in yaml_keys.items():
        require(
            small_candidate_yaml.get(yaml_key) == getattr(small_architecture, attribute),
            f"small_candidate YAML/code mismatch for {yaml_key}",
        )

    scale_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    preset_contract_rows: list[dict[str, Any]] = []
    model_counts: dict[str, dict[str, Any]] = {}

    for preset_name in sorted(MODEL_PRESETS):
        architecture = MODEL_PRESETS[preset_name]
        pretraining = ContextualPretrainingModel(
            d_model=architecture.d_model,
            hyper_dim=architecture.hyper_dim,
            curvature=architecture.curvature,
            use_contextual_encoder=True,
            use_physical_relations=True,
            use_hyperbolic_relations=True,
            channel_memory_size=0,
            n_heads=architecture.n_heads,
            n_context_layers=architecture.n_context_layers,
            ffn_dim=architecture.ffn_dim,
            dropout=architecture.dropout,
            hyper_projection_init_scale=architecture.hyper_projection_init_scale,
            tangent_scale_mode=architecture.tangent_scale_mode,
            hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
        )
        reconstruction = build_ablation_model(
            "full_revised",
            n_features=1,
            n_types=len(PDG_TOKENS),
            hidden_dim=architecture.d_model,
            hyper_dim=architecture.hyper_dim,
            n_queries=architecture.n_queries,
            max_cardinality=architecture.max_cardinality,
            n_heads=architecture.n_heads,
            n_context_layers=architecture.n_context_layers,
            curvature=architecture.curvature,
            ffn_dim=architecture.ffn_dim,
            dropout=architecture.dropout,
            n_queries_by_level=architecture.n_queries_by_level,
            max_cardinality_by_level=architecture.max_cardinality_by_level,
            hyper_projection_init_scale=architecture.hyper_projection_init_scale,
            tangent_scale_mode=architecture.tangent_scale_mode,
            hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
            type_conditioned_daughter_relation_bias=(
                architecture.type_conditioned_daughter_relation_bias
            ),
        )
        pretraining_groups = _count_trainable_by_top_module(pretraining)
        reconstruction_groups = _count_trainable_by_top_module(reconstruction)
        pretraining_total = sum(pretraining_groups.values())
        reconstruction_total = sum(reconstruction_groups.values())
        model_counts[preset_name] = {
            "architecture": asdict(architecture),
            "pretrainingTrainableParameters": pretraining_total,
            "pretrainingTopLevelModules": pretraining_groups,
            "reconstructionTrainableParameters": reconstruction_total,
            "reconstructionTopLevelModules": reconstruction_groups,
        }
        scale_rows.append(
            {
                "preset": preset_name,
                "dModel": architecture.d_model,
                "hyperDim": architecture.hyper_dim,
                "contextLayers": architecture.n_context_layers,
                "queries": architecture.n_queries,
                "pretrainingParameters": pretraining_total,
                "reconstructionParameters": reconstruction_total,
                "evidenceStatus": "observed_cpu_instantiation",
            }
        )

        if preset_name == "small_candidate":
            normalized = {
                "pretraining": {
                    "encoder": pretraining_groups.get("encoder", 0),
                    "relationBias": 0,
                    "contextOrRelationHead": pretraining_groups.get("relation_head", 0),
                    "decoder": 0,
                    "levelDecoders": 0,
                    "leafPidHead": pretraining_groups.get("leaf_pid_head", 0),
                    "auxiliaryHeads": (
                        pretraining_groups.get("candidate_correctness_head", 0)
                        + pretraining_groups.get("corruption_type_head", 0)
                    ),
                },
                "reconstruction": {
                    "encoder": reconstruction_groups.get("encoder", 0),
                    "relationBias": reconstruction_groups.get("flat_relation_bias", 0),
                    "contextOrRelationHead": reconstruction_groups.get("flat_contextualizer", 0),
                    "decoder": reconstruction_groups.get("decoder", 0),
                    "levelDecoders": reconstruction_groups.get("level_decoders", 0),
                    "leafPidHead": reconstruction_groups.get("leaf_pid_head", 0),
                    "auxiliaryHeads": 0,
                },
            }
            for surface, groups in normalized.items():
                total = sum(groups.values())
                require(total > 0, f"empty module composition for {surface}")
                row: dict[str, Any] = {
                    "modelSurface": surface,
                    "totalParameters": total,
                    "evidenceStatus": "observed_cpu_instantiation",
                }
                for group, count in groups.items():
                    row[group] = count
                    row[f"{group}Share"] = count / total
                composition_rows.append(row)

    for key in (
        "d_model",
        "hyper_dim",
        "n_heads",
        "n_context_layers",
        "ffn_dim",
        "n_queries",
        "max_cardinality",
        "curvature",
        "hyperbolic_level_encoding",
    ):
        preset_contract_rows.append(
            {
                "field": key,
                "value": small_candidate_yaml[key],
                "evidenceStatus": "observed_config",
            }
        )
    preset_contract_rows.extend(
        [
            {
                "field": "pretraining_trainable_parameters",
                "value": model_counts["small_candidate"]["pretrainingTrainableParameters"],
                "evidenceStatus": "observed_cpu_instantiation",
            },
            {
                "field": "reconstruction_trainable_parameters",
                "value": model_counts["small_candidate"]["reconstructionTrainableParameters"],
                "evidenceStatus": "observed_cpu_instantiation",
            },
        ]
    )
    return model_counts, scale_rows, composition_rows, preset_contract_rows


def _build_data_datasets(
    inventory: Mapping[str, Any],
    summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    require(inventory.get("event_count") == summary.get("inventory_events"), "inventory event mismatch")
    require(inventory.get("shard_count") == summary.get("inventory_shards"), "inventory shard mismatch")
    entries = inventory.get("entries")
    require(isinstance(entries, list) and entries, "inventory entries are missing")
    entry_shards = Counter(str(entry["category"]) for entry in entries)
    entry_events: Counter[str] = Counter()
    for entry in entries:
        entry_events[str(entry["category"])] += int(entry["event_count"])
    require(dict(sorted(entry_shards.items())) == dict(sorted(summary["inventory_category_shards"].items())), "category shard counts disagree")

    category_rows = []
    for category in sorted(entry_shards):
        events = entry_events[category]
        category_rows.append(
            {
                "category": category,
                "events": events,
                "shards": entry_shards[category],
                "eventShare": events / int(inventory["event_count"]),
                "eventsPerShard": events / entry_shards[category],
                "evidenceStatus": "observed_metadata",
            }
        )

    subset_rows = []
    for selection in summary["nested_training_selections"]:
        subset_rows.append(
            {
                "selection": selection["name"],
                "trainEvents": int(selection["train_events"]),
                "trainShards": int(selection["train_shards"]),
                "validationEvents": int(selection["validation_events"]),
                "sealedTestEventsMetadataOnly": int(selection["test_events"]),
                "sealedTestAccessed": False,
                "categoryShardCounts": ",".join(
                    f"{key}:{value}"
                    for key, value in sorted(selection["category_train_shards"].items())
                ),
                "evidenceStatus": "observed_metadata",
            }
        )
    require([row["trainEvents"] for row in subset_rows] == sorted(row["trainEvents"] for row in subset_rows), "training subsets are not nested by size")

    structural_rows = []
    for row in capacity["levels"]:
        distribution = row["daughter_cardinality_distribution"]
        maximum_daughters = max(int(value) for value in distribution)
        structural_rows.append(
            {
                "level": int(row["level"]),
                "motherP50": float(row["mother_count_quantiles"]["p50"]),
                "motherP90": float(row["mother_count_quantiles"]["p90"]),
                "motherP95": float(row["mother_count_quantiles"]["p95"]),
                "motherP99": float(row["mother_count_quantiles"]["p99"]),
                "maximumMothers": int(row["maximum_mothers"]),
                "daughterP50": float(row["daughter_cardinality_quantiles"]["p50"]),
                "daughterP90": float(row["daughter_cardinality_quantiles"]["p90"]),
                "daughterP95": float(row["daughter_cardinality_quantiles"]["p95"]),
                "daughterP99": float(row["daughter_cardinality_quantiles"]["p99"]),
                "maximumDaughters": maximum_daughters,
                "configuredQueries": int(row["configured_queries"]),
                "configuredMaxCardinality": int(row["configured_max_cardinality"]),
                "evidenceStatus": "observed_train_validation_index",
            }
        )
    require([row["level"] for row in structural_rows] == list(range(1, len(structural_rows) + 1)), "capacity levels are not contiguous")

    index_gate = readiness["index_gate"]
    require(index_gate["sealed_test_opened"] is False, "sealed test was opened in index evidence")
    data_contract_rows = [
        {"field": "inventory_events", "value": int(inventory["event_count"]), "status": "observed"},
        {"field": "inventory_shards", "value": int(inventory["shard_count"]), "status": "observed"},
        {"field": "indexed_train_validation_events", "value": int(index_gate["event_count"]), "status": "observed"},
        {"field": "indexed_nodes", "value": int(index_gate["node_count"]), "status": "observed"},
        {"field": "unique_event_uids", "value": int(index_gate["unique_event_uids"]), "status": "observed"},
        {"field": "sealed_test_opened", "value": False, "status": "observed"},
        {"field": "normalizer_scope", "value": index_gate["normalizer_scope"], "status": "observed"},
        {"field": "target_policy", "value": index_gate["target_policy"], "status": "observed"},
    ]
    facts = {
        "inventoryEvents": int(inventory["event_count"]),
        "inventoryShards": int(inventory["shard_count"]),
        "indexedEvents": int(index_gate["event_count"]),
        "indexedNodes": int(index_gate["node_count"]),
        "sealedTestOpened": False,
        "observedReconstructionLevels": len(structural_rows),
        "capacityQueryOverflowCount": int(capacity["query_overflow_count"]),
        "capacityCardinalityOverflowCount": int(capacity["cardinality_overflow_count"]),
    }
    return facts, {
        "production_category_composition": category_rows,
        "subset_ladder": subset_rows,
        "structural_percentiles": structural_rows,
        "data_contract_rows": data_contract_rows,
    }


def _build_curriculum_datasets(
    scientific_config: Mapping[str, Any],
    curriculum_configs: list[tuple[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    phase_steps = scientific_config.get("curriculum_phase_steps")
    require(isinstance(phase_steps, list) and len(phase_steps) == 4, "scientific curriculum must have four phases")
    require(len(curriculum_configs) == len(phase_steps), "curriculum config count mismatch")
    phase_rows = []
    wide_row: dict[str, Any] = {"plan": "35k scientific pretraining", "evidenceStatus": "planned"}
    for index, ((path, config), steps) in enumerate(zip(curriculum_configs, phase_steps, strict=True), start=1):
        active_weights = sorted(
            key for key, value in config.items()
            if key.endswith("_weight") and isinstance(value, (int, float)) and float(value) != 0.0
        )
        phase_name = Path(path).stem.removeprefix(f"pretrain_stage{index}_")
        phase_rows.append(
            {
                "phaseIndex": index,
                "phase": phase_name,
                "steps": int(steps),
                "activeObjectiveCount": len(active_weights),
                "activeObjectives": ",".join(active_weights),
                "evidenceStatus": "planned",
            }
        )
        wide_row[f"phase{index}Steps"] = int(steps)
    require(sum(int(value) for value in phase_steps) == int(scientific_config["max_steps"]), "curriculum steps do not sum to max_steps")
    facts = {
        "plannedSteps": int(scientific_config["max_steps"]),
        "plannedBatchSize": int(scientific_config["batch_size"]),
        "plannedSeenEvents": int(scientific_config["max_steps"]) * int(scientific_config["batch_size"]),
        "plannedValidationEvents": int(scientific_config["validation_events"]),
        "phaseCount": len(phase_rows),
    }
    return facts, {"curriculum_plan": [wide_row], "curriculum_phase_rows": phase_rows}


def _find_diagnostic_attempt(readiness: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    attempts = readiness["slurm_diagnostic_execution"]["v100"]["attempts"]
    matches = [row for row in attempts if str(row.get("job_id")) == job_id]
    require(len(matches) == 1, f"expected one readiness attempt for job {job_id}")
    return matches[0]


def _build_diagnostic_datasets(
    receipt: Mapping[str, Any],
    telemetry_rows: list[Mapping[str, Any]],
    telemetry_summary: Mapping[str, Any],
    metrics_rows: list[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    job_id = str(receipt["slurm"]["job_id"])
    require(job_id == "15745941", f"unexpected diagnostic job: {job_id}")
    require(receipt["status"] == "completed", "diagnostic receipt is not completed")
    require(receipt["trainer_status"] == 0 and receipt["batch_exit_status"] == 0, "diagnostic did not exit cleanly")
    attempt = _find_diagnostic_attempt(readiness, job_id)
    require(attempt["state"] == "COMPLETED" and attempt["trainer_status"] == 0, "readiness diagnostic did not complete")
    require(attempt["model_preset"] == diagnostic_config["model_preset"] == "small_candidate", "diagnostic model preset mismatch")
    require(int(diagnostic_config["max_steps"]) == 4, "diagnostic config is no longer four steps")
    require(list(diagnostic_config["curriculum_phase_steps"]) == [1, 1, 1, 1], "diagnostic phase schedule changed")

    elapsed_seconds = _iso_seconds(str(receipt["started_at"]), str(receipt["completed_at"]))
    require(elapsed_seconds == _iso_seconds(str(attempt["start"]), str(attempt["end"])), "receipt/readiness duration mismatch")
    require(attempt["elapsed"] == "00:02:12" and elapsed_seconds == 132, "diagnostic elapsed contract changed")

    require(len(telemetry_rows) == int(telemetry_summary["sample_count"]), "telemetry sample count mismatch")
    require(receipt["gpu_telemetry"] == telemetry_summary, "receipt telemetry summary mismatch")
    memory_peak = max(int(row["memory_used_mib"]) for row in telemetry_rows)
    utilization_peak = max(int(row["gpu_utilization_percent"]) for row in telemetry_rows)
    temperature_peak = max(int(row["temperature_c"]) for row in telemetry_rows)
    require(memory_peak == int(telemetry_summary["peak_memory_used_mib"]), "telemetry memory peak mismatch")
    require(utilization_peak == int(telemetry_summary["peak_gpu_utilization_percent"]), "telemetry utilization peak mismatch")
    require(temperature_peak == int(telemetry_summary["peak_temperature_c"]), "telemetry temperature peak mismatch")
    allocation = attempt["allocation"]
    require(memory_peak == int(allocation["periodic_telemetry_peak_memory_mib"]), "readiness memory peak mismatch")
    require(utilization_peak == int(allocation["periodic_telemetry_peak_utilization_percent"]), "readiness utilization peak mismatch")
    require(temperature_peak == int(allocation["periodic_telemetry_peak_temperature_c"]), "readiness temperature peak mismatch")

    ordered_telemetry = sorted(telemetry_rows, key=lambda row: str(row["timestamp"]))
    first_timestamp = str(ordered_telemetry[0]["timestamp"])
    telemetry_dataset = []
    for index, row in enumerate(ordered_telemetry, start=1):
        telemetry_dataset.append(
            {
                "sample": index,
                "elapsedSeconds": _iso_seconds(first_timestamp, str(row["timestamp"])),
                "memoryUsedMiB": int(row["memory_used_mib"]),
                "gpuUtilizationPercent": int(row["gpu_utilization_percent"]),
                "temperatureC": int(row["temperature_c"]),
                "gpuModel": str(row["gpu_name"]),
                "evidenceStatus": "observed_diagnostic",
            }
        )

    validation_rows = [row for row in metrics_rows if row.get("split") == "validation"]
    require(len(validation_rows) == 1, "expected one diagnostic validation metrics row")
    validation = validation_rows[0]
    metric_contract = attempt["metrics"]
    for metric_key in (
        "validation_batches",
        "validation_events",
        "validation_named_view_evaluations",
        "validation_full_training_objective",
    ):
        require(float(validation[metric_key]) == float(metric_contract[metric_key]), f"diagnostic metric mismatch: {metric_key}")
    phase_indices = list(attempt["checkpoint"]["phase_indices_by_step"])
    require(phase_indices == [0, 1, 2, 3], "diagnostic did not record all four phases")

    diagnostic_contract_rows = [
        {"field": "job_id", "value": job_id, "status": "observed_diagnostic"},
        {"field": "model_preset", "value": attempt["model_preset"], "status": "observed_diagnostic"},
        {"field": "elapsed_seconds", "value": elapsed_seconds, "status": "observed_diagnostic"},
        {"field": "optimizer_steps", "value": int(attempt["checkpoint"]["step"]), "status": "observed_diagnostic"},
        {"field": "curriculum_phases_entered", "value": len(phase_indices), "status": "observed_diagnostic"},
        {"field": "validation_batches", "value": int(metric_contract["validation_batches"]), "status": "observed_diagnostic"},
        {"field": "validation_events", "value": int(metric_contract["validation_events"]), "status": "observed_diagnostic"},
        {"field": "validation_named_views", "value": int(metric_contract["validation_named_view_evaluations"]), "status": "observed_diagnostic"},
        {"field": "validation_event_view_evaluations", "value": int(metric_contract["validation_progress_event_views"]), "status": "observed_diagnostic"},
        {"field": "telemetry_samples", "value": len(telemetry_dataset), "status": "observed_diagnostic"},
        {"field": "peak_memory_mib", "value": memory_peak, "status": "observed_diagnostic"},
        {"field": "peak_gpu_utilization_percent", "value": utilization_peak, "status": "observed_diagnostic"},
        {"field": "peak_temperature_c", "value": temperature_peak, "status": "observed_diagnostic"},
        {"field": "scientific_claim", "value": False, "status": "observed_boundary"},
    ]
    facts = {
        "jobId": job_id,
        "modelPreset": attempt["model_preset"],
        "elapsedSeconds": elapsed_seconds,
        "steps": int(attempt["checkpoint"]["step"]),
        "phaseIndices": phase_indices,
        "validationBatches": int(metric_contract["validation_batches"]),
        "validationEvents": int(metric_contract["validation_events"]),
        "validationNamedViews": int(metric_contract["validation_named_view_evaluations"]),
        "validationEventViewEvaluations": int(metric_contract["validation_progress_event_views"]),
        "telemetrySamples": len(telemetry_dataset),
        "peakMemoryMiB": memory_peak,
        "peakGpuUtilizationPercent": utilization_peak,
        "peakTemperatureC": temperature_peak,
        "scientificClaimsAllowed": False,
    }
    return facts, {
        "diagnostic_telemetry": telemetry_dataset,
        "diagnostic_contract_rows": diagnostic_contract_rows,
    }


def _build_inference_datasets(
    reconstruction_config: Mapping[str, Any],
    capacity: Mapping[str, Any],
    architecture_text: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    for phrase in (
        "p4(mother) = sum p4(selected reconstructed daughters)",
        "Greedy exclusivity remains the production default",
        "evaluation-only",
        "Scheduled sampling is seeded and reproducible",
    ):
        require(phrase in architecture_text, f"inference contract phrase is missing: {phrase}")
    maximum_mothers = max(int(row["maximum_mothers"]) for row in capacity["levels"])
    maximum_daughters = max(
        max(int(value) for value in row["daughter_cardinality_distribution"])
        for row in capacity["levels"]
    )
    configured_queries = int(capacity["architecture"]["n_queries"])
    configured_cardinality = int(capacity["architecture"]["max_cardinality"])
    require(capacity["query_overflow_count"] == 0, "query capacity has overflow")
    require(capacity["cardinality_overflow_count"] == 0, "cardinality capacity has overflow")
    capacity_rows = [
        {
            "control": "mother queries",
            "configuredCapacity": configured_queries,
            "observedMaximum": maximum_mothers,
            "capacityMargin": configured_queries - maximum_mothers,
            "evidenceStatus": "observed_train_validation_index",
        },
        {
            "control": "daughter cardinality",
            "configuredCapacity": configured_cardinality,
            "observedMaximum": maximum_daughters,
            "capacityMargin": configured_cardinality - maximum_daughters,
            "evidenceStatus": "observed_train_validation_index",
        },
    ]
    inference_contract_rows = [
        {"control": "training topology source", "value": "truth-guided supervision", "status": "implemented"},
        {"control": "inference mother source", "value": "predicted mothers", "status": "implemented"},
        {"control": "mother four-momentum", "value": "exact daughter sum", "status": "implemented"},
        {"control": "default exclusivity", "value": "greedy", "status": "implemented"},
        {"control": "bounded set packing", "value": "evaluation only", "status": "implemented_diagnostic"},
        {"control": "bounded beam", "value": "evaluation only", "status": "implemented_diagnostic"},
        {"control": "rollout checkpoint metric", "value": reconstruction_config["best_metric"], "status": "planned_config"},
        {"control": "minimum tree validity", "value": float(reconstruction_config["rollout_min_tree_validity"]), "status": "planned_config"},
        {"control": "minimum p4 closure", "value": float(reconstruction_config["rollout_min_p4_closure"]), "status": "planned_config"},
        {"control": "p4 tolerance", "value": float(reconstruction_config["rollout_p4_tolerance"]), "status": "planned_config"},
        {"control": "maximum recursive-source conflicts", "value": int(reconstruction_config["rollout_max_recursive_source_conflicts"]), "status": "planned_config"},
    ]
    facts = {
        "motherP4": "daughter_sum",
        "teacherForcingMothers": "truth_guided_topology_reco_features",
        "inferenceMothers": "predicted",
        "defaultExclusivity": "greedy",
        "setPacking": "evaluation_only",
        "boundedBeam": "evaluation_only",
        "configuredQueries": configured_queries,
        "observedMaximumMothers": maximum_mothers,
        "configuredMaxCardinality": configured_cardinality,
        "observedMaximumDaughters": maximum_daughters,
    }
    return facts, {
        "inference_capacity_controls": capacity_rows,
        "inference_contract_rows": inference_contract_rows,
    }


def _build_staged_budget_dataset(plan_text: str) -> list[dict[str, Any]]:
    long_rows = _extract_seen_event_budgets(plan_text)
    values: dict[str, dict[str, int]] = defaultdict(dict)
    for row in long_rows:
        values[row["stage"]][row["track"]] = int(row["seenEvents"])
    result = []
    for stage in ("screening", "candidate", "confirmation"):
        require(set(values[stage]) == {"pretraining", "reconstruction"}, f"missing staged budget for {stage}")
        result.append(
            {
                "stage": stage,
                "pretrainingSeenEvents": values[stage]["pretraining"],
                "reconstructionSeenEvents": values[stage]["reconstruction"],
                "evidenceStatus": "planned",
            }
        )
    return result


def _one_hot_gate(label: str, state: str, evidence_status: str, detail: str) -> dict[str, Any]:
    allowed = ("observedComplete", "plannedRequired", "blocked", "unavailable", "unknown")
    require(state in allowed, f"unknown gate state: {state}")
    row: dict[str, Any] = {
        "gate": label,
        "detail": detail,
        "evidenceStatus": evidence_status,
    }
    for field in allowed:
        row[field] = int(field == state)
    return row


def _build_readiness_datasets(
    summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    no_submit: Mapping[str, Any],
    test_counts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    require(summary["reduced_campaign_contract"] == "validated", "reduced campaign contract is not validated")
    require(readiness["index_gate"]["status"] == "complete", "index gate is incomplete")
    require(readiness["capacity_gate"]["status"] == "complete", "capacity gate is incomplete")
    require(readiness["slurm_diagnostic_execution"]["v100"]["status"] == "complete", "V100 diagnostic gate is incomplete")
    require(readiness["scientific_submission_allowed"] is False, "scientific submission unexpectedly allowed")
    require(no_submit["submission_authorized"] is False, "no-submit contract unexpectedly authorized")
    require(no_submit["submission_performed"] is False, "no-submit contract records a submission")
    require(no_submit["sealed_test_role_access"] == "forbidden", "no-submit contract permits sealed test")
    blockers = list(no_submit["scientific_submission_blockers"])
    require(len(blockers) == 3, f"expected three scientific blockers, found {len(blockers)}")
    blocker_text = " ".join(blockers).lower()
    for phrase in ("production source object", "fresh exact in-allocation", "commit and review"):
        require(phrase in blocker_text, f"expected blocker is missing: {phrase}")
    h200 = readiness["slurm_diagnostic_execution"]["h200"]
    require(h200["state"] == "not_submitted_exact_gres_unavailable", "H200 state changed")
    require(h200["submission_performed"] is False, "H200 submission was performed")

    gate_rows = [
        _one_hot_gate("reduced 1M publication inventory", "observedComplete", "observed", summary["reduced_campaign_contract"]),
        _one_hot_gate("35k train plus validation index", "observedComplete", "observed", readiness["index_gate"]["status"]),
        _one_hot_gate("small_candidate capacity", "observedComplete", "observed", readiness["capacity_gate"]["status"]),
        _one_hot_gate("CPU test suite", "observedComplete", "observed", f"{test_counts['passed']} passed"),
        _one_hot_gate("V100 small_candidate diagnostic", "observedComplete", "observed_diagnostic", readiness["slurm_diagnostic_execution"]["v100"]["status"]),
        _one_hot_gate("production source object/tree", "blocked", "observed_blocker", blockers[0]),
        _one_hot_gate("fresh in-allocation preflight", "plannedRequired", "planned", blockers[1]),
        _one_hot_gate("clean review/tag/render", "plannedRequired", "planned", blockers[2]),
        _one_hot_gate("exact H200 resource", "unavailable", "observed_metadata", h200["state"]),
        _one_hot_gate("long convergence and held-out quality", "unknown", "unknown", "no full scientific run or sealed-test evaluation"),
    ]
    no_submit_rows = [
        {"field": "contract_version", "value": no_submit["contract_version"], "status": "observed"},
        {"field": "mode", "value": no_submit["mode"], "status": "observed"},
        {"field": "verification_scope", "value": no_submit["verification_scope"], "status": "observed"},
        {"field": "selection_manifest", "value": no_submit["selection_manifest"], "status": "observed"},
        {"field": "model_preset", "value": "small_candidate", "status": "planned_contract"},
        {"field": "seed", "value": int(no_submit["seed"]), "status": "planned_contract"},
        {"field": "submission_authorized", "value": False, "status": "observed"},
        {"field": "submission_performed", "value": False, "status": "observed"},
        {"field": "sealed_test_role_access", "value": no_submit["sealed_test_role_access"], "status": "observed"},
    ]
    for index, blocker in enumerate(blockers, start=1):
        no_submit_rows.append(
            {"field": f"scientific_blocker_{index}", "value": blocker, "status": "observed_blocker"}
        )
    facts = {
        "scientificSubmissionAllowed": False,
        "noSubmitContract": True,
        "blockerCount": len(blockers),
        "blockers": blockers,
        "h200State": h200["state"],
        "h100Substitution": "unknown_from_selected_evidence",
        "cpuTests": dict(test_counts),
    }
    return facts, {"readiness_gates": gate_rows, "no_submit_contract_rows": no_submit_rows}


def _ablation_row(
    arm_id: str,
    family: str,
    arm: str,
    status: str,
    implementation: str,
    evidence_status: str,
) -> dict[str, Any]:
    return {
        "armId": arm_id,
        "family": family,
        "arm": arm,
        "status": status,
        "implementation": implementation,
        "evidenceStatus": evidence_status,
    }


def _build_ablation_datasets(
    configs: Mapping[str, Mapping[str, Any]],
    plan_text: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    required_plan_phrases = (
        "Physical relation bias off/on; then optional hyperbolic relation refinement off/on",
        "Scheduled sampling endpoint 0/10/25/50%",
        "Level encoding none/Euclidean/bounded tangent and radius target alternatives",
        "Never cross-product ablations.",
    )
    for phrase in required_plan_phrases:
        require(phrase in plan_text, f"ablation plan phrase is missing: {phrase}")

    def config_name(path: str) -> str:
        config = configs[path]
        return str(config.get("name", Path(path).stem))

    for path in ABLATION_CONFIG_PATHS:
        require(path in configs, f"required ablation config was not read: {path}")

    rows = [
        _ablation_row("flat_baseline", "representation", "flat baseline", "implemented", config_name("ablations/flat_baseline.yaml"), "observed_code_config"),
        _ablation_row("heterogeneous_only", "representation", "heterogeneous Euclidean adapters without context", "implemented", config_name("ablations/heterogeneous_only.yaml"), "observed_code_config"),
        _ablation_row("contextual_euclidean", "representation", "contextual Euclidean", "implemented", config_name("ablations/contextual_euclidean.yaml"), "observed_code_config"),
        _ablation_row("hyperbolic_node_geometry", "representation", "hyperbolic node projection/objectives", "implemented", config_name("ablations/contextual_hyperbolic_parent_lca.yaml"), "observed_code_config"),
        _ablation_row("context_removal", "context", "remove contextual encoder", "implemented", config_name("ablations/heterogeneous_only.yaml"), "observed_code_config"),
        _ablation_row("physical_relation_bias_only", "relation bias", "physical relation bias on, hyperbolic refinement off", "planned_no_named_arm", "model API supports separate flags; no named YAML arm", "planned"),
        _ablation_row("physical_plus_hyperbolic_relation", "relation bias", "physical plus hyperbolic relation refinement", "implemented", config_name("ablations/plus_hyperbolic_relation_attention.yaml"), "observed_code_config"),
    ]
    for path in (
        "ablations/level_encoding_none.yaml",
        "ablations/level_encoding_euclidean.yaml",
        "ablations/level_encoding_bounded_tangent.yaml",
    ):
        rows.append(_ablation_row(Path(path).stem, "level geometry", config_name(path), "implemented", path, "observed_config"))
    for path in (
        "ablations/radius_generation_height.yaml",
        "ablations/radius_exact_root_depth.yaml",
        "ablations/radius_weak_or_learned.yaml",
    ):
        rows.append(_ablation_row(Path(path).stem, "radius geometry", config_name(path), "implemented", path, "observed_config"))
    for path in (
        "ablations/learned_bounded_tangent_scale.yaml",
        "ablations/lower_radius_tangent_scale.yaml",
    ):
        rows.append(_ablation_row(Path(path).stem, "tangent scale", config_name(path), "implemented", path, "observed_config"))
    rows.append(_ablation_row("scheduled_sampling_mechanism", "exposure bias", "scheduled sampling plus free rollout", "implemented", config_name("ablations/plus_scheduled_sampling.yaml"), "observed_code_config"))
    for endpoint in (0, 10, 25, 50):
        rows.append(
            _ablation_row(
                f"scheduled_sampling_{endpoint}",
                "exposure bias endpoint",
                f"scheduled sampling endpoint {endpoint}%",
                "planned_scientific_comparison",
                "paired 35k screen; winner and 0% control advance",
                "planned",
            )
        )

    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = "implemented" if row["status"] == "implemented" else "planned"
        coverage[row["family"]][category] += 1
    coverage_rows = [
        {
            "family": family,
            "implementedArms": counts["implemented"],
            "plannedArms": counts["planned"],
            "totalArms": counts["implemented"] + counts["planned"],
            "evidenceStatus": "mixed_observed_planned" if counts["planned"] else "observed_config",
        }
        for family, counts in sorted(coverage.items())
    ]
    facts = {
        "armCount": len(rows),
        "implementedArmCount": sum(row["status"] == "implemented" for row in rows),
        "plannedArmCount": sum(row["status"] != "implemented" for row in rows),
        "physicalOnlyNamedArm": False,
        "crossProductsPlanned": False,
    }
    return facts, {"ablation_coverage": coverage_rows, "ablation_arm_rows": rows}


DATASET_CONTRACTS: dict[str, dict[str, str]] = {
    "production_category_composition": {"laterUse": "chart", "grain": "production category", "status": "observed"},
    "structural_percentiles": {"laterUse": "chart", "grain": "reconstruction level", "status": "observed"},
    "subset_ladder": {"laterUse": "chart", "grain": "nested selection", "status": "observed metadata"},
    "model_parameter_scale": {"laterUse": "chart", "grain": "CPU-instantiated preset", "status": "observed"},
    "model_module_composition": {"laterUse": "chart", "grain": "small_candidate model surface", "status": "observed"},
    "curriculum_plan": {"laterUse": "chart", "grain": "35k scientific plan", "status": "planned"},
    "diagnostic_telemetry": {"laterUse": "chart", "grain": "15-second telemetry sample", "status": "observed diagnostic"},
    "inference_capacity_controls": {"laterUse": "chart", "grain": "capacity control", "status": "observed train+validation index"},
    "staged_training_budgets": {"laterUse": "chart", "grain": "promotion stage", "status": "planned"},
    "readiness_gates": {"laterUse": "chart", "grain": "readiness gate", "status": "mixed"},
    "ablation_coverage": {"laterUse": "chart", "grain": "ablation family", "status": "mixed"},
    "data_contract_rows": {"laterUse": "table", "grain": "data contract field", "status": "observed"},
    "model_contract_rows": {"laterUse": "table", "grain": "small_candidate field", "status": "observed"},
    "diagnostic_contract_rows": {"laterUse": "table", "grain": "diagnostic receipt field", "status": "observed diagnostic"},
    "inference_contract_rows": {"laterUse": "table", "grain": "inference control", "status": "mixed"},
    "no_submit_contract_rows": {"laterUse": "table", "grain": "no-submit contract field", "status": "mixed"},
    "ablation_arm_rows": {"laterUse": "table", "grain": "ablation arm", "status": "mixed"},
    "curriculum_phase_rows": {"laterUse": "supporting evidence", "grain": "curriculum phase", "status": "planned"},
}


def _merge_datasets(target: dict[str, list[dict[str, Any]]], incoming: Mapping[str, list[dict[str, Any]]]) -> None:
    overlap = set(target) & set(incoming)
    require(not overlap, f"duplicate dataset ids: {sorted(overlap)}")
    target.update(incoming)


def generate(repo_root: Path, metadata_roots: Mapping[str, str]) -> dict[str, Any]:
    reader = EvidenceReader(repo_root, metadata_roots)

    inventory = reader.read_json("inventory", *REQUIRED_INPUTS["inventory"])
    selection_summary = reader.read_json("selection_summary", *REQUIRED_INPUTS["selection_summary"])
    readiness = reader.read_json("training_readiness", *REQUIRED_INPUTS["training_readiness"])
    capacity = reader.read_json("capacity_small_candidate", *REQUIRED_INPUTS["capacity_small_candidate"])
    receipt = reader.read_json("diagnostic_receipt", *REQUIRED_INPUTS["diagnostic_receipt"])
    telemetry = reader.read_jsonl("diagnostic_telemetry", *REQUIRED_INPUTS["diagnostic_telemetry"])
    telemetry_summary = reader.read_json(
        "diagnostic_telemetry_summary", *REQUIRED_INPUTS["diagnostic_telemetry_summary"]
    )
    metrics = reader.read_jsonl("diagnostic_metrics", *REQUIRED_INPUTS["diagnostic_metrics"])
    no_submit = reader.read_json("no_submit_contract", *REQUIRED_INPUTS["no_submit_contract"])
    scientific_config = reader.read_yaml(
        "pretrain_scientific_config", *REQUIRED_INPUTS["pretrain_scientific_config"]
    )
    diagnostic_config = reader.read_yaml(
        "pretrain_diagnostic_config", *REQUIRED_INPUTS["pretrain_diagnostic_config"]
    )
    reconstruction_config = reader.read_yaml(
        "reconstruction_config", *REQUIRED_INPUTS["reconstruction_config"]
    )
    small_candidate_yaml = reader.read_yaml(
        "small_candidate_config", *REQUIRED_INPUTS["small_candidate_config"]
    )
    plan_text = reader.read_text("training_plan", *REQUIRED_INPUTS["training_plan"])
    status_text = reader.read_text("current_status", *REQUIRED_INPUTS["current_status"])
    architecture_text = reader.read_text(
        "architecture_document", *REQUIRED_INPUTS["architecture_document"]
    )
    model_config_text = reader.read_text(
        "model_config_source", *REQUIRED_INPUTS["model_config_source"]
    )
    model_text = reader.read_text("model_source", *REQUIRED_INPUTS["model_source"])
    ablation_text = reader.read_text("ablation_source", *REQUIRED_INPUTS["ablation_source"])
    pretrain_text = reader.read_text("pretrain_source", *REQUIRED_INPUTS["pretrain_source"])

    require('"small_candidate": ModelArchitecture' in model_config_text, "small_candidate preset source is missing")
    require("class LevelAutoregressiveReconstructor" in model_text, "reconstruction model source is missing")
    require('"full_revised": AblationConfig' in ablation_text, "full_revised ablation source is missing")
    require("class ContextualPretrainingModel" in pretrain_text, "pretraining model source is missing")

    curriculum_configs: list[tuple[str, Mapping[str, Any]]] = []
    for index, (root_key, relative) in enumerate(CURRICULUM_CONFIGS, start=1):
        curriculum_configs.append(
            (relative, reader.read_yaml(f"curriculum_stage_{index}", root_key, relative))
        )
    ablation_configs: dict[str, Mapping[str, Any]] = {}
    for relative in ABLATION_CONFIG_PATHS:
        source_id = f"ablation_config_{Path(relative).stem}"
        ablation_configs[relative] = reader.read_yaml(source_id, "configs", relative)

    facts: dict[str, Any] = {}
    datasets: dict[str, list[dict[str, Any]]] = {}

    data_facts, data_datasets = _build_data_datasets(
        inventory, selection_summary, readiness, capacity
    )
    facts["data"] = data_facts
    _merge_datasets(datasets, data_datasets)

    model_facts, scale_rows, composition_rows, model_contract_rows = _build_model_datasets(
        repo_root, small_candidate_yaml
    )
    facts["models"] = model_facts
    _merge_datasets(
        datasets,
        {
            "model_parameter_scale": scale_rows,
            "model_module_composition": composition_rows,
            "model_contract_rows": model_contract_rows,
        },
    )

    curriculum_facts, curriculum_datasets = _build_curriculum_datasets(
        scientific_config, curriculum_configs
    )
    facts["curriculum"] = curriculum_facts
    _merge_datasets(datasets, curriculum_datasets)

    diagnostic_facts, diagnostic_datasets = _build_diagnostic_datasets(
        receipt,
        telemetry,
        telemetry_summary,
        metrics,
        readiness,
        diagnostic_config,
    )
    facts["diagnostic"] = diagnostic_facts
    _merge_datasets(datasets, diagnostic_datasets)

    inference_facts, inference_datasets = _build_inference_datasets(
        reconstruction_config, capacity, architecture_text
    )
    facts["inference"] = inference_facts
    _merge_datasets(datasets, inference_datasets)

    test_counts = _extract_test_counts(status_text)
    readiness_facts, readiness_datasets = _build_readiness_datasets(
        selection_summary, readiness, no_submit, test_counts
    )
    h100_contract_paths = sorted(
        path for path in reader.tracked
        if path.startswith("artifacts/") and "h100" in path.lower() and path.endswith("job-contract.json")
    )
    readiness_facts["trackedH100JobContractCount"] = len(h100_contract_paths)
    readiness_facts["h100Substitution"] = (
        "not_observed_in_tracked_job_contracts" if not h100_contract_paths else "tracked_contract_present"
    )
    facts["readiness"] = readiness_facts
    _merge_datasets(datasets, readiness_datasets)

    ablation_facts, ablation_datasets = _build_ablation_datasets(ablation_configs, plan_text)
    facts["ablations"] = ablation_facts
    _merge_datasets(datasets, ablation_datasets)
    datasets["staged_training_budgets"] = _build_staged_budget_dataset(plan_text)

    require(set(datasets) == set(DATASET_CONTRACTS), f"dataset contract mismatch: {sorted(set(datasets) ^ set(DATASET_CONTRACTS))}")
    for dataset_id, rows in datasets.items():
        require(isinstance(rows, list) and rows, f"dataset is empty: {dataset_id}")
        require(len(rows) <= 50, f"dataset exceeds bounded row limit: {dataset_id}")

    sources = [reader.accessed[source_id] for source_id in sorted(reader.accessed)]
    require(len(sources) == len(reader.accessed), "duplicate source ids")
    require(
        all(not Path(source["path"]).is_absolute() and ".." not in PurePosixPath(source["path"]).parts for source in sources),
        "unsafe source path generated",
    )

    return {
        "version": EVIDENCE_VERSION,
        "title": REPORT_TITLE,
        "reportDate": REPORT_DATE,
        "generatedAt": GENERATED_AT,
        "status": "partial",
        "evidenceBoundary": {
            "numberLabels": {
                "observed": "directly extracted or CPU-instantiated from tracked evidence",
                "planned": "specified by a tracked execution/configuration plan but not run",
                "unknown": "required scientific evidence is absent",
            },
            "gpuAccessPerformed": False,
            "schedulerAccessPerformed": False,
            "jobMutationPerformed": False,
            "sealedTestPayloadAccessed": False,
            "modelCheckpointOpened": False,
            "allowedInputSuffixes": sorted(ALLOWED_SUFFIXES),
            "accessedSourceCount": len(sources),
        },
        "facts": facts,
        "datasetContracts": DATASET_CONTRACTS,
        "datasets": {key: datasets[key] for key in sorted(datasets)},
        "sources": sources,
        "unknowns": [
            {
                "id": "full_convergence",
                "status": "unknown",
                "statement": "No long GPU convergence result is present in permitted tracked evidence.",
            },
            {
                "id": "held_out_reconstruction_quality",
                "status": "unknown",
                "statement": "The sealed test remains closed; held-out reconstruction quality is absent.",
            },
            {
                "id": "physical_bias_only_named_arm",
                "status": "planned",
                "statement": "The model exposes separate physical/hyperbolic flags, but no named physical-only ablation config exists.",
            },
        ],
    }


def _dataset_source_id(dataset_name: str) -> str:
    require(
        SAFE_DATASET_NAME.fullmatch(dataset_name) is not None,
        f"unsafe artifact dataset identifier: {dataset_name!r}",
    )
    return f"evidence_dataset_{dataset_name}"


def _dataset_select_sql(dataset_name: str) -> str:
    _dataset_source_id(dataset_name)
    return (
        "SELECT value AS row_json\n"
        f"FROM json_each(:evidence_json, '$.datasets.{dataset_name}')\n"
        "ORDER BY CAST(key AS INTEGER)"
    )


def _execute_dataset_select(
    evidence: Mapping[str, Any], dataset_name: str
) -> tuple[str, list[dict[str, Any]]]:
    require(dataset_name in evidence["datasets"], f"artifact dataset is missing: {dataset_name}")
    expected_rows = evidence["datasets"][dataset_name]
    require(isinstance(expected_rows, list), f"artifact dataset is not an array: {dataset_name}")
    sql = _dataset_select_sql(dataset_name)
    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        with sqlite3.connect(":memory:") as connection:
            result = connection.execute(sql, {"evidence_json": evidence_json}).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"SQLite JSON1 extraction failed for artifact dataset {dataset_name}: {exc}"
        ) from exc
    try:
        selected_rows = [json.loads(row_json) for (row_json,) in result]
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"SQLite JSON1 returned invalid row JSON for artifact dataset {dataset_name}"
        ) from exc
    require(
        selected_rows == expected_rows,
        f"SQLite JSON1 rows differ from canonical evidence dataset: {dataset_name}",
    )
    return sql, selected_rows


def _artifact_sources(
    evidence: Mapping[str, Any], bound_dataset_names: Iterable[str]
) -> list[dict[str, Any]]:
    source_paths = [source["path"] for source in evidence["sources"]]
    source_by_id = {source["id"]: source for source in evidence["sources"]}
    bound_datasets = tuple(sorted(set(bound_dataset_names)))
    require(
        set(bound_datasets) == set(DATASET_SUPPORT_SOURCE_IDS),
        "chart/table dataset provenance map does not match bound artifact datasets",
    )
    sources = [
        {
            "id": "report_evidence",
            "label": "Deterministic report evidence snapshot",
            "path": REPORT_EVIDENCE_PATH,
            "query": {
                "engine": "repository",
                "language": "python",
                "description": "Bounded deterministic extraction; chart and table rows are copied unchanged from evidence.json.",
                "tables_used": source_paths,
                "filters": [
                    "explicit exact-file allowlist",
                    "tracked repository evidence or permitted reduced-dataset metadata only",
                    "sealed-test payloads excluded",
                ],
                "metric_definitions": [
                    "observed = extracted or CPU-instantiated from permitted evidence",
                    "planned = specified in tracked configuration or execution plan",
                    "unknown = required scientific result absent",
                ],
            },
        }
    ]
    for dataset_name in bound_datasets:
        sql, selected_rows = _execute_dataset_select(evidence, dataset_name)
        support_ids = DATASET_SUPPORT_SOURCE_IDS[dataset_name]
        missing_source_ids = sorted(set(support_ids) - set(source_by_id))
        require(
            not missing_source_ids,
            f"dataset {dataset_name} references missing upstream sources: {missing_source_ids}",
        )
        support_paths = [source_by_id[source_id]["path"] for source_id in support_ids]
        sources.append(
            {
                "id": _dataset_source_id(dataset_name),
                "label": f"SQL-verified evidence dataset: {dataset_name}",
                "path": REPORT_EVIDENCE_PATH,
                "query": {
                    "engine": "sqlite3",
                    "dialect": "SQLite JSON1",
                    "language": "SQL",
                    "sql": sql,
                    "description": (
                        f"Executed against the canonical evidence JSON and verified to return "
                        f"all {len(selected_rows)} {dataset_name} rows in deterministic array order."
                    ),
                    "parameters": {
                        "evidence_json": (
                            "UTF-8 JSON text for the complete canonical evidence.json document; "
                            "the generator binds this value to :evidence_json."
                        )
                    },
                    "tables_used": [REPORT_EVIDENCE_PATH, *support_paths],
                    "filters": [
                        f"JSON path $.datasets.{dataset_name}",
                        "ORDER BY CAST(json_each.key AS INTEGER) preserves canonical array order",
                    ],
                    "metric_definitions": [
                        "row_json is the unchanged JSON value for one canonical evidence dataset row"
                    ],
                },
            }
        )
    for source in evidence["sources"]:
        sources.append(
            {
                "id": source["id"],
                "label": source["id"].replace("_", " "),
                "path": source["path"],
                "query": {
                    "engine": "repository",
                    "id": source["sha256"],
                    "description": f"{source['provenance']} input verified by SHA-256.",
                    "tables_used": [source["path"]],
                },
            }
        )
    return sources


def _encoding(
    field: str,
    kind: str,
    label: str,
    *,
    unit: str | None = None,
    value_format: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "field": field,
        "type": kind,
        "aggregate": "none",
        "label": label,
    }
    if unit is not None:
        result["unit"] = unit
    if value_format is not None:
        result["format"] = value_format
    return result


def _multi_encoding(
    fields: list[str],
    label: str,
    *,
    unit: str,
    value_format: str,
) -> dict[str, Any]:
    require(len(fields) > 1, "multi-measure chart encoding requires at least two fields")
    return {
        "fields": fields,
        "type": "quantitative",
        "aggregate": "none",
        "label": label,
        "unit": unit,
        "format": value_format,
    }


def _charts() -> list[dict[str, Any]]:
    full_surface = {"surface": "explorer", "compact": False, "showControls": False, "viewMode": "both"}
    zero_y = [{"axis": "y", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "solid"}]
    zero_x = [{"axis": "x", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "solid"}]
    return [
        {
            "id": "production-category-composition",
            "title": "Reduced production category composition",
            "subtitle": "Observed metadata for 1,000,000 events across 200 fixed-size shards",
            "intent": "composition",
            "question": "How is the reduced production inventory distributed across categories?",
            "rationale": "Horizontal bars make seven category labels readable and expose the documented mixed/uubar skew without implying representativeness.",
            "comparisonContext": {"baseline": "zero", "denominator": "1,000,000 inventory events", "grain": "production category", "normalization": "none", "semanticFamily": "observed composition", "unit": "events"},
            "type": "horizontalBar",
            "dataset": "production_category_composition",
            "sourceId": _dataset_source_id("production_category_composition"),
            "encodings": {
                "x": _encoding("category", "nominal", "Production category"),
                "y": _encoding("events", "quantitative", "Events", unit="events", value_format="number"),
                "label": _encoding("events", "quantitative", "Events", unit="events", value_format="number"),
                "tooltip": [
                    _encoding("shards", "quantitative", "Shards", unit="shards", value_format="number"),
                    _encoding("eventShare", "quantitative", "Inventory share", unit="share", value_format="percent"),
                ],
            },
            "xAxisTitle": "Observed events",
            "yAxisTitle": "Production category",
            "palette": {"kind": "semantic", "name": "blue-observed"},
            "valueFormat": "number",
            "unit": "events",
            "layout": "full",
            "maxRows": 7,
            "referenceLines": zero_x,
            "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": full_surface,
        },
        {
            "id": "structural-percentiles",
            "title": "Mother-count percentiles by reconstruction level",
            "subtitle": "Observed train+validation complete-only index; six populated reconstruction levels",
            "intent": "distribution",
            "question": "Where does set cardinality concentrate across reconstruction levels?",
            "rationale": "An ordered multi-series line preserves the level sequence and compares four observed percentiles without treating the hierarchy as time.",
            "comparisonContext": {"baseline": "level-specific median", "denominator": "eligible events in the 85,000-event train+validation index", "grain": "reconstruction level", "normalization": "quantiles within level", "semanticFamily": "structural distribution", "unit": "mothers per event"},
            "type": "line",
            "dataset": "structural_percentiles",
            "sourceId": _dataset_source_id("structural_percentiles"),
            "encodings": {
                "x": _encoding("level", "ordinal", "Reconstruction level"),
                "y": _multi_encoding(
                    ["motherP50", "motherP90", "motherP95", "motherP99"],
                    "Mother count",
                    unit="mothers per event",
                    value_format="number",
                ),
                "tooltip": [
                    _encoding("maximumMothers", "quantitative", "Observed maximum", unit="mothers per event", value_format="number"),
                    _encoding("maximumDaughters", "quantitative", "Maximum daughter cardinality", unit="daughters", value_format="number"),
                ],
            },
            "xAxisTitle": "Reconstruction level",
            "yAxisTitle": "Observed mothers per event",
            "palette": {"kind": "categorical", "name": "blue-purple-orange-pink-percentiles"},
            "valueFormat": "number",
            "unit": "mothers per event",
            "layout": "full",
            "maxRows": 6,
            "referenceLines": zero_y,
            "settings": {"sort": "none", "showPoints": "always", "showValues": False},
            "surface": full_surface,
        },
        {
            "id": "subset-ladder",
            "title": "Nested training subset ladder",
            "subtitle": "Observed manifest metadata; validation is fixed and the 50,000-event test role remains sealed",
            "intent": "composition",
            "question": "How do training events scale while validation and sealed-test metadata remain fixed?",
            "rationale": "Stacked bars show the discrete promotion ladder and held-out role sizes; the neutral test segment is metadata only, not accessed evaluation data.",
            "comparisonContext": {"baseline": "35k selection", "denominator": "events named by each selection manifest", "grain": "nested training selection", "normalization": "none", "semanticFamily": "data promotion ladder", "unit": "events"},
            "type": "stackedBar",
            "dataset": "subset_ladder",
            "sourceId": _dataset_source_id("subset_ladder"),
            "encodings": {
                "x": _encoding("selection", "ordinal", "Selection"),
                "y": _multi_encoding(
                    ["trainEvents", "validationEvents", "sealedTestEventsMetadataOnly"],
                    "Events",
                    unit="events",
                    value_format="number",
                ),
                "tooltip": [
                    _encoding("trainShards", "quantitative", "Train shards", unit="shards", value_format="number"),
                    _encoding("sealedTestAccessed", "nominal", "Sealed test accessed"),
                ],
            },
            "xAxisTitle": "Nested selection",
            "yAxisTitle": "Events by role",
            "palette": {"kind": "semantic", "name": "blue-purple-neutral-data-roles"},
            "valueFormat": "number",
            "unit": "events",
            "layout": "full",
            "maxRows": 3,
            "referenceLines": zero_y,
            "settings": {"groupMode": "stacked", "sort": "none", "showValues": True},
            "surface": full_surface,
        },
        {
            "id": "model-parameter-scale",
            "title": "Trainable parameter scale across CPU-instantiated presets",
            "subtitle": "Actual pretraining and full-revised reconstruction module instantiation",
            "intent": "comparison",
            "question": "How does trainable parameter count scale across the four implemented presets?",
            "rationale": "Grouped bars compare the two model surfaces at each preset and preserve the large scale difference between debug and promotion models.",
            "comparisonContext": {"baseline": "tiny_cpu", "denominator": "trainable tensor elements", "grain": "model preset and training surface", "normalization": "none", "semanticFamily": "model capacity", "unit": "parameters"},
            "type": "bar",
            "dataset": "model_parameter_scale",
            "sourceId": _dataset_source_id("model_parameter_scale"),
            "encodings": {
                "x": _encoding("preset", "ordinal", "Model preset"),
                "y": _multi_encoding(
                    ["pretrainingParameters", "reconstructionParameters"],
                    "Trainable parameters",
                    unit="parameters",
                    value_format="compact",
                ),
                "tooltip": [
                    _encoding("dModel", "quantitative", "d_model", unit="dimensions", value_format="number"),
                    _encoding("contextLayers", "quantitative", "Context layers", unit="layers", value_format="number"),
                ],
            },
            "xAxisTitle": "CPU-instantiated preset",
            "yAxisTitle": "Trainable parameters",
            "palette": {"kind": "semantic", "name": "blue-purple-model-surfaces"},
            "valueFormat": "compact",
            "unit": "parameters",
            "layout": "full",
            "maxRows": 4,
            "referenceLines": zero_y,
            "settings": {"groupMode": "grouped", "sort": "none", "showValues": True},
            "surface": full_surface,
        },
        {
            "id": "model-module-composition",
            "title": "small_candidate trainable-module composition",
            "subtitle": "Share of actual trainable parameters by top-level module for both model surfaces",
            "intent": "composition",
            "question": "Which major modules account for small_candidate trainable capacity?",
            "rationale": "A 100% horizontal stack compares composition despite different surface totals; exact counts remain in the model contract table.",
            "comparisonContext": {"baseline": "within-surface total", "denominator": "trainable parameters in each small_candidate surface", "grain": "model surface", "normalization": "share of surface total", "semanticFamily": "module composition", "unit": "share"},
            "type": "horizontalStackedBar100",
            "dataset": "model_module_composition",
            "sourceId": _dataset_source_id("model_module_composition"),
            "encodings": {
                "x": _encoding("modelSurface", "nominal", "Model surface"),
                "y": _multi_encoding(
                    [
                        "encoderShare",
                        "relationBiasShare",
                        "contextOrRelationHeadShare",
                        "decoderShare",
                        "levelDecodersShare",
                        "leafPidHeadShare",
                        "auxiliaryHeadsShare",
                    ],
                    "Share of trainable parameters",
                    unit="share",
                    value_format="percent",
                ),
                "tooltip": [_encoding("totalParameters", "quantitative", "Surface total", unit="parameters", value_format="number")],
            },
            "xAxisTitle": "Share of trainable parameters",
            "yAxisTitle": "Model surface",
            "palette": {"kind": "categorical", "name": "blue-purple-orange-yellow-pink-green-neutral-modules"},
            "valueFormat": "percent",
            "unit": "share",
            "layout": "full",
            "maxRows": 2,
            "referenceLines": zero_x,
            "settings": {"groupMode": "stacked100", "orientation": "horizontal", "sort": "none", "showPercent": True},
            "surface": full_surface,
        },
        {
            "id": "curriculum-plan",
            "title": "Planned 35k pretraining curriculum",
            "subtitle": "17,500 optimizer steps divided evenly across four progressive phases",
            "intent": "composition",
            "question": "How is the first scientific pretraining budget allocated across curriculum phases?",
            "rationale": "One stacked bar makes the cumulative four-phase allocation explicit without implying that planned steps have executed.",
            "comparisonContext": {"baseline": "17,500-step plan", "denominator": "planned optimizer steps", "grain": "curriculum phase", "normalization": "none", "semanticFamily": "planned curriculum", "unit": "optimizer steps"},
            "type": "stackedBar",
            "dataset": "curriculum_plan",
            "sourceId": _dataset_source_id("curriculum_plan"),
            "encodings": {
                "x": _encoding("plan", "nominal", "Training plan"),
                "y": _multi_encoding(
                    ["phase1Steps", "phase2Steps", "phase3Steps", "phase4Steps"],
                    "Optimizer steps",
                    unit="optimizer steps",
                    value_format="number",
                ),
            },
            "xAxisTitle": "Planned run",
            "yAxisTitle": "Optimizer steps",
            "palette": {"kind": "categorical", "name": "blue-purple-orange-pink-curriculum-phases"},
            "valueFormat": "number",
            "unit": "optimizer steps",
            "layout": "full",
            "maxRows": 1,
            "referenceLines": zero_y,
            "settings": {"groupMode": "stacked", "sort": "none", "showValues": True},
            "surface": full_surface,
        },
        {
            "id": "staged-training-budgets",
            "title": "Planned event budgets by promotion stage",
            "subtitle": "Pretraining and reconstruction budgets increase only after staged evidence gates",
            "intent": "comparison",
            "question": "How does the planned event budget grow from screening through confirmation?",
            "rationale": "Grouped bars compare pretraining and reconstruction at the same promotion stage while preserving planned status.",
            "comparisonContext": {"baseline": "35k screening", "denominator": "planned seen events", "grain": "promotion stage and training track", "normalization": "none", "semanticFamily": "staged promotion", "unit": "seen events"},
            "type": "bar",
            "dataset": "staged_training_budgets",
            "sourceId": _dataset_source_id("staged_training_budgets"),
            "encodings": {
                "x": _encoding("stage", "ordinal", "Promotion stage"),
                "y": _multi_encoding(
                    ["pretrainingSeenEvents", "reconstructionSeenEvents"],
                    "Seen events",
                    unit="seen events",
                    value_format="compact",
                ),
            },
            "xAxisTitle": "Promotion stage",
            "yAxisTitle": "Planned seen events",
            "palette": {"kind": "semantic", "name": "blue-purple-planned-training-tracks"},
            "valueFormat": "compact",
            "unit": "seen events",
            "layout": "full",
            "maxRows": 3,
            "referenceLines": zero_y,
            "settings": {"groupMode": "grouped", "sort": "none", "showValues": True},
            "surface": full_surface,
        },
        {
            "id": "inference-capacity-controls",
            "title": "Inference capacity versus observed structural maxima",
            "subtitle": "Observed on the sealed-test-excluding 85,000-event train+validation index",
            "intent": "comparison",
            "question": "Do configured query and daughter-cardinality limits cover observed indexed structure?",
            "rationale": "Grouped bars expose capacity headroom and the zero-margin daughter-cardinality boundary using the same count unit.",
            "comparisonContext": {"baseline": "observed maximum", "denominator": "count per event or mother", "grain": "inference capacity control", "normalization": "none", "semanticFamily": "capacity guard", "unit": "count"},
            "type": "bar",
            "dataset": "inference_capacity_controls",
            "sourceId": _dataset_source_id("inference_capacity_controls"),
            "encodings": {
                "x": _encoding("control", "nominal", "Capacity control"),
                "y": _multi_encoding(
                    ["configuredCapacity", "observedMaximum"],
                    "Count",
                    unit="count",
                    value_format="number",
                ),
                "tooltip": [_encoding("capacityMargin", "quantitative", "Capacity margin", unit="count", value_format="number")],
            },
            "xAxisTitle": "Inference capacity control",
            "yAxisTitle": "Configured and observed count",
            "palette": {"kind": "semantic", "name": "blue-purple-configured-observed"},
            "valueFormat": "number",
            "unit": "count",
            "layout": "full",
            "maxRows": 2,
            "referenceLines": zero_y,
            "settings": {"groupMode": "grouped", "sort": "none", "showValues": True},
            "surface": full_surface,
        },
        {
            "id": "diagnostic-telemetry",
            "title": "Diagnostic GPU utilization telemetry",
            "subtitle": "Nine observed 15-second samples from job 15745941; runtime evidence only",
            "intent": "trend",
            "question": "What utilization pattern was observed during the bounded four-step diagnostic?",
            "rationale": "A nine-point line is sufficient for the bounded telemetry interval and avoids mixing utilization, memory, and temperature units on one axis.",
            "comparisonContext": {"baseline": "zero utilization", "denominator": "one allocated V100", "grain": "15-second telemetry sample", "normalization": "GPU utilization percent", "semanticFamily": "diagnostic telemetry", "unit": "percent"},
            "type": "line",
            "dataset": "diagnostic_telemetry",
            "sourceId": _dataset_source_id("diagnostic_telemetry"),
            "encodings": {
                "x": _encoding("elapsedSeconds", "quantitative", "Elapsed time", unit="seconds", value_format="number"),
                "y": _encoding("gpuUtilizationPercent", "quantitative", "GPU utilization", unit="percent", value_format="number"),
                "tooltip": [
                    _encoding("memoryUsedMiB", "quantitative", "Memory used", unit="MiB", value_format="number"),
                    _encoding("temperatureC", "quantitative", "Temperature", unit="°C", value_format="number"),
                ],
            },
            "xAxisTitle": "Elapsed seconds from first telemetry sample",
            "yAxisTitle": "GPU utilization (%)",
            "palette": {"kind": "semantic", "name": "blue-observed"},
            "valueFormat": "number",
            "unit": "percent",
            "layout": "full",
            "maxRows": 9,
            "referenceLines": zero_y,
            "settings": {"sort": "none", "showPoints": "always", "showValues": False},
            "surface": full_surface,
        },
        {
            "id": "ablation-coverage",
            "title": "Implemented and planned ablation-arm coverage",
            "subtitle": "Named runnable controls remain separate from planned scientific comparisons",
            "intent": "composition",
            "question": "Which ablation families have runnable named arms, and where are planned gaps?",
            "rationale": "Horizontal stacked counts show implemented versus planned arms by family without ranking families as better or worse.",
            "comparisonContext": {"baseline": "implemented named arm", "denominator": "enumerated evidence-package arms", "grain": "ablation family", "normalization": "none", "semanticFamily": "ablation readiness", "unit": "arms"},
            "type": "horizontalStackedBar",
            "dataset": "ablation_coverage",
            "sourceId": _dataset_source_id("ablation_coverage"),
            "encodings": {
                "x": _encoding("family", "nominal", "Ablation family"),
                "y": _multi_encoding(
                    ["implementedArms", "plannedArms"],
                    "Arms",
                    unit="arms",
                    value_format="number",
                ),
                "tooltip": [_encoding("totalArms", "quantitative", "Total enumerated arms", unit="arms", value_format="number")],
            },
            "xAxisTitle": "Enumerated arms",
            "yAxisTitle": "Ablation family",
            "palette": {"kind": "semantic", "name": "blue-purple-implemented-planned"},
            "valueFormat": "number",
            "unit": "arms",
            "layout": "full",
            "maxRows": 8,
            "referenceLines": zero_x,
            "settings": {"groupMode": "stacked", "orientation": "horizontal", "sort": "none", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": full_surface,
        },
        {
            "id": "readiness-gates",
            "title": "Readiness evidence and unresolved gates",
            "subtitle": "Observed, planned, blocked, unavailable, and unknown states are one-hot evidence labels",
            "intent": "composition",
            "question": "Which gates are evidenced, planned, blocked, unavailable, or scientifically unknown?",
            "rationale": "A 100% horizontal status matrix preserves each gate as one unit and makes label categories explicit without green/red judgment semantics.",
            "comparisonContext": {"baseline": "one state per gate", "denominator": "ten enumerated readiness gates", "grain": "readiness gate", "normalization": "one-hot state share", "semanticFamily": "evidence readiness", "unit": "share"},
            "type": "horizontalStackedBar100",
            "dataset": "readiness_gates",
            "sourceId": _dataset_source_id("readiness_gates"),
            "encodings": {
                "x": _encoding("gate", "nominal", "Readiness gate"),
                "y": _multi_encoding(
                    ["observedComplete", "plannedRequired", "blocked", "unavailable", "unknown"],
                    "Gate state",
                    unit="share",
                    value_format="percent",
                ),
                "tooltip": [_encoding("detail", "text", "Evidence detail")],
            },
            "xAxisTitle": "Evidence-state share",
            "yAxisTitle": "Readiness gate",
            "palette": {"kind": "semantic", "name": "blue-purple-orange-yellow-neutral-readiness"},
            "valueFormat": "percent",
            "unit": "share",
            "layout": "full",
            "maxRows": 10,
            "referenceLines": zero_x,
            "settings": {"groupMode": "stacked100", "orientation": "horizontal", "sort": "none", "showPercent": True, "categoryLabelPolicy": "wrap"},
            "surface": full_surface,
        },
    ]


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "data-contract-table",
            "title": "Exact data and index contract",
            "subtitle": "Observed metadata only; no sealed-test payload access",
            "dataset": "data_contract_rows",
            "sourceId": _dataset_source_id("data_contract_rows"),
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "field", "label": "Contract field", "type": "text"},
                {"field": "value", "label": "Exact value", "type": "text", "role": "value"},
                {"field": "status", "label": "Evidence status", "type": "text"},
            ],
        },
        {
            "id": "model-contract-table",
            "title": "Exact small_candidate model contract",
            "subtitle": "Configured architecture plus CPU-instantiated trainable totals",
            "dataset": "model_contract_rows",
            "sourceId": _dataset_source_id("model_contract_rows"),
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "field", "label": "Model field", "type": "text"},
                {"field": "value", "label": "Exact value", "type": "text", "role": "value"},
                {"field": "evidenceStatus", "label": "Evidence status", "type": "text"},
            ],
        },
        {
            "id": "diagnostic-contract-table",
            "title": "Exact job 15745941 diagnostic contract",
            "subtitle": "Bounded runtime evidence; not convergence or held-out quality",
            "dataset": "diagnostic_contract_rows",
            "sourceId": _dataset_source_id("diagnostic_contract_rows"),
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "field", "label": "Diagnostic field", "type": "text"},
                {"field": "value", "label": "Exact value", "type": "text", "role": "value"},
                {"field": "status", "label": "Evidence status", "type": "text"},
            ],
        },
        {
            "id": "inference-contract-table",
            "title": "Exact inference and rollout controls",
            "subtitle": "Implemented mechanics and planned checkpoint gates are labeled separately",
            "dataset": "inference_contract_rows",
            "sourceId": _dataset_source_id("inference_contract_rows"),
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "control", "label": "Control", "type": "text"},
                {"field": "value", "label": "Exact setting", "type": "text", "role": "value"},
                {"field": "status", "label": "Implementation status", "type": "text"},
            ],
        },
        {
            "id": "no-submit-contract-table",
            "title": "Exact blocked no-submit contract",
            "subtitle": "Scientific execution remains unauthorized and unperformed",
            "dataset": "no_submit_contract_rows",
            "sourceId": _dataset_source_id("no_submit_contract_rows"),
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "field", "label": "Contract field", "type": "text"},
                {"field": "value", "label": "Exact value", "type": "text", "role": "value"},
                {"field": "status", "label": "Evidence status", "type": "text"},
            ],
        },
        {
            "id": "ablation-arm-table",
            "title": "Exact ablation-arm registry",
            "subtitle": "Runnable named controls, planned comparisons, and the physical-only gap",
            "dataset": "ablation_arm_rows",
            "sourceId": _dataset_source_id("ablation_arm_rows"),
            "layout": "full",
            "density": "dense",
            "columns": [
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "arm", "label": "Arm", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "implementation", "label": "Implementation / plan", "type": "text"},
                {"field": "evidenceStatus", "label": "Evidence label", "type": "text"},
            ],
        },
    ]


def _markdown_block(
    block_id: str,
    body: str,
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "id": block_id,
        "type": "markdown",
        "body": body.strip(),
        "layout": "full",
    }
    if source_id is not None:
        block["sourceId"] = source_id
    return block


def _blocks(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    facts = evidence["facts"]
    data = facts["data"]
    diagnostic = facts["diagnostic"]
    curriculum = facts["curriculum"]
    model = facts["models"]["small_candidate"]
    readiness = facts["readiness"]
    ablations = facts["ablations"]
    return [
        _markdown_block("section-01-title", f"# {REPORT_TITLE}"),
        _markdown_block(
            "section-02-technical-summary",
            f"""
## Technical summary — implementation evidence is strong, scientific readiness remains partial

**Result headline.** The repository supports a pilot-ready, relation-aware, level-autoregressive reconstruction path, but it does not yet support a convergence or held-out physics-quality claim.

**Interpretation.** Observed evidence covers a {data['inventoryEvents']:,}-event reduced publication, a sealed-test-excluding {data['indexedEvents']:,}-event train+validation index, CPU-constructed model presets, and a bounded runtime diagnostic. Planned evidence begins with the frozen 35k scientific path.

**Evidence note.** Values below are labeled observed, planned, or unknown. Implemented mechanics are separated from deferred designs and unrun scientific comparisons.

**Method/assumption.** This snapshot copies bounded rows from `evidence.json`; it performs no payload, checkpoint, scheduler, or sealed-test access.

**Limitation/implication.** The report status is partial. Publication integrity and software execution are prerequisites, not substitutes for long training and held-out reconstruction quality.
""",
        ),
        {"id": "block-data-contract-table", "type": "table", "tableId": "data-contract-table", "layout": "full"},
        _markdown_block(
            "section-03-key-findings",
            f"""
## Key findings with visual evidence — scale and hierarchy justify set reconstruction, not long-sequence modeling

**Takeaway.** The observed reduced inventory contains {data['inventoryEvents']:,} events, but its category mixture is skewed; mixed and uubar dominate the metadata composition.

**How to read.** Compare absolute event bars across the seven categories. Each shard contributes the same event count, so event and shard composition agree.

**Evidence note.** This is observed publication metadata, not a claim that the reduced campaign is category-representative.

**Method/assumption.** Counts are recomputed from inventory entries and cross-checked against the tracked selection summary.

**Implication/caveat.** Future metrics must report macro-category and production-mixture-weighted views separately.
""",
            source_id="inventory",
        ),
        {"id": "block-production-category-composition", "type": "chart", "chartId": "production-category-composition", "layout": "full"},
        _markdown_block(
            "key-finding-structure-explainer",
            f"""
### First-level multiplicity is the principal structural pressure

**Takeaway.** The observed complete-only index contains {data['observedReconstructionLevels']} populated reconstruction levels, with the largest mother multiplicity concentrated at level 1.

**How to read.** Follow p50 through p99 across ordered levels; higher percentiles expose the tail that query capacity must cover.

**Evidence note.** Percentiles and maxima come only from train+validation index metadata; sealed test is excluded.

**Method/assumption.** Reconstruction level is generation height, not graph distance.

**Implication/caveat.** Capacity conclusions apply to the selected complete-only index and require revalidation if the target policy or data population changes.
""",
            source_id="capacity_small_candidate",
        ),
        {"id": "block-structural-percentiles", "type": "chart", "chartId": "structural-percentiles", "layout": "full"},
        _markdown_block(
            "section-04-scope-data-definitions",
            """
## Scope, data, and definitions — promotion uses immutable nested subsets and a sealed final test

**Takeaway.** The 35k, 100k, and 250k training subsets are nested; validation stays fixed and test appears only as sealed metadata.

**How to read.** The growing blue segment is training data. Purple validation is fixed. Neutral test is a role-size annotation and was not opened.

**Evidence note.** Whole source/task shards are indivisible, normalizers use train only, and the indexed cohort is complete-only train+validation.

**Method/assumption.** “Observed” here means manifest/index metadata, not evaluated model performance. Reconstruction level means retained-tree generation height.

**Implication/caveat.** No threshold, checkpoint, or ablation may be selected on the sealed test; stress and validation remain separate diagnostic roles.
""",
            source_id="selection_summary",
        ),
        {"id": "block-subset-ladder", "type": "chart", "chartId": "subset-ladder", "layout": "full"},
        _markdown_block(
            "section-05-model-specification",
            f"""
## Model specification — shared relation-aware geometry stays small enough for staged single-GPU study

**Takeaway.** CPU instantiation counts {model['pretrainingTrainableParameters']:,} trainable pretraining parameters and {model['reconstructionTrainableParameters']:,} reconstruction parameters for `small_candidate`.

**How to read.** Compare the two surfaces across presets; the counts include actual trainable tensors constructed from repository code, not estimates.

**Evidence note.** The architecture is a heterogeneous Euclidean set transformer with physical relation bias, shared Poincaré projections/objectives, optional hyperbolic relation refinement, and level-autoregressive pointer decoding.

**Method/assumption.** Truth supervises topology, while composite inputs and four-momenta are reconstructed from daughter state. The counted full-revised module owns allocated top-level parameters even when a compatibility module is inactive on the heterogeneous forward path.

**Implication/caveat.** Mixture-of-experts is deferred and is not part of any implemented production model or parameter total.
""",
            source_id="report_evidence",
        ),
        {"id": "block-model-parameter-scale", "type": "chart", "chartId": "model-parameter-scale", "layout": "full"},
        _markdown_block(
            "model-composition-explainer",
            """
### The shared encoder dominates trainable capacity

**Takeaway.** Encoder parameters dominate both `small_candidate` surfaces; reconstruction additionally allocates contextualization and pointer-decoder capacity.

**How to read.** Each horizontal bar sums to 100% within its model surface. Use the exact-contract table for absolute totals.

**Evidence note.** Module groups are top-level names from actual CPU-instantiated models.

**Method/assumption.** Shares use trainable parameters only; non-trainable channel-memory buffers are excluded.

**Implication/caveat.** Composition does not measure runtime or memory hotspots; representative profiling remains future evidence.
""",
            source_id="report_evidence",
        ),
        {"id": "block-model-module-composition", "type": "chart", "chartId": "model-module-composition", "layout": "full"},
        {"id": "block-model-contract-table", "type": "table", "tableId": "model-contract-table", "layout": "full"},
        _markdown_block(
            "section-06-training-methodology",
            f"""
## Training methodology and details — progressive objectives and free rollout address geometry and exposure bias

**Takeaway.** The first scientific pretraining plan allocates {curriculum['plannedSteps']:,} optimizer steps across four equal progressive phases, for {curriculum['plannedSeenEvents']:,} planned seen events.

**How to read.** The stacked bar is a planned allocation: topology/parent/anticollapse, distance/radius, channel contrast, then candidate correctness/hard negatives.

**Evidence note.** Objectives cover same-mother/branch and LCA relations, parent ranking, exact tree distance, radius/depth, B-side/channel separation, and tangent-space anti-collapse.

**Method/assumption.** Objectives are cumulative unless preregistered otherwise; scheduled sampling and free rollout are required to address exposure bias.

**Implication/caveat.** Planned steps are not observed convergence, and promotion gates require measured validation and rollout behavior.
""",
            source_id="pretrain_scientific_config",
        ),
        {"id": "block-curriculum-plan", "type": "chart", "chartId": "curriculum-plan", "layout": "full"},
        _markdown_block(
            "staged-budget-explainer",
            """
### Compute expands only after evidence gates

**Takeaway.** Screening, candidate, and confirmation budgets increase in discrete stages for both pretraining and reconstruction.

**How to read.** Compare pretraining and reconstruction seen-event budgets within each promotion stage; all bars are planned.

**Evidence note.** Screening uses one paired seed; confirmation requires multi-seed evidence and fixed validation cohorts.

**Method/assumption.** Event budgets, not vague epochs, are authoritative.

**Implication/caveat.** No 1M/10M or distributed scaling is justified until the 250k stage demonstrates stable scientific value and compute scaling.
""",
            source_id="training_plan",
        ),
        {"id": "block-staged-training-budgets", "type": "chart", "chartId": "staged-training-budgets", "layout": "full"},
        _markdown_block(
            "section-07-inference-design",
            """
## Inference design — learned proposals are bounded by exclusivity, topology, and exact daughter-sum kinematics

**Takeaway.** The configured mother-query budget exceeds the observed mother maximum, while daughter-cardinality capacity exactly reaches the observed maximum.

**How to read.** Compare configured and observed bars for the two count-based controls; the tooltip exposes remaining margin.

**Evidence note.** Teacher forcing uses truth topology but reco-derived mother features and daughter-summed p4. Inference uses predicted mothers, learned type/pointer/confidence scores, greedy exclusivity, and light hard constraints.

**Method/assumption.** Exact bounded set packing and beam search are evaluation-only diagnostics. Whole-set scoring and iterative within-mother pointer decoding remain deferred designs, not implemented inference paths.

**Implication/caveat.** Zero cardinality margin warrants monitoring on every promoted cohort; later physical fits/constraints are separate from unconstrained truth-momentum regression.
""",
            source_id="report_evidence",
        ),
        {"id": "block-inference-capacity-controls", "type": "chart", "chartId": "inference-capacity-controls", "layout": "full"},
        {"id": "block-inference-contract-table", "type": "table", "tableId": "inference-contract-table", "layout": "full"},
        _markdown_block(
            "section-08-limitations-robustness",
            f"""
## Limitations and robustness — the four-step V100 run proves execution, not learning

**Takeaway.** Diagnostic job {diagnostic['jobId']} completed {diagnostic['steps']} steps in {diagnostic['elapsedSeconds']} seconds with {diagnostic['telemetrySamples']} telemetry samples; peak utilization was {diagnostic['peakGpuUtilizationPercent']}%.

**How to read.** The utilization trace shows sparse bounded activity. Memory and temperature remain available in tooltips and the exact table.

**Evidence note.** The run entered four curriculum phases and validated on only {diagnostic['validationEvents']} events across {diagnostic['validationNamedViews']} named views.

**Method/assumption.** Receipt, readiness metadata, raw telemetry, and metrics were cross-checked. No checkpoint payload was opened.

**Implication/caveat.** This is diagnostic smoke evidence only: it cannot establish convergence, throughput at scientific scale, calibration, category robustness, or held-out reconstruction quality.
""",
            source_id="report_evidence",
        ),
        {"id": "block-diagnostic-telemetry", "type": "chart", "chartId": "diagnostic-telemetry", "layout": "full"},
        {"id": "block-diagnostic-contract-table", "type": "table", "tableId": "diagnostic-contract-table", "layout": "full"},
        _markdown_block(
            "section-09-staged-ablation-plan",
            f"""
## Staged training and ablation plan — isolate mechanisms sequentially and never cross-product the grid

**Takeaway.** The evidence registry contains {ablations['implementedArmCount']} implemented/named arms and {ablations['plannedArmCount']} planned arms across focused representation, context, relation, geometry, and exposure-bias families.

**How to read.** Blue counts are runnable named controls; purple counts require a scientific comparison or a missing named arm.

**Evidence note.** The ladder isolates flat baseline, heterogeneous/contextual Euclidean encoding, hyperbolic node geometry, combined physical+hyperbolic relation attention, context removal, level/radius/tangent variants, and scheduled-sampling endpoints.

**Method/assumption.** Screening is paired on 35k; only directional, safe effects advance. MoE, whole-set scoring, and iterative-pointer designs stay deferred.

**Implication/caveat.** A physical-relation-only named arm is absent even though the model API exposes separate physical and hyperbolic flags; add that runnable control before claiming the two biases are isolated.
""",
            source_id="report_evidence",
        ),
        {"id": "block-ablation-coverage", "type": "chart", "chartId": "ablation-coverage", "layout": "full"},
        {"id": "block-ablation-arm-table", "type": "table", "tableId": "ablation-arm-table", "layout": "full"},
        _markdown_block(
            "section-10-recommended-next-steps",
            """
## Recommended next steps — clear provenance and execution gates before any scientific run

**Result headline.** Keep the scientific path fail-closed until all three serialized blockers are resolved.

**Interpretation.** Recover and independently verify the production source object/tree; require a fresh exact in-allocation CUDA/GPU preflight; then commit, review, tag, and render from a clean immutable source.

**Evidence note.** The blocked contract authorizes no submission, records no submission, and forbids sealed-test access.

**Method/assumption.** After those gates, run only the 35k `small_candidate` pretraining/reconstruction screen and preregistered ablations.

**Limitation/implication.** H200 exact GRES was unavailable in the observed metadata. No H100 job contract is tracked; absence of a tracked contract is not proof of external scheduler state.
""",
            source_id="no_submit_contract",
        ),
        {"id": "block-no-submit-contract-table", "type": "table", "tableId": "no-submit-contract-table", "layout": "full"},
        _markdown_block(
            "section-11-further-questions",
            """
## Further questions — scientific quality, scaling, and causal attribution remain open

**Takeaway.** The unresolved questions are concentrated in planned prerequisites and scientifically unknown outcomes, not in the mechanics of deterministic evidence generation.

**How to read.** Each readiness row has exactly one evidence state. Colors identify categories and do not encode good/bad judgment.

**Evidence note.** Open questions include reproducible 35k learning, three-seed 100k stability, 250k category/channel robustness, rollout calibration, KLM cohorts, representative throughput, and final sealed-test quality.

**Method/assumption.** Promotion requires paired baselines, fixed validation UIDs, rollout-selected checkpoints, and explicit denominators.

**Implication/caveat.** Do not infer H100 substitution, long-run convergence, held-out quality, or benefit from hyperbolic/physical biases until corresponding evidence is recorded.
""",
            source_id="training_readiness",
        ),
        {"id": "block-readiness-gates", "type": "chart", "chartId": "readiness-gates", "layout": "full"},
    ]


def build_artifact(evidence: Mapping[str, Any]) -> dict[str, Any]:
    require(evidence["title"] == REPORT_TITLE, "evidence title does not match report contract")
    require(evidence["generatedAt"] == GENERATED_AT, "evidence timestamp does not match report contract")
    require(evidence["status"] == "partial", "evidence must remain partial")
    charts = _charts()
    tables = _tables()
    bound_dataset_names = [item["dataset"] for item in (*charts, *tables)]
    sources = _artifact_sources(evidence, bound_dataset_names)
    blockers = evidence["facts"]["readiness"]["blockers"]
    access_issues = [
        {
            "id": "missing-full-convergence",
            "scope": "scientific results",
            "sourceId": "training_plan",
            "message": "No long GPU convergence result is present; diagnostic smoke evidence cannot substitute.",
        },
        {
            "id": "missing-held-out-quality",
            "scope": "held-out reconstruction",
            "sourceId": "selection_summary",
            "message": "The sealed test remains closed, so held-out reconstruction quality is unknown.",
        },
        {
            "id": "blocker-production-source",
            "scope": "scientific submission",
            "sourceId": "no_submit_contract",
            "message": blockers[0],
        },
        {
            "id": "blocker-fresh-preflight",
            "scope": "scientific submission",
            "sourceId": "no_submit_contract",
            "message": blockers[1],
        },
        {
            "id": "blocker-review-tag-render",
            "scope": "scientific submission",
            "sourceId": "no_submit_contract",
            "message": blockers[2],
        },
        {
            "id": "h200-unavailable",
            "scope": "resource availability",
            "sourceId": "training_readiness",
            "message": "Exact H200 GRES was unavailable in the observed readiness metadata; no submission was performed.",
        },
        {
            "id": "physical-only-named-arm-gap",
            "scope": "ablation isolation",
            "sourceId": "report_evidence",
            "dataset": "ablation_arm_rows",
            "message": "A standalone named physical-relation-bias arm is absent; the sequential scientific comparison is planned.",
        },
    ]
    blocks = _blocks(evidence)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "A deterministic technical evidence package for structured-set reconstruction architecture, data, training, inference, diagnostics, and staged scientific readiness.",
            "generatedAt": GENERATED_AT,
            "sources": sources,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "partial",
            "datasets": evidence["datasets"],
            "accessIssues": access_issues,
        },
        "sources": sources,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root; defaults to discovery from this script.",
    )
    parser.add_argument(
        "--metadata-root",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Override one permitted logical metadata root (artifacts/configs/docs/src).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults beside this generator.",
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="Canonical report artifact path; defaults to artifact.json beside this generator.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the current output bytes differ; do not write.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else discover_repo_root(Path(__file__).resolve().parent)
    )
    require((repo_root / "pyproject.toml").is_file(), f"invalid repository root: {repo_root}")
    metadata_roots = parse_metadata_root_overrides(args.metadata_root)
    output = (
        args.output.resolve()
        if args.output is not None
        else Path(__file__).resolve().with_name("evidence.json")
    )
    artifact_output = (
        args.artifact_output.resolve()
        if args.artifact_output is not None
        else Path(__file__).resolve().with_name("artifact.json")
    )
    evidence = generate(repo_root, metadata_roots)
    payload = _json_bytes(evidence)
    artifact_payload = _json_bytes(build_artifact(evidence))
    if args.check:
        require(output.is_file(), f"generated evidence is missing: {output}")
        require(output.read_bytes() == payload, f"generated evidence is stale: {output}")
        require(artifact_output.is_file(), f"generated artifact is missing: {artifact_output}")
        require(
            artifact_output.read_bytes() == artifact_payload,
            f"generated artifact is stale: {artifact_output}",
        )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        artifact_output.parent.mkdir(parents=True, exist_ok=True)
        artifact_output.write_bytes(artifact_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
