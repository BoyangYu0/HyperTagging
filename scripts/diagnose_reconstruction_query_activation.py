#!/usr/bin/env python3
"""Read-only query-activation diagnostics on saved reconstruction checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hypertagging.data.heterogeneous import collate_heterogeneous_events  # noqa: E402
from hypertagging.data.streaming import RuntimeFeatureNormalizer  # noqa: E402
from hypertagging.evaluation.query_activation import (  # noqa: E402
    QUERY_ACTIVATION_DIAGNOSTIC_SCHEMA_VERSION,
    gradient_reachability_diagnostics,
    inference_decision_diagnostics,
    matching_and_margin_diagnostics,
    per_query_probability_distributions,
    require_diagnostic_role,
    require_finite_json,
)
from hypertagging.losses.level_reconstruction import (  # noqa: E402
    level_reconstruction_loss,
)
from hypertagging.models.ablation import build_ablation_model  # noqa: E402
from hypertagging.preprocessing.pid_filter import DETOKENIZE_DICT, PDG_TOKENS  # noqa: E402
from hypertagging.reconstruction.constraints import (  # noqa: E402
    ReconstructionConstraintPolicy,
)
from hypertagging.reconstruction.level_rollout import (  # noqa: E402
    RolloutConfig,
    cached_context_for_level,
    level_rollout,
)
from hypertagging.training.checkpointing import load_training_checkpoint  # noqa: E402
from hypertagging.training.data_module import build_real_data_module  # noqa: E402
from hypertagging.training.fixed_validation import select_validation_events  # noqa: E402
from hypertagging.training.model_config import ModelArchitecture  # noqa: E402
from hypertagging.training.reconstruction_trainer import _with_allowed_types  # noqa: E402
from hypertagging.training.scheduled_sampling import aligned_level_targets  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_from_checkpoint(
    payload: dict[str, Any], data_module: Any, device: torch.device
) -> torch.nn.Module:
    config = payload["config"]
    architecture = ModelArchitecture.from_dict(payload["architecture"])
    model = build_ablation_model(
        config["ablation"],
        n_features=12,
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
        pid_kinematics_mode=config.get("pid_kinematics_mode"),
        pid_temperature=float(config.get("pid_temperature", 1.0)),
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
    return model.to(device).eval()


def _target_family_counts(
    truth: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor],
    *,
    target_level: int,
    target_policy: str,
) -> dict[str, dict[str, int]]:
    eligible = truth["node_mask"][0] & (truth["level_ids"][0] == target_level)
    if target_policy != "diagnostic_all":
        eligible &= truth["valid_reconstruction_target"][0]
    if target_policy == "complete_only":
        eligible &= truth["recursive_reconstructable_complete"][0]
    elif target_policy not in {"reconstructable_partial", "diagnostic_all"}:
        raise ValueError(f"unknown target policy: {target_policy}")
    context_mask = context["node_mask"][0]
    predicted_sources = context["recursive_leaf_source_mask"][0, context_mask]
    result: dict[str, dict[str, int]] = {}
    labels = truth.get("pid_target_labels", truth["pid_labels"])
    for mother in eligible.nonzero(as_tuple=False).flatten().tolist():
        daughters = truth["daughter_adjacency"][0, mother].nonzero(
            as_tuple=False
        ).flatten()
        if daughters.numel() < 2:
            continue
        representable = True
        used_positions: set[int] = set()
        for daughter in daughters.tolist():
            sources = truth["recursive_leaf_source_mask"][0, daughter]
            exact = (predicted_sources == sources.unsqueeze(0)).all(dim=-1)
            candidates = exact.nonzero(as_tuple=False).flatten().tolist()
            available = [position for position in candidates if position not in used_positions]
            if not available:
                representable = False
                break
            used_positions.add(int(available[0]))
        token = int(labels[0, mother])
        family = str(DETOKENIZE_DICT[token])
        counts = result.setdefault(family, {"truth_targets": 0, "representable": 0})
        counts["truth_targets"] += 1
        counts["representable"] += int(representable)
    return result


def _merge_counts(
    destination: dict[str, dict[str, int]], source: dict[str, dict[str, int]]
) -> None:
    for key, values in source.items():
        target = destination.setdefault(key, {name: 0 for name in values})
        for name, value in values.items():
            target[name] = target.get(name, 0) + int(value)


def _diagnose_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, Any],
    data_module: Any,
    events: list[Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model = _model_from_checkpoint(payload, data_module, device)
    config = payload["config"]
    target_policy = str(config["target_policy"])
    policy = ReconstructionConstraintPolicy.from_dict(
        payload["feature_contract"]["reconstruction_constraint_policy"]
    )
    rollout_config = RolloutConfig(
        max_level=8,
        root_types=(),
        confidence_trained=True,
        use_learned_confidence=True,
        constraint_policy=policy,
        rollout_pid_kinematics_mode=str(config["rollout_pid_kinematics_mode"]),
        rollout_pid_temperature=float(config["rollout_pid_temperature"]),
    )
    level_one_logits: list[torch.Tensor] = []
    event_level_diagnostics: list[dict[str, Any]] = []
    rollout_summaries: list[dict[str, Any]] = []
    category_counts: dict[str, dict[str, int]] = {}
    family_counts: dict[str, dict[str, int]] = {}
    category_loss: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gradient_report: dict[str, Any] | None = None

    for event in events:
        truth = data_module.normalize_batch(collate_heterogeneous_events([event]))
        truth = {name: value.to(device) for name, value in truth.items()}
        rollout = level_rollout(model, truth, mode="predicted", config=rollout_config)
        predicted_nodes = int(
            (rollout.batch["node_mask"] & (rollout.batch["level_ids"] > 0)).sum()
        )
        predicted_depth = int(
            rollout.batch["level_ids"][rollout.batch["node_mask"]].max()
        )
        rollout_summaries.append(
            {
                "event_uid": event.event_uid,
                "source_category": event.source_category or "unknown",
                "stop_reason": rollout.stop_reason,
                "predicted_node_count": predicted_nodes,
                "predicted_depth": predicted_depth,
            }
        )
        truth_depth = int(truth["level_ids"][truth["node_mask"]].max())
        for target_level in range(1, truth_depth + 1):
            predicted_context = cached_context_for_level(rollout, target_level)
            level_batch = _with_allowed_types(
                predicted_context,
                target_level,
                data_module.allowed_types_by_level,
                policy,
            )
            with torch.no_grad():
                predicted_output = model(
                    level_batch,
                    target_level=target_level,
                    pid_kinematics_mode_override="soft_expectation",
                    pid_temperature_override=float(config["rollout_pid_temperature"]),
                )
            decision = inference_decision_diagnostics(
                predicted_output, level_batch, rollout_config
            )
            if target_level == 1:
                level_one_logits.append(predicted_output.pointer.object_logits.detach().cpu())

            teacher_batch = _with_allowed_types(
                truth, target_level, data_module.allowed_types_by_level, policy
            )
            teacher_output = model(teacher_batch, target_level=target_level)
            loss_batch = dict(teacher_batch)
            if teacher_output.current_p4 is not None:
                loss_batch["p4"] = teacher_output.current_p4
            loss_output = level_reconstruction_loss(
                teacher_output.pointer,
                loss_batch,
                target_level=target_level,
                target_policy=target_policy,
                matching_production=True,
                constraint_policy=policy,
                object_positive_weight=float(config.get("object_positive_weight", 2.0)),
                pointer_positive_weight=float(config.get("pointer_positive_weight", 4.0)),
            )
            if gradient_report is None:
                gradient_report = gradient_reachability_diagnostics(
                    model, loss_output.total, target_level=target_level
                )
            matching = matching_and_margin_diagnostics(
                teacher_output,
                loss_batch,
                loss_output.matches,
                target_level=target_level,
                target_policy=target_policy,
            )
            category = event.source_category or "unknown"
            for name, value in loss_output.components.items():
                category_loss[category][name].append(float(value.detach().cpu()))
            alignment = aligned_level_targets(
                truth,
                predicted_context,
                target_level=target_level,
                target_policy=target_policy,
            )
            _merge_counts(
                category_counts,
                {
                    category: {
                        "truth_targets": alignment.truth_target_count,
                        "representable": alignment.representable_count,
                    }
                },
            )
            _merge_counts(
                family_counts,
                _target_family_counts(
                    truth,
                    predicted_context,
                    target_level=target_level,
                    target_policy=target_policy,
                ),
            )
            event_level_diagnostics.append(
                {
                    "event_uid": event.event_uid,
                    "source_category": category,
                    "target_level": target_level,
                    "inference": decision,
                    "assignment": matching,
                    "representability": {
                        "truth_targets": alignment.truth_target_count,
                        "representable": alignment.representable_count,
                    },
                    "loss_components": {
                        name: float(value.detach().cpu())
                        for name, value in loss_output.components.items()
                    },
                }
            )

    if not level_one_logits or gradient_report is None:
        raise RuntimeError("diagnostic cohort produced no level-one supervision")
    for counts in list(category_counts.values()) + list(family_counts.values()):
        counts["unrepresentable"] = counts["truth_targets"] - counts["representable"]
    result = {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "step": int(payload["step"]),
            "source_git_commit": str(payload.get("git_commit", "")),
        },
        "per_query_level_one_probabilities": per_query_probability_distributions(
            torch.cat(level_one_logits, dim=0)
        ),
        "event_level_diagnostics": event_level_diagnostics,
        "rollout_summaries": rollout_summaries,
        "representability_by_source_category": category_counts,
        "representability_by_target_family_pdg": family_counts,
        "loss_contribution_by_source_category": {
            category: {
                component: sum(values) / len(values)
                for component, values in components.items()
                if values
            }
            for category, components in category_loss.items()
        },
        "gradient_reachability": gradient_report,
    }
    require_finite_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-events", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_events <= 0:
        raise ValueError("max-events must be positive")
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing diagnostic output")
    checkpoints = [path.resolve(strict=True) for path in args.checkpoint]
    hashes_before = {path: _sha256(path) for path in checkpoints}
    payloads = [load_training_checkpoint(path, map_location="cpu") for path in checkpoints]
    first = payloads[0]
    first_config = first["config"]
    data_module = build_real_data_module(
        args.data.resolve(strict=True),
        seed=int(first_config["seed"]),
        normalization_state=first["normalizer_state"],
        dataset_index=args.dataset_index.resolve(strict=True),
        target_policy=str(first_config["target_policy"]),
        scientific_mode=True,
    )
    require_diagnostic_role(args.role, split_counts=data_module.split_counts)
    if args.role != "validation":
        raise ValueError("saved-cohort checkpoint diagnostics require validation role")
    selected_uids = tuple(
        first["validation_selection"]["rollout_event_uids"][: args.max_events]
    )
    if len(selected_uids) != args.max_events:
        raise ValueError("checkpoint has fewer fixed rollout UIDs than requested")
    for payload in payloads[1:]:
        other = tuple(
            payload["validation_selection"]["rollout_event_uids"][: args.max_events]
        )
        if other != selected_uids:
            raise ValueError("checkpoints do not share the exact fixed rollout cohort")
        if payload["normalizer_state"].keys() != first["normalizer_state"].keys():
            raise ValueError("checkpoint normalizer contracts differ")
    events, restored_uids, selection_contract = select_validation_events(
        data_module.iter_events("validation", shuffle=False),
        limit=args.max_events,
        scientific_mode=True,
        selection_manifest_hash=data_module.selection_manifest_hash,
        seed=int(first_config["seed"]),
        restored_event_uids=selected_uids,
    )
    if restored_uids != selected_uids:
        raise RuntimeError("fixed validation UID restoration changed order")
    device = torch.device(args.device)
    results = [
        _diagnose_checkpoint(path, payload, data_module, events, device=device)
        for path, payload in zip(checkpoints, payloads, strict=True)
    ]
    hashes_after = {path: _sha256(path) for path in checkpoints}
    if hashes_after != hashes_before:
        raise RuntimeError("a diagnostic source checkpoint changed during read-only use")
    report = {
        "schema_version": QUERY_ACTIVATION_DIAGNOSTIC_SCHEMA_VERSION,
        "mode": "read_only_saved_checkpoint_query_activation_diagnostic",
        "data_role": args.role,
        "sealed_test_role_access": "forbidden",
        "split_counts": data_module.split_counts,
        "selection_contract": selection_contract,
        "evaluated_event_uids": list(restored_uids),
        "checkpoint_count": len(results),
        "checkpoints_unchanged": True,
        "results": results,
    }
    require_finite_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": "completed", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
