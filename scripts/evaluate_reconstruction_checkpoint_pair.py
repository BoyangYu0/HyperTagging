#!/usr/bin/env python3
"""Paired, validation-only reconstruction checkpoint confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.diagnose_reconstruction_query_activation import (  # noqa: E402
    _model_from_checkpoint,
)
from hypertagging.data.heterogeneous import collate_heterogeneous_events  # noqa: E402
from hypertagging.evaluation.hierarchical_metrics import (  # noqa: E402
    complete_target_efficiency_counts,
    summarize_rollout,
)
from hypertagging.evaluation.query_activation import require_finite_json  # noqa: E402
from hypertagging.reconstruction.constraints import (  # noqa: E402
    ReconstructionConstraintPolicy,
)
from hypertagging.reconstruction.level_rollout import (  # noqa: E402
    RolloutConfig,
    level_rollout,
)
from hypertagging.training.checkpointing import load_training_checkpoint  # noqa: E402
from hypertagging.training.data_module import build_real_data_module  # noqa: E402
from hypertagging.training.fixed_validation import select_validation_events  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str, *, suffix: str) -> Path:
    path = (ROOT / value).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    if path.suffix != suffix:
        raise ValueError(f"expected {suffix} input: {value}")
    return path


def nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def select_confirmation_events(
    events: Iterable[Any],
    *,
    event_count: int,
    rank_offset: int,
    seed: int,
    selection_manifest_hash: str,
    excluded_uids: set[str],
) -> tuple[list[Any], tuple[str, ...], dict[str, Any]]:
    if event_count <= 0 or rank_offset < 0:
        raise ValueError("event_count must be positive and rank_offset nonnegative")
    eligible = (
        event for event in events
        if str(getattr(event, "event_uid")) not in excluded_uids
    )
    selected, uids, source_contract = select_validation_events(
        eligible,
        limit=event_count + rank_offset,
        scientific_mode=True,
        selection_manifest_hash=selection_manifest_hash,
        seed=seed,
    )
    if len(selected) != event_count + rank_offset:
        raise ValueError("validation role is too small for the requested disjoint cohort")
    selected = selected[rank_offset:]
    uids = uids[rank_offset:]
    if len(selected) != event_count or len(set(uids)) != event_count:
        raise RuntimeError("confirmation cohort is incomplete or contains duplicate UIDs")
    if set(uids) & excluded_uids:
        raise RuntimeError("confirmation cohort overlaps the original rollout cohort")
    return selected, uids, {
        **source_contract,
        "rank_offset": rank_offset,
        "event_count": event_count,
        "excluded_uid_count": len(excluded_uids),
        "excluded_uid_sha256": hashlib.sha256(
            "\n".join(sorted(excluded_uids)).encode("utf-8")
        ).hexdigest(),
    }


def paired_bootstrap(
    differences: list[float],
    *,
    samples: int,
    seed: int,
    lower_index: int,
    upper_index: int,
    minimum_effect_size: float,
) -> dict[str, Any]:
    if not differences or samples <= 0:
        raise ValueError("paired bootstrap needs observations and positive samples")
    if not (0 <= lower_index < upper_index < samples):
        raise ValueError("bootstrap percentile indices are invalid")
    values = np.asarray(differences, dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty(samples, dtype=np.float64)
    cursor = 0
    while cursor < samples:
        batch = min(256, samples - cursor)
        indices = rng.integers(0, len(values), size=(batch, len(values)), dtype=np.int32)
        bootstrap[cursor : cursor + batch] = values[indices].mean(axis=1)
        cursor += batch
    bootstrap.sort()
    centered = bootstrap - observed
    one_sided_p = (1 + int(np.count_nonzero(centered >= observed))) / (samples + 1)
    return {
        "observation_count": len(differences),
        "estimate": observed,
        "confidence_interval": [
            float(bootstrap[lower_index]),
            float(bootstrap[upper_index]),
        ],
        "percentile_indices_zero_based": [lower_index, upper_index],
        "samples": samples,
        "seed": seed,
        "engine": "numpy.random.Generator(PCG64)",
        "one_sided_centered_null_add_one_p": one_sided_p,
        "minimum_effect_size": minimum_effect_size,
        "effect_size_gate_passed": observed >= minimum_effect_size,
        "confidence_lower_bound_positive": float(bootstrap[lower_index]) > 0.0,
    }


def numeric_macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, bool))})
    return {
        key: float(sum(float(row[key]) for row in rows if key in row) / sum(key in row for row in rows))
        for key in keys
    }


def event_metrics(
    model: torch.nn.Module,
    event: Any,
    data_module: Any,
    *,
    device: torch.device,
    rollout_config: RolloutConfig,
    target_policy: str,
) -> dict[str, Any]:
    truth = data_module.normalize_batch(collate_heterogeneous_events([event]))
    truth = {name: value.to(device) for name, value in truth.items()}
    predicted = level_rollout(model, truth, mode="predicted", config=rollout_config)
    summary = summarize_rollout(predicted.batch, truth)
    complete_correct, complete_eligible = complete_target_efficiency_counts(
        predicted.batch, truth, target_policy=target_policy
    )
    return {
        **{
            name: (bool(value) if isinstance(value, bool) else float(value))
            for name, value in summary.items()
        },
        "complete_target_correct": int(complete_correct),
        "complete_target_eligible": int(complete_eligible),
        "complete_target_efficiency": complete_correct / max(complete_eligible, 1),
        "rollout_valid": bool(predicted.valid),
        "stop_reason": str(predicted.stop_reason),
    }


def evaluate(contract_path: Path, output: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prior = contract["prior_decision"]
    prior_path = repo_path(str(prior["path"]), suffix=".json")
    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
    if sha256(prior_path) != prior["sha256"]:
        raise RuntimeError("prior no-promotion decision hash changed")
    if (
        prior_payload.get("decision", {}).get("verdict") != "NO_PROMOTION"
        or prior_payload.get("decision", {}).get("promotion_authorized") is not False
        or prior_payload.get("decision", {}).get("sealed_test_authorized") is not False
    ):
        raise RuntimeError("prior decision does not preserve no-promotion/test denial")
    arms = list(contract["arms"])
    if len(arms) != 2:
        raise ValueError("paired confirmation requires exactly two arms")
    checkpoint_paths = [repo_path(str(arm["checkpoint"]), suffix=".pt") for arm in arms]
    hashes_before = [sha256(path) for path in checkpoint_paths]
    for arm, actual in zip(arms, hashes_before, strict=True):
        if actual != arm["checkpoint_sha256"]:
            raise RuntimeError("checkpoint hash does not match the confirmation contract")
    payloads = [load_training_checkpoint(path, map_location="cpu") for path in checkpoint_paths]
    reference = payloads[0]
    for payload in payloads[1:]:
        for key in ("architecture", "feature_contract"):
            if not nested_equal(reference[key], payload[key]):
                raise RuntimeError(f"checkpoint {key} contracts differ")
        if not nested_equal(reference["normalizer_state"], payload["normalizer_state"]):
            raise RuntimeError("checkpoint normalizer states differ")
        for key in (
            "target_policy",
            "rollout_pid_kinematics_mode",
            "rollout_pid_temperature",
        ):
            if reference["config"][key] != payload["config"][key]:
                raise RuntimeError(f"checkpoint configuration differs for {key}")

    data = contract["data"]
    selection_path = repo_path(str(data["selection_manifest"]), suffix=".json")
    index_path = repo_path(str(data["dataset_index"]), suffix=".json")
    if sha256(selection_path) != data["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest hash changed")
    if sha256(index_path) != data["dataset_index_sha256"]:
        raise RuntimeError("dataset index hash changed")
    config = reference["config"]
    data_module = build_real_data_module(
        selection_path,
        seed=int(config["seed"]),
        normalization_state=reference["normalizer_state"],
        dataset_index=index_path,
        target_policy=str(config["target_policy"]),
        scientific_mode=True,
    )
    if data_module.split_counts.get("test", 0) != 0:
        raise RuntimeError("sealed test data entered the confirmation data module")
    original_uids = {
        str(uid)
        for payload in payloads
        for uid in payload["validation_selection"]["rollout_event_uids"]
    }
    cohort = contract["cohort"]
    events, selected_uids, selection_contract = select_confirmation_events(
        data_module.iter_events("validation", shuffle=False),
        event_count=int(cohort["event_count"]),
        rank_offset=int(cohort["rank_offset"]),
        seed=int(cohort["selection_seed"]),
        selection_manifest_hash=str(data_module.selection_manifest_hash),
        excluded_uids=original_uids,
    )
    device = torch.device(str(contract["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confirmation execution requires an allocated CUDA device")
    models = [_model_from_checkpoint(payload, data_module, device) for payload in payloads]
    policy = ReconstructionConstraintPolicy.from_dict(
        reference["feature_contract"]["reconstruction_constraint_policy"]
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
    per_arm: dict[str, list[dict[str, Any]]] = {str(arm["role"]): [] for arm in arms}
    event_rows: list[dict[str, Any]] = []
    for event in events:
        row = {"event_uid": str(event.event_uid), "arms": {}}
        for arm, model in zip(arms, models, strict=True):
            role = str(arm["role"])
            metrics = event_metrics(
                model,
                event,
                data_module,
                device=device,
                rollout_config=rollout_config,
                target_policy=str(config["target_policy"]),
            )
            row["arms"][role] = metrics
            per_arm[role].append(metrics)
        event_rows.append(row)
    left_role = str(arms[0]["role"])
    right_role = str(arms[1]["role"])
    differences = [
        float(row["arms"][left_role]["edge_f1"])
        - float(row["arms"][right_role]["edge_f1"])
        for row in event_rows
    ]
    analysis = contract["analysis"]
    contrast = paired_bootstrap(
        differences,
        samples=int(analysis["paired_bootstrap_samples"]),
        seed=int(cohort["bootstrap_seed"]),
        lower_index=int(analysis["percentile_lower_index_zero_based"]),
        upper_index=int(analysis["percentile_upper_index_zero_based"]),
        minimum_effect_size=float(analysis["minimum_effect_size"]),
    )
    contrast.update({
        "name": str(analysis["contrast"]),
        "left_role": left_role,
        "right_role": right_role,
        "left_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "right_wins": sum(value < 0 for value in differences),
        "promotion_authorized": False,
    })
    hashes_after = [sha256(path) for path in checkpoint_paths]
    if hashes_after != hashes_before:
        raise RuntimeError("a source checkpoint changed during validation-only evaluation")
    report = {
        "schema_version": "hypertagging-reconstruction-paired-confirmation-result-v1",
        "status": "completed",
        "study_id": contract["study_id"],
        "task_id": contract["task_id"],
        "mode": "validation_only_paired_confirmation",
        "training_performed": False,
        "sealed_test_role_access": "forbidden",
        "promotion_authorized": False,
        "checkpoints_unchanged": True,
        "checkpoint_hashes": {
            str(arm["role"]): digest
            for arm, digest in zip(arms, hashes_after, strict=True)
        },
        "split_counts": data_module.split_counts,
        "selection": {
            **selection_contract,
            "event_uids": list(selected_uids),
            "original_rollout_cohort_disjoint": True,
        },
        "aggregates": {role: numeric_macro(rows) for role, rows in per_arm.items()},
        "paired_primary_contrast": contrast,
        "events": event_rows,
    }
    require_finite_json(report)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["result_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing confirmation result")
    report = evaluate(args.contract.resolve(strict=True), args.output.resolve())
    print(json.dumps({
        "status": report["status"],
        "task_id": report["task_id"],
        "result": str(args.output.resolve()),
        "result_sha256": report["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
