#!/usr/bin/env python3
"""Execute one contract-bound phase-35 reconstruction training arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypertagging.training.fixed_validation import (  # noqa: E402
    excluded_event_uids_contract,
)
from hypertagging.training.data_module import (  # noqa: E402
    BALANCED_LEVEL_REPLAY_LEVELS,
)
from hypertagging.training.reconstruction_trainer import (  # noqa: E402
    ReconstructionConfig,
    train_level_reconstruction,
)
from scripts.slurm.verify_reconstruction_phase35_contract import (  # noqa: E402
    load_evaluation_cohort,
    load_preregistration,
    load_validation_exclusions,
    uid_sequence_sha256,
    verify_contract,
)


PHASE35_REPLAY_SLOT_BUDGET = 140_032
PHASE35_REPLAY_SLOT_COUNTS = {
    1: 23_339,
    2: 23_339,
    3: 23_339,
    4: 23_339,
    5: 23_338,
    6: 23_338,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_payload(payload: Any, path: str = "payload") -> list[str]:
    offenders: list[str] = []

    def visit(value: Any, name: str) -> None:
        if torch.is_tensor(value):
            if not torch.isfinite(value).all().item():
                offenders.append(name)
            return
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), f"{name}.{field.name}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]")
            return
        if isinstance(value, float) and not math.isfinite(value):
            offenders.append(name)

    visit(payload, path)
    return offenders


def finite_checkpoint(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    offenders = finite_payload(payload)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"checkpoint has no model state: {path}")
    model_tensors = [
        value for value in model_state.values() if torch.is_tensor(value)
    ]
    if not model_tensors:
        raise RuntimeError(f"checkpoint has no model tensors: {path}")
    if offenders:
        raise RuntimeError(
            f"checkpoint contains non-finite state: {path}: {offenders[:8]}"
        )

    def tensor_count(value: Any) -> int:
        if torch.is_tensor(value):
            return 1
        if isinstance(value, dict):
            return sum(tensor_count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_count(item) for item in value)
        return 0

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "step": int(payload.get("step", -1)),
        "tensor_count_recursive": tensor_count(payload),
        "model_tensor_count": len(model_tensors),
        "all_checkpoint_tensors_finite": True,
        "all_model_tensors_finite": True,
        "metrics": dict(payload.get("metrics", {})),
    }


def json_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            json_finite(item, f"{path}[{index}]")


def _exact_integer(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
    ):
        raise RuntimeError(f"phase35 metric {name} is not an exact integer")
    return int(value)


def _optimizer_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"phase35 metrics JSON is invalid at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(
                f"phase35 metrics row {line_number} is not an object"
            )
        json_finite(row, f"metrics[{line_number}]")
        if row.get("split") == "validation":
            continue
        if isinstance(row.get("step"), bool) or not isinstance(
            row.get("step"), int
        ):
            raise RuntimeError(
                f"phase35 optimizer metrics row {line_number} has no integer step"
            )
        rows.append(row)
    return rows


def replay_slot_audit(
    metrics_path: Path,
    *,
    config: dict[str, Any],
    trainer_contract: dict[str, object] | None,
) -> dict[str, Any]:
    """Verify every phase35 optimizer presentation against global replay order."""

    levels = tuple(BALANCED_LEVEL_REPLAY_LEVELS)
    batch_size = int(config["batch_size"])
    max_steps = int(config["max_steps"])
    slot_budget = int(config["replay_slot_budget"])
    configured_counts = {
        int(level): int(count)
        for level, count in config["replay_slot_counts_by_level"].items()
    }
    if (
        config.get("level_sampling_mode") != "balanced_level_replay"
        or batch_size != 64
        or max_steps != 2188
        or slot_budget != PHASE35_REPLAY_SLOT_BUDGET
        or slot_budget != batch_size * max_steps
        or configured_counts != PHASE35_REPLAY_SLOT_COUNTS
    ):
        raise RuntimeError("phase35 replay slot preregistration changed")
    if not isinstance(trainer_contract, dict):
        raise RuntimeError("trainer did not expose a balanced replay contract")
    preregistered_contract = config.get("balanced_level_replay_contract")
    if (
        not isinstance(preregistered_contract, dict)
        or trainer_contract != preregistered_contract
    ):
        raise RuntimeError(
            "trainer balanced replay contract does not exactly match the "
            "preregistered real-data contract"
        )
    planned = trainer_contract.get("planned_schedule")
    pool_counts = trainer_contract.get("eligible_pool_counts_by_level")
    pool_hashes = trainer_contract.get("eligible_pool_uid_sha256_by_level")
    string_levels = {str(level) for level in levels}
    if (
        trainer_contract.get("version") != "balanced-level-replay-v1"
        or trainer_contract.get("levels") != list(levels)
        or trainer_contract.get("seed") != int(config["seed"])
        or trainer_contract.get("target_policy") != config["target_policy"]
        or trainer_contract.get("min_daughters") != 2
        or not isinstance(planned, dict)
        or planned.get("start_slot") != 0
        or planned.get("slot_count") != slot_budget
        or planned.get("end_slot_exclusive") != slot_budget
        or planned.get("level_counts")
        != {str(level): configured_counts[level] for level in levels}
        or planned.get("max_minus_min_level_count") != 1
        or not isinstance(pool_counts, dict)
        or set(pool_counts) != string_levels
        or any(_exact_integer(value, name=f"pool_count_{level}") <= 0 for level, value in pool_counts.items())
        or not isinstance(pool_hashes, dict)
        or set(pool_hashes) != string_levels
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in pool_hashes.values()
        )
    ):
        raise RuntimeError("trainer balanced replay contract changed")

    rows = _optimizer_metric_rows(metrics_path)
    observed_steps = [int(row["step"]) for row in rows]
    if observed_steps != list(range(1, max_steps + 1)):
        raise RuntimeError("phase35 optimizer metric steps are not exact and contiguous")
    selected_totals = {level: 0 for level in levels}
    optimized_totals = {level: 0 for level in levels}
    previous_end = 0
    for row in rows:
        step = int(row["step"])
        expected_start = (step - 1) * batch_size
        expected_end = step * batch_size
        start = _exact_integer(
            row.get("balanced_replay_slot_start"),
            name=f"step_{step}_slot_start",
        )
        end = _exact_integer(
            row.get("balanced_replay_slot_end_exclusive"),
            name=f"step_{step}_slot_end",
        )
        if start != expected_start or start != previous_end or end != expected_end:
            raise RuntimeError(
                f"phase35 replay slot range is not contiguous at step {step}"
            )
        previous_end = end
        expected_step_counts = {level: 0 for level in levels}
        for slot in range(expected_start, expected_end):
            expected_step_counts[levels[slot % len(levels)]] += 1
        selected_step: dict[int, int] = {}
        optimized_step: dict[int, int] = {}
        for level in levels:
            selected_step[level] = _exact_integer(
                row.get(f"selected_event_level_{level}_count"),
                name=f"step_{step}_selected_level_{level}",
            )
            optimized_step[level] = _exact_integer(
                row.get(f"optimized_event_level_{level}_count"),
                name=f"step_{step}_optimized_level_{level}",
            )
        if (
            selected_step != expected_step_counts
            or optimized_step != selected_step
            or sum(selected_step.values()) != batch_size
            or _exact_integer(
                row.get("optimized_event_level_count"),
                name=f"step_{step}_optimized_total",
            )
            != batch_size
            or row.get("target_levels") != list(levels)
        ):
            raise RuntimeError(
                f"phase35 selected/optimized replay counts changed at step {step}"
            )
        for level in levels:
            selected_totals[level] += selected_step[level]
            optimized_totals[level] += optimized_step[level]
    if (
        previous_end != slot_budget
        or selected_totals != configured_counts
        or optimized_totals != configured_counts
    ):
        raise RuntimeError("phase35 cumulative replay slot counts changed")
    return {
        "version": "phase35-balanced-replay-runtime-audit-v1",
        "passed": True,
        "optimizer_steps": len(rows),
        "batch_size": batch_size,
        "slot_count": slot_budget,
        "slot_start": 0,
        "slot_end_exclusive": previous_end,
        "selected_slot_counts_by_level": {
            str(level): selected_totals[level] for level in levels
        },
        "optimized_slot_counts_by_level": {
            str(level): optimized_totals[level] for level in levels
        },
        "eligible_pool_counts_by_level": dict(pool_counts),
        "eligible_pool_uid_sha256_by_level": dict(pool_hashes),
        "trainer_contract_match": True,
        "preregistered_contract_match": True,
    }


def first_twenty_gate(metrics_path: Path) -> dict[str, Any]:
    rows = _optimizer_metric_rows(metrics_path)
    rows = [row for row in rows if 1 <= int(row["step"]) <= 20]
    by_step = {int(row["step"]): row for row in rows}
    missing = sorted(set(range(1, 21)) - set(by_step))
    required = (
        "forward_finite",
        "loss_finite",
        "raw_gradient_finite",
        "model_finite",
        "optimizer_finite",
        "scheduler_finite",
    )
    failures = [
        f"step-{step}:{field}"
        for step, row in sorted(by_step.items())
        for field in required
        if row.get(field) != 1.0
    ]
    if missing or failures:
        raise RuntimeError(
            f"first-20 optimizer gate failed: missing={missing} "
            f"failures={failures[:8]}"
        )
    return {
        "passed": True,
        "steps": 20,
        "required_fields": list(required),
        "min_step": 1,
        "max_step": 20,
        "rows": len(rows),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pairs(values: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(level), int(value)) for level, value in values)


def _float_pairs(values: list[list[float]]) -> tuple[tuple[int, float], ...]:
    return tuple((int(level), float(value)) for level, value in values)


def _training_config(
    *,
    config: dict[str, Any],
    runtime: dict[str, str],
    training_output: Path,
    validation_exclusions: tuple[str, ...],
) -> ReconstructionConfig:
    return ReconstructionConfig(
        data=runtime["selection_manifest"],
        output_dir=str(training_output),
        pretrained_encoder=runtime["checkpoint"],
        device="cuda",
        max_steps=int(config["max_steps"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
        learning_rate=float(config["learning_rate"]),
        lr_schedule_total_steps=int(config["lr_schedule_total_steps"]),
        warmup_fraction=float(config["warmup_fraction"]),
        warmup_steps=int(config["warmup_steps"]),
        max_warmup_steps=int(config["max_warmup_steps"]),
        min_lr_ratio=float(config["min_lr_ratio"]),
        encoder_lr_multiplier=float(config["encoder_lr_multiplier"]),
        freeze_pretrained_encoder_steps=int(
            config["freeze_pretrained_encoder_steps"]
        ),
        checkpoint_every=int(config["checkpoint_every"]),
        validate_every=int(config["validate_every"]),
        rollout_validate_every=int(config["rollout_validate_every"]),
        n_queries_by_level=_pairs(config["n_queries_by_level"]),
        max_cardinality=int(config["max_cardinality"]),
        max_cardinality_by_level=_pairs(
            config["max_cardinality_by_level"]
        ),
        scheduled_sampling_probability=float(
            config["scheduled_sampling_probability"]
        ),
        scheduled_sampling_schedule=str(
            config["scheduled_sampling_schedule"]
        ),
        scheduled_sampling_duration_steps=int(
            config["scheduled_sampling_duration_steps"]
        ),
        allow_tiny_bruteforce_matching=False,
        mixed_precision=bool(config["mixed_precision"]),
        amp_dtype=str(config["amp_dtype"]),
        grad_scaler_enabled=bool(config["grad_scaler_enabled"]),
        validation_enabled=bool(config["validation_enabled"]),
        transfer_leaf_pid_head=bool(config["transfer_leaf_pid_head"]),
        require_exact_leaf_pid_transfer=bool(
            config["require_exact_leaf_pid_transfer"]
        ),
        freeze_leaf_pid_head_steps=int(
            config["freeze_leaf_pid_head_steps"]
        ),
        leaf_pid_lr_multiplier=float(config["leaf_pid_lr_multiplier"]),
        target_policy=str(config["target_policy"]),
        max_validation_events=int(config["max_validation_events"]),
        rollout_validation_events=int(
            config["rollout_validation_events"]
        ),
        validation_batch_size=int(config["validation_batch_size"]),
        validation_excluded_event_uids=validation_exclusions,
        auxiliary_teacher_weight=float(config["auxiliary_teacher_weight"]),
        dataset_index=runtime["dataset_index"],
        unrepresentable_target_policy=str(
            config["unrepresentable_target_policy"]
        ),
        level_sampling_mode=str(config["level_sampling_mode"]),
        empirical_type_prior_mode=str(
            config["empirical_type_prior_mode"]
        ),
        model_preset=str(config["model_preset"]),
        query_repulsion_weight=float(config["query_repulsion_weight"]),
        object_positive_weight=float(config["object_positive_weight"]),
        pointer_positive_weight=float(config["pointer_positive_weight"]),
        level_loss_weights=_float_pairs(config.get("level_loss_weights", [])),
        recovery_objective_weight=float(
            config.get("recovery_objective_weight", 1.0)
        ),
        best_metric=str(config["best_metric"]),
        best_mode=str(config["best_mode"]),
        early_stopping_patience=config["early_stopping_patience"],
        initial_state_policy=str(config["initial_state_policy"]),
        rollout_pid_kinematics_mode=str(
            config["rollout_pid_kinematics_mode"]
        ),
        rollout_pid_temperature=float(config["rollout_pid_temperature"]),
        rollout_object_threshold=float(
            config.get("rollout_object_threshold", 0.5)
        ),
        rollout_pointer_threshold=float(
            config.get("rollout_pointer_threshold", 0.5)
        ),
        rollout_continue_through_empty_levels=bool(
            config["rollout_continue_through_empty_levels"]
        ),
        rollout_max_level=int(config["rollout_max_level"]),
        rollout_root_types=tuple(
            int(token) for token in config["rollout_root_types"]
        ),
        rollout_exclusive_final=bool(config["rollout_exclusive_final"]),
        rollout_use_learned_confidence=bool(
            config["rollout_use_learned_confidence"]
        ),
        rollout_min_tree_validity=float(
            config.get("rollout_min_tree_validity", 0.999)
        ),
        rollout_min_p4_closure=float(
            config.get("rollout_min_p4_closure", 1.0)
        ),
        rollout_min_depth_fraction=float(
            config.get("rollout_min_depth_fraction", 0.0)
        ),
        rollout_min_complete_target_efficiency=float(
            config.get("rollout_min_complete_target_efficiency", 0.0)
        ),
        type_conditioned_daughter_relation_bias=bool(
            config["type_conditioned_daughter_relation_bias"]
        ),
        num_workers=int(config["num_workers"]),
        scientific_mode=bool(config["scientific_mode"]),
        log_every=int(config["log_every"]),
        pilot_allow_train_validation_fallback=False,
        minimum_encoder_transfer_coverage=float(
            config["minimum_encoder_transfer_coverage"]
        ),
        allow_low_encoder_transfer_coverage=bool(
            config["allow_low_encoder_transfer_coverage"]
        ),
    )


def _validation_selection_audit(
    checkpoint: Path,
    *,
    exclusions: tuple[str, ...],
    expected_event_uids: tuple[str, ...],
    expected_rollout_event_uids: tuple[str, ...],
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selection = payload.get("validation_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("final checkpoint has no validation selection")
    event_uids = tuple(str(uid) for uid in selection.get("event_uids", ()))
    rollout_uids = tuple(
        str(uid) for uid in selection.get("rollout_event_uids", ())
    )
    if event_uids != expected_event_uids:
        raise RuntimeError(
            "phase35 fixed validation cohort does not exactly match the "
            "preregistered ordered checkpoint-selection UIDs"
        )
    if rollout_uids != expected_rollout_event_uids:
        raise RuntimeError(
            "phase35 rollout cohort does not exactly match the preregistered "
            "ordered checkpoint-selection prefix"
        )
    if expected_rollout_event_uids != expected_event_uids[
        : len(expected_rollout_event_uids)
    ]:
        raise RuntimeError("phase35 expected rollout UIDs are not an ordered prefix")
    if (
        selection.get("split") != "validation"
        or selection.get("rollout_was_run") is not True
        or selection.get("deterministic") is not True
        or selection.get("scientific_mode") is not True
        or selection.get("strategy") != "manifest_validation_role_uid_hash"
    ):
        raise RuntimeError("phase35 final validation selection metadata changed")
    overlap = set(event_uids) & set(exclusions)
    if overlap:
        raise RuntimeError(
            f"phase35 validation cohort reuses prior events: {sorted(overlap)[:3]}"
        )
    expected_identity = excluded_event_uids_contract(exclusions)
    stored_identity = {
        key: selection.get(key) for key in expected_identity
    }
    if stored_identity != expected_identity:
        raise RuntimeError("checkpoint validation exclusion identity changed")
    return {
        "event_count": len(event_uids),
        "rollout_event_count": len(rollout_uids),
        "event_uids_sha256": uid_sequence_sha256(list(event_uids)),
        "rollout_event_uids_sha256": uid_sequence_sha256(list(rollout_uids)),
        "ordered_event_uids_match": True,
        "ordered_rollout_prefix_match": True,
        "prior_cohort_overlap_count": 0,
        **expected_identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    output = args.output.resolve()
    training_output = args.training_output_dir.resolve()
    output_root = Path(runtime["output_root"]).resolve()
    if output_root not in output.parents or output_root not in training_output.parents:
        raise RuntimeError("phase35 outputs escape the arm output root")
    if output.exists():
        raise RuntimeError("refusing to overwrite phase35 result")
    if training_output.exists():
        raise RuntimeError("refusing to reuse phase35 training output")

    source_checkpoint = Path(runtime["checkpoint"]).resolve(strict=True)
    source_before = sha256(source_checkpoint)
    config = dict(contract["config"])
    preregistration = load_preregistration()
    exclusions = load_validation_exclusions(preregistration)
    expected_exclusion = preregistration["validation_exclusion"]
    exclusion_contract = excluded_event_uids_contract(exclusions)
    if (
        exclusion_contract["excluded_event_uid_count"]
        != expected_exclusion["event_uid_count"]
        or exclusion_contract["excluded_event_uids_sha256"]
        != expected_exclusion["event_uids_sha256"]
        or exclusion_contract["excluded_event_uids_hash_scheme"]
        != expected_exclusion["event_uids_hash_scheme"]
    ):
        raise RuntimeError("phase35 runtime validation exclusions changed")
    evaluation_cohort = load_evaluation_cohort(
        preregistration, exclusions=exclusions
    )
    checkpoint_selection_event_uids = tuple(
        str(uid)
        for uid in evaluation_cohort["checkpoint_selection_event_uids"]
    )
    rollout_event_count = int(config["rollout_validation_events"])
    if (
        len(checkpoint_selection_event_uids)
        != int(config["max_validation_events"])
        or len(checkpoint_selection_event_uids) != 2_000
        or rollout_event_count != 1_000
    ):
        raise RuntimeError("phase35 fixed validation limits changed")
    rollout_event_uids = checkpoint_selection_event_uids[:rollout_event_count]

    started = time.perf_counter()
    result = train_level_reconstruction(
        _training_config(
            config=config,
            runtime=runtime,
            training_output=training_output,
            validation_exclusions=exclusions,
        )
    )
    elapsed = time.perf_counter() - started
    if result.steps != int(config["max_steps"]):
        raise RuntimeError(
            f"phase35 training stopped at {result.steps}, "
            f"expected {config['max_steps']}"
        )
    transfer_report = result.transfer_report
    minimum_transfer_coverage = float(
        config["minimum_encoder_transfer_coverage"]
    )
    if (
        transfer_report is None
        or transfer_report.coverage < minimum_transfer_coverage
        or transfer_report.shape_mismatches
        or transfer_report.unexpected_keys
    ):
        raise RuntimeError("phase35 encoder transfer contract failed")
    if bool(config["require_exact_leaf_pid_transfer"]) and (
        not transfer_report.leaf_pid_loaded_keys
        or transfer_report.leaf_pid_missing_keys
        or transfer_report.leaf_pid_unexpected_keys
        or transfer_report.leaf_pid_shape_mismatches
    ):
        raise RuntimeError("phase35 exact leaf PID transfer contract failed")
    split_counts = dict(result.data_module.split_counts)
    if split_counts != contract["data"]["split_counts"]:
        raise RuntimeError(
            f"phase35 runtime split counts changed: {split_counts}"
        )
    source_after = sha256(source_checkpoint)
    if (
        source_before != source_after
        or source_after != runtime["checkpoint_sha256"]
    ):
        raise RuntimeError("phase35 source checkpoint changed during training")

    metrics_path = result.log_path.resolve(strict=True)
    first20 = first_twenty_gate(metrics_path)
    checkpoint_paths = {
        "final": result.checkpoint.resolve(strict=True),
        "best": (training_output / "best.pt").resolve(strict=True),
        "best_rollout_edge_f1": (
            training_output / "best_rollout_edge_f1.pt"
        ).resolve(strict=True),
    }
    checkpoint_audits = {
        name: finite_checkpoint(path)
        for name, path in checkpoint_paths.items()
    }
    if checkpoint_audits["final"]["step"] != int(config["max_steps"]):
        raise RuntimeError("phase35 final checkpoint is not from max_steps")
    allowed_steps = {
        500,
        1000,
        1500,
        2000,
        int(config["max_steps"]),
    }
    for name in ("best", "best_rollout_edge_f1"):
        if checkpoint_audits[name]["step"] not in allowed_steps:
            raise RuntimeError(
                f"{name} checkpoint was selected at an unregistered step"
            )
    validation_selection = _validation_selection_audit(
        checkpoint_paths["final"],
        exclusions=exclusions,
        expected_event_uids=checkpoint_selection_event_uids,
        expected_rollout_event_uids=rollout_event_uids,
    )
    replay_audit = replay_slot_audit(
        metrics_path,
        config=config,
        trainer_contract=result.balanced_level_replay_contract,
    )

    result_payload: dict[str, Any] = {
        "status": "training_completed",
        "study_id": contract["study_id"],
        "task_id": contract["task_id"],
        "arm_role": runtime["arm_role"],
        "contract_sha256": runtime["contract_sha256"],
        "optimizer_steps": result.steps,
        "replay_slot_budget": replay_audit["slot_count"],
        "replay_slot_counts_by_level": replay_audit[
            "optimized_slot_counts_by_level"
        ],
        "balanced_level_replay": replay_audit,
        "elapsed_seconds": elapsed,
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "step": int(runtime["checkpoint_step"]),
            "sha256_before": source_before,
            "sha256_after": source_after,
            "unchanged": True,
        },
        "checkpoints": checkpoint_audits,
        "training_log": {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256(metrics_path),
        },
        "first_20_optimizer_steps": first20,
        "config": config,
        "metrics": dict(result.metrics),
        "transfer_report": {
            **asdict(transfer_report),
            "coverage": transfer_report.coverage,
            "minimum_required_coverage": minimum_transfer_coverage,
            "exact_leaf_pid_transfer_required": bool(
                config["require_exact_leaf_pid_transfer"]
            ),
            "passed": True,
        },
        "data": {
            "split_counts": split_counts,
            "selection_manifest_hash": result.data_module.selection_manifest_hash,
            "split_manifest_hash": result.data_module.split_manifest_hash,
            "dataset_index_hash": (
                result.data_module.dataset_index.get("index_hash", "")
                if result.data_module.dataset_index
                else ""
            ),
            "training_role": "train",
            "evaluation_role": "validation",
            "validation_accessed": True,
            "validation_selection": validation_selection,
            "sealed_test_role_access": "forbidden",
        },
        "post_training_policy": {
            "automatic_promotion": False,
            "sealed_test_evaluation_authorized": False,
            "strict_full_decay_evaluation_required": True,
        },
    }
    json_finite(result_payload)
    atomic_json(output, result_payload)
    print(
        json.dumps(
            {"status": "training_completed", "result": str(output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
