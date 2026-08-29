#!/usr/bin/env python
"""Evaluate hierarchical full/half decay reconstruction on schema-v4 data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

# This is an offline CPU evaluator. Hide accelerators before importing torch or
# any project module, including when the caller has a GPU-visible shell.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair  # noqa: E402
from hypertagging.evaluation.full_decay_metrics import (  # noqa: E402
    evaluate_full_decay,
    evaluate_half_decays,
    summarize_decay_evaluations,
    truth_target_policy_diagnostics,
)
from hypertagging.evaluation.full_decay_runner import (  # noqa: E402
    inference_diagnostics,
    serialize_reconstructed_tree,
    summarize_inference_diagnostics,
)
from hypertagging.evaluation.trained_context import (  # noqa: E402
    load_trained_evaluation_context,
)
from hypertagging.reconstruction.hierarchical_inference import (  # noqa: E402
    HierarchicalInferenceConfig,
    OFFLINE_INFERENCE_POLICY_VERSION,
    reconstruct_full_tree_from_fsps,
)
from hypertagging.reconstruction.level_rollout import (  # noqa: E402
    RolloutConfig,
    rollout_policy_identity,
)


REPORT_VERSION = "hypertagging-offline-full-decay-evaluation-v3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-checkpoint", required=True)
    parser.add_argument("--reconstruction-checkpoint", required=True)
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help=(
            "Schema-v4 training-selection manifest. A source-role-bound index "
            "requires its matching manifest; not raw mDST or GraFEI pairs."
        ),
    )
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--event-selection",
        choices=(
            "auto",
            "checkpoint_rollout",
            "checkpoint_validation",
            "stream",
        ),
        default="auto",
        help=(
            "Event cohort policy. Auto restores checkpoint rollout UIDs for "
            "validation and uses stream order only where no checkpoint cohort exists."
        ),
    )
    parser.add_argument(
        "--source-category",
        action="append",
        default=None,
        help=(
            "Optional exact source category filter (repeatable), useful for "
            "separate B-pair and continuum evaluation."
        ),
    )
    parser.add_argument("--scope", choices=("full", "half", "both"), default="both")
    parser.add_argument(
        "--truth-topology-mode",
        choices=("checkpoint_direct", "contracted_diagnostic"),
        default="checkpoint_direct",
        help=(
            "Primary mode follows original direct checkpoint targets. "
            "Contracted topology is diagnostic and is not the trained pointer target."
        ),
    )
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-level", type=int, default=8)
    parser.add_argument("--object-threshold", type=float, default=0.5)
    parser.add_argument("--pointer-threshold", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--type-probability-threshold", type=float, default=None)
    parser.add_argument("--no-cardinality", action="store_true")
    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument(
        "--use-learned-confidence",
        dest="use_learned_confidence",
        action="store_true",
        default=None,
        help="Explicitly enable the trained confidence head.",
    )
    confidence_group.add_argument(
        "--disable-learned-confidence",
        dest="use_learned_confidence",
        action="store_false",
        help=(
            "Diagnostic override; the default restores the checkpoint-selection "
            "rollout policy."
        ),
    )
    parser.add_argument("--p4-closure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--omit-trees", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument(
        "--allow-finetuned-encoder",
        action="store_true",
        help="Allow a future intentionally fine-tuned reconstruction encoder.",
    )
    parser.add_argument(
        "--diagnostic-external-independent-sample",
        action="store_true",
        help="Relax only checkpoint split/hash identity for a verified external sample.",
    )
    parser.add_argument("--output", required=True, help="Destination JSON report.")
    args = parser.parse_args(argv)
    if args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.max_level <= 0:
        parser.error("--max-level must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    for name in (
        "object_threshold",
        "pointer_threshold",
        "confidence_threshold",
        "type_probability_threshold",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.p4_closure_tolerance < 0:
        parser.error("--p4-closure-tolerance must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = _validated_output_path(
        args.output,
        direct_inputs=(
            args.pretraining_checkpoint,
            args.reconstruction_checkpoint,
            args.dataset_index,
            *args.data,
        ),
        data_arguments=args.data,
    )
    run_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        # Another library may already have initialized the inter-op pool.
        pass

    print("Validating checkpoint lineage on CPU...", file=sys.stderr, flush=True)
    phase_started = time.perf_counter()
    checkpoint_pair = validate_checkpoint_pair(
        args.pretraining_checkpoint,
        args.reconstruction_checkpoint,
        require_exact_frozen_encoder=not args.allow_finetuned_encoder,
    )
    phase_seconds["checkpoint_pair_validation"] = time.perf_counter() - phase_started
    print(
        f"Loading {args.max_events} held-out {args.split} event(s) from schema-v4 data...",
        file=sys.stderr,
        flush=True,
    )
    data: str | list[str] = args.data[0] if len(args.data) == 1 else args.data
    phase_started = time.perf_counter()
    context = load_trained_evaluation_context(
        checkpoint=args.reconstruction_checkpoint,
        data=data,
        dataset_index=args.dataset_index,
        split=args.split,
        max_events=args.max_events,
        device="cpu",
        diagnostic_allow_external_independent_sample=(
            args.diagnostic_external_independent_sample
        ),
        source_categories=args.source_category,
        event_selection=args.event_selection,
    )
    phase_seconds["model_and_data_context_loading"] = (
        time.perf_counter() - phase_started
    )
    if any(parameter.device.type != "cpu" for parameter in context.model.parameters()):
        raise RuntimeError("evaluation model is not CPU-only")
    if context.model.training:
        raise RuntimeError("evaluation model was not restored in eval mode")

    policy = context.constraint_policy
    pointer_threshold = (
        float(policy.minimum_pointer_probability)
        if args.pointer_threshold is None
        else float(args.pointer_threshold)
    )
    checkpoint_config = context.checkpoint.get("config", {})
    target_policy = str(checkpoint_config.get("target_policy", "complete_only"))
    confidence_trained = bool(context.checkpoint.get("confidence_head_trained", False))
    checkpoint_selection_contract = context.checkpoint.get(
        "training_state", {}
    ).get("checkpoint_selection_contract", {})
    checkpoint_rollout_contract = checkpoint_selection_contract.get(
        "rollout_configuration", {}
    )
    checkpoint_uses_learned_confidence = bool(
        checkpoint_rollout_contract.get("learned_confidence", confidence_trained)
    )
    use_learned_confidence = (
        checkpoint_uses_learned_confidence
        if args.use_learned_confidence is None
        else bool(args.use_learned_confidence)
    )
    if use_learned_confidence and not confidence_trained:
        raise ValueError(
            "learned confidence is enabled by the resolved rollout policy, but "
            "the checkpoint does not mark that head as trained"
        )
    rollout_config = RolloutConfig(
        max_level=args.max_level,
        object_threshold=float(args.object_threshold),
        pointer_threshold=pointer_threshold,
        confidence_threshold=float(args.confidence_threshold),
        type_probability_threshold=args.type_probability_threshold,
        min_daughters=int(policy.minimum_daughters),
        use_cardinality=not args.no_cardinality,
        exclusive_final=True,
        use_learned_confidence=use_learned_confidence,
        confidence_trained=confidence_trained,
        cardinality_insufficient_policy=str(policy.cardinality_insufficient_policy),
        constraint_policy=policy,
        rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode,
        rollout_pid_temperature=float(
            checkpoint_config.get("rollout_pid_temperature", 0.5)
        ),
        profile_phases=bool(args.profile_phases),
    )
    scopes = ("full", "half") if args.scope == "both" else (args.scope,)
    event_records: list[dict[str, Any]] = []
    metric_rows: dict[str, list[Any]] = defaultdict(list)
    diagnostic_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_rows_by_category: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    diagnostics_by_category: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    metric_rows_by_target_shape: dict[
        str, dict[str, list[Any]]
    ] = defaultdict(lambda: defaultdict(list))
    inference_wall_seconds = 0.0
    metric_wall_seconds = 0.0
    serialization_wall_seconds = 0.0
    event_processing_started = time.perf_counter()

    for event_index, event in enumerate(context.events):
        print(
            f"[{event_index + 1}/{len(context.events)}] {event.event_uid}",
            file=sys.stderr,
            flush=True,
        )
        truth_batch = context.collated_event_batch(event_index, device="cpu")
        record: dict[str, Any] = {
            "event_uid": event.event_uid,
            "source_category": event.source_category,
            "source_file": event.source_file,
            "truth_active_node_count": int(truth_batch["node_mask"].sum()),
            "truth_fsp_count": int(
                (truth_batch["node_mask"] & (truth_batch["level_ids"] == 0)).sum()
            ),
            "truth_target_policy": truth_target_policy_diagnostics(
                truth_batch,
                target_policy=target_policy,
                minimum_daughters=int(policy.minimum_daughters),
            ),
            "scopes": {},
        }
        for scope in scopes:
            phase_started = time.perf_counter()
            inference = reconstruct_full_tree_from_fsps(
                context.model,
                truth_batch,
                config=HierarchicalInferenceConfig(
                    scope=scope,
                    rollout_config=rollout_config,
                    max_level=args.max_level,
                ),
            )
            inference_wall_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            diagnostics = inference_diagnostics(
                inference,
                p4_tolerance=float(args.p4_closure_tolerance),
            )
            if scope == "full":
                evaluation = evaluate_full_decay(
                    inference.batch,
                    truth_batch,
                    target_policy=target_policy,
                    minimum_daughters=int(policy.minimum_daughters),
                    truth_topology_mode=args.truth_topology_mode,
                )
            else:
                evaluation = evaluate_half_decays(
                    inference.batch,
                    truth_batch,
                    source_category=event.source_category,
                    target_policy=target_policy,
                    minimum_daughters=int(policy.minimum_daughters),
                    truth_topology_mode=args.truth_topology_mode,
                )
            metric_wall_seconds += time.perf_counter() - phase_started
            scope_record: dict[str, Any] = {
                "input_audit": inference.input_audit.as_dict(),
                "inference": diagnostics,
                "metrics": evaluation.as_dict(),
            }
            if not args.omit_trees:
                phase_started = time.perf_counter()
                scope_record["reconstructed_tree"] = serialize_reconstructed_tree(
                    inference
                )
                serialization_wall_seconds += time.perf_counter() - phase_started
            if inference.rollout.host_phase_seconds is not None:
                scope_record["host_phase_seconds"] = dict(
                    inference.rollout.host_phase_seconds
                )
            record["scopes"][scope] = scope_record
            metric_rows[scope].append(evaluation)
            diagnostic_rows[scope].append(diagnostics)
            metric_rows_by_category[event.source_category][scope].append(
                evaluation
            )
            diagnostics_by_category[event.source_category][scope].append(
                diagnostics
            )
            units = evaluation.halves if scope == "half" else (evaluation,)
            for unit in units:
                if not unit.available or unit.truth_retained_depth is None:
                    continue
                shape = (
                    f"fsp_count={len(unit.truth_sources)};"
                    f"retained_depth={unit.truth_retained_depth}"
                )
                metric_rows_by_target_shape[scope][shape].append(unit)
        event_records.append(record)

    phase_seconds["event_processing"] = time.perf_counter() - event_processing_started
    phase_seconds["hierarchical_inference"] = inference_wall_seconds
    phase_seconds["metrics_and_diagnostics"] = metric_wall_seconds
    phase_seconds["tree_serialization"] = serialization_wall_seconds
    phase_seconds["total_before_report_write"] = time.perf_counter() - run_started

    report = {
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "offline_model_evaluation",
        "offline_inference_policy_version": OFFLINE_INFERENCE_POLICY_VERSION,
        "rollout_policy_identity": rollout_policy_identity(
            continue_through_empty_levels=True
        ),
        "evaluator_code_provenance": _evaluator_code_provenance(),
        "not_basf2_reconstruction": True,
        "input_contract": "direct-mdst-tree-v4-preprocessed-training-input",
        "kinematic_reference": "reconstructed_fsp_daughter_sum",
        "physical_momentum_error_available": False,
        "physical_momentum_error_unavailable_reason": (
            "preprocessed_training_schema_does_not_retain_mc_composite_p4"
        ),
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_num_threads": torch.get_num_threads(),
        "timing": {
            "phase_seconds": phase_seconds,
            "event_count": len(event_records),
            "scope_evaluation_count": len(event_records) * len(scopes),
            "events_per_event_processing_second": (
                len(event_records) / phase_seconds["event_processing"]
                if phase_seconds["event_processing"] > 0
                else None
            ),
        },
        "checkpoint_pair": checkpoint_pair.as_dict(),
        "context": context.report_metadata,
        "configuration": {
            "requested_scope": args.scope,
            "evaluated_scopes": list(scopes),
            "max_events": args.max_events,
            "source_categories": args.source_category or [],
            "requested_event_selection": args.event_selection,
            "max_level": args.max_level,
            "object_threshold": args.object_threshold,
            "pointer_threshold": pointer_threshold,
            "confidence_threshold": args.confidence_threshold,
            "type_probability_threshold": args.type_probability_threshold,
            "use_cardinality": not args.no_cardinality,
            "use_learned_confidence": use_learned_confidence,
            "learned_confidence_policy_source": (
                "checkpoint_selection_contract"
                if args.use_learned_confidence is None
                else "explicit_cli_override"
            ),
            "confidence_head_trained": confidence_trained,
            "checkpoint_selection_contract": checkpoint_selection_contract,
            "p4_closure_tolerance": args.p4_closure_tolerance,
            "rollout_pid_kinematics_mode": context.rollout_pid_kinematics_mode,
            "rollout_pid_temperature": rollout_config.rollout_pid_temperature,
            "constraint_policy": policy.to_dict(),
            "forest_root_only_daughters": True,
            "continue_through_empty_levels": True,
            "target_policy": target_policy,
            "truth_topology_mode": args.truth_topology_mode,
            "trees_included": not args.omit_trees,
        },
        "summaries": {
            scope: {
                "decay_metrics": summarize_decay_evaluations(metric_rows[scope]),
                "inference": summarize_inference_diagnostics(
                    diagnostic_rows[scope]
                ),
            }
            for scope in scopes
        },
        "summaries_by_source_category": {
            category: {
                scope: {
                    "decay_metrics": summarize_decay_evaluations(
                        metric_rows_by_category[category][scope]
                    ),
                    "inference": summarize_inference_diagnostics(
                        diagnostics_by_category[category][scope]
                    ),
                }
                for scope in scopes
            }
            for category in sorted(metric_rows_by_category)
        },
        "summaries_by_target_shape": {
            scope: {
                shape: summarize_decay_evaluations(rows)
                for shape, rows in sorted(
                    metric_rows_by_target_shape[scope].items()
                )
            }
            for scope in scopes
        },
        "events": event_records,
    }
    _atomic_write_json(output, report)
    print(f"Wrote {output}", file=sys.stderr, flush=True)
    return 0


def _validated_output_path(
    output: str | os.PathLike[str],
    *,
    direct_inputs: tuple[str | os.PathLike[str], ...],
    data_arguments: list[str] | tuple[str, ...] = (),
) -> Path:
    """Reject output paths that alias any direct or manifest-referenced input."""

    destination = Path(output).expanduser().resolve()
    if destination.exists() and destination.is_dir():
        raise ValueError(f"--output is a directory: {destination}")
    read_inputs = [Path(value).expanduser() for value in direct_inputs]
    for value in data_arguments:
        read_inputs.extend(_manifest_referenced_paths(Path(value).expanduser()))
    for candidate in read_inputs:
        resolved = candidate.resolve()
        aliases = destination == resolved
        if not aliases and destination.exists() and candidate.exists():
            try:
                aliases = os.path.samefile(destination, candidate)
            except OSError:
                aliases = False
        if aliases:
            raise ValueError(
                "--output must not alias a read input: "
                f"{destination} == {resolved}"
            )
    return destination


def _manifest_referenced_paths(path: Path) -> list[Path]:
    """Return local data paths from a training-selection manifest, if present."""

    manifest_path = path.resolve()
    if not manifest_path.is_file() or manifest_path.suffix.lower() != ".json":
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    data_root = payload.get("data_root")
    if not isinstance(entries, list) or not isinstance(data_root, str):
        return []
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = manifest_path.parent / root
    root = root.resolve()
    output: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in ("path", "sidecar_path", "completion_marker_path"):
            value = entry.get(field)
            if not isinstance(value, str):
                continue
            candidate = Path(value).expanduser()
            output.append(
                candidate.resolve()
                if candidate.is_absolute()
                else (root / candidate).resolve()
            )
    return output


def _atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    """Publish JSON through an exclusively owned temporary file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        handle = os.fdopen(file_descriptor, "w", encoding="utf-8")
        file_descriptor = -1
        with handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _evaluator_code_provenance() -> dict[str, Any]:
    """Bind receipts to the Git index without reading untracked payloads."""

    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().strip()
        index_metadata = subprocess.run(
            ("git", "ls-files", "-s", "-z"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        cached_diff = subprocess.run(
            ("git", "diff", "--cached", "--binary", "HEAD", "--"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        cached_paths = subprocess.run(
            ("git", "diff", "--cached", "--name-only", "-z", "HEAD", "--"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        unstaged_paths = subprocess.run(
            ("git", "diff", "--name-only", "-z", "--"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        untracked_paths = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        cached_names = sorted(os.fsdecode(path) for path in cached_paths if path)
        unstaged_names = sorted(os.fsdecode(path) for path in unstaged_paths if path)
        untracked_names = sorted(os.fsdecode(path) for path in untracked_paths if path)
        dirty_paths = sorted(set(cached_names + unstaged_names + untracked_names))
        digest = sha256()
        digest.update(b"git-index-metadata-v1\0")
        digest.update(index_metadata)
        digest.update(b"\0cached-diff-v1\0")
        digest.update(cached_diff)
        index_matches_worktree = not unstaged_names
        provenance_complete = index_matches_worktree and not untracked_names
        return {
            "git_head": head,
            "provenance_version": "git-index-cached-diff-v1",
            "provenance_complete": provenance_complete,
            "index_matches_worktree": index_matches_worktree,
            "worktree_dirty": bool(dirty_paths),
            "dirty_path_count": len(dirty_paths),
            "dirty_paths": dirty_paths,
            "untracked_path_count": len(untracked_names),
            "untracked_paths": untracked_names,
            "worktree_patch_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "git_head": "unknown",
            "provenance_version": "git-index-cached-diff-v1",
            "provenance_complete": False,
            "index_matches_worktree": None,
            "worktree_dirty": None,
            "dirty_path_count": None,
            "dirty_paths": [],
            "untracked_path_count": None,
            "untracked_paths": [],
            "worktree_patch_sha256": "unknown",
        }


if __name__ == "__main__":
    raise SystemExit(main())
