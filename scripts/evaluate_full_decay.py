#!/usr/bin/env python
"""Evaluate hierarchical full/half decay reconstruction on schema-v4 data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
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

from hypertagging.evaluation.retained_tree_checks import RetainedTreeChecks, validate_retained_tree_report  # noqa: E402
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
    summarize_beam_search_diagnostics,
)
from hypertagging.evaluation.beam_decay_metrics import (  # noqa: E402
    evaluate_ranked_decay_candidates,
    summarize_beam_decay_evaluations,
)
from hypertagging.evaluation.trained_context import (  # noqa: E402
    load_trained_evaluation_context,
)
from hypertagging.reconstruction.hierarchical_inference import (  # noqa: E402
    FULL_ROOT_TOKEN,
    HierarchicalInferenceConfig,
    OFFLINE_INFERENCE_POLICY_VERSION,
    project_schema_v4_fsps,
    reconstruct_full_tree_from_fsps,
    reconstruct_beam_from_fsps,
)
from hypertagging.reconstruction.beam_search import BeamSearchConfig  # noqa: E402
from hypertagging.reconstruction.level_rollout import (  # noqa: E402
    RolloutConfig,
    diagnostic_proposal_beam_rollout,
    rollout_policy_identity,
)


REPORT_VERSION = "hypertagging-offline-full-decay-evaluation-v3"
BEAM_REPORT_VERSION = "hypertagging-offline-full-decay-evaluation-v4"


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
        "--event-uid-manifest",
        help=(
            "Exact validation-only event cohort manifest. When supplied, it "
            "replaces checkpoint/stream selection and must contain exactly "
            "--max-events unique event_uids."
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
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Compare the unchanged greedy evaluation with bounded beam top-1 and truth-only oracle@K.",
    )
    beam_defaults = BeamSearchConfig()
    for field in (
        "beam_width",
        "max_candidates_per_query",
        "max_proposals_per_level",
        "max_daughter_options",
        "max_type_options",
        "max_cardinality_options",
        "max_candidate_expansions_per_query",
        "max_nodes_per_hypothesis",
    ):
        option = field if field == "beam_width" else f"beam_{field}"
        parser.add_argument(
            f"--{option.replace('_', '-')}",
            dest=option,
            type=int,
            default=getattr(beam_defaults, field),
        )
    for field in ("score_length_normalization", "empty_level_penalty"):
        parser.add_argument(
            f"--beam-{field.replace('_', '-')}",
            dest=f"beam_{field}",
            type=float,
            default=getattr(beam_defaults, field),
        )
    parser.add_argument(
        "--beam-oracle-k",
        type=int,
        action="append",
        default=None,
        help="Oracle prefix K (repeatable, 1 <= K <= beam width); defaults to 1 and beam width.",
    )
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
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Require PyTorch deterministic algorithms for reproducibility-gated "
            "offline inference."
        ),
    )
    parser.add_argument("--omit-trees", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument(
        "--beam-max-events",
        type=int,
        default=20,
        help="Bounded validation events used by the full-tree beam diagnostic.",
    )
    parser.add_argument(
        "--beam-max-proposals",
        type=int,
        default=12,
        help="Maximum exact proposal-set enumeration size per beam level.",
    )
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
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    args.beam_ranking_diagnostics = (
        not args.beam_search
        and args.beam_width > 1
        and any(arg == "--beam-width" or arg.startswith("--beam-width=") for arg in raw_args)
    )
    if args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.max_level <= 0:
        parser.error("--max-level must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.beam_width <= 0 or args.beam_max_events <= 0:
        parser.error("beam width and beam max events must be positive")
    if args.beam_max_proposals <= 0 or args.beam_max_proposals > 16:
        parser.error("--beam-max-proposals must lie in [1, 16]")
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
    try:
        _beam_config_from_args(args)
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid beam configuration: {exc}")
    if args.beam_oracle_k is not None and any(
        k < 1 or k > args.beam_width for k in args.beam_oracle_k
    ):
        parser.error("--beam-oracle-k must lie between 1 and --beam-width")
    return args


def _beam_config_from_args(args: argparse.Namespace) -> BeamSearchConfig:
    return BeamSearchConfig(
        **{
            field: getattr(args, field if field == "beam_width" else f"beam_{field}")
            for field in asdict(BeamSearchConfig())
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event_uid_manifest = (
        _load_event_uid_manifest(Path(args.event_uid_manifest), args.max_events)
        if args.event_uid_manifest is not None
        else None
    )
    if event_uid_manifest is not None and args.event_selection != "auto":
        raise ValueError(
            "--event-uid-manifest cannot be combined with --event-selection"
        )
    output = _validated_output_path(
        args.output,
        direct_inputs=(
            args.pretraining_checkpoint,
            args.reconstruction_checkpoint,
            args.dataset_index,
            *args.data,
            *((args.event_uid_manifest,) if args.event_uid_manifest else ()),
        ),
        data_arguments=args.data,
    )
    run_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(bool(args.deterministic_algorithms))
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
        event_selection=(
            "explicit_uids" if event_uid_manifest is not None else args.event_selection
        ),
        explicit_event_uids=(
            event_uid_manifest["event_uids"]
            if event_uid_manifest is not None
            else None
        ),
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
    checkpoint_selection_contract = context.checkpoint.get("training_state", {}).get(
        "checkpoint_selection_contract", {}
    )
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
    diagnostics_by_category: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metric_rows_by_target_shape: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    beam_rows: dict[str, list[Any]] = defaultdict(list)
    beam_search_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    beam_top1_diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    beam_rows_by_category: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    beam_search_by_category: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    beam_top1_by_category: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    beam_rows_by_target_shape: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    beam_config = _beam_config_from_args(args)
    oracle_ks = sorted(set(args.beam_oracle_k or (1, beam_config.beam_width)))
    inference_wall_seconds = 0.0
    metric_wall_seconds = 0.0
    serialization_wall_seconds = 0.0
    event_processing_started = time.perf_counter()
    beam_rankings = (
        "learned_confidence_sum",
        "learned_confidence_mean",
        "average_link_probability",
        "normalized_joint_log_probability",
    )
    beam_metric_rows: dict[str, dict[str, list[Any]]] = {
        scope: defaultdict(list) for scope in scopes
    }
    beam_oracle_rows: dict[str, list[Any]] = {scope: [] for scope in scopes}
    beam_event_records: list[dict[str, Any]] = []

    retained_checks = RetainedTreeChecks(target_policy=target_policy, minimum_daughters=int(policy.minimum_daughters))

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
            retained_evaluation = retained_checks.evaluate(inference.batch, truth_batch, scope=scope, source_category=event.source_category)
            retained_checks.add(f"{scope}/greedy", retained_evaluation, event.source_category)
            scope_record: dict[str, Any] = {
                "retained_tree_metrics": retained_evaluation.as_dict(),
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
            if args.beam_search:
                phase_started = time.perf_counter()
                beam = reconstruct_beam_from_fsps(
                    context.model,
                    truth_batch,
                    config=HierarchicalInferenceConfig(
                        scope=scope,
                        rollout_config=rollout_config,
                        max_level=args.max_level,
                    ),
                    beam_config=beam_config,
                )
                inference_wall_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                retained_candidates = []
                candidate_evaluations = []
                candidate_diagnostics = []
                for candidate in beam.candidates:
                    retained_candidate = retained_checks.evaluate(candidate.batch, truth_batch, scope=scope, source_category=event.source_category)
                    retained_candidates.append(retained_candidate)
                    retained_checks.add(f"{scope}/beam_candidate_rank_{len(retained_candidates)}", retained_candidate, event.source_category)
                    evaluation_kwargs = {
                        "target_policy": target_policy,
                        "minimum_daughters": int(policy.minimum_daughters),
                        "truth_topology_mode": args.truth_topology_mode,
                    }
                    if scope == "full":
                        candidate_evaluations.append(
                            evaluate_full_decay(
                                candidate.batch,
                                truth_batch,
                                **evaluation_kwargs,
                            )
                        )
                    else:
                        candidate_evaluations.append(
                            evaluate_half_decays(
                                candidate.batch,
                                truth_batch,
                                source_category=event.source_category,
                                **evaluation_kwargs,
                            )
                        )
                    candidate_diagnostics.append(
                        inference_diagnostics(
                            candidate,
                            p4_tolerance=float(args.p4_closure_tolerance),
                        )
                    )
                beam_evaluation = evaluate_ranked_decay_candidates(
                    candidate_evaluations,
                    scores=beam.scores,
                    oracle_ks=oracle_ks,
                )
                retained_beam = retained_checks.add_beam(f"{scope}/full_depth_beam", retained_candidates, beam.scores, oracle_ks)
                scope_record["retained_tree_beam"] = retained_beam.as_dict()
                retained_checks.add(f"{scope}/beam_top1", retained_candidates[0], event.source_category)
                beam_record = beam_evaluation.as_dict()
                beam_record["search"] = beam.diagnostics
                beam_record["top1_inference"] = candidate_diagnostics[0]
                for index, candidate_record in enumerate(beam_record["candidates"]):
                    candidate_record["inference"] = candidate_diagnostics[index]
                    candidate_record["log_score_sum"] = beam.log_score_sums[index]
                    candidate_record["scored_candidate_count"] = (
                        beam.scored_candidate_counts[index]
                    )
                    candidate_record["scored_decision_count"] = (
                        beam.scored_decision_counts[index]
                    )
                metric_wall_seconds += time.perf_counter() - phase_started
                if not args.omit_trees:
                    phase_started = time.perf_counter()
                    for candidate, candidate_record in zip(
                        beam.candidates, beam_record["candidates"]
                    ):
                        candidate_record["reconstructed_tree"] = (
                            serialize_reconstructed_tree(candidate)
                        )
                    serialization_wall_seconds += time.perf_counter() - phase_started
                scope_record["beam"] = beam_record
                beam_rows[scope].append(beam_evaluation)
                beam_search_rows[scope].append(beam.diagnostics)
                beam_top1_diagnostics[scope].append(candidate_diagnostics[0])
                beam_rows_by_category[event.source_category][scope].append(
                    beam_evaluation
                )
                beam_search_by_category[event.source_category][scope].append(
                    beam.diagnostics
                )
                beam_top1_by_category[event.source_category][scope].append(
                    candidate_diagnostics[0]
                )
                candidate_units = [
                    candidate.halves if scope == "half" else (candidate,)
                    for candidate in candidate_evaluations
                ]
                for unit_index, unit in enumerate(candidate_units[0]):
                    if not unit.available or unit.truth_retained_depth is None:
                        continue
                    shape = (
                        f"fsp_count={len(unit.truth_sources)};"
                        f"retained_depth={unit.truth_retained_depth}"
                    )
                    beam_rows_by_target_shape[scope][shape].append(
                        evaluate_ranked_decay_candidates(
                            [units[unit_index] for units in candidate_units],
                            scores=beam.scores,
                            oracle_ks=oracle_ks,
                        )
                    )
            record["scopes"][scope] = scope_record
            metric_rows[scope].append(evaluation)
            diagnostic_rows[scope].append(diagnostics)
            metric_rows_by_category[event.source_category][scope].append(evaluation)
            diagnostics_by_category[event.source_category][scope].append(diagnostics)
            units = evaluation.halves if scope == "half" else (evaluation,)
            for unit in units:
                if not unit.available or unit.truth_retained_depth is None:
                    continue
                shape = (
                    f"fsp_count={len(unit.truth_sources)};"
                    f"retained_depth={unit.truth_retained_depth}"
                )
                metric_rows_by_target_shape[scope][shape].append(unit)
        if args.beam_ranking_diagnostics and event_index < args.beam_max_events:
            phase_started = time.perf_counter()
            projection = project_schema_v4_fsps(truth_batch)
            diagnostic_config = replace(
                rollout_config,
                max_level=args.max_level,
                root_types=(FULL_ROOT_TOKEN,),
                continue_through_empty_levels=True,
                max_resolution_proposals=args.beam_max_proposals,
            )
            hypotheses = diagnostic_proposal_beam_rollout(
                context.model,
                projection.batch,
                config=diagnostic_config,
                beam_width=args.beam_width,
                lookahead_levels=args.max_level,
            )
            retained_diagnostic_candidates = []
            candidates: list[tuple[Any, dict[str, Any]]] = []
            for hypothesis in hypotheses:
                hypothesis.batch["evaluation_leaf_source_keys"] = (
                    projection.evaluation_leaf_source_keys.clone()
                )
                evaluations: dict[str, Any] = {}
                if "full" in scopes:
                    evaluations["full"] = evaluate_full_decay(
                        hypothesis.batch,
                        truth_batch,
                        target_policy=target_policy,
                        minimum_daughters=int(policy.minimum_daughters),
                        truth_topology_mode=args.truth_topology_mode,
                    )
                if "half" in scopes:
                    evaluations["half"] = evaluate_half_decays(
                        hypothesis.batch,
                        truth_batch,
                        source_category=event.source_category,
                        target_policy=target_policy,
                        minimum_daughters=int(policy.minimum_daughters),
                        truth_topology_mode=args.truth_topology_mode,
                    )
                retained_by_scope = {scope: retained_checks.evaluate(hypothesis.batch, truth_batch, scope=scope, source_category=event.source_category) for scope in scopes}
                retained_diagnostic_candidates.append(retained_by_scope)
                for scope in scopes:
                    retained_checks.add(f"{scope}/proposal_beam_candidate_rank_{len(retained_diagnostic_candidates)}", retained_by_scope[scope], event.source_category)
                candidates.append((hypothesis, evaluations))
            if candidates:
                ranking_selections: dict[str, int] = {}
                for ranking in beam_rankings:
                    selected_index = max(
                        range(len(candidates)),
                        key=lambda index: (
                            candidates[index][0].ranking_scores()[ranking],
                            -index,
                        ),
                    )
                    ranking_selections[ranking] = selected_index
                    for scope in scopes:
                        retained_checks.add(f"{scope}/proposal_beam/{ranking}", retained_diagnostic_candidates[selected_index][scope], event.source_category)
                    for scope in scopes:
                        beam_metric_rows[scope][ranking].append(
                            candidates[selected_index][1][scope]
                        )
                retained_oracle_indices = {scope: max(range(len(candidates)), key=lambda index: _beam_oracle_key(retained_diagnostic_candidates[index][scope])) for scope in scopes}
                for scope in scopes:
                    retained_checks.add(f"{scope}/proposal_beam/oracle_diagnostic", retained_diagnostic_candidates[retained_oracle_indices[scope]][scope], event.source_category)
                    ranked_indices = sorted(range(len(candidates)), key=lambda index: (-candidates[index][0].ranking_scores()["normalized_joint_log_probability"], index))
                    retained_checks.add_beam(f"{scope}/proposal_beam_normalized_joint", [retained_diagnostic_candidates[index][scope] for index in ranked_indices], [candidates[index][0].ranking_scores()["normalized_joint_log_probability"] for index in ranked_indices], [1, args.beam_width])
                oracle_candidate_indices = {
                    scope: max(
                        range(len(candidates)),
                        key=lambda index: _beam_oracle_key(
                            candidates[index][1][scope]
                        ),
                    )
                    for scope in scopes
                }
                for scope, oracle_index in oracle_candidate_indices.items():
                    beam_oracle_rows[scope].append(
                        candidates[oracle_index][1][scope]
                    )
                beam_event_records.append(
                    {
                        "event_uid": event.event_uid,
                        "candidate_count": len(candidates),
                        "ranking_selections": ranking_selections,
                        "retained_tree_oracle_indices_by_scope": retained_oracle_indices,
                        "oracle_candidate_indices_by_scope": oracle_candidate_indices,
                        "oracle_candidate_index": oracle_candidate_indices.get("full"),
                        "candidates": [
                            {
                                "candidate_index": index,
                                "model_only_ranking_scores": hypothesis.ranking_scores(),
                                "proposal_count": hypothesis.proposal_count,
                                "stop_reason": hypothesis.stop_reason,
                                "accepted_proposal_counts_by_level": [
                                    len(items)
                                    for items in hypothesis.accepted_by_level
                                ],
                                "retained_tree_metrics_by_scope": {scope: evaluation.as_dict() for scope, evaluation in retained_diagnostic_candidates[index].items()},
                                "truth_diagnostic_metrics_by_scope": {
                                    scope: evaluation.as_dict()
                                    for scope, evaluation in evaluations.items()
                                },
                                "truth_diagnostic_metrics": (
                                    evaluations["full"].as_dict()
                                    if "full" in evaluations
                                    else None
                                ),
                            }
                            for index, (hypothesis, evaluations) in enumerate(candidates)
                        ],
                    }
                )
            inference_wall_seconds += time.perf_counter() - phase_started
        event_records.append(record)

    phase_seconds["event_processing"] = time.perf_counter() - event_processing_started
    phase_seconds["hierarchical_inference"] = inference_wall_seconds
    phase_seconds["metrics_and_diagnostics"] = metric_wall_seconds
    phase_seconds["tree_serialization"] = serialization_wall_seconds
    phase_seconds["total_before_report_write"] = time.perf_counter() - run_started

    report = {
        "retained_tree_checks": retained_checks.as_dict(),
        "report_version": (BEAM_REPORT_VERSION if args.beam_search else REPORT_VERSION),
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
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
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
            "event_uid_manifest": (
                {
                    key: event_uid_manifest[key]
                    for key in (
                        "path",
                        "sha256",
                        "manifest_version",
                        "event_uid_count",
                        "event_uids_sha256",
                    )
                }
                if event_uid_manifest is not None
                else None
            ),
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
            **(
                {
                    "beam_search_enabled": True,
                    "beam_search": asdict(beam_config),
                    "beam_oracle_ks": oracle_ks,
                }
                if args.beam_search
                else {}
            ),
            **(
                {"beam_search": {
                    "enabled": True,
                    "algorithm": "diagnostic_proposal_set_beam",
                    "beam_width": args.beam_width,
                    "max_events": min(args.beam_max_events, len(context.events)),
                    "max_proposals": args.beam_max_proposals,
                    "max_level": args.max_level,
                    "model_only_rankings": list(beam_rankings),
                    "truth_used_for_ranking": False,
                    "oracle_at_k_is_diagnostic_only": True,
                }} if args.beam_ranking_diagnostics else {}
            ),
        },
        "summaries": {
            scope: {
                "decay_metrics": summarize_decay_evaluations(metric_rows[scope]),
                "inference": summarize_inference_diagnostics(diagnostic_rows[scope]),
                **(
                    {
                        "beam": {
                            **summarize_beam_decay_evaluations(beam_rows[scope]),
                            "search": summarize_beam_search_diagnostics(
                                beam_search_rows[scope]
                            ),
                            "top1_inference": summarize_inference_diagnostics(
                                beam_top1_diagnostics[scope]
                            ),
                        }
                    }
                    if args.beam_search
                    else {}
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
                    **(
                        {
                            "beam": {
                                **summarize_beam_decay_evaluations(
                                    beam_rows_by_category[category][scope]
                                ),
                                "search": summarize_beam_search_diagnostics(
                                    beam_search_by_category[category][scope]
                                ),
                                "top1_inference": summarize_inference_diagnostics(
                                    beam_top1_by_category[category][scope]
                                ),
                            }
                        }
                        if args.beam_search
                        else {}
                    ),
                }
                for scope in scopes
            }
            for category in sorted(metric_rows_by_category)
        },
        "summaries_by_target_shape": {
            scope: {
                shape: {
                    **summarize_decay_evaluations(rows),
                    **(
                        {
                            "beam": summarize_beam_decay_evaluations(
                                beam_rows_by_target_shape[scope][shape],
                                include_event_metrics=False,
                            )
                        }
                        if args.beam_search
                        else {}
                    ),
                }
                for shape, rows in sorted(metric_rows_by_target_shape[scope].items())
            }
            for scope in scopes
        },
        "events": event_records,
        "beam_search": {
            "event_count": len(beam_event_records),
            "evaluated_scopes": list(scopes),
            "top1_summaries_by_scope_and_model_only_ranking": {
                scope: {
                    ranking: summarize_decay_evaluations(rows)
                    for ranking, rows in beam_metric_rows[scope].items()
                }
                for scope in scopes
            },
            "top1_summaries_by_model_only_ranking": {
                ranking: summarize_decay_evaluations(rows)
                for ranking, rows in beam_metric_rows.get("full", {}).items()
            },
            "oracle_at_k_summary_by_scope": {
                scope: (
                    summarize_decay_evaluations(beam_oracle_rows[scope])
                    if beam_oracle_rows[scope]
                    else {}
                )
                for scope in scopes
            },
            "oracle_at_k_summary": (
                summarize_decay_evaluations(beam_oracle_rows.get("full", []))
                if beam_oracle_rows.get("full")
                else {}
            ),
            "events": beam_event_records,
        },
    }
    validate_retained_tree_report(report)
    _atomic_write_json(output, report)
    print(f"Wrote {output}", file=sys.stderr, flush=True)
    return 0


def _beam_oracle_key(evaluation: Any) -> tuple[float, ...]:
    """Truth-only diagnostic ordering; never used to choose deployed output."""

    if hasattr(evaluation, "halves"):
        summary = summarize_decay_evaluations([evaluation])

        def summary_value(name: str) -> float:
            value = summary.get(name, {}).get("value")
            return float(value) if value is not None else -1.0

        return ((summary_value("coherent_retained_forest"),) if getattr(evaluation, "coherent_retained_forest", None) is not None else ()) + (
            summary_value("perfect_lcag"),
            summary_value("lcag_pair_accuracy"),
            summary_value("mother_pid_coverage"),
            summary_value("source_recall"),
            summary_value("source_precision"),
        )

    def value(metric: Any) -> float:
        return float(metric.value) if metric.value is not None else -1.0

    return (
        value(evaluation.perfect_lcag),
        value(evaluation.lcag_pair_accuracy),
        value(evaluation.mother_pid_coverage),
        value(evaluation.source_recall),
        value(evaluation.source_precision),
    )


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
                f"--output must not alias a read input: {destination} == {resolved}"
            )
    return destination


def _load_event_uid_manifest(path: Path, max_events: int) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve(strict=True)
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    event_uids = payload.get("event_uids")
    if (
        payload.get("manifest_version")
        != "hypertagging-reconstruction-evaluation-cohort-v1"
        or payload.get("role") != "validation"
        or payload.get("sealed_test_role_access") != "forbidden"
        or not isinstance(event_uids, list)
        or len(event_uids) != max_events
        or len(event_uids) != len(set(str(uid) for uid in event_uids))
    ):
        raise ValueError("evaluation event UID manifest is invalid")
    normalized = [str(uid) for uid in event_uids]
    if payload.get("event_uid_count") != len(normalized):
        raise ValueError("evaluation event UID manifest count is invalid")
    if payload.get("event_uids_sha256") != _uid_sequence_sha256(normalized):
        raise ValueError("evaluation event UID manifest hash is invalid")
    return {
        **payload,
        "path": str(manifest_path),
        "sha256": sha256(raw).hexdigest(),
        "event_uids": normalized,
    }


def _uid_sequence_sha256(event_uids: list[str]) -> str:
    digest = sha256()
    digest.update(b"hypertagging-evaluation-event-uids-v1\0")
    for uid in event_uids:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


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
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
        head = (
            subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
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
