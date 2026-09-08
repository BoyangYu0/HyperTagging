#!/usr/bin/env python3
"""Run the complete offline reconstruction evaluation as one fail-closed suite.

The suite binds every component to one checkpoint pair and one validation UID
cohort.  It runs strict direct-topology full/half reconstruction twice, the
contracted-topology diagnostic once, and bounded full-depth beam search on a
preregistered prefix of the same cohort.  One summary receipt indexes the raw
reports and exposes all decision-relevant metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = "hypertagging-full-reconstruction-evaluation-suite-v1"
EVALUATOR_REPORT_VERSION = "hypertagging-offline-full-decay-evaluation-v3"
MANIFEST_VERSION = "hypertagging-reconstruction-evaluation-cohort-v1"
MODEL_ONLY_BEAM_RANKINGS = (
    "learned_confidence_sum",
    "learned_confidence_mean",
    "average_link_probability",
    "normalized_joint_log_probability",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_sequence_sha256(event_uids: Sequence[str]) -> str:
    digest = sha256()
    digest.update(b"hypertagging-evaluation-event-uids-v1\0")
    for uid in event_uids:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_validation_manifest(path: Path, expected_count: int) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    event_uids = payload.get("event_uids")
    if (
        payload.get("manifest_version") != MANIFEST_VERSION
        or payload.get("role") != "validation"
        or payload.get("sealed_test_role_access") != "forbidden"
        or not isinstance(event_uids, list)
    ):
        raise ValueError("full-suite event UID manifest is not validation-only")
    normalized = [str(uid) for uid in event_uids]
    if (
        len(normalized) != expected_count
        or len(normalized) != len(set(normalized))
        or payload.get("event_uid_count") != expected_count
        or payload.get("event_uids_sha256") != _uid_sequence_sha256(normalized)
    ):
        raise ValueError("full-suite event UID manifest identity is invalid")
    return {**payload, "event_uids": normalized, "path": resolved}


def _artifact(path: Path, suite_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(suite_root)),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _decision_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Keep scientific results and exclude only timestamps and timing."""

    return {
        "summaries": report["summaries"],
        "summaries_by_source_category": report["summaries_by_source_category"],
        "summaries_by_target_shape": report["summaries_by_target_shape"],
        "events": report["events"],
        "beam_search": report["beam_search"],
    }


def _structural_guardrails(report: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for scope in report["configuration"]["evaluated_scopes"]:
        inference = report["summaries"][scope]["inference"]
        for name, minimum in (
            ("inference_structurally_valid", 0.999),
            ("p4_closure", 1.0),
            ("recursive_detector_sources_disjoint", 1.0),
            ("forest_root_sources_disjoint", 1.0),
        ):
            value = inference[name]["value"]
            checks[f"{scope}_{name}"] = value is not None and float(value) >= minimum
    return checks


def _command(
    *,
    python: Path,
    pretraining_checkpoint: Path,
    reconstruction_checkpoint: Path,
    data: Sequence[Path],
    dataset_index: Path,
    cohort_manifest: Path,
    output: Path,
    scope: str,
    topology: str,
    max_events: int,
    max_level: int,
    object_threshold: float,
    pointer_threshold: float | None,
    threads: int,
    allow_finetuned_encoder: bool,
    include_trees: bool,
    external_independent_sample: bool,
    beam_width: int = 1,
    beam_max_proposals: int = 12,
) -> list[str]:
    command = [
        str(python),
        "scripts/evaluate_full_decay.py",
        "--pretraining-checkpoint",
        str(pretraining_checkpoint),
        "--reconstruction-checkpoint",
        str(reconstruction_checkpoint),
        "--data",
        *(str(path) for path in data),
        "--dataset-index",
        str(dataset_index),
        "--split",
        "validation",
        "--event-uid-manifest",
        str(cohort_manifest),
        "--scope",
        scope,
        "--truth-topology-mode",
        topology,
        "--max-events",
        str(max_events),
        "--max-level",
        str(max_level),
        "--object-threshold",
        str(object_threshold),
        "--threads",
        str(threads),
        "--deterministic-algorithms",
        "--output",
        str(output),
    ]
    if not include_trees:
        command.append("--omit-trees")
    if pointer_threshold is not None:
        command.extend(("--pointer-threshold", str(pointer_threshold)))
    if allow_finetuned_encoder:
        command.append("--allow-finetuned-encoder")
    if external_independent_sample:
        command.append("--diagnostic-external-independent-sample")
    if beam_width > 1:
        command.extend(
            (
                "--beam-width",
                str(beam_width),
                "--beam-max-events",
                str(max_events),
                "--beam-max-proposals",
                str(beam_max_proposals),
            )
        )
    return command


def _run_component(
    *,
    name: str,
    command: list[str],
    output: Path,
    log: Path,
    timeout_seconds: int,
    threads: int,
    expected_uids: Sequence[str],
    expected_scope: str,
    expected_topology: str,
    expected_checkpoint_sha256: str,
    expect_beam: bool,
) -> dict[str, Any]:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    _atomic_json(
        log,
        {
            "component": name,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"full evaluation component {name} failed: {completed.stderr[-1000:]}"
        )
    report = json.loads(output.resolve(strict=True).read_text(encoding="utf-8"))
    configuration = report.get("configuration", {})
    context = report.get("context", {})
    beam = configuration.get("beam_search", {})
    expected_scopes = ["full", "half"] if expected_scope == "both" else [expected_scope]
    if (
        report.get("report_version") != EVALUATOR_REPORT_VERSION
        or report.get("device") != "cpu"
        or report.get("torch_num_threads") != threads
        or report.get("torch_deterministic_algorithms_enabled") is not True
        or configuration.get("requested_scope") != expected_scope
        or configuration.get("evaluated_scopes") != expected_scopes
        or configuration.get("truth_topology_mode") != expected_topology
        or context.get("evaluation_split") != "validation"
        or context.get("evaluation_event_selection") != "explicit_uid_cohort"
        or context.get("evaluated_event_uids") != list(expected_uids)
        or context.get("evaluation_uid_train_overlap") != []
        or report.get("checkpoint_pair", {}).get("compatible") is not True
        or report.get("checkpoint_pair", {}).get("reconstruction_sha256")
        != expected_checkpoint_sha256
        or bool(beam.get("enabled")) is not expect_beam
    ):
        raise RuntimeError(f"full evaluation component {name} violated its contract")
    if expect_beam:
        observed = report.get("beam_search", {})
        rankings = beam.get("model_only_rankings")
        if (
            observed.get("event_count") != len(expected_uids)
            or rankings != list(MODEL_ONLY_BEAM_RANKINGS)
            or beam.get("truth_used_for_ranking") is not False
            or beam.get("oracle_at_k_is_diagnostic_only") is not True
        ):
            raise RuntimeError("beam-search evaluation violated its ranking contract")
    return report


def _metric_catalog() -> dict[str, Any]:
    return {
        "strict_primary": (
            "Checkpoint-direct truth topology on the complete validation cohort; "
            "this is the decision-bearing full/half decay evaluation."
        ),
        "strict_repeat": (
            "An exact deterministic repeat of strict_primary. Scientific payloads "
            "must be byte-equivalent after excluding timestamps and timing."
        ),
        "contracted_topology_diagnostic": (
            "Diagnostic scoring after contracting truth-only non-target nodes. It "
            "does not replace the checkpoint-direct trained-pointer target."
        ),
        "configured_root_completion": (
            "Fraction of events in which inference builds the configured full root."
        ),
        "source_recall": "Fraction of truth detector-source identities recovered.",
        "source_precision": "Fraction of predicted source identities that are true.",
        "lcag_pair_accuracy": (
            "Fraction of comparable leaf pairs whose lowest-common-ancestor relation "
            "matches truth."
        ),
        "perfect_lcag": "Fraction of targets with every comparable LCAG relation exact.",
        "mother_pid_coverage": (
            "Fraction of truth mothers represented by a matched predicted mother with "
            "the exact particle type."
        ),
        "root_pid_accuracy": "Fraction of available targets with the exact root type.",
        "inference_structurally_valid": "Fraction of reconstructed forests passing tree checks.",
        "p4_closure": "Fraction whose composite four-vectors equal daughter sums.",
        "beam_search": (
            "Bounded, full-depth reconstruction on a fixed validation subset. Greedy "
            "and every top-1 model-only ranking use exactly the same events."
        ),
        "learned_confidence_sum": "Sum of learned proposal confidence log-probabilities.",
        "learned_confidence_mean": "Mean learned proposal confidence over accepted links.",
        "average_link_probability": (
            "Arithmetic mean of accepted pointer-link probabilities; this is the "
            "original HyperTagging-style candidate score."
        ),
        "normalized_joint_log_probability": (
            "Length-normalized joint log score over object, type, cardinality, link, "
            "and learned-confidence terms."
        ),
        "oracle_at_k": (
            "Truth-ranked upper bound within retained candidates; diagnostic only and "
            "never a deployable selection rule."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Run this once after the selected reconstruction checkpoint exists.
The default suite evaluates 100 validation events, repeats the strict result,
runs the contracted diagnostic, and evaluates beam width 4 on the first 20
events. The final index is OUTPUT_DIR/full-evaluation-suite.json.""",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pretraining-checkpoint", type=Path, required=True)
    parser.add_argument("--reconstruction-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--event-uid-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--object-threshold", type=float, default=0.5)
    parser.add_argument("--pointer-threshold", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-max-events", type=int, default=20)
    parser.add_argument("--beam-max-proposals", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--allow-finetuned-encoder", action="store_true")
    parser.add_argument("--include-trees", action="store_true")
    parser.add_argument("--skip-contracted-diagnostic", action="store_true")
    parser.add_argument("--diagnostic-external-independent-sample", action="store_true")
    args = parser.parse_args(argv)
    if args.max_events <= 0 or args.max_level <= 0 or args.threads <= 0:
        parser.error("event, level, and thread counts must be positive")
    if not 1 < args.beam_width or not 0 < args.beam_max_events <= args.max_events:
        parser.error("beam width must exceed one and beam events must fit the cohort")
    if not 0 < args.beam_max_proposals <= 16 or args.timeout_seconds <= 0:
        parser.error("beam proposals must lie in [1, 16] and timeout must be positive")
    for name in ("object_threshold", "pointer_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Keep the virtual-environment entrypoint itself: resolving its symlink to
    # the base interpreter drops pyvenv.cfg discovery and therefore all of the
    # environment's installed packages.
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"evaluation Python does not exist: {python}")
    pretraining = args.pretraining_checkpoint.expanduser().resolve(strict=True)
    reconstruction = args.reconstruction_checkpoint.expanduser().resolve(strict=True)
    data = [path.expanduser().resolve(strict=True) for path in args.data]
    dataset_index = args.dataset_index.expanduser().resolve(strict=True)
    manifest = _load_validation_manifest(args.event_uid_manifest, args.max_events)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"refusing to reuse full-suite output: {output_dir}")
    output_dir.mkdir(parents=True)

    cohort_copy = output_dir / "validation-cohort.json"
    _atomic_json(
        cohort_copy,
        {
            "manifest_version": MANIFEST_VERSION,
            "role": "validation",
            "sealed_test_role_access": "forbidden",
            "event_uid_count": args.max_events,
            "event_uids_sha256": manifest["event_uids_sha256"],
            "event_uids": manifest["event_uids"],
        },
    )
    beam_uids = manifest["event_uids"][: args.beam_max_events]
    beam_manifest = output_dir / "beam-validation-cohort.json"
    _atomic_json(
        beam_manifest,
        {
            "manifest_version": MANIFEST_VERSION,
            "role": "validation",
            "sealed_test_role_access": "forbidden",
            "event_uid_count": len(beam_uids),
            "event_uids_sha256": _uid_sequence_sha256(beam_uids),
            "event_uids": beam_uids,
        },
    )

    input_hashes_before = {
        "pretraining_checkpoint": _sha256(pretraining),
        "reconstruction_checkpoint": _sha256(reconstruction),
        "dataset_index": _sha256(dataset_index),
        "data": [_sha256(path) for path in data],
        "source_event_uid_manifest": _sha256(manifest["path"]),
    }
    common = {
        "python": python,
        "pretraining_checkpoint": pretraining,
        "reconstruction_checkpoint": reconstruction,
        "data": data,
        "dataset_index": dataset_index,
        "max_level": args.max_level,
        "object_threshold": args.object_threshold,
        "pointer_threshold": args.pointer_threshold,
        "threads": args.threads,
        "allow_finetuned_encoder": args.allow_finetuned_encoder,
        "include_trees": args.include_trees,
        "external_independent_sample": args.diagnostic_external_independent_sample,
    }
    checkpoint_sha = input_hashes_before["reconstruction_checkpoint"]

    def evaluate(
        name: str,
        *,
        cohort: Path,
        uids: Sequence[str],
        scope: str,
        topology: str,
        beam_width: int = 1,
    ) -> tuple[dict[str, Any], Path, Path]:
        report_path = output_dir / f"{name}.json"
        log_path = output_dir / f"{name}.log.json"
        command = _command(
            **common,
            cohort_manifest=cohort,
            output=report_path,
            scope=scope,
            topology=topology,
            max_events=len(uids),
            beam_width=beam_width,
            beam_max_proposals=args.beam_max_proposals,
        )
        report = _run_component(
            name=name,
            command=command,
            output=report_path,
            log=log_path,
            timeout_seconds=args.timeout_seconds,
            threads=args.threads,
            expected_uids=uids,
            expected_scope=scope,
            expected_topology=topology,
            expected_checkpoint_sha256=checkpoint_sha,
            expect_beam=beam_width > 1,
        )
        return report, report_path, log_path

    primary, primary_path, primary_log = evaluate(
        "strict-checkpoint-direct",
        cohort=cohort_copy,
        uids=manifest["event_uids"],
        scope="both",
        topology="checkpoint_direct",
    )
    repeat, repeat_path, repeat_log = evaluate(
        "strict-checkpoint-direct-repeat",
        cohort=cohort_copy,
        uids=manifest["event_uids"],
        scope="both",
        topology="checkpoint_direct",
    )
    exact_repeat = _decision_payload(primary) == _decision_payload(repeat)
    if not exact_repeat:
        raise RuntimeError("strict reconstruction repeat is not scientifically identical")

    contracted: dict[str, Any] | None = None
    contracted_paths: tuple[Path, Path] | None = None
    if not args.skip_contracted_diagnostic:
        contracted, contracted_path, contracted_log = evaluate(
            "contracted-topology-diagnostic",
            cohort=cohort_copy,
            uids=manifest["event_uids"],
            scope="both",
            topology="contracted_diagnostic",
        )
        contracted_paths = (contracted_path, contracted_log)

    beam, beam_path, beam_log = evaluate(
        "full-depth-beam-search",
        cohort=beam_manifest,
        uids=beam_uids,
        scope="full",
        topology="checkpoint_direct",
        beam_width=args.beam_width,
    )

    input_hashes_after = {
        "pretraining_checkpoint": _sha256(pretraining),
        "reconstruction_checkpoint": _sha256(reconstruction),
        "dataset_index": _sha256(dataset_index),
        "data": [_sha256(path) for path in data],
        "source_event_uid_manifest": _sha256(manifest["path"]),
    }
    inputs_unchanged = input_hashes_before == input_hashes_after
    if not inputs_unchanged:
        raise RuntimeError("an evaluation input changed while the suite was running")
    primary_structure = _structural_guardrails(primary)
    beam_structure = _structural_guardrails(beam)

    artifacts = {
        "validation_cohort": _artifact(cohort_copy, output_dir),
        "beam_validation_cohort": _artifact(beam_manifest, output_dir),
        "strict_primary_report": _artifact(primary_path, output_dir),
        "strict_primary_log": _artifact(primary_log, output_dir),
        "strict_repeat_report": _artifact(repeat_path, output_dir),
        "strict_repeat_log": _artifact(repeat_log, output_dir),
        "beam_report": _artifact(beam_path, output_dir),
        "beam_log": _artifact(beam_log, output_dir),
    }
    if contracted_paths is not None:
        artifacts.update(
            {
                "contracted_diagnostic_report": _artifact(
                    contracted_paths[0], output_dir
                ),
                "contracted_diagnostic_log": _artifact(
                    contracted_paths[1], output_dir
                ),
            }
        )

    suite_report = {
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "evaluation_role": "validation",
        "sealed_test_role_access": "forbidden",
        "configuration": {
            "max_events": args.max_events,
            "max_level": args.max_level,
            "object_threshold": args.object_threshold,
            "requested_pointer_threshold": args.pointer_threshold,
            "resolved_pointer_threshold": primary["configuration"][
                "pointer_threshold"
            ],
            "threads": args.threads,
            "deterministic_algorithms": True,
            "beam_width": args.beam_width,
            "beam_max_events": args.beam_max_events,
            "beam_max_proposals": args.beam_max_proposals,
            "contracted_topology_diagnostic_included": contracted is not None,
            "trees_included": args.include_trees,
        },
        "metric_catalog": _metric_catalog(),
        "input_identity": input_hashes_after,
        "checkpoint_pair": primary["checkpoint_pair"],
        "evaluation_context": primary["context"],
        "resolved_rollout_configuration": primary["configuration"],
        "evaluator_code_provenance": primary["evaluator_code_provenance"],
        "component_timing": {
            "strict_checkpoint_direct": primary["timing"],
            "strict_checkpoint_direct_repeat": repeat["timing"],
            "contracted_topology_diagnostic": (
                contracted["timing"] if contracted is not None else None
            ),
            "full_depth_beam_search": beam["timing"],
        },
        "validation_checks": {
            "inputs_unchanged": inputs_unchanged,
            "strict_repeat_scientifically_identical": exact_repeat,
            "strict_primary_structural_guardrails": primary_structure,
            "strict_primary_all_structural_guardrails_passed": all(
                primary_structure.values()
            ),
            "beam_greedy_structural_guardrails": beam_structure,
            "beam_greedy_all_structural_guardrails_passed": all(
                beam_structure.values()
            ),
            "beam_rankings_are_model_only": True,
            "oracle_at_k_is_diagnostic_only": True,
        },
        "results": {
            "strict_checkpoint_direct": {
                "summaries": primary["summaries"],
                "summaries_by_source_category": primary[
                    "summaries_by_source_category"
                ],
                "summaries_by_target_shape": primary["summaries_by_target_shape"],
            },
            "contracted_topology_diagnostic": (
                {
                    "summaries": contracted["summaries"],
                    "summaries_by_source_category": contracted[
                        "summaries_by_source_category"
                    ],
                    "summaries_by_target_shape": contracted[
                        "summaries_by_target_shape"
                    ],
                }
                if contracted is not None
                else None
            ),
            "beam_search": {
                "event_count": beam["beam_search"]["event_count"],
                "greedy_same_cohort_summary": beam["summaries"]["full"],
                "top1_summaries_by_model_only_ranking": beam["beam_search"][
                    "top1_summaries_by_model_only_ranking"
                ],
                "oracle_at_k_summary_diagnostic_only": beam["beam_search"][
                    "oracle_at_k_summary"
                ],
            },
        },
        "artifacts": artifacts,
    }
    suite_path = output_dir / "full-evaluation-suite.json"
    _atomic_json(suite_path, suite_report)
    print(
        json.dumps(
            {
                "status": "completed",
                "strict_repeat_scientifically_identical": exact_repeat,
                "suite_report": str(suite_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
