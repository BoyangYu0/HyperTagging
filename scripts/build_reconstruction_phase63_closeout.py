#!/usr/bin/env python3
"""Validate immutable Phase63 reports and export complete reconstruction metrics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_phase62_retained_metric_export import metric_names
from scripts.build_reconstruction_phase41_closeout import digest, numeric_rows  # noqa: E402
from hypertagging.evaluation.retained_tree_checks import validate_retained_tree_report
from scripts.run_full_reconstruction_evaluation_suite import _validate_component_report  # noqa: E402

ARMS = {
    "corrected_recovery": "16742232",
    "masked_only": "16760614",
}
BEAM_METRICS = (
    "source_recall",
    "source_precision",
    "lcag_pair_accuracy",
    "mother_pid_coverage",
    "perfect_lcag",
)


def load(path):
    return json.loads(path.read_text())


def verify_receipt(receipt):
    canonical = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (
        hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != receipt["receipt_sha256"]
    ):
        raise ValueError("Receipt hash mismatch")
    if (
        receipt["status"] != "completed"
        or receipt["exit_status"] != 0
        or receipt["sealed_test_accessed"]
    ):
        raise ValueError("Phase63 run did not complete within its authority")


def topology_audit(report):
    """Count exact tree depths and inspect coherent forests without pooling candidates."""
    from collections import Counter
    from scripts.build_phase62_retained_metric_export import structure_counts

    records = []
    for event in report["events"]:
        for scope, record in event["scopes"].items():
            records.append(
                (event["event_uid"], scope, "greedy", record["retained_tree_metrics"])
            )
            for i, candidate in enumerate(
                record.get("retained_tree_beam", {}).get("candidates", [])
            ):
                records.append(
                    (
                        event["event_uid"],
                        scope,
                        f"full_depth_candidate_{i}",
                        candidate["metrics"],
                    )
                )
    for event in report["beam_search"]["events"]:
        for candidate in event["candidates"]:
            for scope, record in candidate["retained_tree_metrics_by_scope"].items():
                records.append(
                    (
                        event["event_uid"],
                        scope,
                        f"proposal_candidate_{candidate['candidate_index']}",
                        record,
                    )
                )
    shapes = Counter()
    forests = []
    for uid, scope, search, record in records:
        for row in record["rows"]:
            if row["perfectLCAG_numerator"]:
                shapes[
                    (
                        scope,
                        search,
                        row["truth_leaf_count"],
                        row["truth_mother_count"],
                        row["truth_retained_depth"],
                    )
                ] += 1
        if record["coherent_retained_forest"]["numerator"]:
            forests.append(
                {
                    "event_uid": uid,
                    "scope": scope,
                    "search": search,
                    "unit_semantics": record["unit_semantics"],
                    "truth_mothers": sum(
                        row["truth_mother_count"] for row in record["rows"]
                    ),
                    "maximum_truth_depth": max(
                        (row["truth_retained_depth"] for row in record["rows"]),
                        default=0,
                    ),
                }
            )
    return {
        "primary_structure_counts": {
            scope: structure_counts(report, scope) for scope in ("full", "half")
        },
        "evaluated_event_scope_candidate_records": len(records),
        "exact_component_shapes": [
            {
                "scope": k[0],
                "search": k[1],
                "leaves": k[2],
                "mothers": k[3],
                "depth": k[4],
                "count": v,
            }
            for k, v in sorted(shapes.items())
        ],
        "coherent_forests": forests,
    }


def build(recovery_source, masking_source):
    source = recovery_source
    sources = {"corrected_recovery": recovery_source, "masked_only": masking_source}
    contracts = {}
    prereg = load(
        source / "configs/reconstruction/ht_reconstruction_phase63_20260928.json"
    )
    cohort_binding = prereg["validation_cohort"]
    assert digest(source / cohort_binding["path"]) == cohort_binding["sha256"]
    cohort = load(source / cohort_binding["path"])
    for key in ("selection_manifest", "dataset_index"):
        assert (
            digest(source / prereg["data_binding"][key])
            == prereg["data_binding"][key + "_sha256"]
        )
    payload = {
        "audit_version": "phase63-closeout-v1",
        "status_date": "2026-09-30",
        "status": "COMPLETED_PAIR_GATES_FAILED",
        "independent_validation": True,
        "strict_selection_overlap": 0,
        "scientific_mode": "CORRECTED_SUPERVISION_PILOT",
        "metric_completeness": "BOTH_COMPLETED_ARMS_ALL_FOURTEEN_VIEWS",
        "evaluation_role": "validation",
        "train_events": 70000,
        "strict_event_count": 100,
        "beam_event_count": 20,
        "sealed_test_accessed": False,
        "promotion_authorized": False,
        "next_study_status": "MASKING_BASELINE_GEOMETRY_DIAGNOSTIC_PROPOSED_NOT_SUBMITTED",
        "source_boundary": "native_immutable_training_and_evaluation",
        "original_job_status": "ONE_COMPLETED_ONE_FAILED_BEFORE_TRAINING",
        "arms": {},
        "source_hashes": [],
        "metric_rows": [],
    }
    for arm, job in ARMS.items():
        source = sources[arm]
        run = source / f"artifacts/runs/ht-reconstruction-phase63-20260928/{arm}/{job}"
        receipt_path = (
            source
            / f"artifacts/slurm/reconstruction-phase63/jobs/{job}/attempt-00/receipt.json"
        )
        receipt = load(receipt_path)
        verify_receipt(receipt)
        contract_path = run / "provenance/submitted-contract.json"
        contract = load(contract_path)
        contracts[arm] = contract
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            == contract["expected_git_sha"]
        )
        subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"],
            cwd=source,
            check=True,
            capture_output=True,
        )
        canonical = {k: v for k, v in contract.items() if k != "contract_sha256"}
        assert (
            hashlib.sha256(
                json.dumps(
                    canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            == receipt["contract_sha256"]
            == contract["contract_sha256"]
        )
        assert (
            contract["arm_role"] == arm
            and contract["sealed_test_role_access"] == "forbidden"
        )
        for binding in contract["hashed_inputs"]:
            assert digest(source / binding["path"]) == binding["sha256"]
        assert load(source / contract["preregistration"]["path"]) == prereg
        payload["source_hashes"].append(digest(contract_path))
        assert receipt["artifacts"]["full_decay_gates"] is not None
        for name, binding in receipt["artifacts"].items():
            if binding is not None:
                assert digest(source / binding["path"]) == binding["sha256"]
        result, gates = (
            load(run / "result.json"),
            load(run / "full-decay-gates.json"),
        )
        assert result["selection_preflight"] == {
            "before_any_training": True,
            "excluded_events": 99000,
            "selection_events": 1000,
            "status": "PASS",
            "strict_overlap": 0,
        }
        audit = result["validation_selection"]
        assert (
            audit["ordered_event_uids_match"]
            and audit["ordered_rollout_prefix_match"]
            and audit["prior_cohort_overlap_count"] == 0
        )
        assert (
            audit["event_uids_sha256"]
            == cohort["checkpoint_selection_event_uids_sha256"]
        )
        assert len(cohort["event_uids"]) == 100
        assert (
            result["optimizer_steps"] == 4376
            and result["balanced_level_replay"]["passed"]
        )
        assert gates["preregistered_gate_checks"]["primary_repeat_identical"]
        for bindings in gates["artifacts"].values():
            for binding in bindings.values():
                if isinstance(binding, dict) and "path" in binding:
                    assert digest(Path(binding["path"])) == binding["sha256"]
        for checkpoint in result["checkpoints"].values():
            assert digest(Path(checkpoint["path"])) == checkpoint["sha256"]
            assert checkpoint["all_checkpoint_tensors_finite"]
        reports = {}
        view_audits = {}
        report_root = run / "full-decay-reports"
        for path in sorted(report_root.glob("*.json")):
            doc = load(path)
            if "summaries" not in doc:
                continue
            name = path.stem
            beam = name.endswith("beam_direct")
            topology = (
                "contracted_diagnostic"
                if name.endswith("contracted_diagnostic")
                else "checkpoint_direct"
            )
            track = (
                "best_depth"
                if name.startswith("independent_depth")
                else "best_tree_validity"
                if name.startswith("independent_tree_validity")
                else "best_complete_target"
                if name.startswith("independent_complete_target")
                else "best"
            )
            _validate_component_report(
                doc,
                name=name,
                threads=1,
                expected_uids=cohort["event_uids"][:20]
                if beam
                else cohort["event_uids"],
                expected_scope="both",
                expected_topology=topology,
                expected_checkpoint_sha256=result["checkpoints"][track]["sha256"],
                expect_beam=beam,
            )
            validate_retained_tree_report(doc)
            reports[name] = doc
            view_audits[name] = topology_audit(doc)
            payload["metric_rows"].extend(
                {"arm": arm, "view": name, **row}
                for row in numeric_rows(
                    metric_names(doc["retained_tree_checks"]["summaries"]),
                    "retained_tree_summaries",
                )
            )
            payload["source_hashes"].append(digest(path))
            for family in (
                "summaries",
                "summaries_by_source_category",
                "summaries_by_target_shape",
            ):
                payload["metric_rows"].extend(
                    {"arm": arm, "view": name, **row}
                    for row in numeric_rows(doc[family], family)
                )
        assert len(reports) == 7
        primary = reports["primary_complete_target_direct"]
        repeat = reports["primary_complete_target_repeat2_direct"]
        for key in (
            "summaries",
            "summaries_by_source_category",
            "summaries_by_target_shape",
            "events",
            "retained_tree_checks",
        ):
            assert primary[key] == repeat[key], "Strict repeat differs"
        beam = reports["primary_complete_target_beam_direct"]
        ranking_metrics = {
            scope: {
                "greedy": beam["summaries"][scope]["decay_metrics"],
                **beam["beam_search"]["top1_summaries_by_scope_and_model_only_ranking"][
                    scope
                ],
                "oracle_at_k": beam["beam_search"]["oracle_at_k_summary_by_scope"][
                    scope
                ],
            }
            for scope in ("full", "half")
        }
        payload["metric_rows"].extend(
            {"arm": arm, "view": "beam_rankings", **row}
            for row in numeric_rows(ranking_metrics, "beam")
        )
        for track, checkpoint in result["checkpoints"].items():
            payload["metric_rows"].extend(
                {"arm": arm, "view": "training_" + track, **row}
                for row in numeric_rows(checkpoint["metrics"])
            )
        forest = {
            key: sum(
                event["scopes"]["full"]["inference"][key] for event in primary["events"]
            )
            for key in (
                "accepted_mother_count",
                "leftover_input_fsp_count",
                "empty_level_count",
                "b_root_count",
            )
        }
        payload["metric_rows"].extend(
            {"arm": arm, "view": "forest_diagnostics", **row}
            for row in numeric_rows(forest)
        )
        metrics = result["checkpoints"]["best"]["metrics"]
        payload["arms"][arm] = {
            "view_topology_audits": view_audits,
            "job_id": job,
            "source_root": str(source),
            "source_sha": contract["expected_git_sha"],
            "contract_sha256": contract["contract_sha256"],
            "retained_primary": primary["retained_tree_checks"]["summaries"],
            "retained_beam": beam["retained_tree_checks"]["summaries"],
            "perfect_component_shapes": {
                scope: [
                    {
                        key: row[key]
                        for key in (
                            "truth_leaf_count",
                            "truth_mother_count",
                            "truth_retained_depth",
                        )
                    }
                    for event in primary["events"]
                    for row in event["scopes"][scope]["retained_tree_metrics"]["rows"]
                    if row["perfectLCAG_numerator"]
                ]
                for scope in ("full", "half")
            },
            "selected": gates["selected_checkpoints"]["primary_complete_target"],
            "optimizer_steps": result["optimizer_steps"],
            "training_elapsed_seconds": result["elapsed_seconds"],
            "endpoints": gates["primary_full_decay_endpoints"],
            "gates": gates["preregistered_gate_checks"],
            "all_gates_passed": gates["all_preregistered_gates_passed"],
            "primary_repeat_identical": True,
            "complete_target_count": {
                "numerator": metrics["complete_target_efficiency_numerator"],
                "denominator": metrics["complete_target_efficiency_denominator"],
            },
            "forest": forest,
            "beam": {
                scope: {
                    rank: {k: value[k] for k in BEAM_METRICS}
                    for rank, value in rankings.items()
                }
                for scope, rankings in ranking_metrics.items()
            },
        }
        payload["source_hashes"].extend(
            [
                digest(receipt_path),
                digest(run / "result.json"),
                digest(run / "full-decay-gates.json"),
            ]
        )
    # Re-run the original continuation guard against its immutable source, without submission.
    code = """import json,sys
from pathlib import Path
from scripts.slurm.submit_reconstruction_phase63_campaign import validate_failed_arm_retry
contract=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(validate_failed_arm_retry(contract, Path(sys.argv[2]))))
"""
    failed_path = (
        recovery_source
        / "artifacts/slurm/reconstruction-phase63/jobs/16742233/attempt-00/receipt.json"
    )
    retry_contract = (
        masking_source
        / "artifacts/runs/ht-reconstruction-phase63-20260928/masked_only/16760614/provenance/submitted-contract.json"
    )
    payload["failed_attempt_lineage"] = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(retry_contract), str(failed_path)],
            cwd=masking_source,
            text=True,
        )
    )
    left, right = contracts["corrected_recovery"], contracts["masked_only"]
    for key in (
        "data",
        "cohort",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "preregistration",
        "pretraining_refinement",
        "evaluation_contract",
        "post_training_gates",
        "resources",
        "gpu_environment",
    ):
        assert left[key] == right[key], f"Unexpected between-arm difference: {key}"
    differing = {
        key
        for key in left["config"].keys() | right["config"].keys()
        if left["config"].get(key) != right["config"].get(key)
    }
    assert differing == {
        "unrepresentable_target_policy",
        "recovery_objective_weight",
    }, differing
    payload["matched_design_verified"] = True
    payload["all_gates_passed_by_arm"] = {
        arm: row["all_gates_passed"] for arm, row in payload["arms"].items()
    }
    payload["promotion_authorized"] = False
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-source", type=Path, required=True)
    parser.add_argument("--masking-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.recovery_source.resolve(), args.masking_source.resolve())
    args.details_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.details_dir / "phase63-all-metrics.json.gz", "wt") as f:
        json.dump(payload, f, sort_keys=True, allow_nan=False)
    payload["metric_row_count"] = len(payload.pop("metric_rows"))
    export = (args.details_dir / "phase63-all-metrics.json.gz").resolve()
    payload["complete_metric_export"] = {
        "path": str(export),
        "sha256": digest(export),
        "bytes": export.stat().st_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"status": payload["status"], "metric_rows": payload["metric_row_count"]}
        )
    )


if __name__ == "__main__":
    main()
