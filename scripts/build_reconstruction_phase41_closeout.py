#!/usr/bin/env python3
"""Verify Phase41 receipts and reduce aggregate metrics for offline publication."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = {"pointer32_control": "16400108", "level1_pointer24": "16400109"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_rows(value, prefix=""):
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            if not re.fullmatch(r"[A-Za-z0-9_.,:; ()+>=-]+", str(key)):
                raise ValueError(f"non-metric key: {key!r}")
            yield from numeric_rows(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from numeric_rows(item, f"{prefix}.{i}")
    elif value is None or isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite metric")
        yield {"metric": prefix, "value": value}
    # Descriptive source strings and operational paths are never published.


def build(source: Path, supplement: Path | None):
    prereg = json.loads((source / "configs/reconstruction/ht_reconstruction_phase41_20260909.json").read_text())
    for key in ("selection_manifest", "dataset_index"):
        assert digest(source / prereg["data_binding"][key]) == prereg["data_binding"][key + "_sha256"]
    payload = {"audit_version": "phase41-closeout-v1", "status_date": "2026-09-09",
               "status": "COMPLETED", "evaluation_role": "validation",
               "train_events": 70000, "strict_event_count": 100, "beam_event_count": 20,
               "sealed_test_accessed": False, "promotion_authorized": False,
               "arms": {}, "source_hashes": [], "metric_rows": [],
               "next_study": "PRETRAINING_TRANSFER", "next_study_status": "PREPARED"}
    for arm, job in ARMS.items():
        run = source / f"artifacts/runs/ht-reconstruction-phase41-20260909/{arm}/{job}"
        receipt_path = source / f"artifacts/slurm/reconstruction-phase41/jobs/{job}/attempt-00/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        canonical = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        assert hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == receipt["receipt_sha256"]
        assert receipt["status"] == "completed" and receipt["exit_status"] == 0
        assert receipt["sealed_test_accessed"] is False
        for binding in receipt["artifacts"].values():
            assert digest(source / binding["path"]) == binding["sha256"]
        gates = json.loads((run / "full-decay-gates.json").read_text())
        result = json.loads((run / "result.json").read_text())
        assert result["optimizer_steps"] == 4376 and result["balanced_level_replay"]["passed"]
        assert gates["preregistered_gate_checks"]["primary_repeat_identical"]
        for track in gates["artifacts"].values():
            for binding in track.values():
                if isinstance(binding, dict) and "path" in binding:
                    assert digest(Path(binding["path"])) == binding["sha256"]
        payload["source_hashes"].extend([digest(receipt_path), digest(run / "result.json"), digest(run / "full-decay-gates.json")])
        payload["arms"][arm] = {"optimizer_steps": result["optimizer_steps"],
            "training_elapsed_seconds": result["elapsed_seconds"],
            "selected": gates["selected_checkpoints"]["primary_complete_target"],
            "endpoints": gates["primary_full_decay_endpoints"],
            "gates": gates["preregistered_gate_checks"],
            "all_gates_passed": gates["all_preregistered_gates_passed"],
            "primary_repeat_identical": True}
        for track, checkpoint in result["checkpoints"].items():
            payload["metric_rows"].extend({"arm": arm, "view": f"training_{track}", **row} for row in numeric_rows(checkpoint["metrics"]))
        for report in sorted((run / "full-decay-reports").glob('*.json')):
            if report.name.endswith('.log.json'):
                continue
            doc = json.loads(report.read_text())
            if "summaries" not in doc:
                continue
            payload["source_hashes"].append(digest(report))
            for family in ("summaries", "summaries_by_source_category", "summaries_by_target_shape"):
                payload["metric_rows"].extend({"arm": arm, "view": report.stem, **row} for row in numeric_rows(doc[family], family))
            if doc.get("beam_search"):
                for family in ("oracle_at_k_summary", "top1_summaries_by_model_only_ranking"):
                    if family in doc["beam_search"]:
                        payload["metric_rows"].extend({"arm": arm, "view": "original_beam_rankings", **row} for row in numeric_rows(doc["beam_search"][family], family))
        if supplement is not None:
            suite = supplement / arm
            summary = json.loads((suite / "full-evaluation-suite.json").read_text())
            assert summary["status"] == "completed"
            for binding in summary["artifacts"].values():
                assert digest(suite / binding["path"]) == binding["sha256"]
            assert summary["validation_checks"]["inputs_unchanged"]
            assert summary["validation_checks"]["strict_repeat_scientifically_identical"]
            original = json.loads((run / "full-decay-reports/primary_complete_target_direct.json").read_text())
            current_rows = {row["metric"]: row["value"] for row in numeric_rows(summary["results"]["strict_checkpoint_direct"]["summaries"])}
            original_rows = {row["metric"]: row["value"] for row in numeric_rows(original["summaries"])}
            assert current_rows.keys() == original_rows.keys()
            delta = max(abs(current_rows[key] - value) for key, value in original_rows.items() if type(value) in (int, float))
            assert delta < 1e-6, "Current strict evaluation materially differs from original"
            payload["arms"][arm]["cross_runtime_max_absolute_metric_delta"] = delta
            payload["arms"][arm]["current_strict_counts_match_original"] = all(current_rows[key] == value for key, value in original_rows.items() if key.endswith("count") or (key.endswith(("numerator", "denominator")) and type(value) in (int, float) and float(value).is_integer()))
            payload["source_hashes"].append(digest(suite / "full-evaluation-suite.json"))
            for report in sorted(suite.glob('*.json')):
                doc = json.loads(report.read_text())
                if "summaries" not in doc:
                    continue
                payload["source_hashes"].append(digest(report))
                for family in ("summaries", "summaries_by_source_category", "summaries_by_target_shape"):
                    payload["metric_rows"].extend({"arm": arm, "view": "current_" + report.stem, **row} for row in numeric_rows(doc[family], family))
                if (doc.get("beam_search", {}).get("evaluated_scopes") == ["full", "half"]
                    and doc.get("configuration", {}).get("beam_search", {}).get("enabled") is True):
                    beam = doc["beam_search"]
                    payload["arms"][arm]["beam"] = {
                        scope: {"greedy": doc["summaries"][scope]["decay_metrics"],
                                **beam["top1_summaries_by_scope_and_model_only_ranking"][scope],
                                "oracle_at_k": beam["oracle_at_k_summary_by_scope"][scope]}
                        for scope in ("full", "half")}
                    # Keep full beam aggregate statistics, including PID confusion and p4 availability.
                    for row in numeric_rows(payload["arms"][arm]["beam"], "beam"):
                        payload["metric_rows"].append({"arm": arm, "view": "current_beam", **row})
                    payload["arms"][arm]["beam"] = {
                        scope: {ranking: {name: metrics[name] for name in (
                            "source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag")}
                            for ranking, metrics in rankings.items()}
                        for scope, rankings in payload["arms"][arm]["beam"].items()}
            assert "beam" in payload["arms"][arm]
    payload["metric_completeness"] = "COMPLETE" if supplement else "ORIGINAL_REGISTERED_REPORTS_ONLY"
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    args = parser.parse_args()
    payload = build(args.source_root, args.supplement)
    if args.details_output:
        args.details_output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    payload["metric_rows"] = [row for row in payload["metric_rows"] if not row["metric"].startswith(("summaries_by_source_category.", "summaries_by_target_shape."))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"rows": len(payload["metric_rows"]), "bytes": args.output.stat().st_size, "completeness": payload["metric_completeness"]}))


if __name__ == "__main__":
    main()
