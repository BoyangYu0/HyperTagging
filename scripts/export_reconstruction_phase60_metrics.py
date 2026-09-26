#!/usr/bin/env python3
"""Export all recorded Phase60 scalars, retaining unavailable reconstruction status."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ARMS = {"pretraining_balance_control": "16700251", "lower_late_pid_pretraining": "16700252"}
VIEWS = ("primary_direct", "strict_repeat", "best_complete_target", "best_depth", "best_tree_validity", "contracted_diagnostic", "beam_direct")


def scalars(value, path=""):
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from scalars(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scalars(item, f"{path}.{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite scalar: {path}")
        yield path, value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip() == "4c076796ef365c6e1fb31f75dc6779c02af3cdf3"
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.source, text=True).strip()
    public = {"study_date": "2026-09-26", "version": "phase60-failure-metrics-v1", "status": "FAILED_PRETRAINING", "metric_completeness": "ALL_AVAILABLE_EXPORTED", "native_source_revision": "4c076796ef365c6e1fb31f75dc6779c02af3cdf3", "reconstruction_metrics_status": "UNAVAILABLE_NO_RECONSTRUCTION_CHECKPOINT", "treatment_executed": False, "sealed_test_accessed": False, "arms": {}}
    all_rows, histories, failures, evidence = [], {}, {}, {}
    for arm, job in ARMS.items():
        run = args.source / f"artifacts/runs/ht-reconstruction-phase60-20260925/{arm}/{job}"
        pretrain = run / "pretraining"
        receipt_path = args.source / f"artifacts/slurm/reconstruction-phase60/jobs/{job}/attempt-00/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "failed" and receipt["exit_status"] == 1 and receipt["sealed_test_accessed"] is False
        assert not list(run.rglob("*.pt")) and not list(run.rglob("*full-decay*"))
        log_path = pretrain / "metrics.jsonl"
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert [row["step"] for row in rows] == list(range(1, 547))
        assert all(row.get("split") != "validation" and row["leaf_pid_training_weight"] == 1 and row["curriculum_phase_index"] == 0 for row in rows)
        failure_path = pretrain / "objective-preflight-step-547.json"
        failure = json.loads(failure_path.read_text())
        assert failure["completed_optimizer_steps"] == 546 and failure["attempted_optimizer_step"] == 547 and failure["optimizer_step_executed"] is False
        report = failure["report"]
        assert report["dominance_threshold"] == 20 and report["pass"] is False
        objectives = report["objectives"]
        ratio = objectives["leaf_pid"]["weighted_shared_encoder_gradient_norm"] / objectives["lca"]["weighted_shared_encoder_gradient_norm"]
        assert math.isclose(ratio, report["weighted_dominance_ratio"], rel_tol=1e-12) and ratio > 20
        selection = json.loads((run / "selection-preflight.json").read_text())
        assert selection == {"status": "PASS", "selection_events": 1000, "strict_overlap": 0, "excluded_events": 99000, "before_any_training": True}
        history = {}
        for row in rows:
            for metric, value in scalars(row):
                all_rows.append((arm, "training", row["step"], metric, value))
                history.setdefault(metric, []).append(value)
        failure_metrics = dict(scalars(failure))
        for metric, value in failure_metrics.items():
            all_rows.append((arm, "failure_before_optimizer", 547, metric, value))
        summaries = {name: {"count": len(values), "first": values[0], "last": values[-1], "min": min(values), "max": max(values), "mean": statistics.fmean(values)} for name, values in sorted(history.items())}
        public["arms"][arm] = {"completed_pretraining_steps": 546, "attempted_pretraining_step": 547, "training_presentations": sum(r["batch_events"] for r in rows), "completed_reconstruction_steps": 0, "validation_metric_records": 0, "strict_events_scored": 0, "beam_candidates_produced": 0, "gradient_dominance_ratio": ratio, "dominance_threshold": 20, "last_training_metrics": dict(scalars(rows[-1])), "history_summary": summaries, "failure_metrics": failure_metrics, "availability": {view: {scope: {"status": "UNAVAILABLE_NO_RECONSTRUCTION_CHECKPOINT", "numerator": None, "denominator": None, "value": None} for scope in ("full", "half")} for view in VIEWS}}
        histories[arm] = rows
        failures[arm] = failure
        evidence[arm] = {str(p.relative_to(args.source)): digest(p) for p in sorted(run.rglob("*")) if p.is_file()}
        evidence[arm][str(receipt_path.relative_to(args.source))] = digest(receipt_path)
    assert failures[next(iter(ARMS))] == failures[list(ARMS)[1]]
    runtime_keys = {key for key in histories[list(ARMS)[0]][0] if key.endswith(("_seconds", "_per_second", "_bytes"))}
    def scientific(rows):
        return [{k: v for k, v in row.items() if k not in runtime_keys} for row in rows]
    assert scientific(histories[list(ARMS)[0]]) == scientific(histories[list(ARMS)[1]])
    public["recorded_scientific_histories_identical"] = True
    public["failure_reports_identical"] = True
    public["source_hashes"] = sorted({digest for arm in evidence.values() for digest in arm.values()})
    public["scalar_rows"] = len(all_rows)
    public["summary_semantics"] = "Per-metric descriptive summaries over logged steps; means are not global micro metrics or independent replicates. Runtime throughput differs across jobs."
    with gzip.open(args.output / "phase60-all-recorded-scalars.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("arm", "record", "step", "metric", "value")); writer.writerows(all_rows)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n")
    verification = {"status": "PASS", "scalar_rows": len(all_rows), "numeric_histories_verified": True, "failure_reports_identical": True, "no_reconstruction_checkpoints_or_reports": True, "source_hashes": evidence, "public_sha256": digest(args.public_output)}
    (args.output / "native-metrics-verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in verification.items() if k != "source_hashes"}))


if __name__ == "__main__":
    main()
