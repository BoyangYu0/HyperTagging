#!/usr/bin/env python3
"""Validate immutable Phase44 reports and export complete reconstruction metrics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase41_closeout import digest, numeric_rows  # noqa: E402
from scripts.run_full_reconstruction_evaluation_suite import _validate_component_report  # noqa: E402

ARMS = {"late_adaptation_control": "16449957", "early_adaptation": "16449958"}
BEAM_METRICS = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag")


def load(path):
    return json.loads(path.read_text())


def verify_receipt(receipt):
    canonical = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != receipt["receipt_sha256"]:
        raise ValueError("Receipt hash mismatch")
    if receipt["status"] != "failed" or receipt["exit_status"] != 1 or receipt["sealed_test_accessed"]:
        raise ValueError("Phase44 run did not complete within its authority")


def verify_evaluation_settings(report, contract, checkpoint, event_count):
    configuration = report['configuration']
    pair = report['checkpoint_pair']
    expected = {'object_threshold': contract['config']['rollout_object_threshold'],
                'pointer_threshold': contract['config']['rollout_pointer_threshold'],
                'max_level': 6, 'max_events': event_count, 'target_policy': 'complete_only',
                'forest_root_only_daughters': True}
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError('Recovered evaluation thresholds or policy changed')
    if (pair.get('pretraining_sha256') != contract['checkpoint_sha256']
        or pair.get('pretraining_step') != 81096
        or pair.get('reconstruction_step') != checkpoint['step']
        or pair.get('exact_frozen_encoder_required') is not
            (checkpoint['step'] <= contract['config']['freeze_pretrained_encoder_steps'])):
        raise ValueError('Recovered evaluation checkpoint lineage changed')


def build(source, recovery, history):
    prereg = load(source / "configs/reconstruction/ht_reconstruction_phase44_20260911.json")
    cohort_binding = prereg["untouched_validation_cohort"]
    assert digest(source / cohort_binding["path"]) == cohort_binding["sha256"]
    cohort = load(source / cohort_binding["path"])
    for key in ("selection_manifest", "dataset_index"):
        assert digest(source / prereg["data_binding"][key]) == prereg["data_binding"][key + "_sha256"]
    payload = {"audit_version": "phase44-closeout-v1", "status_date": "2026-09-11",
               "status": "RECOVERED_DIAGNOSTIC", "original_job_status": "FAILED",
               "recovery_classification": "POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION", "metric_completeness": "COMPLETE", "evaluation_role": "validation",
               "train_events": 70000, "strict_event_count": 100, "beam_event_count": 20,
               "sealed_test_accessed": False, "promotion_authorized": False,
               "next_study_status": "PREPARED", "arms": {}, "source_hashes": [], "metric_rows": []}
    for arm, job in ARMS.items():
        run = source / f"artifacts/runs/ht-reconstruction-phase44-20260911/{arm}/{job}"
        receipt_path = source / f"artifacts/slurm/reconstruction-phase44/jobs/{job}/attempt-00/receipt.json"
        receipt = load(receipt_path)
        verify_receipt(receipt)
        contract_path = run / "provenance/submitted-contract.json"
        contract = load(contract_path)
        canonical = {k: v for k, v in contract.items() if k != "contract_sha256"}
        assert hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == receipt["contract_sha256"] == contract["contract_sha256"]
        assert contract["arm_role"] == arm and contract["sealed_test_role_access"] == "forbidden"
        for binding in contract["hashed_inputs"]:
            assert digest(source / binding["path"]) == binding["sha256"]
        assert load(source / contract["preregistration"]["path"]) == prereg
        payload["source_hashes"].append(digest(contract_path))
        for binding in receipt["artifacts"].values():
            if binding:
                assert digest(source / binding["path"]) == binding["sha256"]
        recovered = recovery / arm
        result_path = recovered / "training-result.json"
        gates_path = recovered / "full-decay-gates.json"
        result, gates = load(result_path), load(gates_path)
        assert result["source_receipt_sha256"] == digest(receipt_path)
        assert result["original_job_status"] == "FAILED"
        assert result["recovery_classification"] == payload["recovery_classification"]
        assert result["deviation"] == {"field": "balanced_level_replay_contract.seed", "preregistered": 20260910, "configured_and_observed": 20260911, "planned_schedule_reconstructed_from_budget": True}
        assert result["source_training_log_sha256"] == digest(run / "training/metrics.jsonl")
        stderr_path = source / f"artifacts/slurm/phase44-{arm}-20260911-{job}.err"
        assert result["source_stderr_sha256"] == digest(stderr_path)
        assert "trainer balanced replay contract does not exactly match the preregistered real-data contract" in stderr_path.read_text()
        assert contract["config"]["seed"] == 20260911
        assert contract["config"]["balanced_level_replay_contract"]["seed"] == 20260910
        assert not result["sealed_test_accessed"] and not result["promotion_authorized"]
        assert result["optimizer_steps"] == 4376 and result["balanced_level_replay"]["passed"]
        assert result["balanced_level_replay"]["preregistered_contract_match"] is False
        assert result["balanced_level_replay"]["reconciled_contract_match"] is True
        assert gates["preregistered_gate_checks"]["primary_repeat_identical"]
        for bindings in gates["artifacts"].values():
            for binding in bindings.values():
                if isinstance(binding, dict) and "path" in binding:
                    assert digest(Path(binding["path"])) == binding["sha256"]
        for checkpoint in result["checkpoints"].values():
            assert digest(Path(checkpoint["path"])) == checkpoint["sha256"]
            assert checkpoint["all_checkpoint_tensors_finite"]
        reports = {}
        report_root = recovered / "full-decay-reports"
        for path in sorted(report_root.glob("*.json")):
            doc = load(path)
            if "summaries" not in doc:
                continue
            name = path.stem
            beam = name.endswith("beam_direct")
            topology = "contracted_diagnostic" if name.endswith("contracted_diagnostic") else "checkpoint_direct"
            track = ("best_depth" if name.startswith("independent_depth") else
                     "best_tree_validity" if name.startswith("independent_tree_validity") else
                     "best_complete_target" if name.startswith("independent_complete_target") else "best")
            _validate_component_report(doc, name=name, threads=1,
                expected_uids=cohort["event_uids"][:20] if beam else cohort["event_uids"],
                expected_scope="both", expected_topology=topology,
                expected_checkpoint_sha256=result["checkpoints"][track]["sha256"], expect_beam=beam)
            verify_evaluation_settings(doc, contract, result["checkpoints"][track], 20 if beam else 100)
            reports[name] = doc
            payload["source_hashes"].append(digest(path))
            for family in ("summaries", "summaries_by_source_category", "summaries_by_target_shape"):
                payload["metric_rows"].extend({"arm": arm, "view": name, **row} for row in numeric_rows(doc[family], family))
        assert len(reports) == 7
        primary = reports["primary_complete_target_direct"]
        repeat = reports["primary_complete_target_repeat2_direct"]
        for key in ("summaries", "summaries_by_source_category", "summaries_by_target_shape", "events"):
            assert primary[key] == repeat[key], "Strict repeat differs"
        beam = reports["primary_complete_target_beam_direct"]
        ranking_metrics = {scope: {"greedy": beam["summaries"][scope]["decay_metrics"],
                                  **beam["beam_search"]["top1_summaries_by_scope_and_model_only_ranking"][scope],
                                  "oracle_at_k": beam["beam_search"]["oracle_at_k_summary_by_scope"][scope]}
                           for scope in ("full", "half")}
        payload["metric_rows"].extend({"arm": arm, "view": "beam_rankings", **row} for row in numeric_rows(ranking_metrics, "beam"))
        for track, checkpoint in result["checkpoints"].items():
            payload["metric_rows"].extend({"arm": arm, "view": "training_" + track, **row} for row in numeric_rows(checkpoint["metrics"]))
        forest = {key: sum(event["scopes"]["full"]["inference"][key] for event in primary["events"])
                  for key in ("accepted_mother_count", "leftover_input_fsp_count", "empty_level_count", "b_root_count")}
        payload["metric_rows"].extend({"arm": arm, "view": "forest_diagnostics", **row} for row in numeric_rows(forest))
        metrics = result["checkpoints"]["best"]["metrics"]
        payload["arms"][arm] = {
            "original_job_status": "FAILED", "recovery_classification": result["recovery_classification"],
            "selected": gates["selected_checkpoints"]["primary_complete_target"],
            "optimizer_steps": result["optimizer_steps"], "training_elapsed_seconds": result["elapsed_seconds"],
            "endpoints": gates["primary_full_decay_endpoints"], "gates": gates["preregistered_gate_checks"],
            "all_gates_passed": gates["all_preregistered_gates_passed"], "primary_repeat_identical": True,
            "complete_target_count": {"numerator": metrics["complete_target_efficiency_numerator"], "denominator": metrics["complete_target_efficiency_denominator"]},
            "forest": forest,
            "beam": {scope: {rank: {k: value[k] for k in BEAM_METRICS} for rank, value in rankings.items()} for scope, rankings in ranking_metrics.items()}}
        payload["source_hashes"].extend([digest(receipt_path), digest(result_path), digest(gates_path)])
    payload["source_hashes"].extend([digest(ROOT / "scripts/recover_reconstruction_phase44.py"), digest(ROOT / "scripts/build_reconstruction_phase44_closeout.py")])
    history_manifest = load(history / 'manifest.json')
    assert history_manifest['classification'] == payload['recovery_classification']
    assert history_manifest['original_job_status'] == 'FAILED'
    assert history_manifest['sealed_test_accessed'] is False
    assert set(history_manifest['exports']) == {'phase44-training-history.csv.gz', 'phase44-training-history.jsonl.gz'}
    export_validation = load(history / 'export-validation.json')
    assert export_validation == {'status': 'PASS', 'csv_jsonl_rows_identical': True, 'scalar_rows': history_manifest['total_scalar_rows']}
    payload['source_hashes'].append(digest(history / 'export-validation.json'))
    for name, binding in history_manifest['exports'].items():
        assert name in {'phase44-training-history.csv.gz', 'phase44-training-history.jsonl.gz'}
        assert digest(history / name) == binding['sha256']
        payload['source_hashes'].append(binding['sha256'])
    assert set(history_manifest['arms']) == set(ARMS)
    for arm, record in history_manifest['arms'].items():
        assert record['training_log_sha256'] == digest(source / f'artifacts/runs/ht-reconstruction-phase44-20260911/{arm}/{ARMS[arm]}/training/metrics.jsonl')
        checkpoint_dir = source / f'artifacts/runs/ht-reconstruction-phase44-20260911/{arm}/{ARMS[arm]}/training'
        assert set(record['checkpoints']) == {p.name for p in checkpoint_dir.glob('*.pt')}
        assert all(c['all_checkpoint_tensors_finite'] and c['sha256'] == digest(checkpoint_dir / name) for name, c in record['checkpoints'].items())
    assert history_manifest['total_scalar_rows'] == sum(r['log_scalar_rows'] + r['checkpoint_scalar_rows'] for r in history_manifest['arms'].values())
    payload['training_history_scalar_rows'] = history_manifest['total_scalar_rows']
    payload['training_history_log_records'] = sum(r['log_records'] for r in history_manifest['arms'].values())
    payload['training_history_checkpoint_records'] = sum(len(r['checkpoints']) for r in history_manifest['arms'].values())
    payload['source_hashes'].append(digest(history / 'manifest.json'))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--training-history-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.source_root, args.recovery_root, args.training_history_dir)
    args.details_dir.mkdir(parents=True, exist_ok=True)
    (args.details_dir / "phase44-all-metrics.json").write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    with gzip.open(args.details_dir / "phase44-all-metrics.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "view", "metric", "value"])
        writer.writeheader()
        writer.writerows(payload["metric_rows"])
    detailed = len(payload["metric_rows"])
    payload["metric_rows"] = [row for row in payload["metric_rows"] if not row["metric"].startswith(("summaries_by_source_category.", "summaries_by_target_shape."))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"aggregate_rows": len(payload["metric_rows"]), "detailed_rows": detailed, "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
