#!/usr/bin/env python3
"""Verify diagnostic receipts and export compact, identity-free study evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(root, destination):
    plan = json.loads((root / "plan.json").read_text())
    conditions, bindings = {}, {}
    for condition in plan["conditions"]:
        name = condition["name"]
        folder = root / name
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt["status"] != "COMPLETED" or receipt["plan_sha256"] != sha(root / "plan.json"):
            raise ValueError(f"Incomplete or mismatched condition {name}")
        for filename, key in (("summary.json", "summary_sha256"), ("events.jsonl", "events_sha256"),
                              ("traces.jsonl.gz", "trace_sha256")):
            if sha(folder / filename) != receipt[key]:
                raise ValueError(f"Evidence hash mismatch {name}/{filename}")
        report = json.loads((folder / "summary.json").read_text())
        if report["processed_events"] != 60 or report["categories"] != {k: 10 for k in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")}:
            raise ValueError("Diagnostic cohort coverage changed")
        tags = report["tag_efficiency"]
        width = str(report["configuration"]["beam_width"])
        exact, inclusive = tags["summary"], tags["inclusive_fsp_grouping"]["summary"]
        conditions[name] = {"configuration": report["configuration"], "condition": condition,
            "processed_events": 60, "categories": report["categories"], "elapsed_seconds": report["elapsed_seconds"],
            "survival": report["survival"], "exact_top1": exact["top1"]["b_reconstruction"]["per_b_correct"],
            "exact_pool": exact["pool_at_k"][width]["b_reconstruction"]["per_b_correct"],
            "inclusive_top1": inclusive["top1"]["b_reconstruction"]["per_b_correct"],
            "inclusive_pool": inclusive["pool_at_k"][width]["b_reconstruction"]["per_b_correct"],
            "pool_unique_groups": inclusive["pool_at_k"][width]["unique_candidate_source_group_count"],
            "continuum": {k: v["top1_fake_b_event_acceptance"] for k, v in tags["continuum_type_coverage"].items()},
            # Initial run summed all integer counters, including event maxima.
            # Those sums are not maxima; exclude them from the public evidence.
            "additive_search_counters": {k: v for k, v in report["search_counters"].items() if not k.startswith("max_")}}
        bindings[name] = {key: receipt[key] for key in ("summary_sha256", "events_sha256", "trace_sha256")}
    parent_path = Path(plan["parent_plan"]["path"])
    if sha(parent_path) != plan["parent_plan"]["sha256"]:
        raise ValueError("Parent plan changed")
    parent = json.loads(parent_path.read_text())
    task_id = next(i for i, task in enumerate(parent["tasks"])
                   if task["view"] == "beam_diagnostic" and task["arm"] == plan["arm"])
    original = json.loads((parent_path.parent / "tasks" / f"{task_id:03d}" / "report.json").read_text())
    reference = json.loads((root / "reference/summary.json").read_text())
    parity = {"beam_tag_report_identical": original["tag_efficiency"]["summaries"]["full/full_depth_beam"] == reference["tag_efficiency"],
              "greedy_tag_report_identical": original["tag_efficiency"]["summaries"]["full/greedy"] == reference["greedy_tag_efficiency"]}
    if not all(parity.values()):
        raise ValueError("Frozen baseline tag report did not reproduce")
    pilot_root = root / "membership-pilot-v1/run"
    pilot_receipt = json.loads((pilot_root / "receipt.json").read_text())
    if pilot_receipt["status"] != "COMPLETED" or sha(pilot_root / "summary.json") != pilot_receipt["summary_sha256"] or sha(pilot_root / "features.pt") != pilot_receipt["features_sha256"]:
        raise ValueError("Pilot evidence mismatch")
    pilot = json.loads((pilot_root / "summary.json").read_text())
    shared_path = root / "membership-pilot-v1/shared-tag-evaluation.json"
    shared = json.loads(shared_path.read_text())
    if shared["feature_cache_sha256"] != pilot_receipt["features_sha256"]:
        raise ValueError("Shared pilot evaluator used a different feature cache")
    shared_counts = {}
    for name, tags in shared["tag_efficiency"].items():
        inclusive = tags["inclusive_fsp_grouping"]["summary"]["top1"]["b_reconstruction"]["per_b_correct"]
        if inclusive["numerator"] != pilot["results"][name]["development"]["accepted_exact_memberships"]:
            raise ValueError("Pilot membership metric disagrees with the shared evaluator")
        shared_counts[name] = {"exact": tags["summary"]["top1"]["b_reconstruction"]["per_b_correct"],
            "inclusive": inclusive, "channel_rows": len(tags["b_channel_coverage"]),
            "channel_nominal_trials": sum(r["evaluated_b_trials"] for r in tags["b_channel_coverage"]),
            "continuum": {k: v["top1_fake_b_event_acceptance"] for k, v in tags["continuum_type_coverage"].items()}}
    guard_path = pilot_root / "membership-guard-audit.json"
    guard = json.loads(guard_path.read_text())
    evidence = {"version": "assembly-diagnosis-evidence-v1", "date": "2026-10-07",
        "status": "COMPLETED_DIAGNOSTIC_ONLY", "primary_coverage": False, "independent_primary_events_added": 0,
        "distinct_diagnostic_events": 60, "b_trials_per_condition": 40, "baseline_parity": parity,
        "conditions": conditions, "direct_membership_pilot": {k: v for k, v in pilot.items() if k != "source_sha256"},
        "pilot_shared_tag_evaluation": shared_counts, "pilot_tiny_target_guard_audit": guard,
        "bindings": {"search_plan_sha256": sha(root / "plan.json"), "conditions": bindings,
                     "pilot_summary_sha256": pilot_receipt["summary_sha256"], "pilot_features_sha256": pilot_receipt["features_sha256"],
                     "pilot_shared_tag_sha256": sha(shared_path), "pilot_guard_audit_sha256": sha(guard_path)},
        "limitations": ["one_frozen_checkpoint_and_seed", "reused_small_diagnostic_cohort",
                        "full_scope_search_only", "clean_root_cover_is_optimistic_not_legal_reachability",
                        "width_changes_proposal_set_and_forest_width", "proposal_arm_changes_a_budget_bundle",
                        "threshold_probe_changes_both_object_and_pointer_gates", "pilot_is_flat_groups_not_exact_trees",
                        "no_physical_FEI_or_population_zero_claim", "initial_raw_max_counter_fields_are_sums_excluded_here"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.input, args.output)
