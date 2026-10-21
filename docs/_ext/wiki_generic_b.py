"""Bounded aggregate publication for GB001, GB002 status, and CR054."""
from __future__ import annotations

import hashlib
import json


SOURCE = "artifacts/codex/generic_b_dashboard_20261019"
FILES = {"generic-b-status-summary.json"}
LIMIT = 32_768


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _point(value, numerator="correct", denominator="nominal"):
    return f"{value[numerator]}/{value[denominator]}"


def _table(headers, rows):
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + headers[0]]
    lines.extend("     - " + header for header in headers[1:])
    for row in rows:
        lines.append("   * - " + str(row[0]))
        lines.extend("     - " + str(value) for value in row[1:])
    return [*lines, ""]


def generate(root, output):
    """Authenticate and publish only the reviewed aggregate source record."""
    source = root / SOURCE
    binding_path = source / "binding.json"
    summary_path = source / "summary.json"
    _require(not binding_path.is_symlink() and not summary_path.is_symlink(), "Unsafe generic-B source")
    _require(binding_path.stat().st_size <= LIMIT and summary_path.stat().st_size <= LIMIT,
             "Generic-B source exceeds limit")
    binding = json.loads(binding_path.read_bytes())
    raw = summary_path.read_bytes()
    _require(binding == {
        "schema": "generic-b-dashboard-public-binding-v2",
        "status_date": "2026-10-21",
        "source_file": "summary.json",
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "private_payloads_copied": False,
        "contains_operational_identifiers": False,
        "contains_cohort_member_identifiers": False,
    }, "Generic-B binding differs")
    value = json.loads(raw)
    _require(set(value) == {"schema", "status_date", "provenance", "gb001", "gb002", "cr054"},
             "Generic-B summary fields differ")
    _require(value["schema"] == "generic-b-dashboard-public-summary-v2", "Generic-B schema differs")
    gb001, gb002, cr054 = value["gb001"], value["gb002"], value["cr054"]
    _require(set(gb001) == {"status", "evidence_class", "learned_model_evaluation",
                            "legal_decoder_reachability_assessed", "population", "static_compatibility",
                            "categories", "overlapping_recursive_blockers", "learned_metrics", "runtime"},
             "GB001 field allowlist differs")
    _require(set(gb002) == {"status", "result_state", "completed_scientific_result",
                            "snapshot_recorded_state", "current_state", "training_completed",
                            "full_evaluation_completed", "learned_metrics", "local_data",
                            "resource_policy", "runtime_attempt_at_snapshot", "replacement_transition",
                            "snapshot_scope", "snapshot_is_live_monitor",
                            "next_scientific_plan", "science_loop"},
             "GB002 field allowlist differs")
    _require(set(cr054) == {"status", "scientific_gate", "evidence_class", "control_reuse",
                            "baseline_checkpoint", "scientific_factor", "coverage", "arms", "validity",
                            "uncertainty", "runtime"}, "CR054 field allowlist differs")
    static = gb001["static_compatibility"]
    _require(gb001["status"] == "COMPLETED_AND_INDEPENDENTLY_REVIEWED"
             and gb001["learned_model_evaluation"] is False
             and gb001["legal_decoder_reachability_assessed"] is False
             and gb001["population"] == {"collisions": 28000, "b_collisions": 20000,
                                           "continuum_collisions": 8000,
                                           "nominal_b_predicates": 40000,
                                           "processing_failures": 0, "unit": "nominal_B_predicates"}
             and static["accepted_membership"] == {"compatible": 34819, "available": 40000,
                                                       "nominal": 40000, "unavailable": 0}
             and static["recursive_topology_pid"] == {"compatible": 6378, "available": 40000,
                                                        "nominal": 40000, "unavailable": 0}
             and static["physical_completeness"]["compatible"] is None
             and static["physical_completeness"]["unavailable"] == 40000
             and all(metric is None for metric in gb001["learned_metrics"].values()),
             "GB001 census scope differs")
    _require(gb002["status"] == "ACTIVE_OPERATIONAL_REPAIR_NO_LEARNED_EVALUATION"
             and gb002["snapshot_recorded_state"] == "OPERATIONAL_PREPARATION_ADMITTED"
             and gb002["current_state"] == "CONTINUOUS_SCIENCE_LOOP_RESUMED_OPERATIONAL_REPAIR_ACTIVE"
             and gb002["snapshot_scope"] == "AS_OF_REVIEWED_SCIENCE_LOOP_RESUMPTION_BOUNDARY"
             and gb002["snapshot_is_live_monitor"] is False
             and gb002["result_state"] == "NO_COMPLETED_TRAINING_OR_LEARNED_EVALUATION"
             and gb002["completed_scientific_result"] is False
             and gb002["training_completed"] is False
             and gb002["full_evaluation_completed"] is False
             and gb002["local_data"] == {
                 "transfer_status": "COMPLETE_HASH_VERIFIED",
                 "transfer_scope": "OPAQUE_BYTES_ONLY",
                 "scientific_payload_authentication_complete": False,
             }
             and gb002["resource_policy"] == {
                 "project_cpu_time_limit_seconds": None,
                 "project_gpu_time_limit_hours": None,
                 "actual_usage_tracking_required": True,
                 "site_scheduler_limits_enforced": True,
                 "per_operation_watchdogs_enforced": True,
                 "recorded_allocated_cpu_seconds": 6272,
                 "recorded_gpu_hours": 0,
                 "recorded_accounting_scope": "AT_REVIEWED_SCIENCE_LOOP_RESUMPTION_BOUNDARY",
             }
             and gb002["runtime_attempt_at_snapshot"] == {
                 "runtime_admission_passed": True,
                 "nominal_events": 3072,
                 "reached": 1,
                 "authenticated": 0,
                 "failed": 1,
                 "unreached": 3071,
                 "failure_class": "EVENT_NUMBER_SOURCE_ENTRY_IDENTITY_MISMATCH",
                 "all_jobs_terminal": True,
                 "training_performed": False,
                 "inference_performed": False,
             }
             and gb002["replacement_transition"] == {
                 "status": "FAILED_REVIEW_REQUIRED",
                 "nominal_events": 3072,
                 "reached": 0,
                 "payload_execution_authorized": False,
                 "recovery_status": "SUPERSEDED_BY_RESUMED_REVIEWED_SCIENCE_LOOP",
             }
             and gb002["next_scientific_plan"] == {
                 "status": "SEALED_PROSPECTIVE_PLAN_NOT_A_RESULT",
                 "ordered_priorities": [
                     "COMPLETE_PAYLOAD_DECODING_AND_MEASURE_NATIVE_BASELINE",
                     "COMPARE_TWO_GENERIC_B_TRAINING_SAMPLE_SIZES_BY_TWO_NETWORK_WIDTHS",
                     "REPORT_LEARNING_CURVES_AND_MEASURED_INFERENCE_IMPROVEMENTS",
                 ],
                 "recoverable_operational_failure_policy": "FOCUSED_REPAIR_THEN_RENEWED_REVIEWED_EXECUTION",
                 "dashboard_update_policy": "SIGNIFICANT_CHANGE_ONLY",
             }
             and gb002["science_loop"] == {
                 "status": "RESUMED_UNDER_REVIEWED_ASTRA_SOL_ROUTING",
                 "active_work": "FOCUSED_OPERATIONAL_REPAIR",
                 "payload_decoding_status": "PENDING",
                 "native_baseline_status": "UNMEASURED",
                 "training_status": "NOT_STARTED",
             }
             and all(metric is None for metric in gb002["learned_metrics"].values()),
             "GB002 unmeasured status differs")
    _require(cr054["status"] == "COMPLETED_AND_INDEPENDENTLY_REVIEWED"
             and cr054["scientific_gate"] == "FAILED"
             and cr054["coverage"]["collisions_per_arm"] == 384
             and cr054["coverage"]["collisions_per_category"] == 64
             and cr054["coverage"]["required_collisions_per_category"] == 2000
             and cr054["coverage"]["complete_post_study_evaluation"] is False
             and cr054["control_reuse"] == "CR053_HISTORICAL_CONTROL_NOT_FRESH_REPLICATION"
             and cr054["uncertainty"]["single_seed_uncertainty"] is None,
             "CR054 review scope differs")
    for arm in cr054["arms"].values():
        _require(arm["native"]["exact_b_recovery"] == {"correct": 0, "nominal": 256, "unavailable": 0}
                 and arm["native"]["inclusive_b_recovery"] == {"correct": 0, "nominal": 256, "unavailable": 0}
                 and arm["b_channel_coverage"] == {"evaluated_channel_groups": 256,
                                                    "evaluated_b_trials": 256,
                                                    "exact_successes": 0,
                                                    "successful_channel_groups": 0,
                                                    "unavailable": 0}
                 and arm["continuum_component_coverage"] == {
                     "ccbar": {"correct": 0, "nominal": 214, "unknown": 0,
                               "parton_ancestry_available": False},
                     "uubar": {"correct": 3, "nominal": 175, "unknown": 0,
                               "parton_ancestry_available": False},
                     "ddbar": {"correct": 6, "nominal": 190, "unknown": 0,
                               "parton_ancestry_available": False},
                     "ssbar": {"correct": 3, "nominal": 179, "unknown": 0,
                               "parton_ancestry_available": False},
                 }
                 and arm["deep_predicates"]["generated_typed"]["correct"] == 0
                 and arm["deep_predicates"]["accepted_typed"]["correct"] == 0,
                 "CR054 learned endpoint differs")
    # This tracked file is already a small sanitized source. Re-encode it to
    # make the public download deterministic rather than copying source bytes.
    published = {
        "schema": "generic-b-dashboard-public-download-v2",
        "source_summary_sha256": binding["source_sha256"],
        "status_date": value["status_date"],
        "provenance": value["provenance"],
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
    }
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (output / "generic-b-status-summary.json").write_bytes(encoded)
    return value | {"metric_download": {"filename": "generic-b-status-summary.json",
                    "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}}


def render(value):
    """Render completed results separately from current operational state."""
    gb001, gb002, cr054 = value["gb001"], value["gb002"], value["cr054"]
    static = gb001["static_compatibility"]
    lines = [
        "GB001: completed generic-sample static target-contract census",
        "-------------------------------------------------------------", "",
        "GB001 is a reviewed TRAIN target-contract metadata census, not a learned-model evaluation.",
        "The fixed 28,000-collision sample contains 20,000 B and 8,000 continuum collisions. Its unit",
        "is a nominal B predicate (40,000 in the B collisions), not a reconstructed B tag.",
        "The 34,819 accepted-membership and 6,378 recursive-topology+PID counts are static compatibility",
        "necessary conditions. They are not reconstruction efficiency, a ceiling, saturation, or legal",
        "decoder reachability. All 40,000 predicates were reconciled with zero processing failures.", "",
    ]
    lines += _table(["Static endpoint", "Compatible", "Available", "Nominal", "Unavailable"], [
        [name.replace("_", " "), point["compatible"] if point["compatible"] is not None else "UNAVAILABLE",
         point["available"], point["nominal"], point["unavailable"]]
        for name, point in static.items()
    ])
    lines += [
        "The percentages implied by these counts describe this fixed predicate census only. Physical",
        "completeness is UNAVAILABLE for all 40,000 predicates because the needed provenance is",
        "unavailable in the census inputs, so original-mDST recovery was not assessed.",
        "Background acceptance, purity, exact/inclusive learned recovery and uncertainty are UNAVAILABLE,",
        "not zero, because GB001 ran no model inference.", "",
    ]
    lines += _table(["Category", "Collisions", "Nominal B predicates", "Accepted static", "Recursive topology+PID static"], [
        [name, row["collisions"], row["nominal_b_predicates"], row.get("accepted_static_compatible", "N/A"),
         row.get("recursive_topology_pid_static_compatible", "N/A")]
        for name, row in gb001["categories"].items()
    ])
    lines += [
        "Recursive blocker counts overlap and must not be summed as a partition: minimum daughters 28,859;",
        "detector/nominal typed charge 20,039; allowed-token charge union 5,878; duplicated detector",
        "resource or source alias 81; recursive detector-source conflict 53.",
        "Recorded resources: 5,162 allocated CPU-seconds, 1,474.132 recorded CPU-seconds, and",
        "2,740,744,192 bytes peak memory. These are execution resources, not model runtime throughput.", "",
        "GB002: local transfer complete; scientific authentication unresolved",
        "--------------------------------------------------------------------", "",
        "This section is an as-of snapshot at the reviewed science-loop resumption boundary, not a live job monitor.",
        f"The recorded {gb002['snapshot_recorded_state'].replace('_', ' ')} state is a historical",
        "and now-superseded operational snapshot. The required data transfer is complete and hash-verified,",
        "but it establishes opaque-byte identity only; scientific payload authentication remains incomplete.",
        "Project-level CPU and GPU time-admission ceilings were removed. Actual usage accounting, site",
        "scheduler limits and finite per-operation watchdogs remain enforced. At this snapshot,",
        "the recorded 6,272 allocated CPU-seconds and zero GPU-hours are operational cost, not reconstruction performance.", "",
        "The reviewed runtime attempt represented in this snapshot passed scheduler and resource admission,",
        "reached the first of 3,072",
        "events, and failed closed before authentication because event-number and source-entry identities were",
        "conflated: 0 authenticated, 1 failed and 3,071 unreached. It performed no training or inference.",
        "The replacement ownership transition then ended FAILED_REVIEW_REQUIRED before activation completed,",
        "with all 3,072 replacement events unreached. The reviewed Astra/Sol science loop has since resumed and",
        "is running a focused operational repair; payload decoding and the native baseline remain pending.", "",
        "The sealed prospective v3 plan prioritizes payload decoding and a native baseline, then compares two",
        "Generic B training sample sizes by two network widths with learning curves, alongside measured inference",
        "improvements. Recoverable operational failures route to focused repair and renewed reviewed execution.",
        "The plan and transition do not constitute execution approval or a completed scientific result.",
        "There is still no completed GB002 training or learned evaluation. Every learned",
        "metric—exact/inclusive B recovery, background, purity, recursive topology/PID, kinematic validity,",
        "uncertainty and runtime—is explicitly UNAVAILABLE. Operational test counts are not performance metrics.", "",
        "Latest learned full reconstruction evaluation — CR054",
        "-----------------------------------------------------", "",
        "CR054 compares the reused CR053 control with initial-feature history while retaining the CR042",
        "connected step-4,376 native baseline. Only auxiliary feature history changes. The control is not",
        "a fresh replication. This previously inspected adaptive-TRAIN assessment has 384 collisions per",
        "arm (64/category), 1,936/category short of the required complete-evaluation coverage.", "",
    ]
    arms = cr054["arms"]
    lines += _table(["Learned endpoint", "Reused control", "Initial feature history", "Unit"], [
        ["Native exact B recovery", _point(arms["control"]["native"]["exact_b_recovery"]),
         _point(arms["initial_feature_history"]["native"]["exact_b_recovery"]), "B slots"],
        ["Native inclusive FSP-group recovery", _point(arms["control"]["native"]["inclusive_b_recovery"]),
         _point(arms["initial_feature_history"]["native"]["inclusive_b_recovery"]), "B slots"],
        ["Event any exact", _point(arms["control"]["native"]["event_any_exact"]),
         _point(arms["initial_feature_history"]["native"]["event_any_exact"]), "B collisions"],
        ["Event both exact", _point(arms["control"]["native"]["event_both_exact"]),
         _point(arms["initial_feature_history"]["native"]["event_both_exact"]), "B collisions"],
        ["Exact accepted-candidate purity", _point(arms["control"]["native"]["accepted_candidate_purity"], "correct", "accepted_candidates"),
         _point(arms["initial_feature_history"]["native"]["accepted_candidate_purity"], "correct", "accepted_candidates"), "accepted B candidates"],
        ["Flat accepted source-group recovery", _point(arms["control"]["flat"]["accepted_b_recovery"]),
         _point(arms["initial_feature_history"]["flat"]["accepted_b_recovery"]), "B slots; not full reconstruction"],
    ])
    lines += _table(["B category", "Control exact", "Control inclusive", "Initial exact", "Initial inclusive", "Nominal slots/arm"], [
        [category, arms["control"]["b_categories"][category]["exact"],
         arms["control"]["b_categories"][category]["inclusive"],
         arms["initial_feature_history"]["b_categories"][category]["exact"],
         arms["initial_feature_history"]["b_categories"][category]["inclusive"],
         arms["control"]["b_categories"][category]["nominal_b_slots"]]
        for category in ("charged", "mixed")
    ])
    lines += _table(["Continuum category", "Control fake-event acceptance", "Initial fake-event acceptance",
                     "Control retained-component recovery", "Initial retained-component recovery", "Collisions/arm"], [
        [category, row["accepted"], arms["initial_feature_history"]["continuum_fake_event_acceptance"][category]["accepted"],
         _point(arms["control"]["continuum_component_coverage"][category]),
         _point(arms["initial_feature_history"]["continuum_component_coverage"][category]), row["collisions"]]
        for category, row in arms["control"]["continuum_fake_event_acceptance"].items()
        if category != "all"
    ] + [["all continuum", 47, 49, "12/758", "12/758", 256]])
    lines += [
        "Exact tag purity is zero among 99 control and 101 initial-history accepted B candidates.",
        "This is distinct from continuum fake-event acceptance (47/256 versus 49/256). The retained-",
        "component denominator is truth components, not collision events or B slots. Continuum parton",
        "ancestry reconstruction is UNAVAILABLE because the schema has no quark roots or ancestry.", "",
        "Channel coverage",
        "~~~~~~~~~~~~~~~~", "",
        "Each arm evaluates 256 signed retained-channel groups across 256 B trials, with zero exact",
        "successes, zero successful groups and zero unavailable trials. These are reduced retained-tree",
        "channel groups; full generator decay descriptions and physical FEI equivalence are unavailable.", "",
        "Recursive/source/PID validity",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
    ]
    retained_names = ("raw_membership", "accepted_membership", "generated", "generated_typed",
                      "native_valid", "accepted", "topology", "topology_pid")
    lines += _table(["Retained-predicate stage", "Control", "Initial feature history", "Unavailable"], [
        [name.replace("_", " "), _point(arms["control"]["retained_predicates"][name]),
         _point(arms["initial_feature_history"]["retained_predicates"][name]),
         f"{arms['control']['retained_predicates'][name]['unavailable']} / {arms['initial_feature_history']['retained_predicates'][name]['unavailable']}"]
        for name in retained_names
    ])
    lines += [
        "Every retained row keeps the full 2,644-predicate nominal denominator. Deep typed generated and",
        "accepted recovery is 0/992 in both arms; topology+PID has one unavailable predicate per arm.",
        "Committed source conflicts and daughter-sum closure violations are zero. Physical momentum",
        "resolution is UNAVAILABLE because retained truth composite four-vectors are absent; closure is",
        "an implementation invariant, not physical kinematic resolution. Beam evaluation was not run.", "",
        "Uncertainty and runtime",
        "~~~~~~~~~~~~~~~~~~~~~~~", "",
        "Native exact recovery difference is 0 with an empirical paired 95% interval [0, 0] over 256 B slots.",
        "Native continuum background difference is +0.0078125 with interval [-0.01171875, 0.02734375]",
        "over 256 continuum collisions. Flat accepted difference is -0.00390625 with interval",
        "[-0.01171875, 0] over 256 B slots. An all-zero empirical interval does not prove population zero.",
        "Training-seed uncertainty is UNAVAILABLE (one seed).", "",
        "New scientific-child cost was 6,750.681 wall-seconds, 7,335.960 user CPU-seconds and 19.000",
        "system CPU-seconds; maximum child RSS was 1,628,692 KiB. The inherited control runtime is separate,",
        "and the treatment adds a frozen feature pass, so no equal-FLOPs or speedup claim is supported.", "",
        "The registered scientific gate failed. The 1/256 versus 0/256 flat accepted count is mediator",
        "evidence, not full reconstruction. No fresh confirmation, physical FEI efficiency, promotion or",
        "complete post-study evaluation is claimed.", "",
        ":download:`Sanitized GB001, GB002 and CR054 aggregates with content hashes <generic-b-status-summary.json>`.", "",
    ]
    return lines
