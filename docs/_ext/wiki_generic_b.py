"""Bounded aggregate publication for Generic-B reconstruction studies."""
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
        "schema": "generic-b-dashboard-public-binding-v4",
        "status_date": "2026-11-05",
        "source_file": "summary.json",
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "private_payloads_copied": False,
        "contains_operational_identifiers": False,
        "contains_cohort_member_identifiers": False,
    }, "Generic-B binding differs")
    value = json.loads(raw)
    _require(set(value) == {"schema", "status_date", "provenance", "gb001", "gb002", "cr054", "gb101"},
             "Generic-B summary fields differ")
    _require(value["schema"] == "generic-b-dashboard-public-summary-v4", "Generic-B schema differs")
    gb001, gb002, cr054, gb101 = value["gb001"], value["gb002"], value["cr054"], value["gb101"]
    _require(gb101["status"] == "COMPLETED_COVERAGE_INCOMPLETE_NULL_OR_HARM"
             and gb101["scientific_gate"] == "FAILED"
             and gb101["checkpoint_promoted"] is False
             and gb101["coverage"] == {
                 "authenticated_local_inventory_per_category": 512,
                 "complete_post_study_evaluation": False,
                 "diagnostic_collisions": 80,
                 "exploratory_development_collisions_per_category": 8,
                 "exploratory_processed_per_category": 8,
                 "exploratory_processed_total": 48,
                 "exploratory_shortfall_per_category": 1992,
                 "historical_pretraining_exposure":
                     "INCOMPLETE_FOR_SAME_512_EVENT_POPULATION_NOT_ADDITIONAL_EVENTS",
                 "naf_inventory": "UNAVAILABLE_AUTHENTICATION",
                 "prior_adaptive_development_reservation_per_category": 512,
                 "strict_eligible_available_per_category": 0,
                 "strict_eligible_available_total": 0,
                 "strict_eligible_processed_per_category": 0,
                 "strict_eligible_processed_total": 0,
                 "strict_eligible_selected_per_category": 0,
                 "strict_eligible_selected_total": 0,
                 "strict_requested_per_category": 2000,
                 "strict_requested_total": 12000,
                 "strict_shortfall_per_category": 2000,
                 "train_diagnostics": 32,
             },
             "GB101 result scope differs")
    separated = gb101["role_separated_endpoints"]
    _require(separated["endpoint_order"] == [
                 "exact_topology_pid", "topology_only", "inclusive_source"]
             and separated["roles"] == {
                 "development": {"collisions": 16, "nominal_b_slots": 32},
                 "train_diagnostic": {"collisions": 32, "nominal_b_slots": 64},
             }
             and set(separated["views"]) == {
                 "latest_beam4", "latest_native_gb101",
                 "unpromoted_parent_beam4", "unpromoted_parent_native"},
             "GB101 role-separated endpoint scope differs")
    for view_name, view in separated["views"].items():
        expected_keys = {"top1_correct_in_endpoint_order", "oracle_at_1_correct_in_endpoint_order"}
        if "beam4" in view_name:
            expected_keys.add("oracle_at_4_correct_in_endpoint_order")
        for role in ("development", "train_diagnostic"):
            _require(set(view[role]) == expected_keys
                     and all(view[role][key] == [0, 0, 0] for key in expected_keys),
                     "GB101 role-separated endpoint values differ")
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
        "schema": "generic-b-dashboard-public-download-v4",
        "source_summary_sha256": binding["source_sha256"],
        "status_date": value["status_date"],
        "provenance": value["provenance"],
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
        "gb101": gb101,
    }
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (output / "generic-b-status-summary.json").write_bytes(encoded)
    return value | {"metric_download": {"filename": "generic-b-status-summary.json",
                    "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}}


def render(value):
    """Render completed results separately from current operational state."""
    gb001, gb002, cr054, gb101 = value["gb001"], value["gb002"], value["cr054"], value["gb101"]
    static = gb001["static_compatibility"]
    selected = gb101["selected_b_development"]
    latest = gb101["latest_beam4"]
    separated = gb101["role_separated_endpoints"]
    endpoint_labels = {
        "exact_topology_pid": "Exact topology+PID",
        "topology_only": "Topology only",
        "inclusive_source": "Inclusive source grouping",
    }
    latest_native = separated["views"]["latest_native_gb101"]
    latest_beam = separated["views"]["latest_beam4"]
    lines = [
        "GB101: full-depth beam search on the latest H3-adapted checkpoint",
        "------------------------------------------------------------------", "",
        "The requested width-4 coherent-forest beam evaluation completed 160 new rollouts on the",
        "80-collision reused exploratory panel. GB101 H3 conflict packing was bound into the nominal",
        "beam lane; alternate forests retained the documented likelihood policy difference. No truth",
        "entered generation, pruning, ranking or stopping, and no checkpoint was promoted.", "",
        "Strict validation coverage is unavailable: 0 of 2,000 eligible collisions per category were",
        "available, selected or processed (0/12,000 total), leaving a strict shortfall of 2,000/category.",
        "All 512 locally inventoried collisions/category are prior adaptive-development reservations;",
        "incomplete historical pretraining exposure applies to that same population and is not another",
        "512/category. This inventory does not establish that the other 1,488/category were inspected.", "",
        "The retained numerical results use 8 exploratory observations/category (48 total), leaving a",
        "diagnostic numerical shortfall of 1,992/category. They do not satisfy the strict validation quota.",
        "Of the signal observations, 16 development collisions supply 32 B slots. Another 32 charged/mixed",
        "collisions are TRAIN diagnostics supplying 64 B slots and are reported separately, not as",
        "generalization. NAF inventory access was unavailable and was not retried for this correction.", "",
    ]
    lines += _table(["Coverage class", "Per category", "All six categories", "Interpretation"], [
        ["Strict requested", "2,000", "12,000", "complete-study target"],
        ["Strict eligible available / selected / processed", "0 / 0 / 0", "0 / 0 / 0", "quota unmet"],
        ["Strict shortfall", "2,000", "12,000", "unavailable"],
        ["Exploratory processed", "8", "48", "reused adaptive observations"],
        ["Exploratory numerical shortfall", "1,992", "11,952", "not strict coverage"],
    ])
    lines += _table(["Latest-checkpoint continuous endpoint", "Beam width 4", "GB101 native greedy", "Role"], [
        ["Selected-B source F1", f"{selected['latest_beam4_source_f1']['point']:.6f}", f"{selected['latest_greedy_source_f1']['point']:.6f}", "development only"],
        ["Selected-B soft-edge F1", f"{selected['latest_beam4_soft_edge_f1']['point']:.6f}", f"{selected['latest_greedy_soft_edge_f1']['point']:.6f}", "development only"],
    ])
    endpoint_order = separated["endpoint_order"]
    lines += _table(["Development endpoint (32 B slots)", "Greedy top-1", "Greedy oracle@1",
                     "Beam top-1", "Beam oracle@1", "Beam oracle@4"], [
        [endpoint_labels[name],
         f"{latest_native['development']['top1_correct_in_endpoint_order'][index]}/32",
         f"{latest_native['development']['oracle_at_1_correct_in_endpoint_order'][index]}/32",
         f"{latest_beam['development']['top1_correct_in_endpoint_order'][index]}/32",
         f"{latest_beam['development']['oracle_at_1_correct_in_endpoint_order'][index]}/32",
         f"{latest_beam['development']['oracle_at_4_correct_in_endpoint_order'][index]}/32"]
        for index, name in enumerate(endpoint_order)
    ])
    lines += _table(["TRAIN diagnostic endpoint (64 B slots)", "Greedy top-1", "Greedy oracle@1",
                     "Beam top-1", "Beam oracle@1", "Beam oracle@4"], [
        [endpoint_labels[name],
         f"{latest_native['train_diagnostic']['top1_correct_in_endpoint_order'][index]}/64",
         f"{latest_native['train_diagnostic']['oracle_at_1_correct_in_endpoint_order'][index]}/64",
         f"{latest_beam['train_diagnostic']['top1_correct_in_endpoint_order'][index]}/64",
         f"{latest_beam['train_diagnostic']['oracle_at_1_correct_in_endpoint_order'][index]}/64",
         f"{latest_beam['train_diagnostic']['oracle_at_4_correct_in_endpoint_order'][index]}/64"]
        for index, name in enumerate(endpoint_order)
    ])
    delta = selected["beam_minus_greedy_source_f1"]
    edge_delta = selected["beam_minus_greedy_soft_edge_f1"]
    lines += [
        f"Beam minus greedy selected-B source-F1 was {delta['point']:.6f} with the shared-seed-5001",
        f"paired 95% interval [{delta['seed5001_interval']['lower']:.6f}, {delta['seed5001_interval']['upper']:.6f}].",
        f"The soft-edge delta was {edge_delta['point']:.6f}, interval",
        f"[{edge_delta['seed5001_interval']['lower']:.6f}, {edge_delta['seed5001_interval']['upper']:.6f}].",
        "Beam reduced continuum accepted events from greedy 5/8, 5/8, 4/8 and 6/8 to",
        "1/8, 0/8, 1/8 and 1/8 for ccbar, uubar, ddbar and ssbar, respectively, but",
        "this came with a large signal-quality loss and zero exact/inclusive oracle recovery. The",
        "unpromoted parent reference is also zero for all three endpoints in both role-separated views;",
        "no pooled 0/96 denominator is used and no oracle result is promoted.", "",
        f"The latest beam averaged {latest['latency_seconds_per_collision']:.3f} seconds/collision,",
        f"made {latest['model_calls']} model calls, returned {latest['returned_candidates']} ranked",
        f"candidates, and hit the per-query expansion limit in {latest['query_limit_events']['numerator']}/80 events.",
        "Four CPU jobs consumed 826 allocated CPU-seconds and zero GPU-hours; the serial paired",
        "aggregation used 296.46 local CPU-seconds. This is a coverage-incomplete null/harm result,",
        "not population performance or physical FEI completeness.", "",
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
        ":download:`Sanitized GB001, GB002, CR054 and GB101 aggregates with content hashes <generic-b-status-summary.json>`.", "",
    ]
    return lines
