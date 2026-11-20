"""Bounded aggregate publication for Generic-B reconstruction studies."""
from __future__ import annotations

import hashlib
import json


SOURCE = "artifacts/codex/generic_b_dashboard_20261019"
FILES = {
    "generic-b-accumulated-findings-v1.json",
    "generic-b-recovered-default-four-v1.json",
    "generic-b-status-summary.json",
    "generic-b-status-summary-v4.json",
    "generic-b-status-summary-v5.json",
    "generic-b-status-summary-v6.json",
    "generic-b-status-summary-v7.json",
    "generic-b-neural-support-v1.json",
    "generic-b-supported-h3-v1.json",
    "generic-b-h2-anchor-v1.json",
    "generic-b-positive-object-v1.json",
    "generic-b-retained-source-v1.json",
    "generic-b-preservation-v1.json",
    "generic-b-source-supervision-v1.json",
    "generic-b-support-capture-v1.json",
    "generic-b-relation-support-v1.json",
    "generic-b-decoder-native-v1.json",
    "generic-b-query-refresh-fit12-shallow-v1.json",
    "generic-b-query-refresh-fit12-deep-v1.json",
    "generic-b-query-refresh-assessment4-shallow-v1.json",
    "generic-b-query-refresh-assessment4-deep-v1.json",
    "generic-b-query-refresh-native-v1.json",
    "generic-b-query-interaction-native-v1.json",
    "generic-b-dynamic-native-v1.json",
    "generic-b-projection-native-v1.json",
    "generic-b-expanded96-native-v1.json",
    "generic-b-expanded96-teacher-v1.json",
    "generic-b-expanded96-endpoints-v1.json",
    "generic-b-expanded96-fit12-shallow-v1.json",
    "generic-b-expanded96-fit12-deep-v1.json",
    "generic-b-expanded96-assessment4-shallow-v1.json",
    "generic-b-expanded96-assessment4-deep-v1.json",
    "generic-b-replay-native-v1.json",
    "generic-b-replay-teacher-v1.json",
    "generic-b-replay-endpoints-v1.json",
    "generic-b-replay-fit12-shallow-v1.json",
    "generic-b-replay-fit12-deep-v1.json",
    "generic-b-replay-assessment4-shallow-v1.json",
    "generic-b-replay-assessment4-deep-v1.json",
    "generic-b-balanced-refresh-native-v1.json",
    "generic-b-balanced-refresh-teacher-comparison-v1.json",
    "generic-b-balanced-refresh-endpoints-by-category-v1.json",
    "generic-b-balanced-refresh-fit12-shallow-v1.json",
    "generic-b-balanced-refresh-fit12-deep-v1.json",
    "generic-b-balanced-refresh-assessment4-shallow-v1.json",
    "generic-b-balanced-refresh-assessment4-deep-v1.json",
    "generic-b-refresh-native-v1.json",
    "generic-b-refresh-shallow-v1.json",
    "generic-b-refresh-deep-v1.json",
    "generic-b-joint-native-v1.json",
    "generic-b-joint-assessment4-deep-v1.json",
    "generic-b-joint-assessment4-shallow-v1.json",
    "generic-b-joint-fit12-deep-v1.json",
    "generic-b-joint-fit12-shallow-v1.json",
    "generic-b-decoder-assessment4-deep-v1.json",
    "generic-b-decoder-assessment4-shallow-v1.json",
    "generic-b-decoder-fit12-deep-v1.json",
    "generic-b-decoder-fit12-shallow-v1.json",
    "generic-b-corrected-depth-v1.json",
}
LIMIT = 40_960
HISTORICAL_V3_BYTES = 25_249
HISTORICAL_V3_SHA256 = "612f24cbfa13d0c6e1440a168da8c31939de6ed7b03dd6cf73954dd98c933d2f"
HISTORICAL_V4_SOURCE_SHA256 = "273adbc36a416dedd2bd296c0e154b842832ab0333874ee4441cd64fa42be12e"
HISTORICAL_V4_BYTES = 28_609
HISTORICAL_V4_SHA256 = "2be7a36e81269fa41b71cfb1c0ccc109366f86fec82e29c45f7f2ec974d0823b"
HISTORICAL_V5_SOURCE_SHA256 = "5f89af37af06e4603b02b3f21ea512f583d3f2a46982724e790511592af8ba3f"
HISTORICAL_V5_BYTES = 33_319
HISTORICAL_V5_SHA256 = "85352c695dadcce57607b4649cf7b43c882a900ec0a1dfdc0e103a0c45b625aa"
HISTORICAL_V6_SOURCE_SHA256 = "49eb9a331299bf99a18d2fbe0ed1270a60be6be0f15ca80f4601e9d790b70bb0"
HISTORICAL_V6_BYTES = 39_142
HISTORICAL_V6_SHA256 = "b1c8b14a63bdb90e16a878ad66a910c8a228e183d89bd5abb6393fafb5409fae"
LATEST_PROVENANCE = {
    "gb101_h1_positive_anchor_scientific_result_sha256",
    "gb101_h1_positive_anchor_closeout_sha256",
    "gb101_h1_positive_anchor_review_sha256",
}


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
    historical_path = source / "historical-summary-v3-source.json"
    _require(not binding_path.is_symlink() and not summary_path.is_symlink()
             and not historical_path.is_symlink(), "Unsafe generic-B source")
    _require(binding_path.stat().st_size <= LIMIT and summary_path.stat().st_size <= LIMIT
             and historical_path.stat().st_size <= LIMIT,
             "Generic-B source exceeds limit")
    binding = json.loads(binding_path.read_bytes())
    raw = summary_path.read_bytes()
    _require(binding == {
        "schema": "generic-b-dashboard-public-binding-v7",
        "status_date": "2026-11-06",
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
    _require(value["schema"] == "generic-b-dashboard-public-summary-v7", "Generic-B schema differs")
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
    diagnosis = gb101["saved_beam_diagnosis"]
    _require(diagnosis["classification"] == "RETAINED_POOL_DEFICIT_ONLY"
             and diagnosis["analysis"] == {
                 "bootstrap_replicates": 2000, "saved_rows": 160, "scheduler_jobs": 0}
             and diagnosis["interpretation"] == {
                 "ranking_opportunity_supported": False,
                 "ranking_rule": "POOL_MINUS_TOP1_SOURCE_F1_LOWER95_STRICTLY_POSITIVE",
                 "retained_pool_deficit_supported": True,
                 "truth_scored_oracle_promoted": False,
             }
             and diagnosis["signal_diagnostics_combined_train_and_development"] == {
                 "b_root_only_below_top1": 2,
                 "no_b_root_anywhere_retained": 40,
                 "no_b_root_top1": 42,
                 "query_expansion_cap_hit": 43,
                 "signal_collisions": 48,
             }
             and diagnosis["telemetry"] == {
                 "causal_pruning_attribution": False,
                 "dropped_forest_quality": False,
                 "h3_nominal_lane_later_survival": False,
                 "saved_candidate_whole_hypothesis_quality": True,
                 "saved_search_pruning_and_saturation_counters": True,
                 "strict_validation_coverage": False,
             }
             and diagnosis["next_scientific_question"]
                 == "ONE_CONTROLLED_PROPOSAL_RETENTION_CHANGE_AT_SATURATED_QUERY_CAP_WITH_MODEL_BEAM_WIDTH_AND_SCORING_FIXED",
             "GB101 saved-beam diagnosis differs")
    cap = gb101["cap256_intervention"]
    _require(cap["classification"] == "NULL_OR_HARM"
             and cap["changed_factor"] == {
                 "max_candidate_expansions_per_query": {"control": 128, "treatment": 256}}
             and cap["source_revisions"] == {
                 "gb101_scientific": "d3e20ee2dcf9d0ab4e62ae3e8c698c9c296b8254",
                 "cap256_implementation": "cc9608b899b6471a850966ee600b319d4260b3e0"}
             and cap["support"] == {
                 "nominal": 80, "processed": 80, "failed": 0, "incompatible": 0,
                 "unavailable": 0, "train_diagnostic": 32, "development": 48,
                 "development_per_category": 8, "strict_eligible_per_category": 0,
                 "strict_requested_per_category": 2000}
             and cap["development_equal_category"]["top1_source_f1"] == {
                 "cap128": 0.020833333333333332, "cap256": 0.020833333333333332,
                 "delta": 0.0, "lower95": 0.0, "upper95": 0.0}
             and cap["development_equal_category"]["retained_pool_oracle_source_f1"] == {
                 "cap128": 0.035256410256410256, "cap256": 0.035256410256410256,
                 "delta": 0.0, "lower95": 0.0, "upper95": 0.0}
             and cap["development_equal_category"]["height3_retained"]["source_f1_delta"] == {
                 "point": 0.002604166666666685, "lower95": -0.007812499999999972,
                 "upper95": 0.015625}
             and cap["development_equal_category"]["height3_retained"]["source_iou_delta"] == {
                 "point": 0.0035849567099567325, "lower95": -0.004870129870129858,
                 "upper95": 0.015625}
             and cap["search_and_cost"]["cap128"]["cap_hit_events"] == 63
             and cap["search_and_cost"]["cap256"]["cap_hit_events"] == 60
             and cap["search_and_cost"]["cap128"]["daughter_partial_states_expanded"] == 102528
             and cap["search_and_cost"]["cap256"]["daughter_partial_states_expanded"] == 163230
             and cap["search_and_cost"]["latency_increase_fraction"] == 0.1622912010890576
             and cap["interpretation"]["checkpoint_or_policy_promoted"] is False
             and cap["interpretation"]["global_cap_irrelevance_established"] is False,
             "GB101 cap256 intervention differs")
    adaptation = gb101["model_adaptation_series"]
    _require(adaptation["classification"] == "ACCUMULATED_REUSED_PANEL_NULL_OR_HARM"
             and adaptation["support"] == {
                 "nominal": 80, "processed": 80, "failed": 0, "development": 48,
                 "development_per_category": 8, "strict_processed": 0,
                 "strict_requested": 12000, "train_contexts": 32, "updates_per_fit": 32}
             and adaptation["parent_selected_b_source_f1"] == 0.056628787878787876
             and [item["name"] for item in adaptation["experiments"]] == [
                 "H2 confidence adaptation", "H2 unweighted daughter pointer",
                 "H2 balanced daughter pointer", "H2 ranking daughter pointer",
                 "H1 ranking pointer", "H1 ranking plus positive anchor"]
             and all(item["verdict"] == "NULL_OR_HARM"
                     for item in adaptation["experiments"])
             and adaptation["positive_anchor_diagnosis"]["positive_bce_after"]
                 == 0.3031325340270996
             and adaptation["positive_anchor_diagnosis"]["negative_bce_after"]
                 == 0.9431222081184387
             and adaptation["positive_anchor_diagnosis"]["continuum_top1_events"]
                 == {"ccbar": 1, "uubar": 1, "ddbar": 4, "ssbar": 1}
             and adaptation["positive_anchor_diagnosis"]["background_guardrail_passed"] is False
             and adaptation["endpoints"] == {
                 "development_nominal_b_slots": 32, "top1_exact_topology_pid": 0,
                 "top1_topology_only": 0, "top1_inclusive_source": 0,
                 "oracle4_exact_topology_pid": 0, "oracle4_topology_only": 0,
                 "oracle4_inclusive_source": 0}
             and adaptation["checkpoint_promoted"] is False,
             "GB101 model-adaptation series differs")
    development = diagnosis["development_equal_category"]
    _require(development["states"]["greedy"]["source_f1"] == 0.22893044455544453
             and development["states"]["beam_top1"]["source_f1"] == 0.020833333333333332
             and development["states"]["retained_pool_oracle"]["source_f1"] == 0.035256410256410256
             and development["contrasts"]["pool_minus_top1"]["source_f1"] == {
                 "point": 0.014423076923076924,
                 "lower95": 0.0,
                 "upper95": 0.04326923076923077,
             }
             and development["contrasts"]["pool_minus_greedy"]["source_f1"] == {
                 "point": -0.1936740342990343,
                 "lower95": -0.28599358974358974,
                 "upper95": -0.10710268897768904,
             }, "GB101 saved-beam quality differs")
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
    historical = json.loads(historical_path.read_bytes())
    _require(historical["schema"] == "generic-b-dashboard-public-download-v3"
             and historical["gb101"]["status"] == gb101["status"]
             and historical["provenance"]["gb101_source_revision"]
                 == value["provenance"]["gb101_source_revision"],
             "Historical Generic-B v3 scope differs")
    # Re-encode the historical source so the prior public bytes are retained
    # without copying a tracked raw source body into the publication artifact.
    historical_encoded = (
        json.dumps(historical, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    _require(len(historical_encoded) == HISTORICAL_V3_BYTES
             and hashlib.sha256(historical_encoded).hexdigest() == HISTORICAL_V3_SHA256,
             "Historical Generic-B v3 download differs")
    (output / "generic-b-status-summary.json").write_bytes(historical_encoded)
    historical_v4 = {
        "schema": "generic-b-dashboard-public-download-v4",
        "source_summary_sha256": HISTORICAL_V4_SOURCE_SHA256,
        "status_date": "2026-11-05",
        "provenance": {
            key: item for key, item in value["provenance"].items()
            if key not in {"gb101_saved_beam_diagnosis_review_sha256",
                           "gb101_saved_beam_diagnosis_sha256",
                           "gb101_cap256_scientific_result_sha256",
                           "gb101_cap256_closeout_sha256",
                           "gb101_cap256_review_sha256", *LATEST_PROVENANCE}
        },
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
        "gb101": {key: item for key, item in gb101.items()
                  if key not in {"saved_beam_diagnosis", "cap256_intervention",
                                 "model_adaptation_series"}},
    }
    historical_v4_encoded = (
        json.dumps(historical_v4, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    _require(len(historical_v4_encoded) == HISTORICAL_V4_BYTES
             and hashlib.sha256(historical_v4_encoded).hexdigest() == HISTORICAL_V4_SHA256,
             "Historical Generic-B v4 download differs")
    (output / "generic-b-status-summary-v4.json").write_bytes(historical_v4_encoded)
    historical_v5 = {
        "schema": "generic-b-dashboard-public-download-v5",
        "source_summary_sha256": HISTORICAL_V5_SOURCE_SHA256,
        "status_date": "2026-11-05",
        "provenance": {
            key: item for key, item in value["provenance"].items()
            if key not in {"gb101_cap256_scientific_result_sha256",
                           "gb101_cap256_closeout_sha256",
                           "gb101_cap256_review_sha256", *LATEST_PROVENANCE}
        },
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
        "gb101": {key: item for key, item in gb101.items()
                  if key not in {"cap256_intervention", "model_adaptation_series"}},
    }
    historical_v5_encoded = (
        json.dumps(historical_v5, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    _require(len(historical_v5_encoded) == HISTORICAL_V5_BYTES
             and hashlib.sha256(historical_v5_encoded).hexdigest() == HISTORICAL_V5_SHA256,
             "Historical Generic-B v5 download differs")
    (output / "generic-b-status-summary-v5.json").write_bytes(historical_v5_encoded)
    historical_v6 = {
        "schema": "generic-b-dashboard-public-download-v6",
        "source_summary_sha256": HISTORICAL_V6_SOURCE_SHA256,
        "status_date": "2026-11-05",
        "provenance": {key: item for key, item in value["provenance"].items()
                       if key not in LATEST_PROVENANCE},
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
        "gb101": {key: item for key, item in gb101.items()
                  if key != "model_adaptation_series"},
    }
    historical_v6_encoded = (
        json.dumps(historical_v6, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    _require(len(historical_v6_encoded) == HISTORICAL_V6_BYTES
             and hashlib.sha256(historical_v6_encoded).hexdigest() == HISTORICAL_V6_SHA256,
             "Historical Generic-B v6 download differs")
    (output / "generic-b-status-summary-v6.json").write_bytes(historical_v6_encoded)
    published = {
        "schema": "generic-b-dashboard-public-download-v7",
        "source_summary_sha256": binding["source_sha256"],
        "status_date": value["status_date"],
        "provenance": value["provenance"],
        "gb001": gb001,
        "gb002": gb002,
        "cr054": cr054,
        "gb101": gb101,
    }
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (output / "generic-b-status-summary-v7.json").write_bytes(encoded)
    supplement = _neural_support(source, output)
    return value | {
        "accumulated_findings": _accumulated_findings(source, output),
        "recovered_default_four": _recovered_default_four(source, output),
        "neural_support": supplement,
        "supported_h3": _supported_h3(source, output),
        "h2_anchor": _h2_anchor(source, output),
        "metric_download": {"filename": "generic-b-status-summary-v7.json",
                            "bytes": len(encoded),
                            "sha256": hashlib.sha256(encoded).hexdigest()},
        "historical_v6_metric_download": {"filename": "generic-b-status-summary-v6.json",
                                           "bytes": len(historical_v6_encoded),
                                           "sha256": hashlib.sha256(historical_v6_encoded).hexdigest()},
        "historical_v5_metric_download": {"filename": "generic-b-status-summary-v5.json",
                                           "bytes": len(historical_v5_encoded),
                                           "sha256": hashlib.sha256(historical_v5_encoded).hexdigest()},
        "historical_v4_metric_download": {"filename": "generic-b-status-summary-v4.json",
                                           "bytes": len(historical_v4_encoded),
                                           "sha256": hashlib.sha256(historical_v4_encoded).hexdigest()},
        "historical_metric_download": {"filename": "generic-b-status-summary.json",
                                       "bytes": len(historical_encoded),
                                       "sha256": hashlib.sha256(historical_encoded).hexdigest()},
    }


def _neural_support(source, output):
    """Project the separately reviewed adaptive neural/support snapshot."""
    path = source / "neural-support-source.json"
    binding_path = source / "neural-support-binding.json"
    _require(not path.is_symlink() and not binding_path.is_symlink()
             and path.stat().st_size <= LIMIT and binding_path.stat().st_size <= LIMIT,
             "Unsafe neural-support source")
    raw = path.read_bytes()
    binding = json.loads(binding_path.read_bytes())
    _require(binding == {
        "schema": "generic-b-neural-support-binding-v1", "status_date": "2026-11-13",
        "source_file": "neural-support-source.json", "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "private_payloads_copied": False, "contains_operational_identifiers": False,
        "contains_cohort_member_identifiers": False,
    }, "Neural-support binding differs")
    value = json.loads(raw)
    support = value["teacher_support_correction"]
    _require(value["schema"] == "generic-b-neural-support-public-source-v1"
             and len(value["studies"]) == 6 and value["promotion"] is False
             and support["nominal_slots"] == 320
             and [support[k] for k in ("charge_compatible", "incompatible", "unavailable", "absent")]
                 == [1, 92, 49, 178]
             and support["compatible_by_category"] == {"charged": 0, "mixed": 1}
             and value["strict_validation"]["processed"] == 0,
             "Neural-support scope differs")
    published = value | {"schema": "generic-b-neural-support-public-download-v1",
                         "source_sha256": binding["source_sha256"]}
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    filename = "generic-b-neural-support-v1.json"
    (output / filename).write_bytes(encoded)
    return value | {"metric_download": {"filename": filename, "bytes": len(encoded),
                                        "sha256": hashlib.sha256(encoded).hexdigest()}}


def _render_neural_support(value):
    rows = []
    for study in value["studies"]:
        for cell in study["selected_B"]:
            if cell["panel"] != "native_assessment4" or cell["metric"] != "f1":
                continue
            rows.append((study["name"], cell["control_id"], cell["treatment_id"],
                         f'{cell["control"]:.6f}', f'{cell["treatment"]:.6f}',
                         f'{cell["delta"]:+.6f}',
                         f'[{cell["paired95"][0]:+.6f}, {cell["paired95"][1]:+.6f}]',
                         study["scientific_gate"]))
    lines = ["GB101 history: neural fit/native divergence and corrected H3 support",
             "-" * len("GB101 history: neural fit/native divergence and corrected H3 support"), "",
             "Reviewed snapshot: 13 November 2026. These are adaptive reused TRAIN mechanism",
             "studies, separate from the original 80-collision greedy/beam evaluation below.",
             "Each native assessment uses four charged/mixed collisions (eight nominal B slots);",
             "the twelve-collision native-fit panel is reported separately in the download.",
             "Intervals use 2,000 paired whole-collision seed-5001 draws and equal category weights.", ""]
    lines += ["Arm identities (the second-pass intervention is compared separately with N and B):", ""]
    lines += _table(["ID", "Exact arm identity"], sorted(value["arm_identities"].items()))
    lines += _table(["Intervention", "Control identity", "Treatment identity", "Control F1", "Treatment F1", "Delta", "Paired 95%", "Gate"], rows)
    lines += ["Projection changes produced uncertain assessment gains; weighting, contextualization",
              "and refreshed states failed safeguards. Better conditional TRAIN fit did not establish",
              "reliable native improvement. Root-anchored refresh reduced selected-B source F1 from",
              "0.285362 to 0.123413; its paired interval excludes zero in the harmful direction.", "",
              "The latest support correction retains all 320 nominal H3 slots: one charge-compatible",
              "mixed target, 92 incompatible, 49 unavailable and 178 absent. Five structural candidates",
              "had previously been counted before detector-charge compatibility was measured.",
              "There is zero charged teacher support. Charge guards remain unchanged; these counts",
              "describe supervision eligibility and do not measure learned or physical completeness.", "",
              "Strict independent validation remains 0/12,000, with 2,000 missing per category.",
              "Every continuum flavour's event and candidate acceptance is unavailable for these",
              "newer studies. Missing depth cells remain unavailable. No model, oracle or policy is",
              "promoted, and no overall PASS is supported. At this historical snapshot teacher",
              "construction failed before learning. The newer supported-H3 result above supersedes",
              "that earlier next action while preserving this snapshot and its download.", "",
              "The download preserves per-category deltas, source F1/IoU, depth source/edge guards,",
              "fit operands, source/review hashes and measured scheduler compute. Its six scientific",
              "operations used " + str(sum(r["allocated_cpu_seconds"] for r in value["studies"])) +
              " allocated CPU-seconds and " + str(sum(r["allocated_gpu_seconds"] for r in value["studies"])) +
              " allocated GPU-seconds; this is scoped accounting, not total campaign spend.", "",
              ":download:`Reviewed neural/support metrics <generic-b-neural-support-v1.json>`", ""]
    return lines



def _supported_h3(source, output):
    """Publish the reviewed single-supported-target matched neural outcome."""
    path = source / "supported-h3-source.json"
    binding_path = source / "supported-h3-binding.json"
    _require(not path.is_symlink() and not binding_path.is_symlink()
             and path.stat().st_size <= LIMIT and binding_path.stat().st_size <= LIMIT,
             "Unsafe supported-H3 source")
    raw = path.read_bytes()
    binding = json.loads(binding_path.read_bytes())
    _require(binding == {
        "schema": "generic-b-supported-h3-binding-v1", "status_date": "2026-11-14",
        "source_file": "supported-h3-source.json", "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "private_payloads_copied": False, "contains_operational_identifiers": False,
        "contains_cohort_member_identifiers": False,
    }, "Supported-H3 binding differs")
    value = json.loads(raw)
    _require(value["schema"] == "generic-b-supported-h3-public-source-v1"
             and value["promotion"] is False and value["overall_PASS"] is False
             and value["coverage"]["strict_validation_processed"] == 0
             and value["teacher_support"]["nominal_slots"] == 320
             and value["actual_counts"]["native_rollouts"] == 32,
             "Supported-H3 scope differs")
    published = value | {"schema": "generic-b-supported-h3-public-download-v1",
                         "source_sha256": binding["source_sha256"]}
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    filename = "generic-b-supported-h3-v1.json"
    (output / filename).write_bytes(encoded)
    return value | {"metric_download": {"filename": filename, "bytes": len(encoded),
                                        "sha256": hashlib.sha256(encoded).hexdigest()}}


def _render_supported_h3(value):
    return ['GB101 history: supported H3 learning changes selected-B quality', '---------------------------------------------------------------', '', 'Supported H3 learning changed selected-B quality without establishing all native margins. Sparse complete supervision, adaptive TRAIN roles and missing backgrounds limit interpretation. No promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Reviewed supported-H3 learning metrics <generic-b-supported-h3-v1.json>`', '']



def _h2_anchor(source, output):
    """Publish reviewed aggregate H2 outcomes and additive saved-depth correction."""
    result = {}
    for key, basename, schema in (("decoder-fit12-deep", "decoder-fit12-deep", "generic-b-decoder-fit12-deep"),
                                  ("decoder-assessment4-shallow", "decoder-assessment4-shallow", "generic-b-decoder-assessment4-shallow"),
                                  ("decoder-assessment4-deep", "decoder-assessment4-deep", "generic-b-decoder-assessment4-deep"),
                                  ("decoder-fit12-shallow", "decoder-fit12-shallow", "generic-b-decoder-fit12-shallow"),
                                  ("joint-fit12-shallow", "joint-fit12-shallow", "generic-b-joint-fit12-shallow"),
                                  ("joint-fit12-deep", "joint-fit12-deep", "generic-b-joint-fit12-deep"),
                                  ("joint-assessment4-shallow", "joint-assessment4-shallow", "generic-b-joint-assessment4-shallow"),
                                  ("joint-assessment4-deep", "joint-assessment4-deep", "generic-b-joint-assessment4-deep"),
                                  ("query-refresh-fit12-shallow", "query-refresh-fit12-shallow", "generic-b-query-refresh-fit12-shallow"),
                                  ("query-refresh-fit12-deep", "query-refresh-fit12-deep", "generic-b-query-refresh-fit12-deep"),
                                  ("query-refresh-assessment4-shallow", "query-refresh-assessment4-shallow", "generic-b-query-refresh-assessment4-shallow"),
                                  ("query-refresh-assessment4-deep", "query-refresh-assessment4-deep", "generic-b-query-refresh-assessment4-deep"),
                                  ("query-refresh-native", "query-refresh-native", "generic-b-query-refresh-native"),
                                  ("query-interaction-native", "query-interaction-native", "generic-b-query-interaction-native"),
                                  ("dynamic-native", "dynamic-native", "generic-b-dynamic-native"),
                                  ("projection-native", "projection-native", "generic-b-projection-native"),
                                  ("expanded96-native", "expanded96-native", "generic-b-expanded96-native"),
                                  ("expanded96-teacher", "expanded96-teacher", "generic-b-expanded96-teacher"),
                                  ("expanded96-endpoints", "expanded96-endpoints", "generic-b-expanded96-endpoints"),
                                  ("expanded96-fit12-shallow", "expanded96-fit12-shallow", "generic-b-expanded96-fit12-shallow"),
                                  ("expanded96-fit12-deep", "expanded96-fit12-deep", "generic-b-expanded96-fit12-deep"),
                                  ("expanded96-assessment4-shallow", "expanded96-assessment4-shallow", "generic-b-expanded96-assessment4-shallow"),
                                  ("expanded96-assessment4-deep", "expanded96-assessment4-deep", "generic-b-expanded96-assessment4-deep"),
                                  ("replay-native", "replay-native", "generic-b-replay-native"),
                                  ("replay-teacher", "replay-teacher", "generic-b-replay-teacher"),
                                  ("replay-endpoints", "replay-endpoints", "generic-b-replay-endpoints"),
                                  ("replay-fit12-shallow", "replay-fit12-shallow", "generic-b-replay-fit12-shallow"),
                                  ("replay-fit12-deep", "replay-fit12-deep", "generic-b-replay-fit12-deep"),
                                  ("replay-assessment4-shallow", "replay-assessment4-shallow", "generic-b-replay-assessment4-shallow"),
                                  ("replay-assessment4-deep", "replay-assessment4-deep", "generic-b-replay-assessment4-deep"),
                                  ("balanced-refresh-native", "balanced-refresh-native", "generic-b-balanced-refresh-native"),
                                  ("balanced-refresh-teacher-comparison", "balanced-refresh-teacher-comparison", "generic-b-balanced-refresh-teacher-comparison"),
                                  ("balanced-refresh-endpoints-by-category", "balanced-refresh-endpoints-by-category", "generic-b-balanced-refresh-endpoints-by-category"),
                                  ("balanced-refresh-fit12-shallow", "balanced-refresh-fit12-shallow", "generic-b-balanced-refresh-fit12-shallow"),
                                  ("balanced-refresh-fit12-deep", "balanced-refresh-fit12-deep", "generic-b-balanced-refresh-fit12-deep"),
                                  ("balanced-refresh-assessment4-shallow", "balanced-refresh-assessment4-shallow", "generic-b-balanced-refresh-assessment4-shallow"),
                                  ("balanced-refresh-assessment4-deep", "balanced-refresh-assessment4-deep", "generic-b-balanced-refresh-assessment4-deep"),
                                  ("refresh-native", "refresh-native", "generic-b-refresh-native"),
                                  ("refresh-shallow", "refresh-shallow", "generic-b-refresh-shallow"),
                                  ("refresh-deep", "refresh-deep", "generic-b-refresh-deep"),
                                  ("joint-native", "joint-native", "generic-b-joint-native"),
                                  ("decoder_native", "decoder-native", "generic-b-decoder-native"),
                                  ("relation_support", "relation-support", "generic-b-relation-support"),
                                  ("support_capture", "support-capture", "generic-b-support-capture"),
                                  ("source_supervision", "source-supervision", "generic-b-source-supervision"),
                                  ("preservation", "preservation", "generic-b-preservation"),
                                  ("retained_source", "retained-source", "generic-b-retained-source"),
                                  ("positive_object", "positive-object", "generic-b-positive-object"),
                                  ("outcome", "h2-anchor", "generic-b-h2-anchor"),
                                  ("depth", "corrected-depth", "generic-b-corrected-depth")):
        path = source / (basename + "-source.json")
        binding_path = source / (basename + "-binding.json")
        _require(not path.is_symlink() and not binding_path.is_symlink()
                 and path.stat().st_size <= LIMIT and binding_path.stat().st_size <= LIMIT,
                 "Unsafe H2/depth aggregate source")
        raw = path.read_bytes()
        binding = json.loads(binding_path.read_bytes())
        _require(binding == {"schema": schema + "-binding-v1", "status_date": ("2026-11-16" if basename.startswith("state-mixture-") else "2026-11-15" if basename.startswith(("joint-", "refresh-", "balanced-refresh-", "replay-", "expanded96-", "projection-", "dynamic-", "query-interaction-", "query-refresh-")) else "2026-11-14"),
                 "source_file": path.name, "source_bytes": len(raw),
                 "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "private_payloads_copied": False, "contains_operational_identifiers": False,
                 "contains_cohort_member_identifiers": False}, "H2/depth binding differs")
        value = json.loads(raw)
        _require(value["schema"] == schema + "-public-source-v1"
                 and value["promotion"] is False and value["overall_PASS"] is False,
                 "H2/depth publication scope differs")
        published = value | {"schema": schema + "-public-download-v1",
                             "source_sha256": binding["source_sha256"]}
        # New downloads use compact JSON; historical download bytes stay exact.
        options = {"separators": (",", ":")} if basename.startswith(("state-mixture-", "decoder-", "joint-", "refresh-", "balanced-refresh-", "replay-", "expanded96-", "projection-", "dynamic-", "query-refresh-")) else {"indent": 2}
        encoded = (json.dumps(published, sort_keys=True, allow_nan=False, **options) + "\n").encode()
        filename = schema + "-v1.json"
        (output / filename).write_bytes(encoded)
        # Every self-contained Generic-B metric family remains complete in its exact JSON download.
        # Avoid duplicating every cell again inside the aggregate status manifest.
        # Historical download bytes remain identical; the live aggregate indexes them once.
        projected = {k: value[k] for k in ("schema", "status_date", "source_sha",
                     "checkpoint_sha256", "promotion", "overall_PASS") if k in value}
        result[key] = projected | {"metric_download": {"filename": filename, "bytes": len(encoded),
                                                       "sha256": hashlib.sha256(encoded).hexdigest()}}
    return result


def _render_h2_anchor(value):
    return ['GB101 earlier: H2 membership fit does not establish native benefit', '------------------------------------------------------------------', '', 'Earlier H2 membership fitting did not establish native benefit. All ten previously failed arm/event depth observations were recovered in the additive saved-depth correction. Native quality, uncertainty and unavailable support remain in the unchanged downloads. No promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Reviewed H2 outcome and lineage <generic-b-h2-anchor-v1.json>`', '', ':download:`Corrected H3 and H2 depth metrics and support <generic-b-corrected-depth-v1.json>`', '']



def _render_positive_object(value):
    return ['GB101 earlier: positive-object fit improves while native quality fails', '----------------------------------------------------------------------', '', 'Positive-object teacher fit improved while native quality failed its margins. Adaptive TRAIN fitting did not establish calibration or generalization. No promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Positive-object native outcomes, depth support and lineage <generic-b-positive-object-v1.json>`', '']


def _render_retained_source(value):
    return ['GB101 earlier: retained-source learning improves selected B, harms deeper structure', '-----------------------------------------------------------------------------------', '', 'Retained-source learning improved selected-B quality while harming deeper structure. The native category/depth tradeoff prevented promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Retained-source outcomes, uncertainty, depth and checkpoint lineage <generic-b-retained-source-v1.json>`', '']

def _render_preservation(value):
    return ['GB101 earlier: preserving generated states reduces drift but harms native quality', '---------------------------------------------------------------------------------', '', 'Preserving generated states reduced teacher drift but harmed native quality. Better teacher fit did not establish native transfer; no promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Preservation outcomes, depth, support, uncertainty and cost <generic-b-preservation-v1.json>`', '']


def _render_source_supervision(value):
    return ['GB101 latest learned comparison: generated-source supervision improves fit without established native benefit', '-------------------------------------------------------------------------------------------------------------', '', 'Generated-source supervision improved teacher fit without establishing native benefit. Supported labels remained separate from truth-free inference. No promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Generated-source supervision quality, depth, support and cost <generic-b-source-supervision-v1.json>`', '']


def _render_support_capture(value):
    return ['GB101 historical capture support: mixed and charged generated-state capture is available with sparse exact support', '------------------------------------------------------------------------------------------------------------------', '', 'Matched mixed and charged generated-state capture became available with sparse complete support. Capture activity is support evidence,not reconstruction efficiency or physical completeness.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Generated-state capture support, native quality and cost <generic-b-support-capture-v1.json>`', '']


def _render_relation_support(value):
    return ['GB101 latest support correction: explicit relations resolve 292 unavailable support occurrences', '-----------------------------------------------------------------------------------------------', '', 'Explicit detector/FSP incidence resolved 292 previously unavailable support occurrences. Partial and unknown membership stayed distinct;complete deep labels were not invented.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`Corrected support, category/depth counts and compute <generic-b-relation-support-v1.json>`', '']


def _render_decoder_native(value):
    return ['GB101 preceding learned result: membership fit improves while native assessment worsens', '---------------------------------------------------------------------------------------', '', 'Decoder adaptation improved teacher fit while native assessment worsened. Adaptive TRAIN only;no promotion.', '', 'Historical quality,coverage,uncertainty,source/checkpoint and cost remain available in the original downloads.', '', ':download:`fit12-shallow depth metrics <generic-b-decoder-fit12-shallow-v1.json>`', '', ':download:`fit12-deep depth metrics <generic-b-decoder-fit12-deep-v1.json>`', '', ':download:`assessment4-shallow depth metrics <generic-b-decoder-assessment4-shallow-v1.json>`', '', ':download:`assessment4-deep depth metrics <generic-b-decoder-assessment4-deep-v1.json>`', '', ':download:`Matched decoder fit, native quality, depth, endpoints and compute <generic-b-decoder-native-v1.json>`', '']


def _render_joint_native(value):
    return ['GB101 prior learned result: joint supervision gives a mixed native response', '---------------------------------------------------------------------------', '', 'Joint complete-only membership, object and PID supervision did not establish all native margins. Adaptive TRAIN results remain exploratory; no promotion.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`joint-fit12-shallow aggregate metrics <generic-b-joint-fit12-shallow-v1.json>`', '', ':download:`joint-fit12-deep aggregate metrics <generic-b-joint-fit12-deep-v1.json>`', '', ':download:`joint-assessment4-shallow aggregate metrics <generic-b-joint-assessment4-shallow-v1.json>`', '', ':download:`joint-assessment4-deep aggregate metrics <generic-b-joint-assessment4-deep-v1.json>`', '', ':download:`joint-native aggregate metrics <generic-b-joint-native-v1.json>`', '']


def _render_refresh_support(value):
    return ['GB101 current checkpoint exposes limited H3 support', '---------------------------------------------------', '', 'Refreshed generated-state support improved teacher fit without establishing native nonharm. Adaptive TRAIN only; no promotion.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`refresh-native aggregate metrics <generic-b-refresh-native-v1.json>`', '', ':download:`refresh-shallow aggregate metrics <generic-b-refresh-shallow-v1.json>`', '', ':download:`refresh-deep aggregate metrics <generic-b-refresh-deep-v1.json>`', '']


def _render_balanced_refresh(value):
    return ['GB101 support refresh improves selected B but harms mixed H3', '------------------------------------------------------------', '', 'Refreshed balanced support retained native category tradeoffs. Adaptive TRAIN results and their uncertainty remain in the unchanged downloads.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`balanced-refresh-native aggregate metrics <generic-b-balanced-refresh-native-v1.json>`', '', ':download:`balanced-refresh-teacher-comparison aggregate metrics <generic-b-balanced-refresh-teacher-comparison-v1.json>`', '', ':download:`balanced-refresh-endpoints-by-category aggregate metrics <generic-b-balanced-refresh-endpoints-by-category-v1.json>`', '', ':download:`balanced-refresh-fit12-shallow aggregate metrics <generic-b-balanced-refresh-fit12-shallow-v1.json>`', '', ':download:`balanced-refresh-fit12-deep aggregate metrics <generic-b-balanced-refresh-fit12-deep-v1.json>`', '', ':download:`balanced-refresh-assessment4-shallow aggregate metrics <generic-b-balanced-refresh-assessment4-shallow-v1.json>`', '', ':download:`balanced-refresh-assessment4-deep aggregate metrics <generic-b-balanced-refresh-assessment4-deep-v1.json>`', '']


def _render_replay(value):
    return ['GB101 balanced replay repairs mixed H3 but harms charged reconstruction', '-----------------------------------------------------------------------', '', 'Balanced support replay retained category-dependent native harm despite better teacher fit. This adaptive TRAIN comparison did not support promotion.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`replay-native aggregate metrics <generic-b-replay-native-v1.json>`', '', ':download:`replay-teacher aggregate metrics <generic-b-replay-teacher-v1.json>`', '', ':download:`replay-endpoints aggregate metrics <generic-b-replay-endpoints-v1.json>`', '', ':download:`replay-fit12-shallow aggregate metrics <generic-b-replay-fit12-shallow-v1.json>`', '', ':download:`replay-fit12-deep aggregate metrics <generic-b-replay-fit12-deep-v1.json>`', '', ':download:`replay-assessment4-shallow aggregate metrics <generic-b-replay-assessment4-shallow-v1.json>`', '', ':download:`replay-assessment4-deep aggregate metrics <generic-b-replay-assessment4-deep-v1.json>`', '']


def _render_expanded96(value):
    return ['GB101 broader TRAIN support retains charged and mixed tradeoff', '--------------------------------------------------------------', '', 'Broader TRAIN96 support at matched 40 updates lowered selected-B F1 from 0.13052 to 0.08236; paired 95% delta [-0.09633, 0]. Mixed H3 improved while charged H3 worsened. No promotion.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`expanded96-native aggregate metrics <generic-b-expanded96-native-v1.json>`', '', ':download:`expanded96-teacher aggregate metrics <generic-b-expanded96-teacher-v1.json>`', '', ':download:`expanded96-endpoints aggregate metrics <generic-b-expanded96-endpoints-v1.json>`', '', ':download:`expanded96-fit12-shallow aggregate metrics <generic-b-expanded96-fit12-shallow-v1.json>`', '', ':download:`expanded96-fit12-deep aggregate metrics <generic-b-expanded96-fit12-deep-v1.json>`', '', ':download:`expanded96-assessment4-shallow aggregate metrics <generic-b-expanded96-assessment4-shallow-v1.json>`', '', ':download:`expanded96-assessment4-deep aggregate metrics <generic-b-expanded96-assessment4-deep-v1.json>`', '']


def _render_projection(value):
    return ['GB101 category-gradient projection retains native transfer harm', '---------------------------------------------------------------', '', 'Category projection activated 18/40 updates but selected-B F1 fell 0.08236 to 0.07045; paired 95% delta [-0.02381, 0]. H3 was unchanged versus expanded96. No promotion.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`category projection aggregate metrics <generic-b-projection-native-v1.json>`', '']


def _render_dynamic(value):
    return ['GB101 dynamic assignment improves selected-B quality with residual charged H3 harm', '----------------------------------------------------------------------------------', '', 'Dynamic assignment raised adaptive selected-B F1 from0.08236 to0.14091 but retained charged H3 harm. Frozen/rematched teacher views and historical metrics remain in the unchanged download.', '', 'Historical quality, coverage, uncertainty, source/checkpoint and cost are retained in the downloads.', '', ':download:`dynamic assignment aggregate metrics <generic-b-dynamic-native-v1.json>`', '']


def _render_query_interaction(value):
    return ['GB101 query interaction improves selected-B quality with residual charged H3 harm', '---------------------------------------------------------------------------------', '', 'Reviewed 15 November 2026. Training the 12 existing query self-attention tensors alongside', 'the prior 72 trainables increased adaptive assessment selected-B F1 from 0.140909 to', '0.220753: delta +0.079844, paired 95% interval [0.000597, 0.159091] versus dynamic96.', 'All 12 added tensors received gradients and changed. Charged H3 source-F1 and soft-edge', 'nonharm still failed versus refreshed32. Fit12 selected-B F1 decreased by 0.01141.', 'The gain is not uniform: the scientific gate FAILED and there is no model promotion.', '', 'Earlier complete quality,depth,uncertainty,coverage,lineage and cost remain in the unchanged downloads.', '', ':download:`query interaction aggregate metrics <generic-b-query-interaction-native-v1.json>`', '']


def _render_query_refresh(value):
    return ['GB101 query-state refresh fit gain does not transfer to adaptive assessment', '---------------------------------------------------------------------------', '', 'Reviewed 15 November 2026. Refreshing generated TRAIN96 states changed assessment selected-B', 'F1 from 0.220753 to 0.220142: delta -0.000611, paired 95% interval [-0.041667, 0.040444].', 'Fit12 selected-B F1 increased from 0.153935 to 0.244525: delta +0.090590, paired 95%', 'interval [0.021052, 0.155464]. Fit and assessment are distinct. Charged assessment F1', 'fell by 0.035256 while mixed rose by 0.034034. Selected-B gain and several category/H3', 'nonharm checks FAILED. No promotion.', '', 'Earlier complete quality,depth,uncertainty,coverage,lineage and cost remain in the unchanged downloads.', '', ':download:`query-state refresh aggregate metrics <generic-b-query-refresh-native-v1.json>`', '', ':download:`query refresh fit12 shallow metrics <generic-b-query-refresh-fit12-shallow-v1.json>`', '', ':download:`query refresh fit12 deep metrics <generic-b-query-refresh-fit12-deep-v1.json>`', '', ':download:`query refresh assessment4 shallow metrics <generic-b-query-refresh-assessment4-shallow-v1.json>`', '', ':download:`query refresh assessment4 deep metrics <generic-b-query-refresh-assessment4-deep-v1.json>`', '']


def _render_state_mixture(value):
    return ['GB101 alternating replay harms native reconstruction despite better teacher fit', '-------------------------------------------------------------------------------', '', 'Reviewed 16 November 2026. Alternating saved old/refreshed TRAIN96 states reduced adaptive', 'assessment selected-B F1 from 0.220142 to 0.086016: delta -0.134126, paired 95% interval', '[-0.208333, -0.059918]. IoU fell from 0.137740 to 0.050424. Charged F1 delta was', '-0.111111 and mixed -0.157140. Fit12 F1 fell from 0.244525 to 0.182960:', 'delta -0.061565, paired 95% interval [-0.159661, 0.037325]. Gain and multiple', 'category/depth nonharm checks FAILED. No promotion.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`alternating state mixture aggregate metrics (complete JSON, versioned public repository) <https://github.com/BoyangYu0/HyperTagging/blob/5806f57ead0c2785f39f588c7a110c0e97540680/docs/metrics/generic-b-state-mixture-native-v1.json>`_', '', '`state mixture fit12 shallow metrics (complete JSON, versioned public repository) <https://github.com/BoyangYu0/HyperTagging/blob/5806f57ead0c2785f39f588c7a110c0e97540680/docs/metrics/generic-b-state-mixture-fit12-shallow-v1.json>`_', '', '`state mixture fit12 deep metrics (complete JSON, versioned public repository) <https://github.com/BoyangYu0/HyperTagging/blob/5806f57ead0c2785f39f588c7a110c0e97540680/docs/metrics/generic-b-state-mixture-fit12-deep-v1.json>`_', '', '`state mixture assessment4 shallow metrics (complete JSON, versioned public repository) <https://github.com/BoyangYu0/HyperTagging/blob/5806f57ead0c2785f39f588c7a110c0e97540680/docs/metrics/generic-b-state-mixture-assessment4-shallow-v1.json>`_', '', '`state mixture assessment4 deep metrics (complete JSON, versioned public repository) <https://github.com/BoyangYu0/HyperTagging/blob/5806f57ead0c2785f39f588c7a110c0e97540680/docs/metrics/generic-b-state-mixture-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_within_update(value):
    return ['GB101 within-update mixing leaves substantial native reconstruction harm', '------------------------------------------------------------------------', '', 'Reviewed 16 November 2026. Within-update mixing raised adaptive assessment selected-B F1', 'from 0.08602 to 0.10778 versus alternating replay: delta +0.02177, paired 95% interval', '[-0.00192, 0.04545]. F1 remained 0.11236 below refreshed-only training (0.22014).', 'Thirteen registered nonharm checks FAILED. No promotion. This modest rebound does not', 'resolve the accumulated replay harm; the cause remains unproven.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`within update native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/657ff5d4adb3eb54f2b8e0d6d410abf411835d91/docs/metrics/generic-b-within-update-native-v1.json>`_', '', '`within update fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/657ff5d4adb3eb54f2b8e0d6d410abf411835d91/docs/metrics/generic-b-within-update-fit12-shallow-v1.json>`_', '', '`within update fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/657ff5d4adb3eb54f2b8e0d6d410abf411835d91/docs/metrics/generic-b-within-update-fit12-deep-v1.json>`_', '', '`within update assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/657ff5d4adb3eb54f2b8e0d6d410abf411835d91/docs/metrics/generic-b-within-update-assessment4-shallow-v1.json>`_', '', '`within update assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/657ff5d4adb3eb54f2b8e0d6d410abf411835d91/docs/metrics/generic-b-within-update-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_composite(value):
    return ['GB101 composite adaptation does not resolve accumulated replay harm', '-------------------------------------------------------------------', '', 'Reviewed 16 November 2026. Training the composite encoder raised adaptive assessment', 'selected-B F1 from 0.10778 to 0.12902 versus decoder-only within-update mixing: delta', '+0.02123, paired 95% interval [-0.14511, 0.18758]. Fourteen registered safeguards FAILED.', 'F1 remained 0.09113 below refreshed-only training (0.22014). No promotion. Neither', 'within-update mixing nor demonstrably utilized composite adaptation resolved the', 'accumulated whole-bank replay harm. The cause remains unproven.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`composite native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/f35547081ccbab26ab40ebeefcc47b005168bb3c/docs/metrics/generic-b-composite-native-v1.json>`_', '', '`composite fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/f35547081ccbab26ab40ebeefcc47b005168bb3c/docs/metrics/generic-b-composite-fit12-shallow-v1.json>`_', '', '`composite fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/f35547081ccbab26ab40ebeefcc47b005168bb3c/docs/metrics/generic-b-composite-fit12-deep-v1.json>`_', '', '`composite assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/f35547081ccbab26ab40ebeefcc47b005168bb3c/docs/metrics/generic-b-composite-assessment4-shallow-v1.json>`_', '', '`composite assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/f35547081ccbab26ab40ebeefcc47b005168bb3c/docs/metrics/generic-b-composite-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_targeted_h3(value):
    return ['GB101 targeted H3 replay also fails native safeguards', '-----------------------------------------------------', '', 'Reviewed 16 November 2026. Targeted complete-H3 replay achieved adaptive assessment', 'selected-B F1 0.15729 versus 0.22014 for refreshed-only training: delta -0.06285,', 'paired 95% interval [-0.12570, 0]. Eleven registered safeguards FAILED. No promotion.', 'Whole-bank replay harm was not resolved by within-update mixing, utilized composite', 'adaptation or targeted H3 replay. These accumulated nulls motivate a gradient-path', 'intervention; they do not establish a causal mechanism.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`targeted h3 native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/6cf2b287ec1b109b8fb386fc373fbe8a5784be6b/docs/metrics/generic-b-targeted-h3-native-v1.json>`_', '', '`targeted h3 fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/6cf2b287ec1b109b8fb386fc373fbe8a5784be6b/docs/metrics/generic-b-targeted-h3-fit12-shallow-v1.json>`_', '', '`targeted h3 fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/6cf2b287ec1b109b8fb386fc373fbe8a5784be6b/docs/metrics/generic-b-targeted-h3-fit12-deep-v1.json>`_', '', '`targeted h3 assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/6cf2b287ec1b109b8fb386fc373fbe8a5784be6b/docs/metrics/generic-b-targeted-h3-assessment4-shallow-v1.json>`_', '', '`targeted h3 assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/6cf2b287ec1b109b8fb386fc373fbe8a5784be6b/docs/metrics/generic-b-targeted-h3-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_head_gradient(value):
    return ['GB101 restricted head gradients recover native quality with remaining depth failures', '------------------------------------------------------------------------------------', '', 'Reviewed 16 November 2026. Restricting supplemental H3 object/type gradients to four', 'level3 head tensors recovered adaptive assessment selected-B F1 to 0.22180 versus', '0.15729 for unrestricted targeted replay: delta +0.06451, paired 95% interval [0, 0.12902].', 'Versus refreshed-only training, delta +0.00166, paired 95% interval [0, 0.00332].', 'Five depth-specific safeguards FAILED. No promotion or established causal mechanism.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`head gradient native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/153be55554160b64de10e87b4e36d3c3ed88c4fd/docs/metrics/generic-b-head-gradient-native-v1.json>`_', '', '`head gradient fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/153be55554160b64de10e87b4e36d3c3ed88c4fd/docs/metrics/generic-b-head-gradient-fit12-shallow-v1.json>`_', '', '`head gradient fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/153be55554160b64de10e87b4e36d3c3ed88c4fd/docs/metrics/generic-b-head-gradient-fit12-deep-v1.json>`_', '', '`head gradient assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/153be55554160b64de10e87b4e36d3c3ed88c4fd/docs/metrics/generic-b-head-gradient-assessment4-shallow-v1.json>`_', '', '`head gradient assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/153be55554160b64de10e87b4e36d3c3ed88c4fd/docs/metrics/generic-b-head-gradient-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_h3_residual(value):
    return ['GB101 residual teacher fit improves without native reconstruction gain', '----------------------------------------------------------------------', '', 'Reviewed 16 November 2026. The matched nonlinear H3 object/type residual leaves', 'adaptive TRAIN assessment selected-B F1 unchanged at 0.221803 versus the restricted', 'head-gradient control: paired delta 0, 95% interval [0, 0]. Six scientific checks FAILED.', 'No promotion. This teacher/native divergence changes the next operation to capture', 'generated states from the completed model before another architecture intervention.', '', 'Archived adaptive TRAIN diagnostic; no promotion or independent validation.', 'Complete scientific metrics,uncertainty,versions and cost remain in these unchanged aggregates:', '', '`h3 residual native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/8ec6579abc603b88084d6a20e927e157743e13c7/docs/metrics/generic-b-h3-residual-native-v1.json>`_', '', '`h3 residual fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/8ec6579abc603b88084d6a20e927e157743e13c7/docs/metrics/generic-b-h3-residual-fit12-shallow-v1.json>`_', '', '`h3 residual fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/8ec6579abc603b88084d6a20e927e157743e13c7/docs/metrics/generic-b-h3-residual-fit12-deep-v1.json>`_', '', '`h3 residual assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/8ec6579abc603b88084d6a20e927e157743e13c7/docs/metrics/generic-b-h3-residual-assessment4-shallow-v1.json>`_', '', '`h3 residual assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/8ec6579abc603b88084d6a20e927e157743e13c7/docs/metrics/generic-b-h3-residual-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_residual_support_refresh(value):
    return ['GB101 refreshed generated support reduces native reconstruction quality', '-----------------------------------------------------------------------', '', 'Reviewed 16 November 2026. A matched refresh of the primary generated-state bank', 'reduced adaptive TRAIN assessment selected-B F1 from 0.221803 to 0.174616.', 'Paired delta -0.047187, 95% interval [-0.094374, 0]. Seventeen scientific checks', 'FAILED. No promotion. This negative comparison changes the next intervention to', 'authenticated partial-H3 parent-type supervision, with unknown daughter/object', 'labels kept masked. The training candidate requires independent admission.', '', 'Archived adaptive TRAIN outcome;no promotion or independent validation.', 'Complete metrics,versions,uncertainty and cost:', '', '`residual support refresh native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/bde7aad29ae9b93dd85317bda3790098b0c71766/docs/metrics/generic-b-residual-support-refresh-native-v1.json>`_', '', '`residual support refresh fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/bde7aad29ae9b93dd85317bda3790098b0c71766/docs/metrics/generic-b-residual-support-refresh-fit12-shallow-v1.json>`_', '', '`residual support refresh fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/bde7aad29ae9b93dd85317bda3790098b0c71766/docs/metrics/generic-b-residual-support-refresh-fit12-deep-v1.json>`_', '', '`residual support refresh assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/bde7aad29ae9b93dd85317bda3790098b0c71766/docs/metrics/generic-b-residual-support-refresh-assessment4-shallow-v1.json>`_', '', '`residual support refresh assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/bde7aad29ae9b93dd85317bda3790098b0c71766/docs/metrics/generic-b-residual-support-refresh-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_partial_type_tradeoff(value):
    return ['Earlier GB101 partial parent-type tradeoff', '------------------------------------------', '', 'Added partial-H3 parent-type learning raised adaptive TRAIN selected-B F1 from', '0.174616 to 0.254731; delta+0.080115,95% interval[0.035256,0.124974].', 'Charged improved+0.194444; mixed declined-0.034214,including mixed H3 edge harm.', 'Twenty-two safeguards failed. No promotion. Fit12 delta+0.036829,interval', '[-0.021834,0.100838]. Forty updates,16 native rollouts,32 nominal B trials;', 'exact/topology/inclusive0/32 including oracle. Independent0/12000;backgrounds', 'and physical completeness unavailable. Teacher fit is separate from native quality.', 'Complete source/checkpoint,depth,teacher,uncertainty and compute remain in the', 'following unchanged immutable historical aggregate downloads:', '', '`partial type tradeoff native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/cfa5b5213f49329514a25ceeec949748ead8aa77/docs/metrics/generic-b-partial-type-tradeoff-native-v1.json>`_', '', '`partial type tradeoff fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/cfa5b5213f49329514a25ceeec949748ead8aa77/docs/metrics/generic-b-partial-type-tradeoff-fit12-shallow-v1.json>`_', '', '`partial type tradeoff fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/cfa5b5213f49329514a25ceeec949748ead8aa77/docs/metrics/generic-b-partial-type-tradeoff-fit12-deep-v1.json>`_', '', '`partial type tradeoff assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/cfa5b5213f49329514a25ceeec949748ead8aa77/docs/metrics/generic-b-partial-type-tradeoff-assessment4-shallow-v1.json>`_', '', '`partial type tradeoff assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/cfa5b5213f49329514a25ceeec949748ead8aa77/docs/metrics/generic-b-partial-type-tradeoff-assessment4-deep-v1.json>`_', '']


def _render_type_isolation_tradeoff(value):
    return ['Earlier GB101 type-head isolation tradeoff', '------------------------------------------', '', 'Reviewed 17 November 2026. Adaptive TRAIN assessment selected-B F1 fell from', '0.254731 to 0.182950: delta -0.071782, 95% interval [-0.130529,-0.013034].', 'Mixed improved by 0.034214; charged fell by 0.177778. Ten safeguards failed.', 'Forty updates, 16 native rollouts, 7352 allocated CPU-seconds and 919 V100-seconds.', 'Exact/topology/inclusive recovery was 0/32, including oracle pools. No promotion;', 'independent coverage 0/12000 and backgrounds/physical completeness unavailable.', 'The complete quality, teacher, depth, uncertainty, source/checkpoint and cost', 'records remain in these unchanged immutable aggregate downloads:', '', '`type isolation tradeoff native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/141fd7c6f489d70ab0f03cab09de1d33b8f138f7/docs/metrics/generic-b-type-isolation-tradeoff-native-v1.json>`_', '', '`type isolation tradeoff fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/141fd7c6f489d70ab0f03cab09de1d33b8f138f7/docs/metrics/generic-b-type-isolation-tradeoff-fit12-shallow-v1.json>`_', '', '`type isolation tradeoff fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/141fd7c6f489d70ab0f03cab09de1d33b8f138f7/docs/metrics/generic-b-type-isolation-tradeoff-fit12-deep-v1.json>`_', '', '`type isolation tradeoff assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/141fd7c6f489d70ab0f03cab09de1d33b8f138f7/docs/metrics/generic-b-type-isolation-tradeoff-assessment4-shallow-v1.json>`_', '', '`type isolation tradeoff assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/141fd7c6f489d70ab0f03cab09de1d33b8f138f7/docs/metrics/generic-b-type-isolation-tradeoff-assessment4-deep-v1.json>`_', '']


def _render_authenticated_teacher_tradeoff(value):
    return ['Earlier GB101 authenticated teacher tradeoff', '--------------------------------------------', '', '105 teacher contexts improved teacher fit while adaptive selected-B F1 fell from', '0.174616 to 0.094952; delta -0.079664, 95% interval [-0.208333,0.049005].', 'Both categories declined; five safeguards failed. No promotion. Forty updates,', '16 native rollouts, 7176 allocated CPU-seconds and 897 V100-seconds.', 'Exact/topology/inclusive 0/32 including oracle; independent0/12000. Backgrounds', 'and physical completeness unavailable. All complete historical aggregates retained:', '', '`authenticated teacher tradeoff native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/136b0e641efb785db3a049f2d823c465b4c7bec1/docs/metrics/generic-b-authenticated-teacher-tradeoff-native-v1.json>`_', '', '`authenticated teacher tradeoff fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/136b0e641efb785db3a049f2d823c465b4c7bec1/docs/metrics/generic-b-authenticated-teacher-tradeoff-fit12-shallow-v1.json>`_', '', '`authenticated teacher tradeoff fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/136b0e641efb785db3a049f2d823c465b4c7bec1/docs/metrics/generic-b-authenticated-teacher-tradeoff-fit12-deep-v1.json>`_', '', '`authenticated teacher tradeoff assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/136b0e641efb785db3a049f2d823c465b4c7bec1/docs/metrics/generic-b-authenticated-teacher-tradeoff-assessment4-shallow-v1.json>`_', '', '`authenticated teacher tradeoff assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/136b0e641efb785db3a049f2d823c465b4c7bec1/docs/metrics/generic-b-authenticated-teacher-tradeoff-assessment4-deep-v1.json>`_', '']


def _render_teacher_context_ablation(value):
    return ['GB101 teacher type-loss ablation restores the control without advancing reconstruction', '--------------------------------------------------------------------------------------', '', 'Reviewed 17 November 2026. Removing only the added teacher type loss raised adaptive', 'TRAIN assessment selected-B source F1 from 0.094952 to 0.174616: delta +0.079664,', '95% paired interval [-0.049005,0.208333] includes zero. It exactly matched the earlier', 'residual reference; five of 22 scientific checks failed. No promotion.', '', 'Archived adaptive TRAIN outcome;no promotion or independent validation.', 'Complete metrics,versions,uncertainty and cost:', '', '`teacher context ablation native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/555fceab6e07a8b66f514ea38f64c09168586cc7/docs/metrics/generic-b-teacher-context-ablation-native-v1.json>`_', '', '`teacher context ablation fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/555fceab6e07a8b66f514ea38f64c09168586cc7/docs/metrics/generic-b-teacher-context-ablation-fit12-shallow-v1.json>`_', '', '`teacher context ablation fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/555fceab6e07a8b66f514ea38f64c09168586cc7/docs/metrics/generic-b-teacher-context-ablation-fit12-deep-v1.json>`_', '', '`teacher context ablation assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/555fceab6e07a8b66f514ea38f64c09168586cc7/docs/metrics/generic-b-teacher-context-ablation-assessment4-shallow-v1.json>`_', '', '`teacher context ablation assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/555fceab6e07a8b66f514ea38f64c09168586cc7/docs/metrics/generic-b-teacher-context-ablation-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_hybrid_teacher_context(value):
    return ['GB101 hybrid contexts improve teacher fit but harm native reconstruction', '------------------------------------------------------------------------', '', 'Adaptive TRAIN selected-B F1 fell0.174616 to0.115278;delta -0.059338,', 'paired95[-0.111111,-0.007566]. Mixed reached zero;15/22 checks failed.', 'Teacher membership/object/type diagnostic fit improved. No promotion;independent0/12000.', '101 hybrid contexts+4 fallbacks,40 updates,16 rollouts;7016 allocated CPU-seconds,', '877 V100-seconds. Complete metrics,versions,uncertainty and limitations:', '', '`hybrid teacher context native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/afb7d173352788556e090d8c321e1ab54946b6ea/docs/metrics/generic-b-hybrid-teacher-context-native-v1.json>`_', '', '`hybrid teacher context fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/afb7d173352788556e090d8c321e1ab54946b6ea/docs/metrics/generic-b-hybrid-teacher-context-fit12-shallow-v1.json>`_', '', '`hybrid teacher context fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/afb7d173352788556e090d8c321e1ab54946b6ea/docs/metrics/generic-b-hybrid-teacher-context-fit12-deep-v1.json>`_', '', '`hybrid teacher context assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/afb7d173352788556e090d8c321e1ab54946b6ea/docs/metrics/generic-b-hybrid-teacher-context-assessment4-shallow-v1.json>`_', '', '`hybrid teacher context assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/afb7d173352788556e090d8c321e1ab54946b6ea/docs/metrics/generic-b-hybrid-teacher-context-assessment4-deep-v1.json>`_', '', 'Historical downloads remain unchanged.', '']


def _render_hybrid_teacher_gradient(value):
    return ['GB101 history: H3 gradient protection fails the native screen', '-------------------------------------------------------------', '', 'Reviewed 17 November 2026. Adaptive TRAIN assessment selected-B source F1 reached', '0.179167 versus0.115278 for the hybrid control:delta +0.063889,paired95[0,0.127778].', 'Against both earlier controls the gain was only0.004550,paired95[-0.015899,0.025000].', 'Eight of33 registered checks failed,including selected-B improvement and mixed-category', 'selected-B/H2 nonharm. No promotion. These four reused TRAIN assessment collisions', 'do not establish generalization;fit12 is reported separately.', '', 'The sole change projects conflicting teacher gradients against ordinary H3 gradients', 'before the shared clip1 and AdamW step. Projection applied on26/40 updates. Raw-gradient', 'protection does not guarantee optimizer or native nonharm;type-gradient components may', 'be introduced despite zero added teacher type loss. All105 contexts,88 trainable tensors,', 'initialization,ordinary losses,40 updates and16 native rollouts remained matched.', '', 'The application failed diagnostic closeout after all fitting and rollouts:an omitted', 'measurement hook left210 teacher observations unmeasured. This failure and the failed', 'native screen remain preserved. Separate checkpoint-only recovery is being prepared;', 'no retraining,native replay or diagnostic values are claimed.', '', 'Exact recursive topology+PID,topology-only and inclusive recovery remain0/32,including', 'actual oracle@1/@4. Oracle pools are not deployable. GB101 H3 packing and the inherited', 'bounded donor search remain:width4,generation1 eight candidates/query,later levels four,', 'six generations. Truth is supervision or post-inference scoring only.', '', 'Independent coverage remains0/12000:0/2000 each charged,mixed,ccbar,uubar,ddbar,ssbar.', 'Each continuum flavour acceptance,native proposal/retention quality and physical', 'completeness remain unavailable. Unknown labels remain masked;6/111 nominal parents', 'are unavailable. Shared seed5001,2000 whole-collision draws and equal-category weighting', 'define paired intervals. Complete links retain category/depth F1,IoU,source precision/recall,', 'membership edges,exact endpoints,latency,memory,uncertainty and versions.', '', 'Actual science:6976 allocated CPU-seconds,872 V100-seconds (0.242222 GPU-hours),', '21248 forwards and104120 beam expansions. Publication cost is separate.', 'Next:recover210 missing teacher diagnostics at saved update0/final40,then review mixed', 'preservation before another matched neural intervention. No automatic variant sweep.', 'Scientific source: ``6dd2136cd9b595e1a09b8c21409f8168e1da9a2c``. Checkpoint SHA-256:', '``e65a857f186915b7f07bb95de9bbc402818d35bbccdaffd0ad61c02b27dfc001``.', '', '`hybrid teacher gradient native (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/333230bd181c21e33b055512b800103bea63ec3e/docs/metrics/generic-b-hybrid-teacher-gradient-native-v1.json>`_', '', '`hybrid teacher gradient fit12 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/333230bd181c21e33b055512b800103bea63ec3e/docs/metrics/generic-b-hybrid-teacher-gradient-fit12-shallow-v1.json>`_', '', '`hybrid teacher gradient fit12 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/333230bd181c21e33b055512b800103bea63ec3e/docs/metrics/generic-b-hybrid-teacher-gradient-fit12-deep-v1.json>`_', '', '`hybrid teacher gradient assessment4 shallow (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/333230bd181c21e33b055512b800103bea63ec3e/docs/metrics/generic-b-hybrid-teacher-gradient-assessment4-shallow-v1.json>`_', '', '`hybrid teacher gradient assessment4 deep (complete aggregate JSON) <https://github.com/BoyangYu0/HyperTagging/blob/333230bd181c21e33b055512b800103bea63ec3e/docs/metrics/generic-b-hybrid-teacher-gradient-assessment4-deep-v1.json>`_', '', 'Earlier reviewed outcomes and all historical downloads remain available.', '']



def _accumulated_findings(source, output):
    """Publish selected reviewed aggregates with explicit adaptive TRAIN scope."""
    path = source / "accumulated-findings-source-v1.json"
    binding_path = source / "accumulated-findings-binding-v1.json"
    _require(not path.is_symlink() and not binding_path.is_symlink()
             and path.stat().st_size <= LIMIT and binding_path.stat().st_size <= LIMIT,
             "Unsafe accumulated-findings aggregate")
    raw = path.read_bytes()
    binding = json.loads(binding_path.read_bytes())
    _require(binding == {
        "schema": "generic-b-accumulated-findings-binding-v1",
        "status_date": "2026-11-20",
        "source_file": path.name, "source_bytes": 35320,
        "source_sha256": "5d7f080fd717cdea508db308a7d86e19cd1bc2fd2fcb78c7f0c9c980c5a3c714",
        "reviewed_handoff_sha256": "5c5e3709aa59b269e4ff46bac66d1352d3cde40a70021ecebb734de9d45933cd",
        "publication_review_sha256": "e6c23a599fcb753db9ac36805c1b5cadc3283cd4445636c8d5cf097c2a31067b",
        "private_payloads_copied": False,
        "contains_operational_identifiers": False,
        "contains_cohort_member_identifiers": False,
    } and len(raw) == binding["source_bytes"]
       and hashlib.sha256(raw).hexdigest() == binding["source_sha256"],
       "Accumulated-findings binding differs")
    value = json.loads(raw)
    _require(value["study_id"] == "gb101_height3_conflict_set_packing_v1"
             and value["independent_coverage"]["processed_total"] == 0
             and value["independent_coverage"]["requested_total"] == 12000
             and len(value["experiments"]) == 3
             and all(e["scientific_gate"] == "failed" and not e["promotion"]
                     for e in value["experiments"])
             and value["support_refresh"]["novel_contexts"] == 0
             and value["support_refresh"]["novel_canonical_membership_masks"] == 0,
             "Accumulated-findings scope differs")
    filename = "generic-b-accumulated-findings-v1.json"
    public = value | {"schema": "generic-b-accumulated-findings-metrics-v1",
                      "source_summary_sha256": binding["source_sha256"]}
    payload = (json.dumps(public, separators=(",", ":"), sort_keys=True,
                          allow_nan=False) + "\n").encode()
    (output / filename).write_bytes(payload)
    return value | {"metric_download": {"filename": filename, "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest()}}


def _render_accumulated_findings(value):
    reviewed = value["accumulated_findings"]
    generated, weighting, assignment = reviewed["experiments"]
    title = "GB101 accumulated update: local H3 gains fail native safeguards"
    lines = [title, "-" * len(title), "",
             "Reviewed 20 November 2026. Three adaptive TRAIN comparisons and a support",
             "refresh change the recommendation: do not repeat unchanged weighting, capture",
             "or assignment continuation. Local H3 edge gains coexist with mixed/depth harms",
             "and failed selected-B requirements. Every overall scientific gate failed.",
             "No checkpoint or truth-scored oracle is promoted.", "",
             "These are adaptive TRAIN diagnostics, never independent validation. Fit12 contains",
             "8 charged and 4 mixed collisions; assessment4 contains 2 charged and 2 mixed",
             "collisions excluded from the current fit but exposed adaptively. Each new arm",
             "completed 20 updates and 16 native rollouts, with 32 nominal B trials. Only four",
             "H3 branch tensors were trainable. Each arm presented 960 states, 1760 targets,",
             "8020 positive axes and 27480 negative axes. Unknown labels remain masked; partial",
             "supervision does not establish complete daughters, topology or PID.", "",
             "The later adaptive policy uses coherent full-depth width4 beam, H1 eight candidates",
             "per query and four at other levels. Keep it separate from the original matched",
             "GB101 greedy/default-four beam comparison below. Model-ranked top-1 and",
             "post-inference oracle@1/@4 pools remain distinct. Exact topology+PID, topology-only",
             "and inclusive source recovery are each 0/32 per arm, including those oracle views.", "",
             "Generated versus teacher contexts", "~" * len("Generated versus teacher contexts"), "",
             "The generated-minus-teacher H3 soft-edge F1 gain is +0.02524692, paired 95%",
             "interval [0, 0.05049384]. The +0.02 point margin passes; the interval touches zero.",
             "Selected-B F1 gains +0.04883417, interval [-0.02733166, 0.12500000], while mixed",
             "selected-B declines -0.02733166 and breaches the -0.02 safeguard. Against the",
             "earlier KL20 control, selected-B gains only +0.01612737; mixed declines -0.01640494",
             "and mixed H2/H3 source harm persists. The teacher-minus-KL20 primary failure is",
             "a separate comparison. Improved conditional fit does not establish native recovery.", "",
             "Corrected equal-category weighting requirements", "~" * len("Corrected equal-category weighting requirements"), "",
             "All four primary requirements fail. Duplicate output names previously overwrote",
             "three primary margins with nonharm margins; correction preserves every measured",
             "value and interval and separates primary/nonharm namespaces. Against target-average",
             "weighting, the corrected adaptive assessment requirements are:", ""]
    labels = ["Equal-category H3 soft-edge F1", "Mixed H3 source F1",
              "Mixed selected-B F1", "Mixed selected-B IoU"]
    checks = list(weighting["primary_checks"].values())
    lines += _table(["Primary requirement", "Difference", "95% paired interval", "Minimum", "Result"], [
        [label, f"{c['value']:.8f}",
         f"[{c['paired_bootstrap95'][0]:.8f}, {c['paired_bootstrap95'][1]:.8f}]",
         c["minimum"], c["status"]] for label, c in zip(labels, checks)])
    lines += ["Weighting did not rescue mixed selected-B or H3 edge quality. This comparison",
              "does not establish category imbalance as the cause.", "",
              "Support novelty and frozen assignment refresh", "~" * len("Support novelty and frozen assignment refresh"), "",
              "Same-fit12 recapture produced zero new contexts and zero new canonical membership",
              "masks. All 48 states were processed; 88 targets, 405 raw positive axes, 401 matched",
              "training positives, 1374 negatives and 3151 unknown raw axes remained. Eleven query",
              "assignments changed (4 charged, 7 mixed). Assignment change is not new support.", "",
              "Refreshed-minus-historical assignments improve assessment H3 soft-edge F1 by",
              "+0.02476080, paired 95% interval [0.02401961, 0.02550198], passing +0.02.",
              "Selected-B F1 declines -0.04279171, interval [-0.14272628, 0.05714286], failing",
              "its primary requirement; seven category/depth nonharm checks also fail. Both arms",
              "reduce fit loss under both banks. These four reused TRAIN assessment collisions",
              "do not establish population harm or generalization.", ""]
    lines += _table(["Assignment assessment category", "Selected-B F1 difference", "H3 source F1 difference", "H3 soft-edge F1 difference"], [
        [category, *[f"{next(r[6] for r in assignment['selected_metric_rows'] if r[:4] == ['assessment4', category, scope, metric]):.8f}"
                     for scope, metric in [("selected_B", "f1"), ("H3", "f1"), ("H3", "soft_edge_f1")]]]
        for category in ("equal_category", "charged", "mixed")])
    lines += ["Coverage, uncertainty and cost", "~" * len("Coverage, uncertainty and cost"), "",
              "Strict independent coverage is 0/12000: 0/2000 in each of charged, mixed, ccbar,",
              "uubar, ddbar and ssbar. Current-continuation background acceptance is UNAVAILABLE",
              "for every continuum flavour; physical completeness is UNAVAILABLE. Earlier eight-per-flavour",
              "background observations apply only to their original reused panel below. Missing",
              "background evidence is not zero fake-B acceptance. No FEI-equivalent efficiency is claimed.", "",
              "Registered equal-category aggregation, seed5001 and 2000 whole-collision bootstrap",
              "draws define the paired intervals. Fit12 and assessment4 remain separate. The download",
              "retains selected category/depth F1/IoU/precision/recall/edge values and intervals,",
              "nominal/known/unavailable support, corrected primary margins, checkpoint and source",
              "hashes, latency, memory and actual compute. It explicitly lists the metric projection.", ""]
    lines += _table(["Operation", "Allocated CPU-seconds", "Allocated V100-seconds", "Scheduler CPU-seconds"], [
        [e["name"], e["compute"]["allocated_cpu_seconds"], e["compute"]["allocated_V100_seconds"],
         e["compute"].get("scheduler_measured_total_CPU_seconds", e["compute"].get("scheduler_measured_CPU_seconds"))]
        for e in (generated, weighting, assignment)])
    lines += ["Support recapture separately used 664 allocated CPU-seconds and 83 allocated GPU-seconds.",
              "Reused controls are not recounted. Allocation is not GPU utilization; application",
              "and scheduler CPU measurements overlap. Publication cost is separate; no new science ran.", "",
              "Next neural operation: acquire four distinct eligible mixed TRAIN collisions only after",
              "external authentication restoration and separate finite admission. Then review exact",
              "associations and generated-state support before any matched support-expansion training.",
              "Prepared drafts are not launchable. Publication does not gate that independent work.", "",
              ":download:`Reviewed accumulated TRAIN findings and selected metrics <generic-b-accumulated-findings-v1.json>`", ""]
    return lines


def _recovered_default_four(source, output):
    """Retain the complete reviewed correction without rewriting historical bytes."""
    path = source / "recovered-default-four-source.json"
    binding_path = source / "recovered-default-four-binding.json"
    _require(not path.is_symlink() and not binding_path.is_symlink()
             and path.stat().st_size <= LIMIT and binding_path.stat().st_size <= LIMIT,
             "Unsafe recovered default-four aggregate")
    raw = path.read_bytes()
    binding = json.loads(binding_path.read_bytes())
    _require(binding == {'contains_cohort_member_identifiers': False, 'contains_operational_identifiers': False, 'private_payloads_copied': False, 'review_sha256': '265abab464417059cc90d03626cb370d3dc8b3ac36b643d7cea8bfb3507a3b9b', 'schema': 'generic-b-recovered-default-four-binding-v1', 'source_bytes': 29567, 'source_file': 'recovered-default-four-source.json', 'source_sha256': '2c3e16bd2fb0467e4a0749578a49b1f9b93ef97e8f52b4222076b2b8833e1468', 'status_date': '2026-11-18'},
             "Recovered default-four binding differs")
    _require(len(raw) == binding["source_bytes"]
             and hashlib.sha256(raw).hexdigest() == binding["source_sha256"],
             "Recovered default-four source differs")
    value = json.loads(raw)
    _require(value["schema"] == "gb101-recovered-default-four-publication-aggregates-v1"
             and value["strict_independent_collisions"] == 0
             and value["requested_independent_collisions"] == 12000
             and value["nominal_B_slots"] == {"development": 32, "train": 64}
             and value["new_inference_performed"] is False,
             "Recovered default-four scope differs")
    filename = "generic-b-recovered-default-four-v1.json"
    # Publish an explicit metric projection, excluding source-only publication
    # routing/status fields. The input remains authenticated above; the normal
    # source-body and complete artifact privacy gates stay fully enabled.
    public_keys = (
        "beam_implementation_sha", "checkpoint_sha256", "coverage_complete",
        "endpoints", "latency_seconds", "limitations", "new_inference_performed",
        "nominal_B_slots", "original_beam_seed5001", "peak_host_rss_kib",
        "requested_independent_collisions", "requested_per_category",
        "retained_exploratory_development_per_category", "reused_beam_rows",
        "reviewed_weight075_neural_result", "reviewed_partial_H3_ordering_result",
        "scientific_source_base_sha",
        "strict_independent_collisions", "strict_shortfall_per_category",
        "study_id", "train_diagnostic_collisions",
    )
    projected = {"schema": "generic-b-recovered-default-four-metrics-v1"}
    projected.update({key: value[key] for key in public_keys})
    payload = (json.dumps(projected, separators=(",", ":"), sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    (output / filename).write_bytes(payload)
    return value | {"metric_download": {"filename": filename, "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest()}}


def _render_recovered_default_four(value):
    recovered = value["recovered_default_four"]
    beam = recovered["original_beam_seed5001"]
    neural = recovered["reviewed_weight075_neural_result"]
    title = "GB101 preserved original beam comparison and earlier neural nulls"
    lines = [title, "-" * len(title), "",
             "Reviewed 18 November 2026. The original 80-collision full-depth, default-four",
             "beam diagnostic is completed. Its saved outputs and full aggregate tables were",
             "recovered without training or inference replay. This is a completed diagnostic;",
             "strict independent coverage remains 0/12000, with 0/2000 in each of charged,",
             "mixed, ccbar, uubar, ddbar and ssbar. No checkpoint is promoted.", "",
             "Original reused panel: 48 development collisions (8/category) and 32 TRAIN",
             "diagnostic collisions. Signal development has 16 collisions and 32 nominal B slots;",
             "TRAIN has 32 collisions and 64 nominal B slots. Exact topology+PID, topology-only",
             "and inclusive source grouping are each 0/32 development and separately 0/64 TRAIN",
             "for both checkpoints, greedy, beam top-1 and available oracle@1/@4. These roles",
             "remain separate; the endpoints do not establish physical tagging efficiency.", "",
             "Original beam versus greedy on the same exploratory development collisions", "~" * len("Original beam versus greedy on the same exploratory development collisions"), ""]
    lines += _table(["State", "Selected-B source F1", "Selected-B soft edge F1"], [
        [name, f"{beam['states'][key]['selected_B_source_f1']['point']:.5f}",
         f"{beam['states'][key]['selected_B_soft_edge_f1']['point']:.5f}"]
        for name, key in [("Latest greedy", "latest_greedy"),
                          ("Latest beam top-1", "latest_beam4_top1"),
                          ("Unpromoted parent greedy", "parent_greedy"),
                          ("Unpromoted parent beam top-1", "parent_beam4_top1")]])
    delta = beam["paired_contrasts"]["latest_beam4_minus_latest_gb101_greedy"]["selected_B_source_f1"]
    lines += [f"Latest paired source-F1 difference: {delta['point']:.5f}; whole-collision paired",
              f"95% interval [{delta['interval'][0]:.5f}, {delta['interval'][1]:.5f}].",
              "Equal-category aggregation, 2000 paired bootstrap draws and seed5001 are authoritative.",
              "Beam top-1 reduced selected-B quality. Lower continuum acceptance does not",
              "establish better signal reconstruction or a usable tagger.", ""]
    lines += _table(["Continuum flavour", "Latest greedy accepted events", "Latest beam accepted events"], [
        [category, f"{beam['backgrounds']['latest_greedy'][category]['accepted_events']}/8",
         f"{beam['backgrounds']['latest_beam4_top1'][category]['accepted_events']}/8"]
        for category in ("ccbar", "uubar", "ddbar", "ssbar")])
    lines += ["Historical reviewed neural weight0.75 comparison", "~" * len("Historical reviewed neural weight0.75 comparison"), "",
              "Historical reviewed snapshot before the latest ordering comparison. Its four",
              "displayed adaptive TRAIN contrasts failed; complete registered tables failed",
              "5/12 margins versus unit weight and 6/12 versus the stale control:", ""]
    lines += _table(["Paired contrast", "Difference", "Paired 95% interval"], [
        [label, f"{neural[key]['value']:.5f}",
         f"[{neural[key]['paired_bootstrap95'][0]:.5f}, {neural[key]['paired_bootstrap95'][1]:.5f}]"]
        for key, label in [("selected_B_delta_unit", "Selected B versus unit control"),
                           ("H3_delta_unit", "H3 versus unit control"),
                           ("selected_B_delta_stale", "Selected B versus stale control"),
                           ("H3_delta_stale", "H3 versus stale control")]])
    lines += ["This neural diagnostic reports exact, topology-only and inclusive recovery 0/32",
              "nominal B trials on adaptive TRAIN only; its continuum acceptance is unavailable.",
              "Rematched teacher losses are not fixed-query losses. No causal attribution,",
              "independent calibration or generalization follows from these reused panels.", "",
              "Neural cost: 5648 allocated CPU-seconds, 706 allocated GPU-seconds,",
              "658.254 scheduler CPU-seconds and 651.132111329 application CPU-seconds;",
              "paired analysis used 15.509 CPU-seconds. Original saved beam mean latency was",
              "1.635639 seconds/collision latest and 1.469861 parent; peak host RSS was",
              "1660688 KiB and 1696428 KiB respectively. Publication cost is separate.", "",
              "Earlier reviewed neural result: partial H3 BCE plus ordering",
              "~" * len("Earlier reviewed neural result: partial H3 BCE plus ordering"), "",
              "Both arms completed 20 updates and 16 native rollouts. Four of twelve",
              "registered margins failed on adaptive TRAIN assessment4; this is not independent validation.",
              "Selected-B F1 delta +0.000715, paired 95% interval [-0.100000, +0.101430];",
              "H3 source F1 delta +0.007692, interval [0.000000, +0.015385]. Both missed +0.02.",
              "Mixed selected-B and H3 soft-edge nonharm also failed. Exact, topology-only",
              "and inclusive recovery remained 0/32 nominal B trials per arm; no promotion.",
              "Fixed-query ordering improved; rematched teacher membership BCE worsened.",
              "Cost: 4088 allocated CPU-seconds, 511 V100-seconds and 481.011 scheduler CPU-seconds.",
              "All twelve margins are in the aggregate download. Independent coverage is 0/12000;",
              "neural background acceptance remains unavailable for each continuum flavour.", "",
              "Historical next action (superseded above): materialize supported terminal fit12 H3 states;",
              "if ordinary tensor inputs are absent, separately review finite same-fit12 capture",
              "before a matched old-bank versus refreshed-bank BCE intervention.",
              "Support construction is not reconstruction efficiency.",
              "Historical native quality, uncertainty, category/depth support and downloads remain",
              "available below. Full beam endpoints, both checkpoints, backgrounds, versions,",
              "cost, limitations and exact values are retained in the authenticated download.", "",
              ":download:`Recovered original beam and reviewed neural aggregates <generic-b-recovered-default-four-v1.json>`", ""]
    return lines


def render(value):
    lines = ['Earlier Generic-B campaign results', '----------------------------------', '', 'GB001: completed generic-sample static target-contract census (historical). The frozen pool', 'contained 20,000 B and 8,000 continuum collisions. Static accepted membership was 34,819/40,000', 'and recursive topology/PID compatibility was 6,378/40,000. This was not a learned-model evaluation', 'and these counts are not reconstruction efficiency. The needed provenance is unavailable in the census inputs;', 'original-mDST recovery was not assessed. Every learned metric in this census remains UNAVAILABLE.', '', 'GB002: local transfer complete; scientific authentication unresolved is a historical and now-superseded operational snapshot,', 'an as-of snapshot at the reviewed science-loop resumption boundary, not a live job monitor.', 'The transfer established opaque-byte identity only: 0 authenticated, 1 failed and 3,071 unreached.', 'The event-number and source-entry identities were conflated; replacement status was FAILED_REVIEW_REQUIRED.', 'The reviewed Astra/Sol science loop has since resumed; in that snapshot payload decoding and the native baseline remain pending.', 'The prospective two Generic B training sample sizes by two network widths do not constitute execution approval.', '', 'The historical heading Latest learned full reconstruction evaluation — CR054 refers to the earlier diagnostic,', 'not the current GB101 comparison displayed above. Channel coverage included 256 signed retained-channel groups.', 'Retained parton compatibility was 12/758; parton ancestry reconstruction is UNAVAILABLE.', 'The flat accepted 1/256 versus 0/256 comparison is not full reconstruction and did not support promotion.', '', 'Historical quality,source/checkpoint versions,uncertainty,cost,unavailable support and limitations remain', 'in the unchanged versioned downloads and historical study pages. No physical tagging-efficiency claim is implied.', '', ':download:`Reviewed v7 GB001, GB002, CR054 and GB101 aggregates with accumulated H2/H1 adaptations <generic-b-status-summary-v7.json>`.', '', ':download:`Historical v6 aggregate with the cap256 intervention retained with its original bytes <generic-b-status-summary-v6.json>`.', '', ':download:`Historical v5 aggregate with the saved-beam diagnosis retained with its original bytes <generic-b-status-summary-v5.json>`.', '', ':download:`Historical corrected v4 aggregate retained with its original bytes <generic-b-status-summary-v4.json>`.', '', ':download:`Historical v3 aggregate retained with its original bytes <generic-b-status-summary.json>`.', '']
    return _render_accumulated_findings(value) + _render_recovered_default_four(value) + _render_hybrid_teacher_gradient(value) + _render_hybrid_teacher_context(value) + _render_teacher_context_ablation(value) + _render_authenticated_teacher_tradeoff(value) + _render_type_isolation_tradeoff(value) + _render_partial_type_tradeoff(value) + _render_residual_support_refresh(value) + _render_h3_residual(value) + _render_head_gradient(value) + _render_targeted_h3(value) + _render_composite(value) + _render_within_update(value) + _render_state_mixture({}) + _render_query_refresh(value["h2_anchor"]["query-refresh-native"]) + _render_query_interaction(value["h2_anchor"]["query-interaction-native"]) + _render_dynamic(value["h2_anchor"]["dynamic-native"]) + _render_projection(value["h2_anchor"]["projection-native"]) + _render_expanded96(value["h2_anchor"]["expanded96-native"]) + _render_replay(value["h2_anchor"]["replay-native"]) + _render_balanced_refresh(value["h2_anchor"]["balanced-refresh-native"]) + _render_refresh_support(value["h2_anchor"]["refresh-native"]) + _render_joint_native(value["h2_anchor"]["joint-native"]) + _render_decoder_native(value["h2_anchor"]["decoder_native"]) + _render_relation_support(value["h2_anchor"]["relation_support"]) + _render_support_capture(value["h2_anchor"]["support_capture"]) + _render_source_supervision(value["h2_anchor"]["source_supervision"]) + _render_preservation(value["h2_anchor"]["preservation"]) + _render_retained_source(value["h2_anchor"]["retained_source"]) + _render_positive_object(value["h2_anchor"]["positive_object"]) + _render_h2_anchor(value["h2_anchor"]) + _render_supported_h3(value["supported_h3"]) + _render_neural_support(value["neural_support"]) + lines
