"""Bounded aggregate publication for Generic-B reconstruction studies."""
from __future__ import annotations

import hashlib
import json


SOURCE = "artifacts/codex/generic_b_dashboard_20261019"
FILES = {
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
    "generic-b-state-mixture-fit12-shallow-v1.json",
    "generic-b-state-mixture-fit12-deep-v1.json",
    "generic-b-state-mixture-assessment4-shallow-v1.json",
    "generic-b-state-mixture-assessment4-deep-v1.json",
    "generic-b-state-mixture-native-v1.json",
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
                                  ("state-mixture-fit12-shallow", "state-mixture-fit12-shallow", "generic-b-state-mixture-fit12-shallow"),
                                  ("state-mixture-fit12-deep", "state-mixture-fit12-deep", "generic-b-state-mixture-fit12-deep"),
                                  ("state-mixture-assessment4-shallow", "state-mixture-assessment4-shallow", "generic-b-state-mixture-assessment4-shallow"),
                                  ("state-mixture-assessment4-deep", "state-mixture-assessment4-deep", "generic-b-state-mixture-assessment4-deep"),
                                  ("state-mixture-native", "state-mixture-native", "generic-b-state-mixture-native"),
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
        result[key] = value | {"metric_download": {"filename": filename, "bytes": len(encoded),
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
    return ['GB101 query interaction improves selected-B quality with residual charged H3 harm', '---------------------------------------------------------------------------------', '', 'Reviewed 15 November 2026. Training the 12 existing query self-attention tensors alongside', 'the prior 72 trainables increased adaptive assessment selected-B F1 from 0.140909 to', '0.220753: delta +0.079844, paired 95% interval [0.000597, 0.159091] versus dynamic96.', 'All 12 added tensors received gradients and changed. Charged H3 source-F1 and soft-edge', 'nonharm still failed versus refreshed32. Fit12 selected-B F1 decreased by 0.01141.', 'The gain is not uniform: the scientific gate FAILED and there is no model promotion.', '', 'All 16 full-depth coherent-forest width4 rollouts completed: fit12 and adaptive assessment4.', 'Exact recursive topology+PID, topology-only and inclusive recovery remain 0/32 nominal', 'B trials per model, including actual oracle@1/@4. Oracle pools are diagnostics only.', 'Assessment4 is ancestrally exposed adaptive TRAIN, not independent validation. Paired', 'intervals use seed5001, 2000 whole-collision draws and equal-category weighting.', 'The download reports selected-B F1/IoU and charged/mixed H3 source and membership-edge', 'quality, frozen/rematched teacher diagnostics, latency, memory and measured cost.', '', 'The unchanged adaptive GB101 descendant retains H3 packing, bounded donor transactions', 'and coherent width4 search: generation1 uses eight candidates/query and later generations', 'four. This differs from default four/query evaluation. Truth enters supervision or scoring', 'only; it never generates, ranks, prunes, selects or stops native reconstruction.', '', 'Independent coverage remains 0/12000: 0/2000 each charged, mixed, ccbar, uubar, ddbar, ssbar.', 'Each continuum flavour event/candidate acceptance, physical completeness, complete H4', 'supervision and native proposal/proper scores remain unavailable. Partial deep labels stay', 'partial. Teacher-fit observations and adaptive native gains do not establish generalization.', '', 'Matched training completed 40 updates and 15360 nominal generated presentations.', 'Actual allocation: 6544 CPU-seconds and 818 V100-seconds (0.227222 GPU-hours). There were', '18824 forwards in total; final native inference used 341 forwards, 111407 beam expansions', 'and 50.47160 seconds. Scheduler, wrapper and application exits were zero. Reused controls', 'add no new allocation. Publication runner time is separate; runner allocations unavailable.', 'Scientific source: ``2840523db8ef070405db79eb38317a8c6bad64df``. Checkpoint SHA-256:', '``81d52cfd6f94f0f09efed95fc806b8f8aec9842700e99a2ba8a64e62022e4c99``.', '', 'Next: prepare one finite TRAIN96 state refresh using this completed checkpoint for capture,', 'while training still starts from the original common initialization. Preserve the same84', 'trainables, optimizer, masks, roles,40 updates,15360 presentations and16 final native', 'rollouts. State-distribution mismatch and refresh benefit remain hypotheses; no sweep.', '', ':download:`query interaction aggregate metrics <generic-b-query-interaction-native-v1.json>`', '', 'Earlier reviewed findings are summarized below; all historical downloads remain available.', '']


def _render_query_refresh(value):
    return ['GB101 query-state refresh fit gain does not transfer to adaptive assessment', '---------------------------------------------------------------------------', '', 'Reviewed 15 November 2026. Refreshing generated TRAIN96 states changed assessment selected-B', 'F1 from 0.220753 to 0.220142: delta -0.000611, paired 95% interval [-0.041667, 0.040444].', 'Fit12 selected-B F1 increased from 0.153935 to 0.244525: delta +0.090590, paired 95%', 'interval [0.021052, 0.155464]. Fit and assessment are distinct. Charged assessment F1', 'fell by 0.035256 while mixed rose by 0.034034. Selected-B gain and several category/H3', 'nonharm checks FAILED. No promotion.', '', 'Old96 supported states/assigned membership targets were 761/1261; refreshed96 were', '733/1208. All five prior complete H3 occurrences were absent after refresh; refreshed', 'complete H2 occurrences were 94. Partial deep membership remains partial. This observed', 'support change does not establish why native improvement failed. Complete H4 supervision', 'and physical completeness remain unavailable.', '', 'All 16 final full-depth native width4 rollouts completed (fit12 plus adaptive assessment4).', 'Exact recursive topology+PID, topology-only and inclusive recovery remain 0/32 nominal B', 'trials, including actual oracle@1/@4. Oracle pools are diagnostics. The unchanged adaptive', 'GB101 descendant retains H3 packing and bounded donor transactions; generation1 uses', 'eight candidates/query, later generations four. Truth only supplies supervision or scoring;', 'it never generates, ranks, prunes, selects or stops native reconstruction.', '', 'Assessment4 is adaptive TRAIN, not independent validation. Intervals use shared seed5001,', '2000 whole-collision draws and equal-category weighting. Independent coverage is 0/12000:', '0/2000 each charged, mixed, ccbar, uubar, ddbar, ssbar. Each continuum flavour event and', 'candidate acceptance is unavailable. Native proposal/proper scores remain unavailable.', 'The download retains selected-B F1/IoU, per-generation source and membership-edge quality,', 'teacher frozen/rematched diagnostics, source/checkpoint versions, support, cost and limits.', '', 'One operation completed 96 captures, 40 updates, 15360 nominal presentations and16 final', 'native rollouts:10536 allocated CPU-seconds;1317 V100-seconds (0.365833 GPU-hours).', 'There were23784 forwards and794566 total capture/final beam expansions. Scheduler,', 'wrapper and application exits were zero; completed controls add no new allocation.', 'Publication runner time is separate and runner CPU/GPU allocations are unavailable.', 'Scientific source: ``d0f205a7e57ceb9b510a418f4d3ccb9494db403c``. Checkpoint SHA-256:', '``94d28883b67477b00c402db0d1d571d5a12a6bcc91ff6e78bee48fd992ec207d``.', '', 'Next: prepare fixed alternating old/refreshed saved TRAIN96 states,40 updates and15360', 'nominal presentations,84 trainable tensors and16 final native rollouts. Preserve original', 'common initialization,optimizer,masks,roles and inference. No recapture or ratio sweep.', 'Replay benefit remains unmeasured; independent admission precedes submission.', '', ':download:`query-state refresh aggregate metrics <generic-b-query-refresh-native-v1.json>`', '', ':download:`query refresh fit12 shallow metrics <generic-b-query-refresh-fit12-shallow-v1.json>`', '', ':download:`query refresh fit12 deep metrics <generic-b-query-refresh-fit12-deep-v1.json>`', '', ':download:`query refresh assessment4 shallow metrics <generic-b-query-refresh-assessment4-shallow-v1.json>`', '', ':download:`query refresh assessment4 deep metrics <generic-b-query-refresh-assessment4-deep-v1.json>`', '', 'Earlier reviewed results and their historical downloads follow.', '']


def _render_state_mixture(value):
    return ['GB101 alternating replay harms native reconstruction despite better teacher fit', '-------------------------------------------------------------------------------', '', 'Reviewed 16 November 2026. Alternating saved old/refreshed TRAIN96 states reduced adaptive', 'assessment selected-B F1 from 0.220142 to 0.086016: delta -0.134126, paired 95% interval', '[-0.208333, -0.059918]. IoU fell from 0.137740 to 0.050424. Charged F1 delta was', '-0.111111 and mixed -0.157140. Fit12 F1 fell from 0.244525 to 0.182960:', 'delta -0.061565, paired 95% interval [-0.159661, 0.037325]. Gain and multiple', 'category/depth nonharm checks FAILED. No promotion.', '', 'Frozen-query and rematched teacher losses improved on all three saved banks, while native', 'reconstruction worsened. All five saved complete H3 occurrences were retained and all31', 'scheduled presentations were used. Retaining supported H3 alone did not yield a native', 'gain; the cause remains unproven. Partial deep labels remain partial, unknown membership', 'masked. Complete H4 supervision and physical completeness remain unavailable.', '', 'All16 final full-depth native width4 rollouts completed: fit12 plus adaptive assessment4.', 'Exact recursive topology+PID, topology-only and inclusive recovery remain0/32 nominal B', 'trials, including actual oracle@1/@4. The unchanged adaptive GB101 descendant retains H3', 'packing and bounded donor transactions; generation1 uses8 candidates/query, later levels4.', 'Truth is supervision or post-inference scoring only. Oracle pools cannot support promotion.', '', 'The download reports selected-B F1/IoU, source precision/recall/F1 and membership-edge', 'quality by generation and category, paired intervals, frozen/rematched teacher losses,', 'support, latency, memory, endpoints and lineage. Compact generation rows have explicit', 'column and arm/category registries; null remains unavailable. Native proposal/proper scores', 'remain unavailable. Assessment4 is adaptive TRAIN, not independent validation. Intervals', 'use shared seed5001,2000 whole-collision draws and equal-category weighting.', '', 'Independent coverage remains0/12000:0/2000 each charged,mixed,ccbar,uubar,ddbar,ssbar.', 'Every continuum flavour event and candidate acceptance remains unavailable.', '', 'The operation completed40 updates,15360 nominal presentations and16 native rollouts:', '6584 allocated CPU-seconds;823 V100-seconds (0.228611 GPU-hours),20880 forwards and', '111210 beam expansions. Scheduler,wrapper and application exits were zero. No recapture,', 'control replay or retries. Publication runner time is separate;runner allocations unavailable.', 'Scientific source: ``ffb20b625c402545d51cbe2f46bfe446e87956cd``. Checkpoint SHA-256:', '``293d9b287f98677ae4cee216006225c26a2364fc6a7750719071afbbeb450064``.', '', 'Next: prepare within-update old/refreshed mixing of the exact same presentation multiset:', '40 updates,15360 presentations,84 trainable tensors and16 final native rollouts. Preserve', 'initialization,optimizer,masks,roles and inference. Scheduling benefit remains unmeasured;', 'independent review precedes submission. No ratio/order sweep or causal claim.', '', ':download:`alternating state mixture aggregate metrics <generic-b-state-mixture-native-v1.json>`', '', ':download:`state mixture fit12 shallow metrics <generic-b-state-mixture-fit12-shallow-v1.json>`', '', ':download:`state mixture fit12 deep metrics <generic-b-state-mixture-fit12-deep-v1.json>`', '', ':download:`state mixture assessment4 shallow metrics <generic-b-state-mixture-assessment4-shallow-v1.json>`', '', ':download:`state mixture assessment4 deep metrics <generic-b-state-mixture-assessment4-deep-v1.json>`', '', 'Earlier reviewed results and their historical downloads follow.', '']


def render(value):
    lines = ['Earlier Generic-B campaign results', '----------------------------------', '', 'GB001: completed generic-sample static target-contract census (historical). The frozen pool', 'contained 20,000 B and 8,000 continuum collisions. Static accepted membership was 34,819/40,000', 'and recursive topology/PID compatibility was 6,378/40,000. This was not a learned-model evaluation', 'and these counts are not reconstruction efficiency. The needed provenance is unavailable in the census inputs;', 'original-mDST recovery was not assessed. Every learned metric in this census remains UNAVAILABLE.', '', 'GB002: local transfer complete; scientific authentication unresolved is a historical and now-superseded operational snapshot,', 'an as-of snapshot at the reviewed science-loop resumption boundary, not a live job monitor.', 'The transfer established opaque-byte identity only: 0 authenticated, 1 failed and 3,071 unreached.', 'The event-number and source-entry identities were conflated; replacement status was FAILED_REVIEW_REQUIRED.', 'The reviewed Astra/Sol science loop has since resumed; in that snapshot payload decoding and the native baseline remain pending.', 'The prospective two Generic B training sample sizes by two network widths do not constitute execution approval.', '', 'The historical heading Latest learned full reconstruction evaluation — CR054 refers to the earlier diagnostic,', 'not the current GB101 comparison displayed above. Channel coverage included 256 signed retained-channel groups.', 'Retained parton compatibility was 12/758; parton ancestry reconstruction is UNAVAILABLE.', 'The flat accepted 1/256 versus 0/256 comparison is not full reconstruction and did not support promotion.', '', 'Historical quality,source/checkpoint versions,uncertainty,cost,unavailable support and limitations remain', 'in the unchanged versioned downloads and historical study pages. No physical tagging-efficiency claim is implied.', '', ':download:`Reviewed v7 GB001, GB002, CR054 and GB101 aggregates with accumulated H2/H1 adaptations <generic-b-status-summary-v7.json>`.', '', ':download:`Historical v6 aggregate with the cap256 intervention retained with its original bytes <generic-b-status-summary-v6.json>`.', '', ':download:`Historical v5 aggregate with the saved-beam diagnosis retained with its original bytes <generic-b-status-summary-v5.json>`.', '', ':download:`Historical corrected v4 aggregate retained with its original bytes <generic-b-status-summary-v4.json>`.', '', ':download:`Historical v3 aggregate retained with its original bytes <generic-b-status-summary.json>`.', '']
    return _render_state_mixture(value["h2_anchor"]["state-mixture-native"]) + _render_query_refresh(value["h2_anchor"]["query-refresh-native"]) + _render_query_interaction(value["h2_anchor"]["query-interaction-native"]) + _render_dynamic(value["h2_anchor"]["dynamic-native"]) + _render_projection(value["h2_anchor"]["projection-native"]) + _render_expanded96(value["h2_anchor"]["expanded96-native"]) + _render_replay(value["h2_anchor"]["replay-native"]) + _render_balanced_refresh(value["h2_anchor"]["balanced-refresh-native"]) + _render_refresh_support(value["h2_anchor"]["refresh-native"]) + _render_joint_native(value["h2_anchor"]["joint-native"]) + _render_decoder_native(value["h2_anchor"]["decoder_native"]) + _render_relation_support(value["h2_anchor"]["relation_support"]) + _render_support_capture(value["h2_anchor"]["support_capture"]) + _render_source_supervision(value["h2_anchor"]["source_supervision"]) + _render_preservation(value["h2_anchor"]["preservation"]) + _render_retained_source(value["h2_anchor"]["retained_source"]) + _render_positive_object(value["h2_anchor"]["positive_object"]) + _render_h2_anchor(value["h2_anchor"]) + _render_supported_h3(value["supported_h3"]) + _render_neural_support(value["neural_support"]) + lines
