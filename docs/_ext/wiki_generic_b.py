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
    title = "GB101 history: supported H3 learning changes selected-B quality"
    lines = [title, "-" * len(title), "",
             "Reviewed outcome: 14 November 2026. Adaptive reused TRAIN evidence only.",
             "Matched anchor-only terminal40 is the control; single-H3 teacher terminal40 is",
             "the treatment. Both start from the same checkpoint and train the same 69 tensors",
             "for 40 updates. Historical parent/common0 is a separate safeguard comparator.", ""]
    rows = []
    for panel, label in (("assessment4", "Adaptive assessment: 4 collisions"),
                         ("fit12", "Native fit: 12 collisions")):
        for metric in ("f1", "iou"):
            cell = value["selected_B"][panel][metric]
            delta = cell["treatment_minus_control"]
            bounds = delta["paired_bootstrap95"]
            rows.append((label, metric.upper(),
                         f'{cell["control"]["registered_equal_category_point"]:.6f}',
                         f'{cell["treatment"]["registered_equal_category_point"]:.6f}',
                         f'{delta["registered_equal_category_point_delta"]:+.6f}',
                         f'[{bounds[0]:+.6f}, {bounds[1]:+.6f}]'))
    lines += _table(["Panel", "Selected-B source metric", "Control", "Treatment", "Delta", "Paired 95%"], rows)
    lines += ["Assessment F1 is unchanged for charged and rises by +0.235548 for mixed.",
              "Native-fit F1 rises by +0.106322 for charged but falls by -0.062958 for mixed.",
              "Equal-category aggregation retains that mixed harm; it is not hidden by the",
              "aggregate fit delta. Intervals use 2,000 shared seed-5001 whole-collision draws.", "",
              "The one supported mixed H3 teacher's conditional BCE falls from 3.136909 at",
              "initialization to 0.000637 in treatment; the matched control ends at 3.796123.",
              "There is no supported charged H3 teacher. The 320 nominal supervision slots",
              "remain one charge-compatible, 92 incompatible, 49 unavailable and 178 absent.", "",
              "Exact recursive topology+PID, topology-only and inclusive recovery remain zero:",
              "0/8 assessment and 0/24 fit B trials per arm, for selected top1 and oracle@1/@4.",
              "Oracle pools are post-inference diagnostics and are never promoted.",
              "Ten arm/event records lack reviewed depth scoring (nine fit, one assessment);",
              "heights 1-6 and proposal-quality missingness remain explicit pending independent",
              "review of an additive saved-output correction. Missing observations are not zero.", "",
              "This adaptive GB101 descendant uses coherent-forest width4 with inherited",
              "eight candidates/query at level1 and a protected H3 packing continuation lane.",
              "It is distinct from the original default-four full-depth evaluation below.",
              "No new normalization fit or sealed-role access occurred.", "",
              "Strict independent coverage is 0/12,000: all six categories remain 2,000 short.",
              "For ccbar, uubar, ddbar and ssbar, event and candidate acceptance are unavailable",
              "for this newer study. Background nonharm and generalization are unestablished.",
              "The scientific gate remains unresolved; no overall PASS or model promotion.", "",
              "Measured execution: 80 updates, 32 native rollouts, 154,054 query expansions;",
              "5,000 allocated CPU-seconds, 625 allocated GPU-seconds and 610.620 measured",
              "scheduler CPU-seconds. Peak device memory was 288,529,920 bytes; peak host RSS",
              "was 2,955,756 KiB. Per-arm inference latency and checkpoint hashes are in the download.", "",
              "Next: recover depth metrics from saved outputs, then independently review a finite",
              "category-balanced anchor-objective proposal before further training.", "",
              ":download:`Reviewed supported-H3 learning metrics <generic-b-supported-h3-v1.json>`", ""]
    return lines



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
        _require(binding == {"schema": schema + "-binding-v1", "status_date": ("2026-11-15" if basename.startswith(("joint-", "refresh-", "balanced-refresh-", "replay-", "expanded96-")) else "2026-11-14"),
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
        options = {"separators": (",", ":")} if basename.startswith(("decoder-", "joint-", "refresh-", "balanced-refresh-", "replay-", "expanded96-")) else {"indent": 2}
        encoded = (json.dumps(published, sort_keys=True, allow_nan=False, **options) + "\n").encode()
        filename = schema + "-v1.json"
        (output / filename).write_bytes(encoded)
        result[key] = value | {"metric_download": {"filename": filename, "bytes": len(encoded),
                                                   "sha256": hashlib.sha256(encoded).hexdigest()}}
    return result


def _render_h2_anchor(record):
    value, depth = record["outcome"], record["depth"]
    title = "GB101 earlier: H2 membership fit does not establish native benefit"
    lines = [title, "-" * len(title), "",
             "Reviewed 14 November 2026. All evidence is adaptive reused TRAIN, including the",
             "four-collision assessment panel; it does not measure independent generalization.",
             "The completed H3 teacher model is the control; adding 15 supported H2 teacher",
             "membership contexts is the treatment. The same initialization, 69 trainable",
             "tensors, optimizer, 40 updates and terminal checkpoint rule are retained.", ""]
    rows = []
    for cell in value["selected_B"]:
        c = cell["contrast"]
        lo, hi = c["paired_bootstrap95"]
        rows.append((cell["panel"], cell["metric"].upper(),
                     f'{cell["states"][value["control_identity"]]:.6f}',
                     f'{cell["states"][value["treatment_identity"]]:.6f}',
                     f'{c["registered_equal_category_point_delta"]:+.6f}',
                     f'[{lo:+.6f}, {hi:+.6f}]'))
    lines += _table(["Panel", "Selected-B metric", "Control", "H2 treatment", "Delta", "Paired 95%"], rows)
    lines += ["Assessment selected-B F1 changes by +0.076923 for charged but -0.051159 for",
              "mixed. The equal-category F1 delta is +0.012882 with paired 95% interval",
              "[-0.059441, +0.085205]; IoU delta is -0.004222. Native benefit is inconclusive",
              "and the scientific gate failed. These are collision-paired intervals from",
              "2,000 shared seed-5001 draws, with equal category weights.", ""]
    rows = []
    for category in ("charged", "mixed"):
        for population, metric in (("teacher_H2", "H2"), ("teacher_H2", "H2:object"),
                                   ("generated20", "H2"), ("generated20", "H2:object")):
            cell = value["fit_operands"][category + ":" + population + ":" + metric + ":vs_matched_control40"]
            rows.append((category, population, metric,
                         f'{cell["reference"]:.6f}', f'{cell["treatment"]:.6f}'))
    lines += _table(["Category", "Fit population", "Loss", "Control", "H2 treatment"], rows)
    lines += ["Teacher H2 membership fit improves strongly in both categories. Mixed teacher",
              "positive-object loss rises from 0.798053 to 0.965922; mixed generated-state H2",
              "membership and object losses also worsen. Better teacher membership fitting",
              "does not establish improved native reconstruction or its causal mechanism.", "",
              "Saved-depth correction and limits", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "All ten previously failed arm/event depth observations were recovered from",
              "retained operands without new inference. Earlier H3 selected-B gains remain,",
              "but corrected mixed assessment H1 source F1 falls by 0.080000 and H4 source",
              "F1 by 0.123016. This materially limits the interpretation of the H3 gain.",
              "The corrected H2 comparison below uses the same saved control selection identities.",
              "Its saved control depths are observed for all 16 collisions. Five H2 treatment",
              "depth rows remain unavailable (four fit and one assessment); this operation",
              "did not recover them. H1-H6 equal-category H2 deltas therefore remain unavailable.",
              "H6 has no retained truth-target population; that absence is distinct from",
              "an observation failure. Charged-only assessment observations do not fill the",
              "missing mixed support or establish aggregate nonharm.",
              "Recovered observation failures are distinct from genuinely absent or unsupported",
              "heights. Missing observations remain unavailable; neither nonharm nor a PASS",
              "is inferred from missing support. Detailed support and intervals are downloadable.", ""]
    rows = []
    for values in depth["data"]:
        row = dict(zip(depth["columns"], values))
        if row["panel"] == "assessment4" and row["metric"] in ("f1", "soft_edge_f1"):
            ci = row.get("paired_bootstrap95")
            rows.append((row["comparison"], "H" + str(row["height"]), row["metric"],
                         "unavailable" if row["delta"] is None else f'{row["delta"]:+.6f}',
                         "unavailable" if ci is None or any(v is None for v in ci) else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]',
                         row["known_pairs"], row["nominal_pairs"]))
    lines += _table(["Comparison", "Height", "Metric", "Equal-category delta", "Paired 95%", "Known pairs", "Nominal pairs"], rows)
    lines += ["Exact recursive topology+PID, topology-only and inclusive source recovery remain",
              "0/32 B trials per arm: assessment 0/8 and fit 0/24, separately for selected top1",
              "and oracle@1/@4 pools. Oracle pools never determine a deployed candidate.",
              "Selected-B membership metrics and retained-forest depth metrics describe distinct",
              "views. Proposal correctness remains unavailable without retained stage operands.", "",
              "Strict independent coverage is 0/12,000. Charged, mixed, ccbar, uubar, ddbar and",
              "ssbar are each 2,000 short; eligible independent availability is unestablished.",
              "Event and candidate acceptance for each continuum flavour remain unavailable.",
              "There is no promotion, background-safety claim, or physical-completeness claim.", "",
              "The matched adaptive GB101 descendant retains H3 packing as its nominal lane",
              "and coherent-forest width4 search, with inherited eight candidates/query at",
              "level1 and four elsewhere. It is distinct from the original default-four",
              "full-depth evaluation retained below. Truth is post-inference scoring only.",
              "Existing train-only normalization, checkpoint architecture and trained head",
              "state remain bound. Source and checkpoint hashes are in the aggregate download.", "",
              "Actual H2 execution: 40 updates, 16 new rollouts; the completed 16 control rows",
              "were reused. There were 90,800 query expansions and 357 native model forwards.",
              "Measured cost: 3,488 allocated CPU-seconds, 436 allocated GPU-seconds and",
              "424.925 scheduler CPU-seconds. Saved scoring used 12.311695 child CPU-seconds",
              "without new model calls. New-rollout latency totals 49.341 seconds (3.084 mean);",
              "peak host RSS was 2,845,504 KiB and peak device memory 222,499,328 bytes.", "",
              "The following positive-object contrast is now completed and reported above.",
              "This earlier snapshot and its downloads retain the original comparison.", "",
              ":download:`Reviewed H2 outcome and lineage <generic-b-h2-anchor-v1.json>`", "",
              ":download:`Corrected H3 and H2 depth metrics and support <generic-b-corrected-depth-v1.json>`", "",
              "Earlier published snapshots and downloads follow unchanged; their missingness",
              "statements describe those historical snapshots and are superseded by this correction.", ""]
    return lines



def _render_positive_object(value):
    title = "GB101 earlier: positive-object fit improves while native quality fails"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. The H2 membership model is the control; adding",
        "0.1 times positive-object teacher loss is the treatment. Forty updates and",
        "16 new native rollouts completed; the 16 control rows were reused. Both arms",
        "share the original nonlinear initialization, optimizer, 69 trainable tensors,",
        "support, normalization and fixed terminal checkpoint. The object head stayed",
        "frozen; its gradients reached the H2 representation on all 40 updates.", "",
        "All 12 fit and four assessment collisions are adaptive reused TRAIN. These",
        "measurements do not establish independent generalization. Scientific gate",
        "failed; the model is not promoted.", ""]
    rows = []
    for cell in value["selected_B"]:
        c = cell["contrast"]; lo, hi = c["paired_bootstrap95"]
        rows.append((cell["panel"], cell["metric"],
            f'{cell["states"][value["control_identity"]]:.6f}',
            f'{cell["states"][value["treatment_identity"]]:.6f}',
            f'{c["registered_equal_category_point_delta"]:+.6f}',
            f'[{lo:+.6f}, {hi:+.6f}]'))
    lines += _table(["Panel", "Selected-B metric", "Control", "Object treatment", "Delta", "Paired 95%"], rows)
    lines += ["Assessment charged F1 falls 0.076923 while mixed rises 0.025906. Aggregate",
        "F1 falls 0.025508 (paired 95% [-0.104528, +0.053512]); IoU falls 0.006939.",
        "Intervals use 2,000 shared seed-5001 whole-collision draws and equal category",
        "weights. Reused-panel uncertainty does not account for adaptive study selection.", ""]
    rows = []
    for category in ("charged", "mixed"):
        for population in ("teacher_H2", "generated20"):
            key = population + ":" + category
            c = value["positive_object_fit"][value["control_identity"]][key]
            t = value["positive_object_fit"][value["treatment_identity"]][key]
            rows.append((category, population, f'{c["conditional"]:.6f}',
                f'{t["conditional"]:.6f}', t["supported_queries"], t["nominal_query_slots"],
                f'{t["nominal_known_contribution"]:.6f}'))
    lines += _table(["Category", "Fit population", "Control object loss", "Treatment object loss",
        "Supported queries", "Nominal query slots", "Known nominal contribution"], rows)
    lines += ["Conditional positive-object fit improved in all four cells; unobserved nominal",
        "slots are unavailable, not zero loss. Teacher H2 has only 15 supported unique",
        "contexts; teacher H3 has one compatible target among 320 nominal slots.",
        "Better teacher and saved-generated fit did not establish native benefit.", "",
        "Retained selected-forest depth quality", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    rows = []
    d = value["depth"]
    for data in d["data"]:
        r = dict(zip(d["columns"], data))
        if r["panel"] != "assessment4" or r["metric"] not in ("f1", "soft_edge_f1"):
            continue
        for category in ("charged", "mixed"):
            c = dict(zip(d["category_columns"], r[category])); delta=c["registered_point_delta"]; ci=c["paired_bootstrap95"]
            rows.append((category, "H" + str(r["height"]), r["metric"],
                "unavailable" if delta is None else f'{delta:+.6f}',
                "unavailable" if ci is None or any(x is None for x in ci) else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]',
                c["known_pairs"], c["nominal_pairs"]))
    lines += _table(["Category", "Height", "Metric", "Delta", "Paired 95%", "Known pairs", "Nominal pairs"], rows)
    lines += ["Charged assessment H3 source F1 falls 0.035714, paired 95% [-0.071429, 0].",
        "Charged native and H3 safeguards failed. Missing mixed depth support remains",
        "unavailable; absent H6 target population is distinct from an observation failure.",
        "Source F1/IoU, precision/recall and hard/soft membership edge contrasts for all",
        "six heights and both panels are included in the download. Proposal correctness",
        "is unavailable without the necessary retained stage operands.", "",
        "Exact recursive topology+PID, topology-only and inclusive recovery stay 0/32",
        "nominal B slots per arm (fit 0/24, assessment 0/8), separately for selected",
        "top1 and actual oracle@1/@4 pools. Oracle results never select a deployed output.",
        "Strict independent coverage remains 0/12,000: charged, mixed, ccbar, uubar,",
        "ddbar and ssbar each have 0/2,000 processed. Eligible independent availability",
        "is unestablished. Each continuum flavour's event and candidate acceptance is",
        "unavailable. No background safety or physical completeness is claimed.", "",
        "This adaptive GB101 descendant retains H3 packing, coherent full-depth width4",
        "beam search and the bounded donor transaction. It keeps eight candidates/query",
        "at level1 and four elsewhere, with unchanged terminal selection. It differs",
        "from the original default-four full-depth evaluation preserved below.",
        "Truth is supervision or post-inference scoring only. Checkpoint architecture,",
        "reference-conditioned heads, trained PID/confidence and normalization are preserved.", "",
        "Native completion used 335 model forwards and 92,147 beam expansions. Native",
        "latency total was 39.039 seconds (2.440 mean); peak host RSS 2,228,748 KiB,",
        "peak device memory 83,362,816 bytes. Training plus native completion consumed",
        "3,256 allocated CPU-seconds, 407 allocated GPU-seconds and 397.050 scheduler",
        "CPU-seconds; prior controls and preparation are excluded. The first completion",
        "attempt failed before rollout after saving the trained checkpoint; completion",
        "restored it without retraining.", "",
        "Next: prepare a matched frozen-replay versus refreshed-generated-context contrast.",
        "The cause of teacher/native divergence remains unproven. Independent capture/support",
        "review must precede frozen training counts and execution admission; no training is",
        "admitted by preparation. Source and checkpoint hashes accompany the download.", "",
        ":download:`Positive-object native outcomes, depth support and lineage <generic-b-positive-object-v1.json>`", "",
        "Earlier reviewed H2, corrected H3 and original full-depth results follow; their",
        "downloads remain available.", ""]
    return lines


def _render_retained_source(value):
    title = "GB101 earlier: retained-source learning improves selected B, harms deeper structure"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. Masked retained-source supervision is learnable and",
        "improves selected-B quality on four adaptive TRAIN assessment collisions. Charged",
        "H3/H4 source and edge quality harmed, so the scientific gate failed. No promotion.",
        "This is exploratory evidence from a reused panel, not independent generalization.", "",
        "Both arms restore the same full 436-tensor checkpoint and train the same 36",
        "native tensors for 40 updates. The control preserves common-initialization",
        "detector outputs; the treatment adds frozen-matched masked retained-source BCE.",
        "Initialization, optimizer, normalization, terminal checkpoint and inference",
        "controls are matched. Only 42/168 nominal target slots are eligible; 126 remain",
        "unavailable. Retained-source membership is distinct from physical completeness.", ""]
    rows=[]
    for r in value["selected_B"]:
        lo,hi=r["ci95"]
        rows.append((r["panel"],r["metric"],f'{r["states"][value["control_identity"]]:.6f}',
            f'{r["states"][value["treatment_identity"]]:.6f}',f'{r["delta"]:+.6f}',f'[{lo:+.6f}, {hi:+.6f}]'))
    lines += _table(["Panel","Selected-B metric","Control","Retained-source treatment","Delta","Paired 95%"],rows)
    lines += ["Assessment selected-B F1 rises 0.155309 to 0.287848: delta +0.132539",
        "(paired 95% [+0.034558, +0.230519]). IoU rises +0.072292",
        "[+0.007888, +0.136696]. Charged F1 rises +0.071021 and mixed +0.194057.",
        "Intervals use 2,000 shared seed-5001 whole-collision draws, stratified by category",
        "with equal category weights. They do not adjust for adaptive study selection.", "",
        "Auxiliary masked fit, separate from native reconstruction", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    rows=[]
    for metric in ("fixed_nominal_BCE","eligible_only_equal_cell_BCE"):
        r=value["fit"][metric];rows.append((metric,f'{r["control"]:.6f}',f'{r["treatment"]:.6f}'))
    lines += _table(["Fit metric","Control","Treatment"],rows)
    rows=[]
    for r in value["fit"]["per_cell"]:
        rows.append((r["category"],r["level"],f'{r["control_BCE"]:.6f}',f'{r["treatment_BCE"]:.6f}',
            f'{r["control_soft_dice"]:.6f}',f'{r["treatment_soft_dice"]:.6f}'))
    lines += _table(["Category","Height","Control BCE","Treatment BCE","Control soft Dice","Treatment soft Dice"],rows)
    lines += ["The fixed-nominal BCE keeps missing slots in its denominator; it is a known",
        "contribution, not evidence that unavailable targets have zero loss. Eligible-only",
        "BCE and soft Dice describe observed support. Unknown source membership is masked.", "",
        "Native depth response and nonharm", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    rows=[]
    d=value["depth"]
    for record in d["data"]:
        r=dict(zip(d["columns"],record))
        if r["panel"]!="assessment4" or r["metric"] not in ("f1","soft_edge_f1"):continue
        for category in ("charged","mixed"):
            c=dict(zip(d["category_columns"],r[category]));v=c["delta"];ci=c["ci95"]
            rows.append((category,"H"+str(r["height"]),r["metric"],"unavailable" if v is None else f'{v:+.6f}',
                "unavailable" if ci is None or any(v is None for v in ci) else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]',c["known_pairs"],c["nominal_pairs"]))
    lines += _table(["Category","Height","Metric","Delta","Paired 95%","Known pairs","Nominal pairs"],rows)
    lines += ["Charged H3 source F1 falls 0.082540 and soft-edge F1 falls 0.082129; H4",
        "source F1 falls 0.064394 and soft-edge F1 falls 0.062290. These harms fail",
        "registered safeguards despite the selected-B gain. H5/H6 support is incomplete.",
        "The download retains both panels, six heights, source F1/IoU/precision/recall,",
        "hard/soft membership edge contrasts, nominal support, bounds and intervals.",
        "Proposal correctness and calibrated PID/confidence proper scores remain unavailable.", "",
        "Assessment exact recursive topology+PID, topology-only and inclusive source",
        "recovery remain 0/8 nominal B slots per arm, separately for selected top1 and",
        "actual oracle@1/@4. Fit diagnostics remain separate in the download. Oracle",
        "selection never defines a deployable output. All 32 requested rollouts completed.",
        "Independent coverage remains 0/12,000: charged, mixed, ccbar, uubar, ddbar and",
        "ssbar each have 0/2,000 processed. Eligible independent availability is unestablished.",
        "Each continuum flavour's event and candidate acceptance is unavailable, not zero.",
        "No background safety, confirmation or physical completeness is established.", "",
        "Matched inference retains GB101 nominal H3 packing with coherent full-depth",
        "width4 forest beam and a bounded postdecode donor transaction. Level1 uses eight",
        "candidates/query and later levels four; terminal selection is unchanged. This",
        "adaptive descendant is distinct from the original default-four evaluation below.",
        "Truth is supervision or post-inference scoring only; it never ranks or stops",
        "inference. Native architecture, reference-conditioned type heads, trained PID",
        "and confidence state, recursive source uniqueness and normalization are preserved.", "",
        "Training and inference used 1,760 allocated CPU-seconds, 220 allocated GPU-seconds",
        "and 222.944 scheduler CPU-seconds. Native beam expansions total 188,895. Native",
        "inference totals 43.736 seconds for control and 51.893 for treatment across 16",
        "rollouts each. Process-cumulative memory and model-call counts are in the download;",
        "the memory measurements are not isolated arm comparisons. Earlier studies and",
        "publication overhead are excluded from these scientific operation totals.", "",
        "The subsequent matched generated-state preservation result is reported above.",
        "Its reviewed null changes the next recommendation to generated-input supervision.",
        "Mixed generated-state support and direct H4 preservation anchors are absent.", "",
        ":download:`Retained-source outcomes, uncertainty, depth and checkpoint lineage <generic-b-retained-source-v1.json>`", "",
        "Earlier reviewed studies and their historical downloads follow.", ""]
    return lines

def _render_preservation(value):
    title = "GB101 earlier: preserving generated states reduces drift but harms native quality"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. Frozen generated-state preservation reduced drift but",
        "worsened selected-B quality on four reused adaptive TRAIN assessment collisions.",
        "The scientific gate failed; no promotion or independent generalization claim.", "",
        "Both arms restore the same 436-tensor checkpoint, train the same 36 tensors for",
        "40 updates, and retain identical detector anchor and retained-source objectives.",
        "The changed factor is coefficient zero versus one on pointer/object/type KL",
        "against saved common-initialization generated outputs. Fourteen frozen generated",
        "contexts are observed of 96 nominal; 82 remain unreached. Mixed generated-state",
        "support and direct H4 anchors are absent. Detector labels cover 42/168 targets;",
        "126 remain unavailable. Missing terms are not observed zero error.", ""]
    arms = ["retained_source_control40", "generated_preservation40"]
    lines += _table(["Metric", "Control", "Preservation"], [
        ("Fixed-nominal generated KL", *[f'{value["generated_KL"][a]:.6f}' for a in arms]),
        ("Fixed-nominal retained BCE", f'{value["fit_BCE"]["control"]:.6f}', f'{value["fit_BCE"]["treatment"]:.6f}')])
    rows=[]
    for panel,key in [("assessment4","assessment_selected_B"),("fit12","fit_selected_B")]:
        for metric,v in value[key].items():
            rows.append((panel,metric,f'{v["registered_equal_category_point_delta"]:+.6f}',str(v["paired_bootstrap95"])))
    lines += _table(["Panel", "Selected-B metric", "Treatment minus control", "Paired 95%"],rows)
    lines += ["Assessment selected-B F1 falls by 0.064932 (paired 95% [-0.104888, -0.024975]);",
        "charged changes by -0.077110 and mixed by -0.052753. IoU falls by 0.030571.",
        "Charged H3/H4 mean source-F1 change is -0.037037 [-0.074074, 0.000000].",
        "Intervals use 2,000 shared seed-5001 whole-collision draws and equal category",
        "weights. Repeated adaptive study selection is not covered by these intervals.", ""]
    depth=value["per_generation"];rows=[]
    for row in depth["rows"]:
        v=dict(zip(depth["columns"],row))
        if v["panel"]!="assessment4" or v["metric"] not in ("f1","soft_edge_f1"):continue
        rows.append((v["category"],"H"+str(v["height"]),v["metric"],v["delta"] if v["delta"] is not None else "unavailable",str(v["paired95"]),v["known_pairs"],v["nominal_pairs"]))
    lines += _table(["Category", "Height", "Metric", "Delta", "Paired 95%", "Known pairs", "Nominal pairs"],rows)
    lines += ["The download retains fit12 separately, source F1/IoU/precision/recall, hard and",
        "soft membership edge quality by generation, support, endpoints and uncertainty.",
        "Proposal correctness and calibrated PID/confidence proper scores remain unavailable.", "",
        "Exact recursive topology+PID, topology-only and inclusive source recovery remain",
        "0/8 assessment B trials per arm (4 charged, 4 mixed), separately for selected top1",
        "and actual oracle@1/@4. Fit12 remains 0/24 per arm. Oracle pools are post-inference",
        "diagnostics and cannot select a deployable model. All 32 native rollouts completed.", "",
        "Independent coverage remains 0/12,000: charged, mixed, ccbar, uubar, ddbar and",
        "ssbar each have 0/2,000 processed; eligible independent availability is unestablished.",
        "Each continuum flavour's event and candidate acceptance is unavailable, not zero.",
        "Background nonharm, confirmation and physical completeness are unestablished.", "",
        "Inference retains GB101 H3 nominal packing and full-depth coherent-forest width4",
        "beam, unchanged terminal selection and bounded postdecode donor transaction.",
        "Level1 uses eight candidates/query, later levels four: this adaptive descendant",
        "is distinct from the original default-four greedy/beam evaluation below.",
        "Truth is loss supervision or post-inference scoring only. Checkpoint architecture,",
        "reference-conditioned type heads, trained PID/confidence, source uniqueness and",
        "train-only normalization are preserved. No normalization fitting or sealed access.", "",
        "Execution completed 80 updates, 3,580 model forwards, 7,160 encoder passes and",
        "191,546 beam expansions. Allocated compute: 3,048 CPU-seconds and 381 GPU-seconds;",
        "scheduler CPU usage: 376.021 seconds. Native latency totals are 50.576 seconds for",
        "control and 48.539 for preservation over 16 rollouts each. Peak device memory was",
        "293,159,424 bytes and host RSS 2,472,904 KiB; memory peaks are process-cumulative,",
        "not isolated arm comparisons. Publication overhead is separate.", "",
        "Frozen source: ``7ea90bb71cc0fc83358d78715b1adf5e09d9bbcf``. Lower generated KL",
        "alone does not repair native reconstruction; this contrast does not identify cause.",
        "The subsequent reviewed generated-source supervision outcome is reported above.", "",
        ":download:`Preservation outcomes, depth, support, uncertainty and cost <generic-b-preservation-v1.json>`", "",
        "Earlier reviewed results and all historical downloads follow.", ""]
    return lines


def _render_source_supervision(value):
    title = "GB101 latest learned comparison: generated-source supervision improves fit without established native benefit"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. Matched coefficient-zero versus coefficient-one",
        "generated-input retained-source supervision improved fixed-nominal generated BCE",
        "from 0.037021 to 0.013112. Both arms restore the same 436-tensor unpromoted",
        "checkpoint and train the same 36 tensors for 40 updates, with detector anchors",
        "and native inference controls matched. No model promotion: native improvement",
        "and nonharm gates failed. These are reused adaptive TRAIN observations.", ""]
    rows=[]
    for panel, metrics in value["selected_B"].items():
        for metric in ("f1", "iou", "precision", "recall"):
            v=metrics[metric]
            rows.append((panel,metric,f'{v["registered_equal_category_point_delta"]:+.6f}',str(v["paired_bootstrap95"])))
    lines += _table(["Panel", "Selected-B metric", "Treatment minus control", "Paired 95%"],rows)
    lines += ["Assessment selected-B F1 delta is -0.037019, paired 95% [-0.160256, 0.086219];",
        "the interval spans zero, so aggregate native harm is not established. Charged",
        "changes by +0.066017 and mixed by -0.140054. Assessment has two collisions per",
        "category, separate from fit12. Intervals use 2,000 shared seed5001 whole-collision",
        "draws and equal-category weighting; they do not account for adaptive selection.", ""]
    depth=value["per_generation"];rows=[]
    for row in depth["rows"]:
        v=dict(zip(depth["columns"],row))
        if v["panel"]!="assessment4" or v["metric"] not in ("f1","soft_edge_f1"):continue
        rows.append((v["category"],"H"+str(v["height"]),v["metric"],v["delta"] if v["delta"] is not None else "unavailable",str(v["paired95"]),v["known_pairs"],v["nominal_pairs"]))
    lines += _table(["Category", "Height", "Metric", "Delta", "Paired 95%", "Known", "Nominal"],rows)
    lines += ["The download retains source F1/IoU, precision/recall and hard/soft membership",
        "edge quality for fit and assessment separately, with known and nominal support.",
        "Exact recursive topology+PID, topology-only and inclusive source recovery remain",
        "0/8 assessment B trials per arm (4 charged, 4 mixed), for selected top1 and actual",
        "oracle@1/@4 separately. Fit remains 0/24 per arm. Oracle pools are diagnostics.", "",
        "There are 26 eligible generated target occurrences among 672 nominal slots;",
        "646 are unavailable. Fourteen of 96 generated states were observed; 82 were",
        "unreached. Mixed generated support and direct H4 anchors are absent. Detector",
        "supervision covers 42/168 targets. Missing terms are not zero error.", "",
        "Independent coverage remains 0/12,000: each of charged, mixed, ccbar, uubar,",
        "ddbar and ssbar has 0/2,000 processed. Each continuum flavour's event/candidate",
        "acceptance, independent calibration and physical completeness remain unavailable.",
        "Existing development reservations cannot be relabelled TRAIN or independent.", "",
        "All 80 updates and 32 full-depth width4 coherent-forest beam rollouts completed,",
        "with GB101 H3 packing and the unchanged bounded postdecode donor transaction.",
        "Level1 uses eight candidates/query, later levels four; this adaptive descendant",
        "differs from the original default-four evaluation retained below. Truth enters",
        "supervision or post-inference scoring only. Architecture, reference-conditioned",
        "types, trained PID/confidence, train-only normalization and source uniqueness persist.", "",
        "Measured beam expansions: 195,553. Native latency totals: control 49.291 seconds,",
        "supervision 46.474 seconds, 16 rollouts each. Peak device allocation 281,338,880",
        "bytes is process-cumulative, not an isolated arm comparison. Allocated compute:",
        "3,232 CPU-seconds and 404 GPU-seconds; scheduler CPU usage 396.977 seconds.",
        "Publication overhead is separate and its runner CPU/GPU allocation is unavailable.", "",
        "Frozen source: ``bc196f7df9e3f5fcedd7bf2ae839f3710cdd8d92``; common checkpoint",
        "SHA-256: ``202b8a4999ab64827ad91bab7359ab52be157ec61c1e0967c3893fd780c78b91``.",
        "Together with the preservation result below, improved fit alone does not establish",
        "native benefit. Next: acquire role-separated mixed and matched charged generated",
        "support with checkpoint provenance and model soft PID. NAF metadata access has",
        "been restored; this does not establish new eligible independent coverage.", "",
        ":download:`Generated-source supervision quality, depth, support and cost <generic-b-source-supervision-v1.json>`", "",
        "Earlier results and all historical downloads follow.", ""]
    return lines


def _render_support_capture(value):
    title = "GB101 historical capture support: mixed and charged generated-state capture is available with sparse exact support"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. Saved 400 native H2/H3/H4 states from 32 exposed",
        "TRAIN collisions, 16 charged and 16 mixed, with model soft PID and checkpoint",
        "provenance. All 384 nominal primary context slots are accounted: 372 reached",
        "and 12 unreached. The 28 donor states are separate from primary coverage.", "",
        "This changes support readiness, not learned reconstruction quality. Of 709",
        "target-state occurrences, the current consumer reports 380 valid, 292 unavailable",
        "and 37 incompatible. Ten native-eligible exact-daughter occurrences cover four",
        "distinct H2 targets in four events; none are available at H3/H4. These are source",
        "membership support counts, not recursive topology/PID recovery or efficiency.", "",
        "All 13 mapping-unavailable events have unequal FSP and detector-source axis",
        "lengths. An explicit incidence relation is required; positional pairing cannot",
        "repair this. Corrected support yield remains unmeasured. Missing donor provenance",
        "also remains unavailable. Detector likelihood and model soft PID are distinct.", ""]
    rows=[]
    for category, item in value["native_by_category"].items():
        rows.append((category,item["nominal_events"],item["known_B_slots"],
            f'{item["selected_B_F1_nominal_lower"]:.6f}',
            f'{item["selected_B_IoU_nominal_lower"]:.6f}'))
    lines += _table(["TRAIN category", "Collisions", "Known B slots", "Selected-B F1", "Selected-B IoU"], rows)
    lines += ["These unpaired exposed-TRAIN observations establish no improvement, independent",
        "validation, calibration or promotion. Correlated states do not multiply events.",
        "The latest paired learned comparison follows below; all historical full-depth",
        "greedy, beam top1 and oracle results and downloads are retained.", "",
        "Independent coverage remains 0/12,000: 0/2,000 each for charged, mixed, ccbar,",
        "uubar, ddbar and ssbar. Each continuum flavour's event and candidate acceptance",
        "and physical completeness remain unavailable. No TRAIN data is relabelled as",
        "independent assessment and no sealed reservation is opened.", "",
        "Capture completed 32 rollouts and 707 forwards with zero optimizer updates and",
        "zero normalization fits. Beam expansions: 171,616. Native inference latency totals",
        "92.970 seconds (2.905 seconds per collision). Peak device allocation: 81,809,920",
        "bytes; peak host RSS: 2,095,544 KiB. Allocated compute: 1,208 CPU-seconds and",
        "151 GPU-seconds; scheduler CPU usage: 141.695 seconds. Publication costs are",
        "separate; runner CPU/GPU allocations are unavailable.", "",
        "Frozen capture source: ``a67f133f4ab531366ac5a7111ba77dcb21dac165``.",
        "The download records checkpoint hash and aggregate depth/support telemetry.",
        "Truth enters supervision or scoring only, after unchanged native inference.",
        "Next: relation-aware saved-state supervision, followed by an independently",
        "reviewed generated-state decoder/membership experiment with known-label masks.",
        "Training is not yet admitted; no model promotion.", "",
        ":download:`Generated-state capture support, native quality and cost <generic-b-support-capture-v1.json>`", ""]
    return lines


def _render_relation_support(value):
    title = "GB101 latest support correction: explicit relations resolve 292 unavailable support occurrences"
    lines = [title, "-" * len(title), "",
        "Reviewed 14 November 2026. Explicit saved FSP-to-detector incidence relations",
        "resolve all 292 previously mapping-unavailable target-state occurrences, without",
        "replaying capture or inference. All 400 states and 709 occurrences were processed:",
        "672 valid and 37 incompatible. Thirteen events have unequal source axes; no",
        "positional pairing or replacement observations were used.", "",
        "Complete eligible daughter support rises from four to 13 distinct H2 targets",
        "in 12 exposed TRAIN collisions, with 41 occurrences. None exists at H3/H4.",
        "The unchanged 32-collision roster contains 16 charged and 16 mixed collisions.",
        "All 384 primary slots remain accounted:372 reached,12 unreached;28 donor states",
        "are separate. Donor-chain provenance remains separately available or unavailable.", "",
        "Known membership labels comprise 1,059 positive and 23,809 disjoint-negative",
        "operands;8,312 unknown and674 incompatible operands remain masked. These",
        "are correlated target-state observations, not independent events, recursive",
        "topology/PID recovery or physical completeness. Partial labels do not invent",
        "complete deep targets. Truth is used only after inference for supervision.", "",
        "This correction changes training readiness, not measured reconstruction quality.",
        "There were zero new forwards, rollouts, updates or normalization fits. Native",
        "quality observations and the latest paired learned comparison remain below.",
        "No paired gain interval, independent validation, neural benefit or promotion",
        "is established. Independent coverage remains0/12,000,0/2,000 per category.",
        "Each continuum flavour's event/candidate acceptance and physical completeness",
        "remain unavailable. No exposed TRAIN data is relabelled independent validation.", "",
        "The correction used36 allocated CPU-seconds and zero GPU. Scheduler CPU9.532",
        "seconds and application CPU5.590611 seconds are alternative usage counters.",
        "Peak host RSS493,580 KiB. Runtime cgroup and scratch telemetry were not sampled.",
        "Publication runner allocation is unavailable and separate from scientific cost.", "",
        "Corrected support source: ``868f8b5444697fe26da73d914fde38081adf30d9``.",
        "Next: independently review matched detector-anchor training with generated",
        "relation-masked decoder-membership coefficient0 versus1,40 updates per arm.",
        "Training remains unadmitted. Historical support counts below describe the",
        "original consumer; this correction supersedes their mapping-unavailable yield.", "",
        ":download:`Corrected support, category/depth counts and compute <generic-b-relation-support-v1.json>`", ""]
    return lines


def _render_decoder_native(value):
    title = "GB101 preceding learned result: membership fit improves while native assessment worsens"
    lines = [title, "-" * len(title), "",
        "Reviewed 15 November 2026. Two matched 40-update decoder arms completed",
        "with generated membership coefficient 0 versus 1, identical detector anchors,",
        "60 trainable tensors,470 assignments across372 primary states,384 nominal slots.",
        "Generated TRAIN32 and detector fit12 were disjoint; assessment4 was excluded",
        "from all new fitting. This is adaptive ancestral TRAIN, not independent validation.", "",
        "Generated nominal loss improved from0.9103 to0.2242, while detector eligible",
        "BCE worsened from0.7789 to0.8948. Saved-state fit did not translate into benefit.",
        "Assessment selected-B F1 fell from0.1813 to0.1411: delta -0.0402, paired95%",
        "interval[-0.0769,-0.0035]. Charged/mixed deltas:+0.0617/-0.1422.",
        "Selected-B IoU delta:-0.0219, interval[-0.0455,+0.0017]. Shared seed5001",
        "whole-collision draws (2,000 replicates) use registered equal-category weights.",
        "Only four assessment collisions (two/category) limit interpretation.", "",
        "All32 native rollouts completed (fit12 plus assessment4 in each arm). Exact",
        "recursive topology+PID,topology-only and inclusive source recovery remained",
        "0/32 nominal B trials per arm, including oracle@1/@4 pools. Selected-B and",
        "depth source F1/IoU,membership edge quality,charged/mixed estimates,intervals",
        "and support counts are retained in the download. Oracle results are not deployable.",
        "The fixed full-depth GB101 descendant width4 policy and trained-state lineage",
        "were preserved. No inference/search intervention or model promotion occurred.", "",
        "Selected-B gain,mixed nonharm and mixed H3 source-F1 gates failed. Do not",
        "promote or repeat unchanged membership-only adaptation. Joint supported",
        "membership/object/type supervision is the next controlled hypothesis; this",
        "result does not prove that unsupervised head drift caused the native harm.",
        "Proposal quality and PID/confidence proper scores remain unavailable.",
        "Independent coverage remains0/12,000:0/2,000 in each required category.",
        "Each continuum flavour's event/candidate acceptance and physical completeness",
        "remain unavailable. TRAIN evidence is not relabelled independent validation.", "",
        "Actual allocation:9,928 CPU-seconds and1,241 GPU-seconds (0.344722 GPU-hours).",
        "Scheduler CPU1,228.928 seconds and application CPU1,222.206 seconds are",
        "alternative counters. Total32,893 forwards and196,296 beam expansions.",
        "Native control/treatment measured wall time:49.460/47.636 seconds for16",
        "rollouts each; native forwards371/358; expansions94,166/102,130.",
        "Peak host RSS2,577,160 KiB;device allocation275,480,064 bytes (cumulative peak).",
        "Publication runner allocation is unavailable and separate from scientific cost.", "",
        "Scientific source: ``3b399e65a1d74c771c3565180e33dcebc631b109``.",
        "Initialization SHA256: ``202b8a4999ab64827ad91bab7359ab52be157ec61c1e0967c3893fd780c78b91``.",
        "Next: prepare the matched joint-supervision candidate for independent review.",
        "Historical downloads and earlier comparisons below remain available.", "",
        ":download:`fit12-shallow depth metrics <generic-b-decoder-fit12-shallow-v1.json>`", "",
        ":download:`fit12-deep depth metrics <generic-b-decoder-fit12-deep-v1.json>`", "",
        ":download:`assessment4-shallow depth metrics <generic-b-decoder-assessment4-shallow-v1.json>`", "",
        ":download:`assessment4-deep depth metrics <generic-b-decoder-assessment4-deep-v1.json>`", "",
        ":download:`Matched decoder fit, native quality, depth, endpoints and compute <generic-b-decoder-native-v1.json>`", ""]
    return lines


def _render_joint_native(value):
    return ['GB101 prior learned result: joint supervision gives a mixed native response', '----------------------------------------------------------------------------', '', 'Reviewed 15 November 2026. Two matched 40-update arms shared 72 trainable', 'tensors, optimizer controls, detector anchors and generated membership labels.', 'Only the generated object/type loss coefficient differed: zero versus one.', '37 complete primary H2 occurrences cover 13 targets in 12 TRAIN collisions;', 'four donor occurrences were excluded. At that training freeze no complete H3/H4 labels were established; current capture above adds limited H3 support.', 'Unknown membership stays masked; partial deep labels are not complete targets.', '', 'Positive-only joint object/type nominal loss decreased from 0.2595 to 0.0279.', 'This supported teacher fit is not independent calibration. On four adaptive', 'TRAIN assessment collisions, selected-B F1 increased from 0.0875 to 0.1037:', 'delta +0.0162, paired 95% interval [0, 0.0324]. Charged/mixed deltas were', '+0.0054/+0.0269. Selected-B IoU delta was +0.0113, interval [0, 0.0226].', 'Assessment H2/H3 source-F1 deltas were +0.1920/+0.1065; soft-edge-F1 deltas', 'were +0.0708/+0.0528. Registered exploratory point-margin checks passed.', 'Shared seed5001 whole-collision draws (2,000 replicates) and equal category', 'weights retain the registered uncertainty. The interval touches zero.', '', 'The response is mixed: fit12 selected-B F1 decreased 0.0162 and H3 source-F1', 'fell 0.0835, paired 95% interval [-0.1446, -0.0268]. Detector eligible BCE', 'was 0.8951/0.9071 for control/treatment. Neither uniform improvement nor a', 'cause for the deep harm is established. No model promotion occurred.', '', 'All 32 native rollouts completed: fit12 plus assessment4 for each final40', 'checkpoint, 32 nominal B trials per arm. Exact recursive topology+PID,', 'topology-only and inclusive recovery remained 0/32, including oracle@1/@4.', 'Source F1/IoU, precision/recall, membership edge quality and category/depth', 'support are retained in downloads. Oracle pools are not deployable results.', 'This fixed full-depth coherent-forest width4 GB101 descendant uses eight', 'level1 candidates/query; it is not the original default-four evaluation.', 'Checkpoint architecture, normalization, reference type heads, trained PID', 'and confidence state, recursive source uniqueness and truth-free input persist.', '', 'Assessment4 and fit12 remain adaptive ancestral TRAIN; assessment4 was excluded', 'from all new fitting. Independent coverage remains 0/12,000: 0/2,000 for each', "required category. Every continuum flavour's event/candidate acceptance,", 'physical completeness, native proposal quality and native PID/confidence proper', 'scores remain unavailable. These findings do not establish generalization.', '', 'Actual science allocation: 9,600 CPU-seconds and 1,200 V100-seconds (0.333333', 'GPU-hours). Scheduler CPU 1,190.697 seconds and application CPU 1,184.033', 'seconds are alternative counters. Total 32,886 forwards and 203,315 beam', 'expansions. Control/treatment native wall time: 48.747/49.273 seconds for 16', 'rollouts each; native forwards 372/350; expansions 105,362/97,953. Cumulative', 'peak host RSS 2,540,276 KiB; device allocation 293,099,520 bytes. Publication', 'runner allocation is unavailable and separate from scientific cost.', '', 'Scientific source: ``13b9c4fc1f64c747c9aba310fb4bc49e197c3348``.', 'Initialization SHA256: ``202b8a4999ab64827ad91bab7359ab52be157ec61c1e0967c3893fd780c78b91``.', 'Final40 checkpoint hashes and full aggregate metrics are in the download.', 'Next: prepare matched current-checkpoint TRAIN32 support capture, at most 64', 'rollouts and 768 nominal primary slots, for independent scientific admission.', 'That historical preparation is now completed; current support yield is reported above. Do not repeat', 'unchanged training. Historical results and downloads remain available below.', '', ':download:`joint-fit12-shallow aggregate metrics <generic-b-joint-fit12-shallow-v1.json>`', '', '', ':download:`joint-fit12-deep aggregate metrics <generic-b-joint-fit12-deep-v1.json>`', '', '', ':download:`joint-assessment4-shallow aggregate metrics <generic-b-joint-assessment4-shallow-v1.json>`', '', '', ':download:`joint-assessment4-deep aggregate metrics <generic-b-joint-assessment4-deep-v1.json>`', '', '', ':download:`joint-native aggregate metrics <generic-b-joint-native-v1.json>`', '']


def _render_refresh_support(value):
    return ['GB101 current checkpoint exposes limited H3 support', '---------------------------------------------------', '', 'Reviewed 15 November 2026. Matched post-training capture completed 64 full-depth', 'native width4 rollouts on exposed TRAIN32, using both frozen final40 checkpoints.', 'All 768 nominal primary slots are accounted: 720 reached, 48 unreached and zero', 'unprocessed. The 58 donor states remain separate; ambiguous links: zero.', 'No optimizer update or normalization fit occurred in this capture.', '', 'Both checkpoints expose 37 complete eligible primary H2 daughter-set occurrences', 'covering 13 targets in 12 TRAIN collisions. Treatment additionally exposes four', 'complete eligible primary H3 occurrences of ONE target in ONE mixed collision;', 'control exposes zero. Four correlated states are not four independent targets.', 'Neither checkpoint establishes complete H4 primary support. These observations', 'correct the earlier readiness statement; recursive topology/PID support and', 'physical completeness are still unavailable. Unknown membership stays masked;', 'partial deep labels never become complete labels.', '', 'Selected-B F1 is 0.17187012 control and 0.21582420 treatment: delta +0.04395409,', 'paired 95% interval [-0.01572835, 0.09788245]. Charged/mixed deltas are', '+0.03434187/+0.05356630. Selected-B IoU is 0.10799378/0.13543342:', 'delta +0.02743964, interval [-0.01184368, 0.06219930]. Shared seed5001', 'whole-collision draws, 2,000 replicates and equal charged/mixed weights apply.', 'Both intervals include zero. These are adaptive TRAIN support observations,', 'not independent efficacy or generalization. No promotion occurred.', '', 'Exact recursive topology+PID, topology-only and inclusive recovery remain', '0/64 nominal B trials per checkpoint, including oracle@1/@4 pools. Downloads', 'retain category endpoints and per-generation source F1/IoU, precision/recall', 'and membership-sensitive edge quality, including unavailable cells and nominal', 'bounds. No all-nominal depth gain is claimed. Native proposal quality and', 'PID/confidence proper-score closeouts remain unavailable.', '', 'Independent coverage remains 0/12,000: 0/2,000 for charged, mixed and each', "continuum flavour. Each flavour's event and candidate acceptance is unavailable.", 'The unchanged coherent-forest beam is the GB101 descendant policy: width4,', 'eight level1 candidates/query and four at later levels. It preserves H3 packing,', 'architecture flags, normalization, reference-conditioned type heads, trained', 'PID/confidence state and recursive source uniqueness. Truth joins follow', 'inference and never generate, rank, prune or stop the native search.', '', 'Actual capture allocation: 2,496 CPU-seconds and 312 V100-seconds (0.086667', 'GPU-hours); 1,377 forwards and 425,182 beam expansions, zero optimizer updates.', 'Latency and memory observations are retained in the aggregate download.', 'Publication runner allocations are unavailable and separate from science.', 'Scientific source: ``8b00caca7a2aaa627cdc8def447ed7684fcb950e``.', 'Both final40 checkpoint SHA256 values are in the download.', '', 'At that capture stage, the learned comparison was joint supervision below:', 'small adaptive assessment gain with fit-panel H3 harm. Capture adds support', 'readiness evidence, not another training result. Next: prepare common treatment40', 'initialization and matched stale-versus-refreshed joint supervision, two finite', '40-update arms and 32 native rollouts, for independent scientific admission.', 'This package does not isolate H3 causality. Historical downloads remain available.', '', ':download:`refresh-native aggregate metrics <generic-b-refresh-native-v1.json>`', '', '', ':download:`refresh-shallow aggregate metrics <generic-b-refresh-shallow-v1.json>`', '', '', ':download:`refresh-deep aggregate metrics <generic-b-refresh-deep-v1.json>`', '']


def _render_balanced_refresh(value):
    return ['GB101 support refresh improves selected B but harms mixed H3', '------------------------------------------------------------', '', 'Reviewed 15 November 2026. The latest learned comparison completed two matched', '40-update arms and 32 full-depth native rollouts. Both shared treatment40', 'initialization, 72 trainable decoder/object/type tensors, AdamW controls,', 'detector anchors and inference. The changed factor was stale versus refreshed', 'generated joint supervision. Unknown membership remained masked. Stale/refreshed', 'banks had 470/448 membership targets and 37/41 complete object/type occurrences;', 'the four H3 occurrences cover only one target in one mixed TRAIN collision.', 'No complete H4 supervision is established.', '', 'On four adaptive TRAIN assessment collisions, selected-B F1 increased from', '0.07875458 to 0.13052399: delta +0.05176941, paired 95% interval', '[0.04798327, 0.05555556]. Charged improved while mixed selected-B F1 declined.', 'The registered scientific gate FAILED: assessment mixed H3 source-F1 declined', '0.03296703, below the -0.02 nonharm margin, with paired 95% interval', '[-0.14285714, 0.07692308]. This is a tradeoff, not uniform improvement.', '', 'Fit12 mixed selected-B F1 declined 0.03393834; mixed H3 source-F1 declined', '0.11507708 and soft-edge F1 declined 0.08428858. Assessment H4 source-F1', 'declined 0.06118881. Undefined H5/H6 quality stays unavailable. Shared seed5001', 'whole-collision draws (2,000 replicates) and equal charged/mixed weights apply.', 'Downloads retain source precision, recall, F1, IoU and membership-sensitive edge', 'quality by generation and category, with unavailable support and nominal bounds.', '', 'On the same refreshed bank, membership loss improved 0.30517581 to 0.12408081', 'and joint loss improved 0.06037292 to 0.00772501. Supported mixed H3 teacher', 'object-positive BCE improved 0.70299052 to 0.14120397; type NLL improved', '9.57405901 to 0.02744599. These positive-only teacher scores are not calibration.', 'Stale-bank membership loss worsened 0.12330238 to 0.19203068; detector eligible', 'BCE worsened 0.64305113 to 0.65951485. Teacher/native divergence and cross-bank', 'fit loss motivate replay; they do not establish a cause for the native harm.', '', 'Exact recursive topology+PID, topology-only and inclusive recovery remain', '0/32 nominal B trials per arm, including actual oracle@1/@4 pools. Each arm', 'processed fit12 plus assessment4, with 24 plus 8 nominal B slots. No oracle', 'promotion occurred. Assessment remains adaptive ancestral TRAIN excluded from', 'this fitting. Independent coverage is 0/12,000 (0/2,000 each charged, mixed,', "ccbar, uubar, ddbar and ssbar). Each continuum flavour's event and candidate", 'acceptance, physical completeness and native proposal/proper scores remain', 'unavailable. No model promotion or generalization claim is supported.', '', 'Inference retains the GB101 descendant coherent-forest beam: width4 with', 'eight candidates/query at level1 and four thereafter, unchanged H3 packing,', 'checkpoint architecture flags, normalization, reference-conditioned type heads,', 'trained PID/confidence provenance and recursive source uniqueness. This is not', 'the original default-four/query evaluation. Truth joins follow native inference', 'and never generate, rank, prune or stop search.', '', 'Actual science allocation: 10,144 CPU-seconds and 1,268 V100-seconds', '(0.352222 GPU-hours); 33,344 forwards, 156,473 beam expansions, 80 optimizer', 'updates and 32 rollouts. Scheduler CPU: 1,255.299 seconds; application CPU,', 'latency and cumulative memory observations are retained in the download.', 'Publication runner allocations are unavailable and separate from science.', 'Scientific source: ``ee322bc7c8a7b2abbd12c6aadda6bf8c6b66ed9d``.', 'Final40 checkpoint SHA256 values are retained in the download.', '', 'Next: independently admit ONE fixed 50/50 stale/refreshed joint-supervision', 'replay arm, common initialization, 40 updates and 16 native rollouts. Reuse the', 'completed refreshed-only control and stale reference without replay. Accumulate', 'both bank gradients before one clip and optimizer step; detector loss unchanged.', 'Balanced replay efficacy is unmeasured. No coefficient sweep or promotion.', 'Historical scientific results and downloads remain available below.', '', ':download:`balanced-refresh-native aggregate metrics <generic-b-balanced-refresh-native-v1.json>`', '', '', ':download:`balanced-refresh-teacher-comparison aggregate metrics <generic-b-balanced-refresh-teacher-comparison-v1.json>`', '', '', ':download:`balanced-refresh-endpoints-by-category aggregate metrics <generic-b-balanced-refresh-endpoints-by-category-v1.json>`', '', '', ':download:`balanced-refresh-fit12-shallow aggregate metrics <generic-b-balanced-refresh-fit12-shallow-v1.json>`', '', '', ':download:`balanced-refresh-fit12-deep aggregate metrics <generic-b-balanced-refresh-fit12-deep-v1.json>`', '', '', ':download:`balanced-refresh-assessment4-shallow aggregate metrics <generic-b-balanced-refresh-assessment4-shallow-v1.json>`', '', '', ':download:`balanced-refresh-assessment4-deep aggregate metrics <generic-b-balanced-refresh-assessment4-deep-v1.json>`', '']


def _render_replay(value):
    return ['GB101 balanced replay repairs mixed H3 but harms charged reconstruction', '-----------------------------------------------------------------------', '', 'Reviewed 15 November 2026. One fixed 50/50 stale/refreshed replay arm completed', '40 updates and 16 full-depth native rollouts. Completed refreshed-only and stale', 'reference results were reused. Common treatment40 initialization, 72 trainable', 'decoder/object/type tensors, detector loss, optimizer, masks and native policy', 'were preserved; both bank half-weight gradients accumulated before each step.', '', 'Assessment selected-B F1 fell from 0.13052399 to 0.08229167, delta -0.04823232,', 'paired 95% interval [-0.09090909, -0.00555556]. Mixed-H3 source F1 improved', '+0.09379856 versus refreshed-only, but charged-H3 and selected-B nonharm failed.', 'The scientific gate FAILED. Stale-bank teacher fit improved without a uniform', 'native repair. The prior support-refresh gain and this replay harm are both', 'retained; balanced replay is not promoted. These are adaptive TRAIN outcomes.', '', 'Downloads retain selected-B F1/IoU and source precision, recall, F1/IoU,', 'membership-sensitive edge quality by generation and charged/mixed category,', 'nominal bounds, support and whole-collision intervals. Shared seed5001 draws,', '2,000 replicates and equal-category weighting apply. Fit12 and assessment4', 'remain distinct. Assessment4 is excluded from current fitting but remains', 'ancestrally exposed adaptive TRAIN, not independent validation.', '', 'Exact recursive topology+PID, topology-only and inclusive recovery remain', '0/32 nominal B trials per model across fit12 plus assessment4, including actual', 'oracle@1/@4 pools. All 16 new rollouts completed. No oracle promotion occurred.', 'Complete eligible H3 supervision still covers four occurrences of one target', 'in one mixed TRAIN collision; complete H4 support is unavailable. Occurrences', 'are correlated and are not additional independent targets or physical decays.', '', 'Independent coverage remains 0/12,000: 0/2,000 each charged, mixed, ccbar, uubar,', "ddbar and ssbar. Every continuum flavour's event and candidate acceptance,", 'physical completeness and native proposal/proper scores remain unavailable.', 'Missing metrics are not zero. No generalization or promotion is supported.', '', 'The unchanged descendant coherent-forest beam uses width4, eight candidates', 'per query at generation1 and four thereafter, H3 packing and bounded donor', 'transactions. Architecture flags, normalization, reference-conditioned type', 'heads, trained PID/confidence state and recursive source uniqueness are kept.', 'Truth joins follow inference; truth never generates, ranks, prunes or stops it.', '', 'New science allocation: 9,040 CPU-seconds and 1,130 V100-seconds', '(0.313889 GPU-hours), 31,458 forwards and 85,612 beam expansions. Native rollout', 'operation wall was 43.270365 seconds total; it is not inference-only latency.', 'Application peak host RSS was 2,629,316 KiB and cumulative device peak', '236,640,768 bytes. Reused control compute is not charged again.', 'Scientific source: ``ea865e3e0a998898b6c04016b9653f2b7f8bc516``.', 'Publication runner allocations remain unavailable and separate from science.', '', 'Next: prepare 64 additional eligible distinct TRAIN collisions, 32/category,', 'with the fixed treatment40 generating checkpoint and unchanged capture policy.', 'Freeze exact roles, source mappings and 768 nominal H2/H3/H4 capture slots for', 'independent admission. Support yield is unmeasured; this preparation does not', 'claim scarcity caused the observed harm. Follow actual yield with a matched', 'unique-support neural intervention. Historical findings and downloads follow.', '', ':download:`replay-native aggregate metrics <generic-b-replay-native-v1.json>`', '', '', ':download:`replay-teacher aggregate metrics <generic-b-replay-teacher-v1.json>`', '', '', ':download:`replay-endpoints aggregate metrics <generic-b-replay-endpoints-v1.json>`', '', '', ':download:`replay-fit12-shallow aggregate metrics <generic-b-replay-fit12-shallow-v1.json>`', '', '', ':download:`replay-fit12-deep aggregate metrics <generic-b-replay-fit12-deep-v1.json>`', '', '', ':download:`replay-assessment4-shallow aggregate metrics <generic-b-replay-assessment4-shallow-v1.json>`', '', '', ':download:`replay-assessment4-deep aggregate metrics <generic-b-replay-assessment4-deep-v1.json>`', '']


def _render_expanded96(value):
    return ['GB101 broader TRAIN support retains charged and mixed tradeoff', '--------------------------------------------------------------', '', 'Reviewed 15 November 2026. Expanding generated support from32 to96 distinct', 'TRAIN collisions completed40 updates,15360 nominal slot presentations and16', 'full-depth native rollouts. Initialization,72 trainable decoder/object/type', 'tensors,AdamW,detector anchors,masks,assignments and inference stayed fixed.', 'The completed refreshed32 and stale controls were reused without new compute.', '', 'Assessment selected-B F1 fell from0.13052399 to0.08235931;paired95% interval', 'for the difference[-0.09632937,0.00000000]. Mixed-H3 source F1 improved', '+0.09371550,while charged-H3 fell-0.28465909. Selected-B gain and charged', 'selected-B/H3 nonharm failed. The scientific gate FAILED;no promotion.', 'Teacher losses improved,but broader support did not resolve native category', 'tradeoffs. This accumulated evidence changes the next intervention away from', 'repeating support-distribution-only training. Gradient interference is an', 'unproven hypothesis;no causal explanation is established.', '', 'Downloads retain selected-B F1/IoU,source precision/recall/F1/IoU and membership', 'edge quality by generation/category,paired intervals and nominal bounds.', 'Fit12 and assessment4 remain separate. Shared seed5001 whole-collision draws,', '2000 replicates and equal-category weighting apply. Assessment4 is excluded', 'from current fitting but is ancestrally exposed adaptive TRAIN;it cannot', 'establish independent validation. Exact recursive topology+PID,topology-only', 'and inclusive recovery remain0/32 nominal B trials per model,including', 'oracle@1/@4. All16 new rollouts completed. No oracle promotion occurred.', '', 'Support acquisition added64 distinct TRAIN collisions,32/category;all768', 'nominal slots are accounted for:732 reached,36 unreached. Combined TRAIN96', 'has1099 reached of1152 nominal slots,31 complete eligible H2 targets and two', 'complete eligible H3 targets. Complete H4 labels remain unavailable. Partial', 'deep labels stay explicit;unknown membership remains masked. Target counts', 'refer to retained detector support,not complete physical decays.', '', 'Independent coverage remains0/12000:0/2000 each charged,mixed,ccbar,uubar,', "ddbar and ssbar. Each continuum flavour's event/candidate acceptance,", 'physical completeness and native proposal/proper scores remain unavailable.', 'No generalization or model promotion is supported. Truth joins follow native', 'inference;truth never generates,ranks,prunes or stops the coherent forest.', 'The unchanged adaptive GB101 descendant uses width4,H3 packing,bounded donor', 'transactions,eight candidates/query at generation1 and four thereafter.', 'This is distinct from the original requested default four/query diagnostic.', '', 'New TRAIN96 science allocation:5864 CPU-seconds and733 V100-seconds', '(0.203611 GPU-hours),18797 forwards and98374 beam expansions. Support capture', 'separately used2336 CPU-seconds,292 V100-seconds,1363 forwards and408053', 'beam expansions,zero updates. Reused controls are not charged again.', 'Scientific source: ``cc941f9e47ad1c9d66673b3f345ccba4fa2ae3c4``.', 'Application latency and peak memory are retained in the aggregate download;', 'native operation wall includes more than inference. Publication runner', 'allocations remain unavailable and separate from scientific compute.', '', 'Next: prepare one matched40-update TRAIN96 category-gradient projection arm,', '15360 nominal presentations and16 native rollouts,for independent admission.', 'Keep the common initialization,supported labels and inference fixed. Measure', 'opposing generated gradients and realized AdamW displacement;raw projection', 'does not guarantee finite-step nonharm. No coefficient or architecture sweep.', 'Historical findings and downloads follow.', '', ':download:`expanded96-native aggregate metrics <generic-b-expanded96-native-v1.json>`', '', '', ':download:`expanded96-teacher aggregate metrics <generic-b-expanded96-teacher-v1.json>`', '', '', ':download:`expanded96-endpoints aggregate metrics <generic-b-expanded96-endpoints-v1.json>`', '', '', ':download:`expanded96-fit12-shallow aggregate metrics <generic-b-expanded96-fit12-shallow-v1.json>`', '', '', ':download:`expanded96-fit12-deep aggregate metrics <generic-b-expanded96-fit12-deep-v1.json>`', '', '', ':download:`expanded96-assessment4-shallow aggregate metrics <generic-b-expanded96-assessment4-shallow-v1.json>`', '', '', ':download:`expanded96-assessment4-deep aggregate metrics <generic-b-expanded96-assessment4-deep-v1.json>`', '']


def render(value):
    """Render completed results separately from current operational state."""
    gb001, gb002, cr054, gb101 = value["gb001"], value["gb002"], value["cr054"], value["gb101"]
    static = gb001["static_compatibility"]
    selected = gb101["selected_b_development"]
    latest = gb101["latest_beam4"]
    diagnosis = gb101["saved_beam_diagnosis"]
    cap = gb101["cap256_intervention"]
    adaptation = gb101["model_adaptation_series"]
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
        "Saved-beam diagnosis: retained-pool quality, not final ranking",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
        "The reviewed 160-row saved-output diagnosis separates quality retained anywhere in the width-4",
        "pool from the deployable top-1 choice. On development observations, equal-category source F1 is",
        "0.228930 for greedy, 0.020833 for beam top-1 and 0.035256 for the post-inference retained-pool",
        "oracle. Pool minus top-1 is +0.014423 with the shared-seed-5001 paired 95% interval",
        "[0.000000, 0.043269]. Because its lower bound is not strictly positive, the registered",
        "ranking-opportunity rule does not pass. This does not establish that ranking can never help.", "",
        "Pool minus greedy source F1 is -0.193674, interval [-0.285994, -0.107103], establishing an",
        "exploratory retained-pool deficit. Source IoU and membership-edge F1 show the same pattern:",
        "pool-minus-greedy -0.134416 [-0.201777, -0.073319] and -0.056456",
        "[-0.086067, -0.029583], respectively. The oracle uses truth only after inference and remains",
        "diagnostic and unpromoted.", "",
    ]
    diagnosis_states = diagnosis["development_equal_category"]["states"]
    lines += _table(["Development equal-category state", "Source F1", "Source IoU", "Membership-edge F1"], [
        ["Greedy", f"{diagnosis_states['greedy']['source_f1']:.6f}",
         f"{diagnosis_states['greedy']['source_iou']:.6f}",
         f"{diagnosis_states['greedy']['soft_edge_f1']:.6f}"],
        ["Beam top-1", f"{diagnosis_states['beam_top1']['source_f1']:.6f}",
         f"{diagnosis_states['beam_top1']['source_iou']:.6f}",
         f"{diagnosis_states['beam_top1']['soft_edge_f1']:.6f}"],
        ["Retained-pool oracle", f"{diagnosis_states['retained_pool_oracle']['source_f1']:.6f}",
         f"{diagnosis_states['retained_pool_oracle']['source_iou']:.6f}",
         f"{diagnosis_states['retained_pool_oracle']['soft_edge_f1']:.6f}"],
    ])
    category_f1 = diagnosis["development_source_f1_by_category"]
    lines += _table(["Development category", "Greedy source F1", "Beam top-1", "Retained-pool oracle"], [
        [category, f"{row['greedy']:.6f}", f"{row['beam_top1']:.6f}",
         f"{row['retained_pool_oracle']:.6f}"]
        for category, row in category_f1.items()
    ])
    continuum = diagnosis["continuum_acceptance"]
    lines += _table(["Continuum flavour", "Top-1 events", "Top-1 candidates",
                     "Retained-union events", "Retained-union candidates"], [
        [category, f"{row['top1_events']}/{row['processed']}", row["top1_candidates"],
         f"{row['union_k4_events']}/{row['processed']}", row["union_k4_candidates"]]
        for category, row in continuum.items()
    ])
    lines += [
        "The retained union is a post-inference diagnostic pool, not deployable background acceptance.",
        "Across the combined TRAIN and development signal diagnostics, 40/48 collisions have no B root",
        "anywhere in the retained pool, 42/48 have none at top-1, and only 2/48 have a B root below",
        "top-1. The query-expansion cap was hit in 43/48. These counts are not development-only, and cap",
        "saturation is associative rather than proof of causal pruning loss. Quality of forests pruned",
        "before the saved pool and later survival of the H3 nominal lane are unavailable.", "",
        "The successful saved-output analysis used 8.44 process CPU-seconds, 8.49 wall-seconds and",
        "739,076 KiB peak RSS. The preserved compatibility failure used 6.10 CPU-seconds, 6.14",
        "wall-seconds and 739,120 KiB peak RSS. No new inference, training or scheduler job was run.",
        "The resulting cap intervention is reported below; it supersedes the earlier recommendation to",
        "increase this query cap, while retaining the saved-pool diagnosis as evidence.", "",
        "Controlled query-expansion cap intervention",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
        "The reviewed intervention changed only the per-query expansion cap from 128 to 256. The GB100",
        "H3-adapted checkpoint, GB101 nominal-lane adapter, width-4 beam, normalization, trained heads and",
        "query-log scoring remained fixed. All 80 reused exploratory collisions completed: 32 TRAIN",
        "diagnostics and 48 development observations (8/category), with zero failed, incompatible or",
        "unavailable rows inside this panel. Strict eligible coverage remains 0/2,000 per category.", "",
    ]
    top1 = cap["development_equal_category"]["top1_source_f1"]
    pool = cap["development_equal_category"]["retained_pool_oracle_source_f1"]
    h3 = cap["development_equal_category"]["height3_retained"]
    lines += _table(["Development equal-category endpoint", "Cap 128", "Cap 256", "Delta and seed-5001 paired 95% interval"], [
        ["Top-1 selected-B source F1", f"{top1['cap128']:.6f}", f"{top1['cap256']:.6f}",
         f"{top1['delta']:+.6f} [{top1['lower95']:.6f}, {top1['upper95']:.6f}]"],
        ["Retained-pool oracle source F1", f"{pool['cap128']:.6f}", f"{pool['cap256']:.6f}",
         f"{pool['delta']:+.6f} [{pool['lower95']:.6f}, {pool['upper95']:.6f}]"],
        ["Height-3 retained proposal source F1", "—", "—",
         f"{h3['source_f1_delta']['point']:+.6f} [{h3['source_f1_delta']['lower95']:.6f}, {h3['source_f1_delta']['upper95']:.6f}]"],
        ["Height-3 retained proposal source IoU", "—", "—",
         f"{h3['source_iou_delta']['point']:+.6f} [{h3['source_iou_delta']['lower95']:.6f}, {h3['source_iou_delta']['upper95']:.6f}]"],
    ])
    lines += [
        "Top-1 membership-edge F1 also had zero delta and interval [0, 0]. These degenerate empirical",
        "intervals on a small reused panel do not prove population equivalence. The +0.02 practical-gain",
        "rule and strictly-positive-interval rule both fail; no checkpoint, beam policy or oracle is promoted.",
        "Exact topology+PID, topology-only and inclusive-source recovery remain 0/32 development and",
        "0/64 TRAIN diagnostics for both top-1 and oracle@4.", "",
    ]
    background128 = cap["continuum_acceptance"]["cap128"]
    background256 = cap["continuum_acceptance"]["cap256"]
    lines += _table(["Continuum flavour", "Cap128 top-1 events/candidates", "Cap256 top-1 events/candidates",
                     "Cap128 retained union", "Cap256 retained union", "Nominal"], [
        [category,
         f"{background128[category]['top1_events']} / {background128[category]['top1_candidates']}",
         f"{background256[category]['top1_events']} / {background256[category]['top1_candidates']}",
         f"{background128[category]['union_events']} / {background128[category]['union_candidates']}",
         f"{background256[category]['union_events']} / {background256[category]['union_candidates']}",
         background128[category]["nominal"]]
        for category in ("ccbar", "uubar", "ddbar", "ssbar")
    ])
    cost = cap["search_and_cost"]
    lines += _table(["Measured search/cost quantity", "Cap 128", "Cap 256"], [
        ["Events hitting query cap", f"{cost['cap128']['cap_hit_events']}/80", f"{cost['cap256']['cap_hit_events']}/80"],
        ["Total query-cap hits", cost["cap128"]["cap_hits"], cost["cap256"]["cap_hits"]],
        ["Daughter partial states expanded", cost["cap128"]["daughter_partial_states_expanded"], cost["cap256"]["daughter_partial_states_expanded"]],
        ["Candidates generated", cost["cap128"]["candidates_generated"], cost["cap256"]["candidates_generated"]],
        ["Mean seconds/collision", f"{cost['cap128']['mean_latency_seconds']:.6f}", f"{cost['cap256']['mean_latency_seconds']:.6f}"],
    ])
    lines += [
        "Mean latency increased 16.229%. New inference consumed 546 allocated CPU-seconds, 348.904",
        "measured scheduler CPU-seconds and zero GPU-hours; peak batch RSS was 1,244,760 KiB. Successful",
        "aggregation used 267.26 wall-seconds, 263.03 CPU-seconds and 1,286,588 KiB peak RSS. An initial",
        "missing-PYTHONPATH aggregation launch is preserved; inference was not rerun for that repair.", "",
        "Increasing this cap changed search work but did not remedy the retained-pool deficit and increased",
        "ccbar top-1 acceptance from 1/8 to 2/8. Do not adopt cap256 or continue a cap sweep on this evidence.",
        "Other search limits still bind, so global cap irrelevance is not established. A finite H3",
        "nominal-lane survival/retention investigation is a candidate for separate Astra design, not an",
        "established causal diagnosis or an admitted experiment.", "",
        "Accumulated H2/H1 model adaptations",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
        "Six finite model adaptations reused the same 80-collision protected-H3 panel and fixed beam",
        "controls. Each fit used 32 cached TRAIN contexts and 32 updates. The table reports development",
        "selected-B source F1 against the protected-H3 parent (0.056629), with whole-collision",
        "seed-5001 paired intervals. Every result is exploratory: strict eligible coverage remains",
        "0/12,000, and none passed the registered gain and background rules.", "",
    ]
    lines += _table(["Adaptation", "Selected-B source F1", "Delta and paired 95% interval",
                     "Allocated CPU/GPU seconds", "Decision"], [
        [item["name"], f"{item['selected_b_source_f1']:.6f}",
         f"{item['delta']:+.6f} [{item['lower95']:.6f}, {item['upper95']:.6f}]",
         f"{item['allocated_cpu_seconds']} / {item['gpu_seconds']}", item["verdict"]]
        for item in adaptation["experiments"]
    ])
    lines += [
        "The latest positive-anchor fit improved positive-label BCE from 1.081553 to 0.303133 and",
        "ranking loss, but negative-label BCE worsened from 0.184610 to 0.943122 while mean negative",
        "probability rose from 0.105112 to 0.367449. Its source-F1 gain was only +0.010417",
        "[0, 0.031250], below the +0.02 margin; membership-edge F1 changed by +0.001429.", "",
        "Dedicated top-1 continuum acceptance was ccbar 1/8, uubar 1/8, ddbar 4/8 and ssbar 1/8.",
        "Relative to the protected-H3 parent, ddbar increased by 2/8 and failed the background",
        "guardrail. This is an association, not causal attribution. Exact topology+PID, topology-only",
        "and inclusive-source recovery stayed 0/32 development B slots for top-1 and oracle@4.",
        "No adaptation checkpoint, beam policy or oracle was promoted.", "",
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
        ":download:`Reviewed v7 GB001, GB002, CR054 and GB101 aggregates with accumulated H2/H1 adaptations <generic-b-status-summary-v7.json>`.",
        ":download:`Historical v6 aggregate with the cap256 intervention retained with its original bytes <generic-b-status-summary-v6.json>`.",
        ":download:`Historical v5 aggregate with the saved-beam diagnosis retained with its original bytes <generic-b-status-summary-v5.json>`.",
        ":download:`Historical corrected v4 aggregate retained with its original bytes <generic-b-status-summary-v4.json>`.",
        ":download:`Historical v3 aggregate retained with its original bytes <generic-b-status-summary.json>`.", "",
    ]
    return _render_expanded96(value["h2_anchor"]["expanded96-native"]) + _render_replay(value["h2_anchor"]["replay-native"]) + _render_balanced_refresh(value["h2_anchor"]["balanced-refresh-native"]) + _render_refresh_support(value["h2_anchor"]["refresh-native"]) + _render_joint_native(value["h2_anchor"]["joint-native"]) + _render_decoder_native(value["h2_anchor"]["decoder_native"]) + _render_relation_support(value["h2_anchor"]["relation_support"]) + _render_support_capture(value["h2_anchor"]["support_capture"]) + _render_source_supervision(value["h2_anchor"]["source_supervision"]) + _render_preservation(value["h2_anchor"]["preservation"]) + _render_retained_source(value["h2_anchor"]["retained_source"]) + _render_positive_object(value["h2_anchor"]["positive_object"]) + _render_h2_anchor(value["h2_anchor"]) + _render_supported_h3(value["supported_h3"]) + _render_neural_support(value["neural_support"]) + lines
