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
    return value | {
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
    return lines
