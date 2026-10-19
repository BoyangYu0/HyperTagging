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
        "schema": "generic-b-dashboard-public-binding-v1",
        "status_date": "2026-10-19",
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
    _require(value["schema"] == "generic-b-dashboard-public-summary-v1", "Generic-B schema differs")
    gb001, gb002, cr054 = value["gb001"], value["gb002"], value["cr054"]
    _require(set(gb001) == {"status", "evidence_class", "learned_model_evaluation",
                            "legal_decoder_reachability_assessed", "population", "static_compatibility",
                            "categories", "overlapping_recursive_blockers", "learned_metrics", "runtime"},
             "GB001 field allowlist differs")
    _require(set(gb002) == {"status", "result_state", "completed_scientific_result",
                            "snapshot_recorded_state", "current_state", "training_completed",
                            "full_evaluation_completed", "learned_metrics"},
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
    _require(gb002["status"] == "HISTORICAL_SUPERSEDED_OPERATIONAL_SNAPSHOT"
             and gb002["snapshot_recorded_state"] == "OPERATIONAL_PREPARATION_ADMITTED"
             and gb002["current_state"] == "VERSIONED_B_ONLY_REPLANNING"
             and gb002["result_state"] == "NO_COMPLETED_TRAINING_OR_LEARNED_EVALUATION"
             and gb002["completed_scientific_result"] is False
             and gb002["training_completed"] is False
             and gb002["full_evaluation_completed"] is False
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
        "schema": "generic-b-dashboard-public-download-v1",
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
        "GB002: superseded snapshot and current replanning status",
        "--------------------------------------------------------", "",
        f"The recorded {gb002['snapshot_recorded_state'].replace('_', ' ')} state is a historical",
        "and now-superseded operational snapshot. It does not describe the current plan and is not an execution",
        "approval. The current GB002 state is versioned B-only replanning, with no completed training or",
        "learned evaluation. This is not a completed scientific result. Every learned",
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
