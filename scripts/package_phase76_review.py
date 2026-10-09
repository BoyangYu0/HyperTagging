"""Lossless aggregate-only delivery of the single completed successor."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def package(root, destination):
    value = json.loads((root / "phase76-review-public.json").read_text())
    if (
        value["status"] != "COMPLETED"
        or value["stage"] != "development"
        or value["primary_eligible"]
        or set(value["arms"]) != {"partial_context_on", "partial_context_off"}
    ):
        raise ValueError("Wrong completed development study")
    contract = json.loads((root / "successor-campaign-v1/contract.json").read_text())
    value["controls"] = {
        "context_width": 128,
        "hyperbolic_width": 32,
        "depth": 4,
        "common_head_width": 256,
        "encoder_initialization_sha256": contract["initial_checkpoint"]["sha256"],
        "shared_inherited_pretraining_updates": 1000,
        "shared_inherited_pretraining_presentations": 8000,
        "new_pretraining_updates": 0,
        "settings": contract["settings"],
        "resources_per_arm": contract["resources"],
        "original_corpus_events": 70000,
        "train_subset_events": 1536,
        "fresh_heldout_events": 600,
        "primary_reservations": 0,
        "no_checkpoint_selection": True,
    }
    admission = json.loads((root / "data-admission.json").read_text())
    value["data_inventory_before_designation"] = admission["inventory"]
    value["lineage"] = {
        "contract_sha256": hashlib.sha256(
            (root / "successor-campaign-v1/contract.json").read_bytes()
        ).hexdigest(),
        "cache_sha256": contract["cache"]["sha256"],
        "development_manifest_sha256": admission["development"]["sha256"],
        "training_manifest_sha256": admission["training"]["sha256"],
    }
    value["designation_scope_note"] = (
        "Fresh non-primary Phase76 assessment excludes all Phase74/75 and historical reservations. Original70000 corpus and1536 pool fixed; no taupair or sealed test."
    )
    value["controls"]["arm_settings"] = contract["arm_settings"]
    value["diagnostics"] = {}
    for label, folder in (
        ("exploratory_phase75", "diagnostic-v3"),
        ("fixed_phase76", "endpoint-diagnostic-v1"),
    ):
        diagnostic = json.loads((root / folder / "diagnostic-summary.json").read_text())
        clean = {"scope": label, "arms": {}}
        for arm, roles in diagnostic["arms"].items():
            clean["arms"][arm] = {}
            for role, record in roles.items():
                counters = {
                    k: v for k, v in record.items() if k not in ("checkpoint", "trace")
                }
                counters["checkpoint_sha256"] = record["checkpoint"]["sha256"]
                counters["detached_trace_sha256"] = record["trace"]["sha256"]
                clean["arms"][arm][role] = counters
        value["diagnostics"][label] = clean
    for arm, record in value["arms"].items():
        for role, metrics in record["roles"].items():
            diagnostic = value["diagnostics"]["fixed_phase76"]["arms"][arm][role][
                "merges"
            ]
            metrics["relation_ignored_pairs"] = {
                "detector": diagnostic["detector_relation_ignored_pairs"],
                "generated": diagnostic["generated_relation_ignored_pairs"],
                "status": "MEASURED_AFTER_GENERATION",
            }
    findings = json.loads((root / "diagnostic-findings-v2.json").read_text())[
        "findings"
    ]
    value["diagnostic_population_breakdown"] = {
        arm: {
            role: {
                k: v for k, v in record.items() if k not in ("by_size", "by_category")
            }
            for role, record in roles.items()
        }
        for arm, roles in findings.items()
    }
    from scripts.review_phase76_terminal import paired_source_errors

    value["paired_source_errors"] = {}
    for role in ("train", "heldout"):
        rows = [
            json.loads(
                (
                    root
                    / "endpoint-diagnostic-v1"
                    / f"{arm}-{role}-joined-private.json"
                ).read_text()
            )
            for arm in ("partial_context_on", "partial_context_off")
        ]
        value["paired_source_errors"][role] = paired_source_errors(*rows)
    value["diagnostic_interpretation"] = (
        "Proposal omissions dominate; partial-context contamination is association, not causal proof. Off removes the entire partial-context block, including singletons. Predicted-membership context remains active. Old development diagnostics are exploratory only."
    )
    value["coverage"] = {
        "heldout_by_category": dict.fromkeys(
            ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"), 100
        ),
        "primary_requirement_met": False,
    }
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    packed = gzip.compress(data, mtime=0)
    if len(data) > 5_000_000:
        raise ValueError("Decoded publication bound exceeded")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "phase76-review.json.gz").open("xb") as f:
        f.write(packed)
    binding = {
        "version": "phase76-public-binding-v1",
        "file": "phase76-review.json.gz",
        "compressed_sha256": hashlib.sha256(packed).hexdigest(),
        "decoded_sha256": hashlib.sha256(data).hexdigest(),
        "decoded_bytes": len(data),
        "source_sha": value["source_sha"],
    }
    (destination / "binding.json").write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(binding)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    package(a.root, a.output)
