"""Complete training-only diagnostic delivery and compact dashboard."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase78_review_20261009"
FILES = {
    "phase78-review.json",
    "phase78-integrity.json",
    "phase78-decoder.txt",
    *{f"phase78-events-{i}.json" for i in range(3)},
}


def sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(root, output):
    source = root / SOURCE
    if not source.exists():
        return None
    bind = json.loads((source / "binding.json").read_text())
    p = source / bind["file"]
    if (
        p.name != "phase78-review.json.gz"
        or p.is_symlink()
        or p.stat().st_size > 5_000_000
    ):
        raise ValueError("Unsafe Phase78 source")
    packed = p.read_bytes()
    if hashlib.sha256(packed).hexdigest() != bind["compressed_sha256"]:
        raise ValueError("Phase78 compressed hash mismatch")
    with gzip.open(p, "rb") as f:
        data = f.read(5_000_001)
    if (
        len(data) > 5_000_000
        or len(data) != bind["decoded_bytes"]
        or hashlib.sha256(data).hexdigest() != bind["decoded_sha256"]
    ):
        raise ValueError("Phase78 decoded hash mismatch")
    value = json.loads(data)
    if (
        value["status"] != "COMPLETED"
        or value["stage"] != "training_only_diagnostic"
        or value["primary_eligible"]
        or value["decision"]["scientific_training_jobs"]
        or value["decision"]["fresh_development_designations"]
    ):
        raise ValueError("Wrong diagnostic-only scope")
    if {role: r["events"] for role, r in value["populations"].items()} != {
        "train": 1536,
        "tiny": 24,
    }:
        raise ValueError("Diagnostic population changed")
    if any(r["max_logit_difference"] != 0 for r in value["populations"].values()):
        raise ValueError("Replay mismatch")
    if (
        len(value["event_parts"]) != 3
        or value["probe_fits"] != 2
        or value["validation_events_evaluated"] != 0
    ):
        raise ValueError("Diagnostic fitting/scope changed")
    part_integrity = []
    for i, part in enumerate(value["event_parts"]):
        name = f"phase78-events-{i}.json.gz"
        path = source / name
        if part["file"] != name or path.is_symlink() or path.stat().st_size > 5_000_000:
            raise ValueError("Unsafe Phase78 event part")
        compressed = path.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != part["compressed_sha256"]:
            raise ValueError("Phase78 event part hash mismatch")
        with gzip.open(path, "rb") as stream:
            decoded = stream.read(5_000_001)
        if (
            len(decoded) > 5_000_000
            or len(decoded) != part["decoded_bytes"]
            or hashlib.sha256(decoded).hexdigest() != part["decoded_sha256"]
        ):
            raise ValueError("Phase78 event decoded hash mismatch")
        rows = json.loads(decoded)
        if (
            len(rows["events"]) != 512
            or rows["start_event_index"] != i * 512
            or rows["role"] != "training"
        ):
            raise ValueError("Phase78 event coverage changed")
        envelope = sibling("wiki_phase72_compact").encode(
            decoded, compressed, 5_000_000
        )
        (output / name.removesuffix(".gz")).write_bytes(envelope)
        part_integrity.append(
            {
                "file": name.removesuffix(".gz"),
                "envelope_sha256": hashlib.sha256(envelope).hexdigest(),
                "decoded_sha256": part["decoded_sha256"],
                "events": 512,
            }
        )
    encoded = sibling("wiki_phase72_compact").encode(data, packed, 5_000_000)
    (output / "phase78-review.json").write_bytes(encoded)
    (output / "phase78-integrity.json").write_text(
        json.dumps(
            {
                "decoded_sha256": bind["decoded_sha256"],
                "decoded_bytes": len(data),
                "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_sha": bind["source_sha"],
                "all_supported_metrics_retained": True,
                "event_parts": part_integrity,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (output / "phase78-decoder.txt").write_text(
        (output / "phase74-decoder.txt").read_text().replace("phase74", "phase78")
    )
    return {
        k: value[k]
        for k in (
            "status",
            "stage",
            "decision",
            "compute",
            "source_sha",
            "paired_probe_minus_native_event_auc",
        )
    } | {
        "populations": {
            role: {
                "events": r["events"],
                "aggregate": r["aggregate"]["all"],
                "assessment": r.get("partitions", {}).get("assessment", {}).get("all"),
            }
            for role, r in value["populations"].items()
        },
        "metric_download": {
            "filename": "phase78-review.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }


def render(value):
    table = sibling("wiki_phase74").table
    lines = [
        "Phase78 completed training-only partition diagnosis",
        "----------------------------------------------------",
        "",
        "Two finite diagnostic probes were fitted on frozen features, including a shuffled-label null. No new scientific model training or validation evaluation was selected.",
        "The probe fitting and assessment events are disjoint, but both were used by the frozen scientific model: these are training-role diagnostics, not independent confirmation.",
        "See :doc:`../../phase78` for interpretation and :doc:`../../phase77` for the preceding proposal diagnosis.",
        "Complete supported category, size, pair, query, uncertainty and native endpoint metrics remain in",
        ":download:`all event metrics part0 <phase78-events-0.json>`, :download:`part1 <phase78-events-1.json>` and :download:`part2 <phase78-events-2.json>`; the same decoder verifies each bounded envelope.",
        ":download:`phase78-review.json <phase78-review.json>`, :download:`integrity <phase78-integrity.json>` and :download:`decoder <phase78-decoder.txt>`.",
        "",
        "Pair separation and assignment",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
    ]
    rows = []
    for role, r in value["populations"].items():
        a = r["aggregate"] if role == "tiny" else r["assessment"]
        for name in (
            "proposal_conditional_same",
            "refinement_conditional_same",
            "encoder_cosine",
            "pair_probe",
            "shuffled_probe",
        ):
            v = a["scores"][name]
            rows.append(
                [
                    role
                    + (
                        " probe-assessment"
                        if role == "train"
                        else " (fitted checkpoint)"
                    ),
                    name,
                    v["event_auc"]["count"],
                    f"{v['event_auc']['mean']:.4f}",
                    v["same_pairs"],
                    v["cross_pairs"],
                ]
            )
    lines += table(
        [
            "Population",
            "Pair score",
            "B collisions",
            "Mean event AUC",
            "Same-B pairs",
            "Cross-B pairs",
        ],
        rows,
    )
    lines += [
        "Tiny probe values transfer a main-checkpoint feature probe to a different fitted encoder; they are recorded but do not diagnose tiny representation sufficiency.",
        "Pair endpoints condition on two known B constituents. They are not source-set recovery or physical efficiency; background controls are retained separately.",
        "Shared detector-source pairs are excluded and counted; no event is dropped. Size strata overlap when an event contains two differently sized B targets.",
        "",
        value["decision"]["reason"],
        "",
        value["decision"]["recommendation"],
        "",
        "Native main exact membership remains 0 of 1024; tiny raw is 32 of 32 and accepted 27 of 32. Five tiny targets violate the unchanged charge guard.",
        "No physical trees, physical beam pool, deep proposal survival or p4 closure are defined for these flat heads. No primary evaluation or model promotion follows.",
        "",
    ]
    return lines
