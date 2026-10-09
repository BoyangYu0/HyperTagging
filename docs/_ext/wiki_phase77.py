"""Complete training-only diagnostic delivery and compact dashboard."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase77_review_20261009"
FILES = {"phase77-review.json", "phase77-integrity.json", "phase77-decoder.txt"}


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
        p.name != "phase77-review.json.gz"
        or p.is_symlink()
        or p.stat().st_size > 5_000_000
    ):
        raise ValueError("Unsafe Phase77 source")
    packed = p.read_bytes()
    if hashlib.sha256(packed).hexdigest() != bind["compressed_sha256"]:
        raise ValueError("Phase77 compressed hash mismatch")
    with gzip.open(p, "rb") as f:
        data = f.read(5_000_001)
    if (
        len(data) > 5_000_000
        or len(data) != bind["decoded_bytes"]
        or hashlib.sha256(data).hexdigest() != bind["decoded_sha256"]
    ):
        raise ValueError("Phase77 decoded hash mismatch")
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
    if any(r["max_probability_difference"] != 0 for r in value["populations"].values()):
        raise ValueError("Replay mismatch")
    encoded = sibling("wiki_phase72_compact").encode(data, packed, 5_000_000)
    (output / "phase77-review.json").write_bytes(encoded)
    (output / "phase77-integrity.json").write_text(
        json.dumps(
            {
                "decoded_sha256": bind["decoded_sha256"],
                "decoded_bytes": len(data),
                "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_sha": bind["source_sha"],
                "all_supported_metrics_retained": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (output / "phase77-decoder.txt").write_text(
        (output / "phase74-decoder.txt").read_text().replace("phase74", "phase77")
    )
    return {
        k: v
        for k, v in value.items()
        if k not in ("training_history", "historical_native_endpoints", "populations")
    } | {
        "populations": {
            role: {
                k: v
                for k, v in r.items()
                if k
                in (
                    "events",
                    "stages",
                    "by_category",
                    "by_size",
                    "foreground_rank_auc",
                    "max_probability_difference",
                    "bindings",
                    "gradient_probes",
                    "event_node_squared_proxy",
                )
            }
            for role, r in value["populations"].items()
        },
        "metric_download": {
            "filename": "phase77-review.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }


def render(value):
    table = sibling("wiki_phase74").table
    lines = [
        "Phase77 completed training-only proposal diagnosis",
        "---------------------------------------------------",
        "",
        "No new training or validation evaluation was selected. This replay diagnoses existing training errors; it is not fresh confirmation.",
        "All 1536 original training-pool identities and 24 tiny-fit views are reproduced exactly. The tiny views reuse training identities.",
        "See :doc:`../../phase77` for interpretation and :doc:`../../phase76` for the preceding negative controlled experiment.",
        "Complete category/size/margin/loss/gradient data, paired uncertainty, prior training curves and native endpoint metrics are retained in",
        ":download:`phase77-review.json <phase77-review.json>`, :download:`integrity <phase77-integrity.json>` and :download:`decoder <phase77-decoder.txt>`.",
        "",
    ]
    rows = []
    for stage, r in value["populations"]["train"]["stages"].items():
        rows.append(
            [
                stage,
                r["B_correct"],
                r["B_to_unassigned"],
                r["B_to_other_B"],
                f"{r['background_to_B']} / {r['background_nodes']}",
                f"{r['native_raw_exact']} / {r['nominal_B_trials']}",
            ]
        )
    lines += table(
        [
            "Stage",
            "Correct B source",
            "Unassigned B source",
            "Other-B source",
            "Background to B",
            "Raw exact B",
        ],
        rows,
    )
    lines += [
        "The B-source denominator is 9437 across 512 B collisions and 1024 correlated B trials. Background counts cover all six categories.",
        "Refinement native accepted membership is 0 of 1024; tiny raw is 32 of 32 and accepted is 27 of 32. Five tiny targets violate the unchanged charge guard.",
        "Main-train native continuum acceptance remains 96 of 1024. No physical hierarchy, beam-pool or p4-closure metric is defined for these flat groups.",
        "",
        "Decision and supervision diagnosis",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "The one prespecified aggregate-B probability decision recovers 709 previously unassigned B sources but adds 882 background assignments.",
        "Correct-slot recall gains only 392 sources: the remaining 317 recovered foreground sources go to the other B. Exact recovery stays 0 of 1024.",
        "Foreground precision falls by 3.12 percentage points (paired training-collision interval: -3.65 to -2.61 points). This is not a calibrated population interval.",
        "Even perfect oracle foreground separation yields 0 exact sets with the existing B-slot assignments. Oracle correction of B-slot swaps alone yields 37 of 1024;",
        "both oracles use truth after generation and cannot be deployed or interpreted as legal physical reachability.",
        "Under the hard-error event permutation, conditional B-slot assignment is correct for 5676 of 9437 B sources. Native total-loss matching and hard-error matching differ in 120 of 512 B collisions;",
        "these optimize different criteria, and permutation/unknown-label tests pass. No target, presence-gating or gradient-path implementation defect was demonstrated.",
        "B sources contribute more CE loss and squared logit-gradient norm than background despite fewer nodes. Counts alone do not support a class-weight intervention.",
        "Convergence is unproven. No threshold, loss-weight, context, projection, width, data or training-duration sweep is launched.",
        "",
        "Training category controls",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
    ]
    lines += table(
        [
            "Category",
            "Events",
            "B sources",
            "B unassigned",
            "Other B",
            "Background sources",
            "Background to B",
        ],
        [
            [
                cat,
                r["proposal"]["events"],
                r["proposal"]["B_nodes"],
                r["proposal"]["B_to_unassigned"],
                r["proposal"]["B_to_other_B"],
                r["proposal"]["background_nodes"],
                r["proposal"]["background_to_B"],
            ]
            for cat, r in value["populations"]["train"]["by_category"].items()
        ],
    )
    lines += ["", "Source-set size controls", "~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    lines += table(
        [
            "B size",
            "B trials",
            "B sources",
            "Unassigned",
            "Other B",
            "Correct conditional B slot",
        ],
        [
            [
                size,
                r["proposal"]["trials"],
                r["proposal"]["B_nodes"],
                r["proposal"]["B_to_unassigned"],
                r["proposal"]["B_to_other_B"],
                r["proposal"]["conditional_slot_correct"],
            ]
            for size, r in sorted(
                value["populations"]["train"]["by_size"].items(),
                key=lambda x: int(x[0].split("-")[0].rstrip("+")),
            )
        ],
    )
    c = value["compute"]
    lines += [
        "",
        "Compute and recommendation",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        f"Replay: {c['wall_seconds']:.2f} measured wall seconds, {c['process_cpu_seconds']:.2f} process CPU seconds, peak RSS {c['peak_rss_kib']} KiB; 1560 forward views and 96 additional gradient views.",
        f"Encoder parameters {c['parameters']['encoder']}; decoder parameters {c['parameters']['decoder']}. No optimizer updates, new fitting presentations, GPU, or fresh development reservation.",
        "Node-squared workload proxies and prior actual sampling/updates remain downloadable; they are not FLOPs.",
        value["decision"]["recommendation"],
        "One final checkpoint and seed; paired B trials and sparse successes limit inference. Keep primary scale-up closed. No automatic successor.",
        "",
    ]
    return lines
