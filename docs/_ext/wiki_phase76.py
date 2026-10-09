"""Bounded complete successor download and compact dashboard projection."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase76_review_20261009"
FILES = {"phase76-review.json", "phase76-integrity.json", "phase76-decoder.txt"}


def sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def generate(root, output):
    source = root / SOURCE
    if not source.exists():
        return None
    bind = json.loads((source / "binding.json").read_text())
    p = source / bind["file"]
    if (
        p.name != "phase76-review.json.gz"
        or p.is_symlink()
        or p.stat().st_size > 5_000_000
    ):
        raise ValueError("Unsafe successor source")
    packed = p.read_bytes()
    if hashlib.sha256(packed).hexdigest() != bind["compressed_sha256"]:
        raise ValueError("Successor compressed hash mismatch")
    with gzip.open(p, "rb") as f:
        data = f.read(5_000_001)
    if (
        len(data) > 5_000_000
        or len(data) != bind["decoded_bytes"]
        or hashlib.sha256(data).hexdigest() != bind["decoded_sha256"]
    ):
        raise ValueError("Successor decoded hash mismatch")
    value = json.loads(data)
    if (
        value["status"] != "COMPLETED"
        or value["stage"] != "development"
        or value["primary_eligible"]
        or set(value["arms"]) != {"partial_context_on", "partial_context_off"}
    ):
        raise ValueError("Wrong successor scope")
    for r in value["arms"].values():
        if {k: len(v) for k, v in r["curves"].items()} != {
            "tiny": 1000,
            "downstream": 1500,
        }:
            raise ValueError("Incomplete successor update records")
        if (
            any(
                x["processed"] != 100
                for x in r["roles"]["heldout"]["by_category"].values()
            )
            or r["roles"]["heldout"]["counts"]["events"] != 600
        ):
            raise ValueError("Successor coverage changed")
    compact = sibling("wiki_phase72_compact")
    encoded = compact.encode(data, packed, 5_000_000)
    (output / "phase76-review.json").write_bytes(encoded)
    (output / "phase76-integrity.json").write_text(
        json.dumps(
            {
                "decoded_sha256": bind["decoded_sha256"],
                "decoded_bytes": len(data),
                "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_sha": bind["source_sha"],
                "all_metrics_and_curves_retained": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    decoder = (output / "phase74-decoder.txt").read_text().replace("phase74", "phase76")
    (output / "phase76-decoder.txt").write_text(decoder)
    return {
        **{
            k: v
            for k, v in value.items()
            if k not in ("arms", "diagnostics", "diagnostic_population_breakdown")
        },
        "arms": {
            a: {
                **{k: v for k, v in r.items() if k not in ("curves", "roles")},
                "roles": {
                    role: {
                        k: v
                        for k, v in counts.items()
                        if k
                        not in (
                            "by_retained_channel",
                            "by_truth_fsp_size",
                            "positive_cases",
                        )
                    }
                    for role, counts in r["roles"].items()
                },
                "complete_channel_size_and_case_records": "phase76-review.json",
            }
            for a, r in value["arms"].items()
        },
        "metric_download": {
            "filename": "phase76-review.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }


def render(value):
    table = sibling("wiki_phase74").table
    lines = [
        "Phase76 completed membership refinement study",
        "----------------------------------------------",
        "",
        "This separately authorized two-arm follow-up tests generated partial-context conditioning. No further campaign is scheduled.",
        "Two matched arms use the same fixed pretraining checkpoint and fresh 600-event development cohort.",
        "Fixed final checkpoints and threshold 0.5; no primary evaluation or physical hierarchy claim.",
        "",
        "See :doc:`../../phase76` for interpretation and :doc:`../../phase75` for the preceding negative gradient study.",
        "Complete metrics, channel/size/category counts, all 5,000 update records and compute are in",
        ":download:`phase76-review.json <phase76-review.json>` with :download:`integrity <phase76-integrity.json>`",
        "and :download:`decoder <phase76-decoder.txt>`.",
        "",
    ]
    lines += [
        "Source-backed failure diagnosis",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "Before training, detached model traces showed proposal-stage omissions dominating the joint control:",
        "6000 missing target-source assignments of 9437 in training; 4172 were assigned unassigned.",
        "Refinement reduced extra sources from 3364 to 2961 but increased missing sources to 6232.",
        "Of 4096 generated merges in training B collisions, 3559 contained unassigned sources and 319 were clean within-B.",
        "The 3559 include 1890 mixed B/unassigned and 1669 background-only merges; only the former mix background into B context.",
        "Final groups mix background into context for 1711 of 9437 B-source nodes. These are exploratory associations, not causal proof.",
        "Both arms retain differentiable predicted-membership context and the same generation procedure, bounds and relation objective.",
        "Off zeros the whole generated partial-context block, including singleton context. No truth conditions inference.",
        "Complete stage/category/size/missing/extra diagnostics and uncertainty are retained in the download.",
        "",
    ]
    rows = []
    for a, r in value["arms"].items():
        t = r["roles"]["tiny"]["counts"]
        tr = r["roles"]["train"]["counts"]
        h = r["roles"]["heldout"]["counts"]
        rows.append(
            [
                {
                    "partial_context_on": "Context on",
                    "partial_context_off": "Context off",
                }[a],
                f"{t['raw_exact_memberships']} of 32",
                f"{t['accepted_exact_memberships']} of 32",
                f"{tr['raw_exact_memberships']} of 1024",
                f"{h['raw_exact_memberships']} of 400",
                f"{h['accepted_exact_memberships']} of 400",
                f"{h['continuum_accepted_events']} of 400",
            ]
        )
    lines += table(
        [
            "Arm",
            "Tiny raw",
            "Tiny accepted",
            "Train raw",
            "Held-out raw",
            "Held-out accepted",
            "Continuum accepted",
        ],
        rows,
    )
    lines += [
        "Relation, latent-depth and compute diagnostics",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
    ]
    rows = []
    for a, r in value["arms"].items():
        h = r["roles"]["heldout"]["counts"]
        rows.append(
            [
                {
                    "partial_context_on": "Context on",
                    "partial_context_off": "Context off",
                }[a],
                f"{h['detector_relation_correct']} of {h['detector_relation_trials']}",
                f"{h['generated_state_relation_correct']} of {h['generated_state_relation_trials']}",
                f"{h['latent_deep_source_sets_proposed']} of {h['latent_deep_source_sets_trials']}",
                f"{h['latent_deep_source_sets_retained']} of {h['latent_deep_source_sets_trials']}",
            ]
        )
    lines += table(
        [
            "Arm",
            "Detector relations",
            "Generated relations",
            "Latent proposed",
            "Latent retained",
        ],
        rows,
    )
    lines += [
        "Generated-state supports differ across models. Latent source-set merges are diagnostic,",
        "not verified legal physical hierarchy proposals. Ignored-pair counts are measured in the complete download.",
        "",
    ]
    lines += table(
        [
            "Arm",
            "Encoder parameters",
            "Fit/evaluation wall seconds",
            "Peak RSS KiB",
            "Fit node-pair proxy",
        ],
        [
            [
                {
                    "partial_context_on": "Context on",
                    "partial_context_off": "Context off",
                }[a],
                r["compute"]["encoder_parameters"],
                round(r["compute"]["wall_seconds"], 2),
                r["compute"]["peak_rss_kib"],
                r["compute"]["detector_node_pairs"],
            ]
            for a, r in value["arms"].items()
        ],
    )
    lines += [
        "Both arms have 20,000 fit presentations and 44,320 encoder passes including evaluation.",
        "The native encoder counter covers fitting only; the download separately derives evaluation accounting.",
        "No gradient projection is used. Equal updates and pair budgets are not measured equal FLOPs.",
        "",
    ]
    lines += ["Paired source-level errors", "~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    rows = []
    for role, stages in value.get("paired_source_errors", {}).items():
        for stage, metrics in stages.items():
            for metric in ("source_recall", "source_precision"):
                r = metrics[metric]
                rows.append(
                    [
                        role,
                        stage,
                        metric,
                        str(r.get("rates", "UNAVAILABLE")),
                        str(r["paired_collision_stratified_bootstrap95"]),
                    ]
                )
    lines += table(
        ["Role", "Stage", "Metric", "Context on, off", "Off minus on interval"], rows
    )
    lines += [
        "These ratios aggregate source counts before division. Intervals resample whole collisions within categories.",
        "Training intervals are descriptive; one seed and repeated fitting do not measure seed variability.",
        "",
    ]
    lines += ["Paired effects and limitations", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    lines += table(
        ["Endpoint", "Context off minus on", "Paired collision interval"],
        [
            [k, v["difference"], str(v["paired_collision_stratified_bootstrap95"])]
            for k, v in value["paired_effects"].items()
        ],
    )
    lines += [
        "One seed, correlated B trials and sparse successes limit inference. Exact event-binomial",
        "bounds accompany the bootstrap in the download. Disabling partial context is not a",
        "guarantee of held-out efficiency. Physical tree/pool/deep-survival",
        "and p4-closure metrics remain UNAVAILABLE. No automatic primary or wider-model scale-up.",
        "",
    ]
    return lines
