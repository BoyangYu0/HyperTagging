"""Bounded complete successor download and compact dashboard projection."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase75_review_20261009"
FILES = {"phase75-review.json", "phase75-integrity.json", "phase75-decoder.txt"}


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
        p.name != "phase75-review.json.gz"
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
        or set(value["arms"]) != {"joint", "project_conflicting_relation"}
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
    (output / "phase75-review.json").write_bytes(encoded)
    (output / "phase75-integrity.json").write_text(
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
    decoder = (output / "phase74-decoder.txt").read_text().replace("phase74", "phase75")
    (output / "phase75-decoder.txt").write_text(decoder)
    return {
        **{k: v for k, v in value.items() if k != "arms"},
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
                "complete_channel_size_and_case_records": "phase75-review.json",
            }
            for a, r in value["arms"].items()
        },
        "metric_download": {
            "filename": "phase75-review.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }


def render(value):
    table = sibling("wiki_phase74").table
    lines = [
        "Phase75 completed gradient-routing development",
        "----------------------------------------------",
        "",
        "This is the sole bounded successor from the Phase74 review; no further campaign is scheduled.",
        "Two matched arms use the same fixed pretraining checkpoint and fresh 600-event development cohort.",
        "Fixed final checkpoints and threshold 0.5; no primary evaluation or physical hierarchy claim.",
        "",
        "See :doc:`../../phase75` for interpretation and :doc:`../../phase74` for the preceding factorial.",
        "Complete metrics, channel/size/category counts, all 5,000 update records and compute are in",
        ":download:`phase75-review.json <phase75-review.json>` with :download:`integrity <phase75-integrity.json>`",
        "and :download:`decoder <phase75-decoder.txt>`.",
        "",
    ]
    rows = []
    for a, r in value["arms"].items():
        t = r["roles"]["tiny"]["counts"]
        tr = r["roles"]["train"]["counts"]
        h = r["roles"]["heldout"]["counts"]
        rows.append(
            [
                {"joint": "Joint", "project_conflicting_relation": "Projection"}[a],
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
                {"joint": "Joint", "project_conflicting_relation": "Projection"}[a],
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
        "not verified legal physical hierarchy proposals. Ignored ambiguous-pair counts were not retained.",
        "",
    ]
    lines += table(
        [
            "Arm",
            "Encoder parameters",
            "Fit/evaluation wall seconds",
            "Peak RSS KiB",
            "Opposing gradient events",
        ],
        [
            [
                {"joint": "Joint", "project_conflicting_relation": "Projection"}[a],
                r["compute"]["encoder_parameters"],
                round(r["compute"]["wall_seconds"], 2),
                r["compute"]["peak_rss_kib"],
                f"{r['compute']['negative_gradient_dot_events']} of {r['compute']['gradient_probe_events']}",
            ]
            for a, r in value["arms"].items()
        ],
    )
    lines += [
        "Both arms have 20,000 fit presentations and 44,320 encoder passes including evaluation.",
        "The native encoder counter covers fitting only; the download separately derives evaluation accounting.",
        "Both probe gradient conflicts; projection adds correction work. Equal updates are not measured equal FLOPs.",
        "",
    ]
    lines += ["Paired effects and limitations", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    lines += table(
        ["Endpoint", "Projection minus control", "Paired collision interval"],
        [
            [k, v["difference"], str(v["paired_collision_stratified_bootstrap95"])]
            for k, v in value["paired_effects"].items()
        ],
    )
    lines += [
        "One seed, correlated B trials and sparse successes limit inference. Exact event-binomial",
        "bounds accompany the bootstrap in the download. A local gradient projection is not a",
        "guarantee of optimizer descent or held-out efficiency. Physical tree/pool/deep-survival",
        "and p4-closure metrics remain UNAVAILABLE. No automatic primary or wider-model scale-up.",
        "",
    ]
    return lines
