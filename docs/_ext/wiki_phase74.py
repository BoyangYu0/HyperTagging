"""Authenticated compact Phase74 development results; no private event publication."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = "artifacts/codex/phase74_review_20261009"
FILES = {"phase74-review.json", "phase74-integrity.json", "phase74-decoder.txt"}


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
    binding = json.loads((source / "binding.json").read_text())
    path = source / binding["file"]
    if (
        path.is_symlink()
        or path.name != "phase74-review.json.gz"
        or path.stat().st_size > 5_000_000
    ):
        raise ValueError("Unsafe Phase74 source")
    packed = path.read_bytes()
    if hashlib.sha256(packed).hexdigest() != binding["compressed_sha256"]:
        raise ValueError("Phase74 compressed hash mismatch")
    with gzip.open(path, "rb") as f:
        data = f.read(5_000_001)
    if (
        len(data) > 5_000_000
        or len(data) != binding["decoded_bytes"]
        or hashlib.sha256(data).hexdigest() != binding["decoded_sha256"]
    ):
        raise ValueError("Phase74 decoded binding mismatch")
    value = json.loads(data)
    if (
        value["stage"] != "development"
        or value["primary_eligible"]
        or value["sealed_test_access"]
        or len(value["arms"]) != 4
    ):
        raise ValueError("Phase74 scope changed")
    for arm, r in value["arms"].items():
        if r["roles"]["heldout"]["counts"]["events"] != 600 or any(
            x["processed"] != 100 for x in r["roles"]["heldout"]["by_category"].values()
        ):
            raise ValueError("Phase74 coverage mismatch")
    compact = sibling("wiki_phase72_compact")
    encoded = compact.encode(data, packed, 5_000_000)
    (output / "phase74-review.json").write_bytes(encoded)
    (output / "phase74-integrity.json").write_text(
        json.dumps(
            {
                "decoded_sha256": binding["decoded_sha256"],
                "decoded_bytes": len(data),
                "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_sha": binding["source_sha"],
                "all_aggregate_and_history_values_preserved": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    decoder = """# Python 3 standard-library decoder for phase74-review.json
import base64,hashlib,json,pathlib,zlib
p=pathlib.Path("phase74-review.json")
e=json.loads(p.read_text())
assert e["encoding"]=="bounded-gzip-base32-json-v1"
n=e["decoded_bytes"]
assert isinstance(n,int) and 0<=n<=5000000 and len(e["data"])<=10485760
z=zlib.decompressobj(31)
b=z.decompress(base64.b32decode(e["data"]),n+1)
assert len(b)==n and z.eof and not z.unconsumed_tail and not z.unused_data
assert hashlib.sha256(b).hexdigest()==e["decoded_sha256"]
pathlib.Path("phase74-decoded.json").write_bytes(b)
"""
    (output / "phase74-decoder.txt").write_text(decoder)
    # Histories stay complete in the bounded download, not duplicated in status.json.
    return {
        **{k: v for k, v in value.items() if k != "arms"},
        "arms": {
            a: {
                **{
                    k: v
                    for k, v in r.items()
                    if k not in ("histories", "roles", "positive_cases")
                },
                "roles": {
                    role: {
                        k: v
                        for k, v in value.items()
                        if k not in ("by_retained_channel", "by_truth_fsp_size")
                    }
                    for role, value in r["roles"].items()
                },
                "positive_cases": [
                    case for case in r["positive_cases"] if case["role"] == "heldout"
                ],
                "complete_channel_size_and_case_records": "phase74-review.json",
            }
            for a, r in value["arms"].items()
        },
        "metric_download": {
            "filename": "phase74-review.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }


def table(headers, rows):
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: auto", ""]
    for row in [headers, *rows]:
        for i, cell in enumerate(row):
            lines.append(("   * - " if i == 0 else "     - ") + str(cell))
    return lines + [""]


def render(value):
    arms = ("128-existing", "128-assembly", "256-existing", "256-assembly")
    lines = [
        "Phase74 completed development factorial",
        "-------------------------------------------",
        "",
        "All arms pass tiny raw memorization, but no reliable held-out winner is established.",
        "One retained source-set recovery is not a physical B-tree tag; background acceptance rises.",
        "The six-category 600-event assessment is development, not the 12,000-event primary requirement.",
        "",
        "See :doc:`../../phase74` for the scientific audit and bounded successor status.",
        "Download every aggregate, channel/size/category count and all 14,000 training-step records in",
        ":download:`phase74-review.json <phase74-review.json>` with",
        ":download:`integrity <phase74-integrity.json>` and :download:`decoder <phase74-decoder.txt>`.",
        "",
    ]
    rows = []
    for a in arms:
        r = value["arms"][a]
        t = r["roles"]["tiny"]["counts"]
        tr = r["roles"]["train"]["counts"]
        h = r["roles"]["heldout"]["counts"]
        rows.append(
            [
                a,
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
        "Relation and latent-depth diagnostics",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "Generated-state supports differ because the states are model-generated.",
        "Latent exact source-set retention is not legal physical hierarchy reachability.",
        "",
    ]
    rows = []
    for a in arms:
        h = value["arms"][a]["roles"]["heldout"]["counts"]
        rows.append(
            [
                a,
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
            "Latent deep proposed",
            "Latent deep retained",
        ],
        rows,
    )
    lines += [
        "Compute and uncertainty",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "Each arm presents 28,000 fitting events over 1,536 unique training-subset events;",
        "the original 70,000 corpus is unchanged. Equal updates are not equal FLOPs.",
        "",
    ]
    lines += table(
        ["Arm", "Encoder parameters", "Wall seconds", "Peak RSS KiB"],
        [
            [
                a,
                value["arms"][a]["compute"]["encoder_parameters"],
                round(value["arms"][a]["compute"]["wall_seconds"], 2),
                value["arms"][a]["compute"]["peak_rss_kib"],
            ]
            for a in arms
        ],
    )
    lines += [
        "Collision-paired factorial intervals and exact sparse event bounds are downloadable.",
        "There is one seed; rare successes and correlated B trials limit inference.",
        "No wider model, primary campaign or production hierarchy is eligible.",
        "Physical exact-tree top1, beam pool, physical deep survival and p4 closure are UNAVAILABLE.",
        "",
    ]
    return lines
