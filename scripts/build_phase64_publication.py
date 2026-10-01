"""Build review-only aggregate inputs and an exact publication metric registry."""

import argparse, gzip, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encode(x):
    return (
        json.dumps(x, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    a = p.parse_args()
    output = ROOT / "artifacts/codex/phase64_publication_20261001"
    output.mkdir(exist_ok=True)
    registry = {}
    files = []
    for src in sorted((a.artifacts / "metrics").glob("public-*.json")):
        d = json.loads(src.read_text())
        view = d["view"]
        rows = d["rows"]
        registry[view] = [[r["arm"], r["metric"]] for r in rows]
        names = list(dict.fromkeys(r["metric"] for r in rows))
        lookup = {k: i for i, k in enumerate(names)}
        arms = ["radial_control", "radial_reconditioned"]
        payload = {
            "version": "phase64-aggregate-columnar-v1",
            "view": view,
            "arms": arms,
            "metric_names": names,
            "columns": ["arm_index", "metric_index", "value"],
            "records": [
                [arms.index(r["arm"]), lookup[r["metric"]], r["value"]] for r in rows
            ],
        }
        raw = encode(payload)
        name = "phase64-" + view.replace("_", "-") + ".json"
        (output / (name + ".gz")).write_bytes(gzip.compress(raw, mtime=0))
        files.append(
            {
                "filename": name,
                "source": str((output / (name + ".gz")).relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "metric_count": len(rows),
                "view": view,
            }
        )
    close = json.loads((a.artifacts / "closeout.json").read_text())
    # Operational paths, UIDs, scheduler identities and individual trees are intentionally omitted.
    arms = {}
    for arm, x in close["arms"].items():
        arms[arm] = {
            k: x[k]
            for k in (
                "endpoints",
                "complete_target_count",
                "gates",
                "all_gates_passed",
                "retained_primary",
                "retained_beam",
                "optimizer_steps",
                "primary_repeat_identical",
                "perfect_component_shapes",
                "forest",
                "beam",
            )
        }
        arms[arm]["topology_by_view"] = {
            view: {
                "exact_component_shapes": v["exact_component_shapes"],
                "coherent_forest_count": len(v["coherent_forests"]),
                "coherent_forests_by_shape": [
                    {
                        k: f[k]
                        for k in (
                            "scope",
                            "search",
                            "unit_semantics",
                            "truth_mothers",
                            "maximum_truth_depth",
                        )
                    }
                    for f in v["coherent_forests"]
                ],
            }
            for view, v in x["view_topology_audits"].items()
        }
    summary = {
        "version": "phase64-public-review-v1",
        "status_date": "2026-10-01",
        "status": close["status"],
        "arms": arms,
        "paired": json.loads((a.artifacts / "paired.json").read_text()),
        "files": files,
        "source_hashes": close["source_hashes"],
        "native_source_revision": "ac3e35bbe93908d4e05f89a7c82df5bebad94c00",
        "sealed_test_accessed": False,
        "train_events": 70000,
        "selection_events": 1000,
        "strict_events": 100,
        "beam_events": 20,
        "historical_reserved": 53300,
        "remaining_validation": 45600,
        "physical_momentum_resolution": "UNAVAILABLE_SCHEMA_NO_MC_COMPOSITE_P4",
        "next_study_status": "PREPARATION_IN_PROGRESS",
    }
    radial = a.artifacts / "posttraining-radial.json"
    if radial.exists():
        summary["geometry"] = json.loads(radial.read_text())["models"]
    manifest = a.artifacts / "metrics/export-manifest.json"
    if manifest.exists():
        summary["export_counts"] = {
            k: v
            for k, v in json.loads(manifest.read_text()).items()
            if k in ("native_scalar_count", "public_metric_count")
        }
    submission = (
        ROOT / "artifacts/codex/reconstruction_phase65_submission_20261001.json"
    )
    if submission.exists():
        receipt = json.loads(submission.read_text())
        summary["next_study_status"] = "SUBMITTED_AND_RELEASED_PENDING_PRIORITY"
        summary["next_study"] = {
            "source_sha": receipt["source_sha"],
            "native_receipt_sha256": receipt["native_receipt_sha256"],
            "task_count": 2,
            "optimizer_steps_per_task": 4376,
            "seed": 20261001,
            "train_events": 70000,
            "remaining_unreserved_validation": 44500,
            "gpu_startup_verified": all(
                j["gpu_startup_verified"] for j in receipt["jobs"]
            ),
            "scheduler_states": [j["state"] for j in receipt["jobs"]],
        }

    def schema(value, path=()):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                yield from schema(item, path + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from schema(item, path + (index,))
        else:
            yield [
                list(path),
                "text" if isinstance(value, str) else "scalar",
                value if isinstance(value, str) else None,
            ]

    public_summary = {key: value for key, value in summary.items() if key != "files"}
    (ROOT / "docs/_ext/phase64_summary_registry.json.gz").write_bytes(
        gzip.compress(encode(list(schema(public_summary))), mtime=0)
    )
    (output / "summary.json").write_bytes(encode(summary))
    (ROOT / "docs/_ext/phase64_metric_registry.json.gz").write_bytes(
        gzip.compress(encode(registry), mtime=0)
    )
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": sum(f["bytes"] for f in files),
                "rows": sum(f["metric_count"] for f in files),
            }
        )
    )


if __name__ == "__main__":
    main()
