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
    output = ROOT / "artifacts/codex/phase68_publication_20261005"
    output.mkdir(exist_ok=True)
    registry = {}
    files = []
    for src in sorted((a.artifacts / "native-metrics").glob("public-*.json")):
        d = json.loads(src.read_text())
        view = d["view"]
        rows = d["rows"]
        registry[view] = [[r["arm"], r["metric"]] for r in rows]
        names = list(dict.fromkeys(r["metric"] for r in rows))
        lookup = {k: i for i, k in enumerate(names)}
        arms = ["refined_encoder", "pre_refinement_encoder"]
        payload = {
            "version": "phase68-aggregate-columnar-v1",
            "view": view,
            "arms": arms,
            "metric_names": names,
            "columns": ["arm_index", "metric_index", "value"],
            "records": [
                [arms.index(r["arm"]), lookup[r["metric"]], r["value"]] for r in rows
            ],
        }
        raw = encode(payload)
        name = "phase68-" + view.replace("_", "-") + ".json"
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
                "structure_counts": v["primary_structure_counts"],
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
        "version": "phase68-public-review-v1",
        "status_date": "2026-10-05",
        "status": close["status"],
        "arms": arms,
        "paired": json.loads((a.artifacts / "paired.json").read_text()),
        "files": files,
        "source_hashes": close["source_hashes"],
        "native_source_revision": "ee4c7255024aa75295c35e32e438c26cde1ed468",
        "sealed_test_accessed": False,
        "train_events": 70000,
        "selection_events": 1000,
        "strict_events": 100,
        "beam_events": 20,
        "historical_reserved": 57700,
        "remaining_validation": 41200,
        "physical_momentum_resolution": "UNAVAILABLE_SCHEMA_NO_MC_COMPOSITE_P4",
        "next_study_status": "PREPARATION_IN_PROGRESS",
    }
    native = json.loads((a.artifacts / "native-lineage.json").read_text())
    summary["context_exposure"] = {arm: row["context_totals"] for arm, row in native["arms"].items()}
    summary["encoder_execution"] = {arm: {track: {"step": value["step"], "encoder_keys": value["encoder_keys"], "changed_key_count": len(value["changed_encoder_keys"]), "encoder_l2_change": value["encoder_l2_change"]} for track,value in row["tracks"].items()} for arm,row in native["arms"].items()}
    radial = a.artifacts / "geometry.json"
    if radial.exists():
        summary["geometry"] = json.loads(radial.read_text())["models"]
    manifest = a.artifacts / "native-metrics/export-manifest.json"
    if manifest.exists():
        summary["export_counts"] = {
            k: v
            for k, v in json.loads(manifest.read_text()).items()
            if k in ("native_scalar_count", "public_metric_count")
        }
    checkpoint_manifest = json.loads((a.artifacts / "native-metrics/checkpoint-export-manifest.json").read_text())
    summary["export_counts"]["native_scalar_count"] += checkpoint_manifest["additional_native_scalar_count"]
    summary["export_counts"]["public_metric_count"] += checkpoint_manifest["additional_public_metric_count"]
    summary["export_counts"]["saved_checkpoint_count"] = checkpoint_manifest["saved_checkpoint_count"]
    submission = (
        ROOT / "artifacts/codex/reconstruction_phase69_submission_20261005.json"
    )
    if submission.exists():
        receipt = json.loads(submission.read_text())
        summary["next_study_status"] = "SUBMITTED_AND_SCHEDULING_VERIFIED"
        summary["next_study"] = {
            "source_sha": receipt["source_sha"],
            "native_receipt_sha256": receipt["native_receipt_sha256"],
            "task_count": 2,
            "optimizer_steps_per_task": 4376,
            "seed": 20261005,
            "train_events": 70000,
            "remaining_unreserved_validation": 40100,
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
    (ROOT / "docs/_ext/phase68_summary_registry.json.gz").write_bytes(
        gzip.compress(encode(list(schema(public_summary))), mtime=0)
    )
    (output / "summary.json").write_bytes(encode(summary))
    (ROOT / "docs/_ext/phase68_metric_registry.json.gz").write_bytes(
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
