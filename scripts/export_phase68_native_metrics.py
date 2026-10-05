"""Export every native scalar privately and aggregate-only public metric views."""

import argparse, gzip, json, math, statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_reconstruction_phase68_closeout import ARMS, digest
from scripts.build_phase62_retained_metric_export import metric_names
from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}.{index}")
    elif value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(path)
        yield {"metric": path, "value": value}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    public = {}
    manifest = {
        "version": "phase68-complete-native-export-v1",
        "arms": {},
        "files": [],
        "private_event_level": True,
    }
    for arm, job in ARMS.items():
        run = (
            a.source / f"artifacts/runs/ht-reconstruction-phase68-20261004/{arm}/{job}"
        )
        result = json.loads((run / "result.json").read_text())
        ref = result["pretraining_refinement"]
        assert digest(Path(ref["checkpoint"])) == ref["checkpoint_sha256"]
        assert ref["additional_pretraining_steps"] == 0
        assert ref["mode"] == "frozen_checkpoint_reuse_no_training"
        transfer = result["transfer_report"]
        assert not any(
            transfer[k]
            for k in (
                "missing_keys",
                "unexpected_keys",
                "shape_mismatches",
                "leaf_pid_missing_keys",
                "leaf_pid_unexpected_keys",
                "leaf_pid_shape_mismatches",
            )
        )
        lineage = {}
        for track, cp in result["checkpoints"].items():
            lineage[track] = validate_checkpoint_pair(
                ref["checkpoint"], cp["path"], require_exact_frozen_encoder=False
            ).as_dict()
        (a.output / f"{arm}-checkpoint-lineage.json").write_text(
            json.dumps(lineage, indent=2) + "\n"
        )
        count = 0
        inventory = []
        export = a.output / f"{arm}-all-native-scalars.jsonl.gz"
        with gzip.open(export, "wt") as out:
            for path in sorted(run.rglob("*")):
                if path.suffix not in (".json", ".jsonl"):
                    continue
                rel = path.relative_to(run).as_posix()
                inventory.append(
                    {"path": rel, "sha256": digest(path), "bytes": path.stat().st_size}
                )
                docs = (
                    [json.loads(x) for x in path.read_text().splitlines()]
                    if path.suffix == ".jsonl"
                    else [json.loads(path.read_text())]
                )
                for index, doc in enumerate(docs):
                    for row in leaves(doc):
                        out.write(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "native_file": rel,
                                    "record": index,
                                    **row,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        count += 1
                    if (
                        path.parent.name == "full-decay-reports"
                        and isinstance(doc, dict)
                        and "summaries" in doc
                    ):
                        aggregate = {
                            k: doc[k]
                            for k in (
                                "summaries",
                                "summaries_by_source_category",
                                "summaries_by_target_shape",
                            )
                        }
                        aggregate["retained"] = doc["retained_tree_checks"]["summaries"]
                        beam = doc.get("beam_search", {})
                        aggregate["beam"] = {
                            k: v for k, v in beam.items() if k != "events"
                        }
                        public.setdefault(path.stem, []).extend(
                            {"arm": arm, **row}
                            for row in leaves(metric_names(aggregate))
                        )
            histories = {}
            for stage in ("training",):
                rows = [
                    json.loads(x)
                    for x in (run / stage / "metrics.jsonl").read_text().splitlines()
                ]
                for i, doc in enumerate(rows):
                    # A history record is an aggregate logged batch/view, never an event row.
                    histories.setdefault(stage, []).extend(
                        {"arm": arm, "record": i, **r} for r in leaves(doc)
                    )
            history_file = a.output / f"{arm}-training-history.json.gz"
            with gzip.open(history_file, "wt") as f:
                json.dump(histories, f, separators=(",", ":"), allow_nan=False)
        manifest["arms"][arm] = {
            "native_scalar_count": count,
            "native_inventory": inventory,
            "encoder_loaded_keys": len(transfer["loaded_keys"]),
            "leaf_pid_loaded_keys": len(transfer["leaf_pid_loaded_keys"]),
            "history_records": {
                s: len((run / s / "metrics.jsonl").read_text().splitlines())
                for s in ("training",)
            },
        }
        manifest["files"].append(
            {
                "name": export.name,
                "sha256": digest(export),
                "bytes": export.stat().st_size,
            }
        )
        print(arm, count, flush=True)
    for view, rows in public.items():
        path = a.output / f"public-{view}.json"
        path.write_text(
            json.dumps(
                {"version": "phase68-aggregate-metrics-v1", "view": view, "rows": rows},
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
    history_rows = []
    for arm, job in ARMS.items():
        run = (
            a.source / f"artifacts/runs/ht-reconstruction-phase68-20261004/{arm}/{job}"
        )
        for stage in ("training",):
            groups = {}
            for line in (run / stage / "metrics.jsonl").read_text().splitlines():
                row = json.loads(line)
                for key, value in row.items():
                    if type(value) in (int, float):
                        groups.setdefault(
                            row.get("split", "training") + "." + key, []
                        ).append(value)
            for key, values in groups.items():
                for stat, value in {
                    "count": len(values),
                    "first": values[0],
                    "last": values[-1],
                    "minimum": min(values),
                    "maximum": max(values),
                    "mean": statistics.fmean(values),
                }.items():
                    history_rows.append(
                        {
                            "arm": arm,
                            "metric": stage + "." + key + "." + stat,
                            "value": value,
                        }
                    )
    (a.output / "public-training_history_summary.json").write_text(
        json.dumps(
            {
                "version": "phase68-aggregate-metrics-v1",
                "view": "training_history_summary",
                "rows": history_rows,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    manifest["public_metric_count"] = sum(len(r) for r in public.values()) + len(
        history_rows
    )
    manifest["native_scalar_count"] = sum(
        x["native_scalar_count"] for x in manifest["arms"].values()
    )
    (a.output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("public_metric_count", "native_scalar_count")}
        )
    )


if __name__ == "__main__":
    main()
