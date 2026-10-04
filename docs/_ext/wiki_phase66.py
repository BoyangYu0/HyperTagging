"""Strict aggregate-only Phase66 publication; never opens private run artifacts."""

import gzip, hashlib, json, math, re
from pathlib import Path

ARMS = ("scheduled_mixed", "teacher_only")
VIEWS = (
    "checkpoint_metrics",
    "training_history_summary",
    "independent_complete_target_direct",
    "independent_depth_direct",
    "independent_tree_validity_direct",
    "primary_complete_target_beam_direct",
    "primary_complete_target_contracted_diagnostic",
    "primary_complete_target_direct",
    "primary_complete_target_repeat2_direct",
)


def summary_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from summary_leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from summary_leaves(item, path + (index,))
    else:
        yield path, value


def validate_summary(value):
    registry = json.loads(
        gzip.decompress(
            Path(__file__).with_name("phase66_summary_registry.json.gz").read_bytes()
        )
    )
    expected = {tuple(row[0]): row[1:] for row in registry}
    observed = dict(summary_leaves(value))
    if observed.keys() != expected.keys():
        raise ValueError("Phase66 summary schema differs from reviewed fields")
    for path, scalar in observed.items():
        kind, fixed = expected[path]
        if kind == "text":
            if scalar != fixed:
                raise ValueError(
                    "Phase66 summary text differs from reviewed vocabulary"
                )
        elif scalar is not None and (
            type(scalar) not in (int, float, bool) or not math.isfinite(scalar)
        ):
            raise ValueError("Invalid Phase66 summary scalar")


def public_pid_labels(value):
    """Name aggregate PID class indices without credential-like token keys."""
    names = {"predicted_token": "predicted_pid_class_index", "truth_token": "truth_pid_class_index"}
    if isinstance(value, dict):
        return {names.get(key, key): public_pid_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_pid_labels(item) for item in value]
    return value


def generate(root, output, summary):
    registry = json.loads(
        gzip.decompress(
            Path(__file__).with_name("phase66_metric_registry.json.gz").read_bytes()
        )
    )
    if (
        set(registry) != set(VIEWS)
        or summary.get("version") != "phase66-public-review-v1"
    ):
        raise ValueError("Incomplete Phase66 registry")
    downloads = []
    bundle_names, bundle_lookup, bundle_records = [], {}, []
    if {f["view"] for f in summary["files"]} != set(VIEWS):
        raise ValueError("Incomplete Phase66 views")
    for info in summary["files"]:
        view = info["view"]
        filename = "phase66-" + view.replace("_", "-") + ".json"
        source = "artifacts/codex/phase66_publication_20261004/" + filename + ".gz"
        if info["filename"] != filename or info["source"] != source:
            raise ValueError("Invalid Phase66 source")
        input_path = root / source
        if any(
            p.is_symlink() for p in (input_path, *input_path.parents)
        ) or not input_path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe Phase66 input path")
        data = gzip.decompress(input_path.read_bytes())
        if (
            len(data) != info["bytes"]
            or hashlib.sha256(data).hexdigest() != info["sha256"]
        ):
            raise ValueError("Phase66 export hash mismatch")
        value = json.loads(data)
        seen = []
        if (
            set(value)
            != {"version", "view", "arms", "metric_names", "columns", "records"}
            or value["version"] != "phase66-aggregate-columnar-v1"
            or value["view"] != view
            or value["arms"] != list(ARMS)
            or value["columns"] != ["arm_index", "metric_index", "value"]
        ):
            raise ValueError("Phase66 schema mismatch")
        names = value["metric_names"]
        if len(set(names)) != len(names) or any(
            not re.fullmatch(r"[A-Za-z0-9_.;=-]+", name) for name in names
        ):
            raise ValueError("Invalid Phase66 metric identity")
        for row in value["records"]:
            if (
                not isinstance(row, list)
                or len(row) != 3
                or type(row[0]) is not int
                or row[0] not in (0, 1)
                or type(row[1]) is not int
                or not 0 <= row[1] < len(names)
            ):
                raise ValueError("Invalid Phase66 metric identity")
            v = row[2]
            if v is not None and (
                type(v) not in (int, float, bool) or not math.isfinite(v)
            ):
                raise ValueError("Invalid Phase66 value")
            seen.append([ARMS[row[0]], names[row[1]]])
        if (
            seen != registry[view]
            or len({tuple(x) for x in seen}) != len(seen)
            or len(seen) != info["metric_count"]
        ):
            raise ValueError("Incomplete Phase66 metric export")
        for arm_index, metric_index, scalar in value["records"]:
            name = names[metric_index]
            if name not in bundle_lookup:
                bundle_lookup[name] = len(bundle_names)
                bundle_names.append(name)
            bundle_records.append(
                [VIEWS.index(view), arm_index, bundle_lookup[name], scalar]
            )
        destination = output / filename
        if destination.is_symlink() or (
            destination.exists() and destination.stat().st_nlink != 1
        ):
            raise ValueError("Unsafe Phase66 output path")
        destination.write_bytes(data)
        downloads.append(
            {
                k: info[k]
                for k in ("filename", "view", "sha256", "bytes", "metric_count")
            }
        )
    # Summary includes fixed review evidence only. Strip native source references before projection.
    public = {
        k: summary[k]
        for k in (
            "version",
            "status_date",
            "status",
            "arms",
            "paired",
            "source_hashes",
            "native_source_revision",
            "sealed_test_accessed",
            "train_events",
            "selection_events",
            "strict_events",
            "beam_events",
            "historical_reserved",
            "remaining_validation",
            "physical_momentum_resolution",
            "next_study_status",
        )
    }
    for key in ("geometry", "export_counts", "next_study", "context_exposure", "encoder_execution"):
        if key in summary:
            public[key] = summary[key]
    validate_summary(public)
    public = public_pid_labels(public)
    bundle = {
        "version": "phase66-complete-aggregate-bundle-v1",
        "views": list(VIEWS),
        "arms": list(ARMS),
        "metric_names": bundle_names,
        "columns": ["view_index", "arm_index", "metric_index", "value"],
        "records": bundle_records,
        "source_hashes": public["source_hashes"],
    }
    bundle_data = (
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(bundle_data) > 10 * 1024 * 1024:
        raise ValueError("Phase66 bundle exceeds publication capacity")
    bundle_path = output / "phase66-all-aggregate-metrics.json"
    if bundle_path.is_symlink() or (
        bundle_path.exists() and bundle_path.stat().st_nlink != 1
    ):
        raise ValueError("Unsafe Phase66 bundle path")
    bundle_path.write_bytes(bundle_data)
    bundle_binding = {
        "filename": bundle_path.name,
        "sha256": hashlib.sha256(bundle_data).hexdigest(),
        "bytes": len(bundle_data),
        "metric_count": len(bundle_records),
    }
    public["complete_aggregate_bundle"] = bundle_binding
    public["downloads"] = downloads
    data = (
        json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    (output / "phase66-review-metrics.json").write_bytes(data)
    return {
        "status": public["status"],
        "complete_aggregate_bundle": bundle_binding,
        "metric_count": sum(x["metric_count"] for x in downloads),
        "review_download": {
            "filename": "phase66-review-metrics.json",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
        "downloads": downloads,
    }


def render(record):
    if not record:
        return []
    lines = [
        "Phase66: teacher-only target support does not yield recursive benefit",
        "-" * 75,
        "",
        "Mixed contexts versus teacher-only: full LCAG 22/2966 versus 21/2966; exact full components 3/151 versus 1/151; coherent forests 3/100 versus 1/100. Neither passes all original gates. All exact successes are depth one.",
        "Teacher-only improves nontrivial full-source recall but loses precision. Paired LCAG uncertainty includes zero. Hold 70,000 training events; one fresh-seed Phase67 confirmation is bounded separately. No model is promoted.",
        "",
        "See :doc:`../../phase66` for geometry, the cumulative evidence, uncertainty and next-study gates.",
        "",
        ":download:`Complete Phase66 aggregate metric bundle <phase66-all-aggregate-metrics.json>`; :download:`review, uncertainty and download manifest <phase66-review-metrics.json>`.",
        "",
        "All registered native full/B-half views, saved-checkpoint metrics, model-only rankings and labelled oracle diagnostics retain their denominators. Unavailable physical momentum resolution is distinct from daughter-sum closure.",
        "",
    ]
    for d in record["downloads"]:
        lines += [
            f":download:`{d['view'].replace('_', ' ')} — {d['metric_count']} values <{d['filename']}>`.",
            "",
        ]
    return lines
