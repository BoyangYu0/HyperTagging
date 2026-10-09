"""Authenticate and designate fresh development identities, never primary/test."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.check_development_cohort_isolation import check_isolation  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(binding_):
    if sha(binding_["path"]) != binding_["sha256"]:
        raise ValueError("Hash mismatch: " + binding_["path"])
    return json.loads(Path(binding_["path"]).read_text())


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def uid_digest(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def ranked(values, role, seed):
    return sorted(
        values,
        key=lambda u: hashlib.sha256(f"phase74-{role}:{seed}:{u}".encode()).digest(),
    )


def prepare(artifacts, output, prior_discovery_root=None):
    prior_discovery_root = prior_discovery_root or output.parent
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Bounded two-CPU allocation required")
    import pyarrow.parquet as pq

    design = json.loads((output / "design-before-data.json").read_text())
    if (
        design["stage"] != "development"
        or design["heldout_per_category"] != 100
        or design["train_per_category"] != 256
    ):
        raise ValueError("Design differs from preregistered bounded designation")
    parent_path = artifacts / "phase72_review_20261008/phase73/contract.json"
    parent = read(
        {
            "path": str(parent_path),
            "sha256": "76d908f1e00734d849aa4721a4e92fcf41f3eadd7585bbf87014949527c0555e",
        }
    )
    selection, index = [read(parent["inputs"][key]) for key in ("selection", "index")]
    if selection["selection_includes_test"] or index["normalizer_scope"] != "train":
        raise ValueError("Data role/normalizer violation")
    excluded = set()
    exclusions = []
    # Authenticated transitive historical union used to reserve Phase72.
    cohort_binding = parent["exclusion_bindings"][0]
    old = read(cohort_binding)
    native_root = Path(cohort_binding["path"]).parents[2]
    previous = old["source_bindings"]["previous_validation_universe"]
    history_binding = {
        "path": str(native_root / previous["path"]),
        "sha256": previous["sha256"],
    }
    history = read(history_binding)
    excluded.update(history["event_uids"])
    exclusions.append(history_binding)
    registry_path = (
        ROOT / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    registry = json.loads(registry_path.read_text())
    for record in registry["reservations"]:
        b = record["cohort_manifest"]
        source = {
            "path": str(artifacts / b["artifact_relative_path"]),
            "sha256": b["sha256"],
        }
        reservation = read(source)
        if len(set(reservation["event_uids"])) != record["reserved_count"]:
            raise ValueError("Reservation count mismatch")
        excluded.update(reservation["event_uids"])
        exclusions.append(source)
    # Include every tracked historical selection/evaluation cohort, even if redundant.
    for path in sorted((ROOT / "configs/reconstruction").glob("*cohort*.json")):
        document = json.loads(path.read_text())
        for key in (
            "event_uids",
            "evaluation_event_uids",
            "checkpoint_selection_event_uids",
        ):
            excluded.update(document.get(key, []))
        exclusions.append(binding(path))
    excluded.update(parent["development_uids"])
    exclusions.append(parent["inputs"]["cohort"])
    candidates = json.loads(
        (prior_discovery_root / "candidate-discovery.json").read_text()
    )
    for candidate in candidates:
        document = read(candidate["candidate"])
        excluded.update(document[candidate.get("field", "event_uids")])
        exclusions.append(candidate["candidate"])
    categories = {"train": {}, "validation": {}}
    shards = []
    for entry in selection["entries"]:
        role = entry["split"]
        if (
            role not in categories
            or index["source_groups"][entry["source_file"]] != role
        ):
            raise ValueError("Source-role mismatch")
        path = Path(selection["data_root"]) / entry["path"]
        if (
            sha(path) != entry["parquet_sha256_reference"]
            or entry["schema_version"] != "direct-mdst-tree-v4"
        ):
            raise ValueError("Shard binding/schema mismatch")
        rows = pq.read_table(
            path,
            columns=["event_uid", "source_category", "source_file"],
            use_threads=False,
        ).to_pylist()
        if len(rows) != entry["event_count"]:
            raise ValueError("Shard row count mismatch")
        for row in rows:
            uid = row["event_uid"]
            if uid in categories["train"] or uid in categories["validation"]:
                raise ValueError("Duplicate identity across shards")
            if (
                row["source_category"] != entry["category"]
                or row["source_file"] != entry["source_file"]
            ):
                raise ValueError("Category/source provenance mismatch")
            categories[role][uid] = row["source_category"]
        shards.append({**binding(path), "role": role, "category": entry["category"]})
    if len(categories["train"]) != 70000 or len(categories["validation"]) != 160000:
        raise ValueError("Original corpus/universe changed")
    groups = {}
    inventory = {}
    training = {}
    for cat in design["categories"]:
        pool = {
            uid for uid, category in categories["validation"].items() if category == cat
        }
        available = pool - excluded - set(categories["train"])
        inventory[cat] = {
            "validation": len(pool),
            "excluded": len(pool & excluded),
            "available": len(available),
        }
        if len(available) < design["heldout_per_category"]:
            raise ValueError("Insufficient fresh development capacity: " + cat)
        groups[cat] = ranked(available, "heldout", design["seed"])[
            : design["heldout_per_category"]
        ]
        train_pool = {
            uid for uid, category in categories["train"].items() if category == cat
        }
        training[cat] = ranked(train_pool, "train", design["seed"])[
            : design["train_per_category"]
        ]
    dev = [u for cat in design["categories"] for u in groups[cat]]
    train = [u for cat in design["categories"] for u in training[cat]]
    candidate = {
        "version": design.get(
            "designation_version", "phase74-fresh-development-designation-v1"
        ),
        "stage": "development",
        "role": "validation",
        "sealed_test_role_access": "forbidden",
        "event_uids": dev,
        "event_uids_sha256": uid_digest(dev),
        "by_category": groups,
        "selection_seed": design["seed"],
        "checkpoint_selection": "fixed_final",
        "selection_uses_truth_or_outcomes": False,
        "primary_reservation": False,
    }
    isolation = check_isolation(
        candidate, [{"event_uids": sorted(excluded | set(categories["train"]))}]
    )
    if (
        isolation["excluded_development_count"]
        or len(dev) != 600
        or len(set(train)) != 1536
    ):
        raise ValueError("Cohort isolation/cardinality failed")
    write(
        output / "exclusion-union.json",
        {
            "event_uids": sorted(excluded | set(categories["train"])),
            "validation_excluded_count": len(excluded & set(categories["validation"])),
            "bindings": exclusions,
            "registry_before": binding(registry_path),
            "design": binding(output / "design-before-data.json"),
        },
    )
    write(output / "development-cohort.json", candidate)
    write(
        output / "training-cohort.json",
        {
            "event_uids": train,
            "by_category": training,
            "role": "train",
            "stage": "development",
            "original_corpus_events": 70000,
            "event_uids_sha256": uid_digest(train),
        },
    )
    write(
        output / "cohort-isolation-request.json",
        {
            "candidate": binding(output / "development-cohort.json"),
            "exclusions": [binding(output / "exclusion-union.json")],
        },
    )
    write(
        output / "data-admission.json",
        {
            "status": "PASS_FULL_IDENTITY_ADMISSION_NOT_MODEL_ADMISSION",
            "stage": "development",
            "inventory": inventory,
            "training_count": len(train),
            "development_count": len(dev),
            "original_corpus_count": 70000,
            "shards": shards,
            "inputs": parent["inputs"],
            "development": binding(output / "development-cohort.json"),
            "training": binding(output / "training-cohort.json"),
            "exclusion_union": binding(output / "exclusion-union.json"),
            "design": binding(output / "design-before-data.json"),
            "isolation": isolation,
            "selection": "no_outcome_labels_read_only_UID_category_source",
            "fresh_primary_reserved": 0,
            "sealed_test_access": False,
            "previous_failed_admission_superseded_not_validated": binding(
                prior_discovery_root / "cohort-isolation-failure.json"
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "inventory": inventory,
                "train": len(train),
                "heldout": len(dev),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-discovery-root", type=Path)
    args = parser.parse_args()
    prepare(args.artifacts.resolve(), args.output.resolve(), args.prior_discovery_root)
