"""Evaluation-only metadata designation; never rerank TRAIN or fit normalization.

Production calls require a preregistered request and bounded scheduler allocation.
No payload labels are read: Parquet reads select UID/category/source only. Registry
mutation is serialized through one artifact-root advisory lock and checked again
before atomic replacement. All cooperating designation writers must use this lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import phase84_train_isolation as isolation  # noqa: E402
from scripts.check_development_cohort_isolation import check_isolation  # noqa: E402
from scripts.prepare_phase74_development_data import (  # noqa: E402
    binding,
    read,
    sha,
    uid_digest,
    write,
)  # noqa: E402

CATEGORIES = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
SEED = 20261010086
CANONICAL_RESEARCH_REPO = Path(
    "/home/b/Boyang.Yu/HyperTagging_uni/HyperTagging_phase74_training_20261008"
)


def validate_registry_repo(value):
    repo = Path(value).resolve()
    if repo != CANONICAL_RESEARCH_REPO:
        raise ValueError(
            "Registry mutation restricted to canonical research repository"
        )
    return repo


def invalid_historical_ids(bound):
    ids = isolation._uids(read(bound))
    if len(ids) != 60:
        raise ValueError("Historical invalid cohort must retain exactly60 identities")
    return ids


def authenticate_preregistration(bound):
    plan = read(bound)
    controls = plan["data_and_compute_control"]
    if (
        plan.get("stage") != "development"
        or plan.get("sealed_test_access") is not False
        or plan.get("arms") != ["connection_off", "connection_on"]
        or controls.get("selection_seed") != SEED
        or controls.get("development_events") != 600
        or controls.get("events_per_category") != 100
        or controls.get("additional_training_updates") != 0
        or controls.get("fixed_presence_threshold") != 0.5
        or controls.get("checkpoint_selection")
        != "fixed6000 only, both pre-existing finals"
        or set(controls.get("checkpoints", {})) != {"connection_off", "connection_on"}
    ):
        raise ValueError("Evaluation preregistration controls mismatch")
    for item in list(controls["checkpoints"].values()) + [
        controls["training_contract"],
        plan["mechanism_evidence"]["report_binding"],
    ]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(
                "Preregistered checkpoint/training/evidence binding changed"
            )
    return plan


def validate_selection_index(selection_binding, index_binding):
    """Use native pure schema/hash/selection linking before any metadata rows."""
    from hypertagging.data.dataset_index import load_dataset_index
    from hypertagging.training.data_module import _require_source_role_manifest_binding

    selection = read(selection_binding)
    read(index_binding)
    index = load_dataset_index(index_binding["path"], verify_sources=False)
    if index["selection_contract"]["mode"] != "source_role_manifest":
        raise ValueError("Original authenticated source-role manifest required")
    _require_source_role_manifest_binding(
        selection_binding["path"], index, required_splits=("train", "validation")
    )
    if (
        index["event_count"] != 230000
        or index["split_counts"].get("train") != 70000
        or index["split_counts"].get("validation") != 160000
        or index["split_counts"].get("test", 0)
    ):
        raise ValueError("Native index role counts differ from fixed universe")
    return selection, index


def select_metadata(rows, excluded, *, per_category=100, seed=SEED):
    """Pure metadata function; unknown roles, aliases and duplicates fail closed."""
    seen, train, validation = set(), {}, {}
    for row in rows:
        uid, role, category = row["uid"], row["role"], row["category"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("Duplicate or invalid identity")
        seen.add(uid)
        if role not in ("train", "validation"):
            raise ValueError("Forbidden source role")
        (train if role == "train" else validation)[uid] = category
    groups, inventory = {}, {}
    for category in CATEGORIES:
        pool = {u for u, c in validation.items() if c == category}
        available = pool - set(excluded) - set(train)
        inventory[category] = {
            "validation": len(pool),
            "excluded": len(pool & set(excluded)),
            "available": len(available),
        }
        if len(available) < per_category:
            raise ValueError("Insufficient fresh capacity: " + category)
        groups[category] = sorted(
            available,
            key=lambda u: (
                hashlib.sha256(f"phase85-heldout:{seed}:{u}".encode()).digest(),
                u,
            ),
        )[:per_category]
    return groups, inventory, train, validation


def exclusion_union(
    training_uids, repo_root, artifact_root, *, historical_binding=None
):
    """Reconstruct every authenticated UID binding behind the full-union audit."""
    audit = isolation.authenticate_train_isolation(
        training_uids,
        repo_root,
        artifact_root=artifact_root,
        historical_binding=historical_binding,
    )
    union = set(training_uids)
    for item in audit["bindings"]:
        document = read(item)
        if any(key in document for key in isolation.IDENTITY_FIELDS):
            union.update(isolation._uids(document))
    if not set(training_uids) <= union:
        raise ValueError("Missing training exclusions")
    return union, audit


def append_registry(
    registry_path, expected_sha, candidate_binding, candidate, artifact_root
):
    """Caller holds the artifact-root lock; CAS rejects concurrent noncooperators."""
    registry_path, artifact_root = Path(registry_path), Path(artifact_root).resolve()
    if sha(registry_path) != expected_sha:
        raise ValueError("Registry changed during designation")
    registry = read({"path": str(registry_path), "sha256": expected_sha})
    ids = set(candidate["event_uids"])
    if (
        candidate.get("stage") != "development"
        or candidate.get("primary_reservation") is not False
        or len(ids) != 600
        or len(candidate["event_uids"]) != 600
        or candidate.get("role") != "validation"
        or set(candidate.get("by_category", {})) != set(CATEGORIES)
        or any(len(candidate["by_category"][c]) != 100 for c in CATEGORIES)
        or set(u for c in CATEGORIES for u in candidate["by_category"][c]) != ids
    ):
        raise ValueError("Invalid development reservation")
    rel = Path(candidate_binding["path"]).resolve().relative_to(artifact_root)
    for record in registry["reservations"]:
        old = record["cohort_manifest"]
        path = artifact_root / old["artifact_relative_path"]
        if not path.resolve().is_relative_to(artifact_root):
            raise ValueError("Reservation escapes artifact root")
        if ids & isolation._uids(read({"path": str(path), "sha256": old["sha256"]})):
            raise ValueError("Duplicate reserved identity")
    if read(candidate_binding) != candidate:
        raise ValueError("Candidate binding differs")
    registry["reservations"].append(
        {
            "version": "phase85-pair-connection-development-designation-v1",
            "status": "DESIGNATED_DEVELOPMENT",
            "stage": "development",
            "role": "validation",
            "cohort_manifest": {
                "artifact_relative_path": str(rel),
                "sha256": candidate_binding["sha256"],
            },
            "reserved_count": 600,
            "primary_count": 0,
            "selection_count": 0,
            "by_category": {c: 100 for c in CATEGORIES},
            "sealed_test_accessed": False,
            "requirement": "Exclude all identities from future fitting, normalization, selection, development and primary reservations; fixed-final paired development assessment only.",
        }
    )
    temporary = registry_path.with_name(registry_path.name + ".phase85-new")
    with temporary.open("x") as stream:
        json.dump(registry, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if sha(registry_path) != expected_sha:
        raise ValueError("Registry changed before atomic replacement")
    os.replace(temporary, registry_path)
    return binding(registry_path)


def prepare(request, output):
    wall_start, cpu_start = time.monotonic(), time.process_time()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Two-CPU scheduler allocation required")
    if (
        request.get("stage") != "development"
        or request.get("seed") != SEED
        or request.get("per_category") != 100
        or request.get("selection_uses_outcomes") is not False
    ):
        raise ValueError("Preregistered designation controls differ")
    import pyarrow.parquet as pq

    authenticate_preregistration(request["preregistration"])
    output = Path(output).resolve()
    artifact_root = Path(request["artifact_root"]).resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("Output must remain in artifact root")
    output.mkdir(exist_ok=False, parents=True)
    write(output / "request.json", request)
    # Full union must consult canonical current repository, not stale frozen source.
    repo = validate_registry_repo(request["registry_repo_root"])
    registry_path = (
        repo / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    if str(registry_path) != str(Path(request["registry"]["path"]).resolve()):
        raise ValueError("Registry path is not canonical declared repository")
    lock = artifact_root / ".development-designation.lock"
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read(request["registry"])
        selection, index = validate_selection_index(
            request["selection"], request["index"]
        )
        if selection["selection_includes_test"] or index["normalizer_scope"] != "train":
            raise ValueError("Role/normalization contract violation")
        preserved = request["preserved_training_bindings"]
        if not preserved or any(sha(b["path"]) != b["sha256"] for b in preserved):
            raise ValueError("Preserved training/cache/normalizer binding changed")
        rows, shards = [], []
        for entry in selection["entries"]:
            role = entry["split"]
            if (
                role not in ("train", "validation")
                or index["source_groups"][entry["source_file"]] != role
                or entry["schema_version"] != "direct-mdst-tree-v4"
            ):
                raise ValueError("Source role/schema mismatch")
            path = Path(selection["data_root"]) / entry["path"]
            if sha(path) != entry["parquet_sha256_reference"]:
                raise ValueError("Shard hash mismatch")
            metadata = pq.read_table(
                path,
                columns=["event_uid", "source_category", "source_file"],
                use_threads=False,
            ).to_pylist()
            if len(metadata) != entry["event_count"]:
                raise ValueError("Shard count mismatch")
            for offset, row in enumerate(metadata):
                if (
                    row["source_category"] != entry["category"]
                    or row["source_file"] != entry["source_file"]
                ):
                    raise ValueError("Metadata provenance mismatch")
                rows.append(
                    {
                        "uid": row["event_uid"],
                        "role": role,
                        "category": row["source_category"],
                        "shard": str(path),
                        "row_index": offset,
                    }
                )
            shards.append(
                {
                    **binding(path),
                    "role": role,
                    "category": entry["category"],
                    "count": len(metadata),
                }
            )
        train_ids = [r["uid"] for r in rows if r["role"] == "train"]
        union, audit = exclusion_union(train_ids, repo, artifact_root)
        union.update(invalid_historical_ids(request["invalid_historical_cohort"]))
        groups, inventory, train, validation = select_metadata(rows, union)
        if len(train) != 70000 or len(validation) != 160000:
            raise ValueError("Authenticated original corpus/universe count changed")
        dev = [u for c in CATEGORIES for u in groups[c]]
        selected = set(dev)
        candidate = {
            "version": "phase85-fresh-development-v1",
            "stage": "development",
            "role": "validation",
            "event_uids": dev,
            "event_uids_sha256": uid_digest(dev),
            "by_category": groups,
            "selection_seed": SEED,
            "checkpoint_selection": "fixed_final6000",
            "selection_uses_truth_or_outcomes": False,
            "primary_reservation": False,
            "sealed_test_role_access": "forbidden",
            "selection_rows": [r for r in rows if r["uid"] in selected],
            "request": binding(output / "request.json"),
        }
        check = check_isolation(candidate, [{"event_uids": sorted(union)}])
        if check["excluded_development_count"]:
            raise ValueError("Development overlap")
        write(
            output / "exclusion-union.json",
            {
                "event_uids": sorted(union),
                "bindings": audit["bindings"],
                "authenticated_full_union": audit,
                "validation_excluded_count": len(union & set(validation)),
                "registry_before": request["registry"],
                "invalid_historical_cohort": request["invalid_historical_cohort"],
            },
        )
        write(
            output / "original-training-identities.json",
            {
                "event_uids": sorted(train),
                "role": "train",
                "event_uids_sha256": uid_digest(train),
            },
        )
        write(output / "development-cohort.json", candidate)
        write(
            output / "cohort-isolation-request.json",
            {
                "candidate": binding(output / "development-cohort.json"),
                "exclusions": [binding(output / "exclusion-union.json")],
            },
        )
        if any(sha(b["path"]) != b["sha256"] for b in preserved):
            raise ValueError("Training binding changed during designation")
        after = append_registry(
            registry_path,
            request["registry"]["sha256"],
            binding(output / "development-cohort.json"),
            candidate,
            artifact_root,
        )
        write(
            output / "data-admission.json",
            {
                "status": "PASS_FULL_IDENTITY_ADMISSION_NOT_MODEL_ADMISSION",
                "stage": "development",
                "preregistration": request["preregistration"],
                "development": binding(output / "development-cohort.json"),
                "exclusion_union": binding(output / "exclusion-union.json"),
                "registry_before": request["registry"],
                "invalid_historical_cohort": request["invalid_historical_cohort"],
                "registry_after": after,
                "selection": request["selection"],
                "index": request["index"],
                "shards": shards,
                "inventory": inventory,
                "original_training": binding(
                    output / "original-training-identities.json"
                ),
                "registry_repo_root": str(repo),
                "artifact_root": str(artifact_root),
                "training_count": len(train),
                "development_count": len(dev),
                "preserved_training_bindings": preserved,
                "isolation": check,
                "labels_read": False,
                "sealed_test_access": False,
                "fresh_primary_reserved": 0,
                "resource_accounting": {
                    "wall_seconds": time.monotonic() - wall_start,
                    "process_cpu_seconds": time.process_time() - cpu_start,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "metadata_rows": len(rows),
                    "shard_bytes_hashed": sum(
                        Path(b["path"]).stat().st_size for b in shards
                    ),
                    "optimizer_updates": 0,
                    "model_forwards": 0,
                },
            },
        )


def revalidate(admission_binding):
    """Admission after reservation: exclude exactly this immutable manifest only.

    Every other latest registry/cohort binding remains checked, including new
    reservations made after designation. Source shards and preserved TRAIN bytes
    are reauthenticated. This does not load event payloads or run a model.
    """
    admission = read(admission_binding)
    authenticate_preregistration(admission["preregistration"])
    candidate = read(admission["development"])
    train = isolation._uids(read(admission["original_training"]))
    _, audit = exclusion_union(
        train, admission["registry_repo_root"], admission["artifact_root"]
    )
    registry_path = (
        Path(admission["registry_repo_root"])
        / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    registry = json.loads(registry_path.read_text())
    candidate_path = Path(admission["development"]["path"]).resolve()
    matching = [
        r
        for r in registry["reservations"]
        if (
            Path(admission["artifact_root"])
            / r["cohort_manifest"]["artifact_relative_path"]
        ).resolve()
        == candidate_path
    ]
    if (
        len(matching) != 1
        or matching[0]["cohort_manifest"]["sha256"]
        != admission["development"]["sha256"]
    ):
        raise ValueError("Missing/duplicate/changed canonical reservation")
    excluded = set(train) | invalid_historical_ids(
        admission["invalid_historical_cohort"]
    )
    for item in audit["bindings"]:
        if Path(item["path"]).resolve() == candidate_path:
            continue
        document = read(item)
        if any(k in document for k in isolation.IDENTITY_FIELDS):
            excluded.update(isolation._uids(document))
    result = check_isolation(candidate, [{"event_uids": sorted(excluded)}])
    if result["excluded_development_count"]:
        raise ValueError("Latest full union conflicts with reserved cohort")
    for item in (
        admission["shards"]
        + admission["preserved_training_bindings"]
        + [admission["selection"], admission["index"]]
    ):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Runtime data/source binding changed")
    return {
        "status": "PASS_FULL_CURRENT_RESERVED_COHORT_ADMISSION",
        "admission": admission_binding,
        "registry": binding(registry_path),
        "isolation": result,
        "full_union_audit": audit,
        "development": admission["development"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--revalidate-admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revalidate_admission:
        write(args.output, revalidate(binding(args.revalidate_admission)))
    elif args.request:
        prepare(json.loads(args.request.read_text()), args.output)
    else:
        parser.error("request or revalidate-admission required")
