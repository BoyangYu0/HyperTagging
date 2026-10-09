"""Authenticate the complete inherited validation union and current reservations.

Only identity fields are consumed. The pinned Phase76 exclusion receipt supplies
historical primary/selection/development bindings, not an arbitrary caller list.
Its union also contains the full training corpus, so never use that combined UID
list itself as a validation exclusion set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARTIFACT_ROOT = Path("/project/agkuhr/users/boyang/data/HyperTagging_artifacts")
HISTORICAL_RELATIVE = "phase76_membership_20261009/exclusion-union.json"
HISTORICAL_SHA256 = "dde1894fb71f5402f7e93a95c77071b281ba86f8fc6a6223d9a8b9b16e15367a"
FINAL_PHASE76_DESIGNATION = {
    "artifact_relative_path": "phase76_membership_20261009/development-cohort.json",
    "sha256": "de1425f76d5f7dee9d99978003f4b1fe20951df7ce947c8f03f8ee3ce2b88fe5",
}
IDENTITY_FIELDS = (
    "event_uids",
    "evaluation_event_uids",
    "checkpoint_selection_event_uids",
    "development_event_uids",
)


def _read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Changed isolation binding: {path}")
    return json.loads(raw), {"path": str(path), "sha256": digest}


def _uids(document):
    found = False
    result = set()
    for key in IDENTITY_FIELDS:
        if key not in document:
            continue
        found = True
        values = document[key]
        if not isinstance(values, list) or any(
            not isinstance(x, str) or not x for x in values
        ):
            raise ValueError(f"Invalid identity field: {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate identities in field: {key}")
        result.update(values)
    if not found:
        raise ValueError("Historical/current cohort has no supported identity fields")
    return result


def authenticate_train_isolation(
    training_uids, repo_root, *, artifact_root=ARTIFACT_ROOT, historical_binding=None
):
    """Return a hash-bound membership-only audit; raise on any gap or overlap.

    Production defaults pin the full Phase76 historical receipt. An explicit
    alternative binding is for independently authenticated fixtures/contracts;
    callers must freeze that binding, never infer its hash from the same file.
    Current repository cohort files and every current supplementary reservation
    are added independently, including designations made after the baseline.
    """
    values = list(training_uids)
    if (
        not values
        or any(not isinstance(x, str) or not x for x in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Training identities must be nonempty and unique")
    train = set(values)
    repo_root, artifact_root = Path(repo_root), Path(artifact_root)
    baseline = historical_binding or {
        "path": str(artifact_root / HISTORICAL_RELATIVE),
        "sha256": HISTORICAL_SHA256,
    }
    old, old_binding = _read(baseline["path"], baseline["sha256"])
    if not old.get("bindings") or not isinstance(
        old.get("validation_excluded_count"), int
    ):
        raise ValueError("Incomplete historical exclusion receipt")
    bindings = [old_binding]
    historical = set()
    historical_cohorts = {}
    for item in old["bindings"]:
        document, checked = _read(item["path"], item["sha256"])
        historical.update(_uids(document))
        bindings.append(checked)
        p = Path(item["path"])
        if p.parent.name == "reconstruction" and "cohort" in p.name:
            historical_cohorts[p.name] = item["sha256"]
    if len(historical) != old["validation_excluded_count"]:
        raise ValueError("Historical validation union coverage mismatch")
    if not historical <= set(old["event_uids"]):
        raise ValueError("Historical binding identities absent from frozen union")
    union = set(historical)
    current_cohorts = sorted(
        (repo_root / "configs/reconstruction").glob("*cohort*.json")
    )
    if not set(historical_cohorts) <= {p.name for p in current_cohorts}:
        raise ValueError("Historical tracked cohort missing from current repository")
    for p in current_cohorts:
        document, checked = _read(p, historical_cohorts.get(p.name))
        union.update(_uids(document))
        bindings.append(checked)
    registry_path = (
        repo_root / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    registry, registry_binding = _read(registry_path)
    bindings.append(registry_binding)
    if historical_binding is None and FINAL_PHASE76_DESIGNATION not in [
        r["cohort_manifest"] for r in registry["reservations"]
    ]:
        raise ValueError("Final Phase76 designation missing from current registry")
    reservations = []
    for record in registry["reservations"]:
        manifest = record["cohort_manifest"]
        rel = Path(manifest["artifact_relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Reservation path must remain within artifact root")
        document, checked = _read(artifact_root / rel, manifest["sha256"])
        ids = _uids(document)
        if len(ids) != record["reserved_count"] or record["role"] != "validation":
            raise ValueError("Reservation count or role mismatch")
        union.update(ids)
        bindings.append(checked)
        reservations.append(
            {**checked, "count": len(ids), "training_overlap": len(ids & train)}
        )
    if not reservations:
        raise ValueError("Empty current supplementary registry")
    overlap = union & train
    if overlap:
        raise ValueError(
            f"Training overlaps authenticated validation exclusions: {len(overlap)}"
        )
    return {
        "status": "PASS_FULL_INHERITED_AND_CURRENT_VALIDATION_ISOLATION",
        "scope": "Pinned historical union plus every current tracked cohort and supplementary reservation; membership only, no new cohort or model admission",
        "training_count": len(train),
        "training_overlap": 0,
        "training_uid_set_sha256": hashlib.sha256(
            "\n".join(sorted(train)).encode()
        ).hexdigest(),
        "historical_validation_count": len(historical),
        "validation_union_count": len(union),
        "validation_union_uid_set_sha256": hashlib.sha256(
            "\n".join(sorted(union)).encode()
        ).hexdigest(),
        "current_cohort_count": len(current_cohorts),
        "reservations": reservations,
        "bindings": list({(x["path"], x["sha256"]): x for x in bindings}.values()),
        "validation_outcome_fields_consumed": False,
    }
