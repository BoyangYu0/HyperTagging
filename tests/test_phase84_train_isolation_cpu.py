"""Membership-only isolation union; no tensors, checkpoints or event outcomes."""

import hashlib
import json

import pytest

from scripts.phase84_train_isolation import authenticate_train_isolation


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path):
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    old = write(
        repo / "configs/reconstruction/old_cohort.json",
        {
            "event_uids": ["primary"],
            "checkpoint_selection_event_uids": ["selection"],
        },
    )
    hist = write(artifacts / "history.json", {"event_uids": ["historical"]})
    baseline = write(
        artifacts / "baseline.json",
        {
            "event_uids": ["historical", "primary", "selection", "train"],
            "validation_excluded_count": 3,
            "bindings": [old, hist],
        },
    )
    write(
        repo / "configs/reconstruction/new_cohort.json",
        {"evaluation_event_uids": ["new-cohort"]},
    )
    development = write(artifacts / "fresh.json", {"event_uids": ["new-dev"]})
    registry = (
        repo / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    write(
        registry,
        {
            "reservations": [
                {
                    "cohort_manifest": {
                        "artifact_relative_path": "fresh.json",
                        "sha256": development["sha256"],
                    },
                    "reserved_count": 1,
                    "role": "validation",
                }
            ]
        },
    )
    return repo, artifacts, baseline, registry


def run(case, train=("train",)):
    repo, artifacts, baseline, _ = case
    return authenticate_train_isolation(
        train, repo, artifact_root=artifacts, historical_binding=baseline
    )


def test_complete_union_includes_selection_current_and_new_reservations(case):
    receipt = run(case)
    assert receipt["historical_validation_count"] == 3
    assert receipt["validation_union_count"] == 5
    assert receipt["training_overlap"] == 0
    assert receipt["validation_outcome_fields_consumed"] is False
    expected = hashlib.sha256(
        "\n".join(
            sorted(["historical", "primary", "selection", "new-cohort", "new-dev"])
        ).encode()
    ).hexdigest()
    assert receipt["validation_union_uid_set_sha256"] == expected
    # The baseline's train corpus entries must not become validation exclusions.
    assert receipt["training_count"] == 1


@pytest.mark.parametrize(
    "uid", ["historical", "primary", "selection", "new-cohort", "new-dev"]
)
def test_overlap_any_scope_fails(case, uid):
    with pytest.raises(ValueError, match="overlaps"):
        run(case, [uid])


def test_historical_manifest_change_fails(case):
    repo, _, _, _ = case
    write(repo / "configs/reconstruction/old_cohort.json", {"event_uids": []})
    with pytest.raises(ValueError, match="Changed isolation binding"):
        run(case)


def test_current_reservation_change_fails(case):
    _, artifacts, _, _ = case
    write(artifacts / "fresh.json", {"event_uids": ["changed"]})
    with pytest.raises(ValueError, match="Changed isolation binding"):
        run(case)


def test_incomplete_authenticated_baseline_fails(case):
    repo, artifacts, baseline, registry = case
    old = json.loads((artifacts / "baseline.json").read_text())
    old["bindings"].pop()
    rebound = write(artifacts / "baseline.json", old)
    with pytest.raises(ValueError, match="coverage mismatch"):
        run((repo, artifacts, rebound, registry))


def test_current_registry_count_mismatch_fails(case):
    registry = case[3]
    document = json.loads(registry.read_text())
    document["reservations"][0]["reserved_count"] = 2
    write(registry, document)
    with pytest.raises(ValueError, match="count or role"):
        run(case)


def test_training_duplicates_fail(case):
    with pytest.raises(ValueError, match="unique"):
        run(case, ["train", "train"])
