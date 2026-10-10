"""Synthetic-only development designation and reservation regressions."""

import json

import pytest

from scripts import phase85_fresh_development_data as mod


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return mod.binding(path)


def preregistration(tmp_path):
    dependency = put(tmp_path / "dependency.json", {})
    return put(
        tmp_path / "preregistration.json",
        {
            "stage": "development",
            "sealed_test_access": False,
            "arms": ["connection_off", "connection_on"],
            "mechanism_evidence": {"report_binding": dependency},
            "data_and_compute_control": {
                "selection_seed": mod.SEED,
                "development_events": 600,
                "events_per_category": 100,
                "additional_training_updates": 0,
                "fixed_presence_threshold": 0.5,
                "checkpoint_selection": "fixed6000 only, both pre-existing finals",
                "checkpoints": {
                    "connection_off": dependency,
                    "connection_on": dependency,
                },
                "training_contract": dependency,
            },
        },
    )


def rows(n=3):
    return [
        {"uid": f"{c}-{i}", "category": c, "role": "validation"}
        for c in mod.CATEGORIES
        for i in range(n)
    ] + [{"uid": "train", "category": "charged", "role": "train"}]


def test_selection_deterministic_order_independent_and_excluded():
    data = rows()
    a = mod.select_metadata(data, {"charged-0"}, per_category=2)
    b = mod.select_metadata(list(reversed(data)), {"charged-0"}, per_category=2)
    assert a == b
    assert "charged-0" not in a[0]["charged"]
    assert all(len(v) == 2 for v in a[0].values())
    assert a[2] == {"train": "charged"}


@pytest.mark.parametrize("mutation", ["duplicate", "test", "empty", "shortage"])
def test_selection_rejects_invalid(mutation):
    data = rows()
    if mutation == "duplicate":
        data.append(data[0])
    if mutation == "test":
        data[0]["role"] = "test"
    if mutation == "empty":
        data[0]["uid"] = ""
    with pytest.raises(ValueError):
        mod.select_metadata(
            data,
            {"charged-0", "charged-1"} if mutation == "shortage" else set(),
            per_category=2,
        )


def candidate(tmp_path):
    document = {
        "stage": "development",
        "role": "validation",
        "primary_reservation": False,
        "sealed_test_role_access": "forbidden",
        "event_uids": [f"{c}-{i}" for c in mod.CATEGORIES for i in range(100)],
        "by_category": {c: [f"{c}-{i}" for i in range(100)] for c in mod.CATEGORIES},
    }
    return document, put(tmp_path / "new/cohort.json", document)


def test_atomic_append_and_duplicate_rejection(tmp_path):
    registry = tmp_path / "registry.json"
    before = put(registry, {"reservations": []})
    doc, bound = candidate(tmp_path)
    after = mod.append_registry(registry, before["sha256"], bound, doc, tmp_path)
    assert after["sha256"] != before["sha256"]
    saved = json.loads(registry.read_text())["reservations"]
    assert saved[0]["primary_count"] == 0
    assert saved[0]["reserved_count"] == 600
    with pytest.raises(ValueError, match="Duplicate"):
        mod.append_registry(registry, after["sha256"], bound, doc, tmp_path)


def test_concurrent_registry_change_rejected(tmp_path):
    registry = tmp_path / "registry.json"
    before = put(registry, {"reservations": []})
    doc, bound = candidate(tmp_path)
    registry.write_text('{"reservations": [], "new": true}')
    with pytest.raises(ValueError, match="changed"):
        mod.append_registry(registry, before["sha256"], bound, doc, tmp_path)


def test_candidate_hash_change_rejected(tmp_path):
    registry = tmp_path / "registry.json"
    before = put(registry, {"reservations": []})
    doc, bound = candidate(tmp_path)
    with open(bound["path"], "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Hash mismatch"):
        mod.append_registry(registry, before["sha256"], bound, doc, tmp_path)


def test_no_local_designation(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="scheduler"):
        mod.prepare({}, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_full_union_uses_authenticated_receipt_not_supplied_shortlist(
    tmp_path, monkeypatch
):
    old = put(tmp_path / "old.json", {"event_uids": ["primary", "olddev"]})
    recent = put(tmp_path / "recent.json", {"development_event_uids": ["newdev"]})
    called = []

    def authenticate(train, repo, **kwargs):
        called.append((train, repo, kwargs))
        return {"bindings": [old, recent], "status": "PASS_FULL"}

    monkeypatch.setattr(mod.isolation, "authenticate_train_isolation", authenticate)
    union, audit = mod.exclusion_union(["train"], tmp_path, tmp_path)
    assert union == {"train", "primary", "olddev", "newdev"}
    assert len(called) == 1
    assert audit["status"] == "PASS_FULL"
    put(tmp_path / "recent.json", {"development_event_uids": ["changed"]})
    with pytest.raises(ValueError, match="Hash mismatch"):
        mod.exclusion_union(["train"], tmp_path, tmp_path)


def test_runtime_revalidation_excludes_only_own_binding(tmp_path, monkeypatch):
    doc, bound = candidate(tmp_path)
    train = put(tmp_path / "train.json", {"event_uids": ["train"]})
    other = put(tmp_path / "other.json", {"event_uids": ["historical"]})
    repo = tmp_path / "repo"
    registry = (
        repo / "configs/reconstruction/supplementary_validation_reservations.json"
    )
    put(
        registry,
        {
            "reservations": [
                {
                    "cohort_manifest": {
                        "artifact_relative_path": "new/cohort.json",
                        "sha256": bound["sha256"],
                    }
                }
            ]
        },
    )
    selection = put(tmp_path / "selection.json", {})
    admission = put(
        tmp_path / "admission.json",
        {
            "development": bound,
            "preregistration": preregistration(tmp_path),
            "invalid_historical_cohort": put(
                tmp_path / "invalid.json",
                {"event_uids": [f"invalid-{i}" for i in range(60)]},
            ),
            "original_training": train,
            "registry_repo_root": str(repo),
            "artifact_root": str(tmp_path),
            "shards": [],
            "preserved_training_bindings": [train],
            "selection": selection,
            "index": selection,
        },
    )
    monkeypatch.setattr(
        mod, "exclusion_union", lambda *args: (set(), {"bindings": [bound, other]})
    )
    assert (
        mod.revalidate(admission)["status"]
        == "PASS_FULL_CURRENT_RESERVED_COHORT_ADMISSION"
    )
    # Re-reservation through a DIFFERENT manifest is a conflict, not subtracted.
    conflicting = put(
        tmp_path / "conflict.json", {"event_uids": [doc["event_uids"][0]]}
    )
    monkeypatch.setattr(
        mod,
        "exclusion_union",
        lambda *args: (set(), {"bindings": [bound, conflicting]}),
    )
    with pytest.raises(ValueError, match="conflicts"):
        mod.revalidate(admission)


def test_candidate_category_contract_rejected(tmp_path):
    registry = tmp_path / "registry.json"
    before = put(registry, {"reservations": []})
    doc, bound = candidate(tmp_path)
    doc["by_category"]["charged"] = doc["by_category"]["mixed"]
    bound = put(tmp_path / "new/cohort.json", doc)
    with pytest.raises(ValueError, match="Invalid development"):
        mod.append_registry(registry, before["sha256"], bound, doc, tmp_path)


def test_publisher_and_other_registry_repositories_rejected(tmp_path):
    with pytest.raises(ValueError, match="canonical research"):
        mod.validate_registry_repo(tmp_path)
    assert (
        mod.validate_registry_repo(mod.CANONICAL_RESEARCH_REPO)
        == mod.CANONICAL_RESEARCH_REPO
    )


def test_invalid_historical_cohort_count_and_hash(tmp_path):
    b = put(tmp_path / "invalid.json", {"event_uids": [str(i) for i in range(60)]})
    assert len(mod.invalid_historical_ids(b)) == 60
    changed = put(tmp_path / "invalid.json", {"event_uids": ["a"]})
    with pytest.raises(ValueError, match="Hash mismatch"):
        mod.invalid_historical_ids(b)
    with pytest.raises(ValueError, match="exactly60"):
        mod.invalid_historical_ids(changed)


def test_preregistration_controls_and_dependencies(tmp_path):
    bound = preregistration(tmp_path)
    assert mod.authenticate_preregistration(bound)["stage"] == "development"
    doc = mod.read(bound)
    doc["data_and_compute_control"]["selection_seed"] += 1
    invalid = put(tmp_path / "wrongplan.json", doc)
    with pytest.raises(ValueError, match="controls mismatch"):
        mod.authenticate_preregistration(invalid)
    put(tmp_path / "dependency.json", {"changed": True})
    with pytest.raises(ValueError, match="binding changed"):
        mod.authenticate_preregistration(bound)
