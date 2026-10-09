"""Native TRAIN loading guards without loading real checkpoints or events."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from scripts.phase84_hierarchy_train_context import (
    checked,
    train_uids,
    train_paths,
    validate_payload,
    validate_module,
    select_events,
)
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.evaluation.trained_context import load_trained_evaluation_context


def payload():
    spec = feature_spec_v4()
    return {
        "feature_specification": {"feature_spec_hash": spec["feature_spec_hash"]},
        "feature_contract": {
            "model_feature_contract_hash": spec["model_feature_contract_hash"],
            "reconstruction_constraint_policy": {"version": "fixture"},
            "pid_reconstruction_mode": "soft_expectation",
        },
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "architecture": {"fixture": True},
        "data_compatible_performance": True,
        "normalizer_state": {
            key: {} for key in ("track", "cluster", "common", "composite")
        },
        "config": {"target_policy": "complete_only", "max_events": None},
        "preprocessing_schema_version": "direct-mdst-tree-v4",
        "data_order_contract": {"dataset_index_hash": "index"},
        "split_manifest_hash": "splits",
    }


def test_binding_immutable_and_explicit(tmp_path):
    p = tmp_path / "source"
    p.write_text("frozen")
    b = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert checked(b) == p
    p.write_text("changed")
    with pytest.raises(ValueError, match="Immutable"):
        checked(b)
    with pytest.raises(ValueError, match="Missing"):
        checked({"path": str(p), "unrelated": "field"})


@pytest.mark.parametrize(
    "doc",
    [
        {"role": "validation", "event_uids": ["a"]},
        {"role": "train", "event_uids": []},
        {"role": "train", "event_uids": ["a", "a"]},
        {"role": "train", "event_uids": [None]},
        {"role": "train", "event_uids": [""]},
        {"role": "train", "event_uids": "a"},
    ],
)
def test_only_fixed_unique_train_identity_documents(doc):
    with pytest.raises(ValueError):
        train_uids(doc)


def test_train_identity_order_retained():
    assert train_uids({"role": "train", "event_uids": ["b", "a"]}) == ("b", "a")


def selection(tmp_path):
    return {
        "data_root": str(tmp_path),
        "selection_includes_test": False,
        "normalizer_scope": "train",
        "entries": [
            {"split": "train", "path": "train.parquet"},
            {"split": "validation", "path": "validation.parquet"},
        ],
    }


def test_train_paths_never_include_validation(tmp_path):
    d = selection(tmp_path)
    assert train_paths(d) == (tmp_path / "train.parquet",)
    for bad in ("../outside", "/absolute"):
        changed = deepcopy(d)
        changed["entries"][0]["path"] = bad
        with pytest.raises(ValueError, match="escapes"):
            train_paths(changed)
    d["selection_includes_test"] = True
    with pytest.raises(ValueError, match="Sealed"):
        train_paths(d)


@pytest.mark.parametrize(
    "change",
    [
        "spec",
        "feature",
        "pid_vocab",
        "architecture",
        "policy",
        "legacy",
        "compatible",
        "normalizer",
        "pid_mode",
        "pid_conflict",
        "prefix",
        "target",
    ],
)
def test_native_payload_guards(change):
    p = payload()
    assert validate_payload(p) == "soft_expectation"
    if change == "spec":
        p["feature_specification"]["feature_spec_hash"] = "wrong"
    elif change == "feature":
        p["feature_contract"]["model_feature_contract_hash"] = "wrong"
    elif change == "pid_vocab":
        p["pid_vocabulary_version"] = "wrong"
    elif change == "architecture":
        p["architecture"] = {}
    elif change == "policy":
        p["feature_contract"]["reconstruction_constraint_policy"] = {}
    elif change == "legacy":
        p["legacy_conflated_fraction"] = 0.1
    elif change == "compatible":
        p["data_compatible_performance"] = False
    elif change == "normalizer":
        del p["normalizer_state"]["track"]
    elif change == "pid_mode":
        p["feature_contract"]["pid_reconstruction_mode"] = ""
    elif change == "pid_conflict":
        p["config"]["pid_kinematics_mode"] = "hard"
    elif change == "prefix":
        p["config"]["max_events"] = 96
    elif change == "target":
        p["config"]["target_policy"] = "diagnostic_all"
    with pytest.raises(ValueError):
        validate_payload(p)


def test_native_index_split_schema_binding():
    p = payload()
    m = SimpleNamespace(
        source_schema_versions=("direct-mdst-tree-v4",),
        dataset_index={"index_hash": "index"},
        split_manifest_hash="splits",
    )
    validate_module(p, m)
    for name, value in [
        ("source_schema_versions", ("v3",)),
        ("dataset_index", {"index_hash": "other"}),
        ("split_manifest_hash", "other"),
    ]:
        modified = deepcopy(m)
        setattr(modified, name, value)
        with pytest.raises(ValueError):
            validate_module(p, modified)


@dataclass
class FixtureModule:
    input_paths: tuple
    events: tuple

    def iter_events(self, split, *, shuffle, event_uids):
        assert split == "train" and shuffle is False
        assert self.input_paths == (Path("/train"),)
        return iter(self.events)


def test_select_train_shards_exact_uids_and_original_tree_identity():
    a = SimpleNamespace(
        event_uid="a",
        level_ids=[0, 1, 4],
        recursive_leaf_source_mask="full source columns",
    )
    b = SimpleNamespace(
        event_uid="b",
        level_ids=[0, 2],
        recursive_leaf_source_mask="full source columns",
    )
    module = FixtureModule((Path("/train"), Path("/validation")), (a, b))
    restricted, events = select_events(module, ("b", "a"), (Path("/train"),))
    assert events == (b, a) and events[0] is b
    assert events[0].level_ids == [0, 2]
    assert module.input_paths == (Path("/train"), Path("/validation"))
    assert restricted.input_paths == (Path("/train"),)
    with pytest.raises(ValueError, match="coverage"):
        select_events(module, ("a", "missing"), (Path("/train"),))
    with pytest.raises(ValueError, match="absent"):
        select_events(module, ("a", "b"), (Path("/external"),))
    duplicate = FixtureModule(module.input_paths, (a, a))
    with pytest.raises(ValueError, match="uniqueness"):
        select_events(duplicate, ("a",), (Path("/train"),))


def test_shared_heldout_loader_remains_train_rejecting():
    with pytest.raises(ValueError, match="validation or test"):
        load_trained_evaluation_context(
            checkpoint="unused", data="unused", dataset_index="unused", split="train"
        )


def test_full_exclusion_check_precedes_checkpoint_load(tmp_path, monkeypatch):
    import json
    import scripts.phase84_hierarchy_train_context as loader

    inputs = {}
    for name in (
        "checkpoint",
        "pretraining_checkpoint",
        "selection",
        "dataset_index",
        "uid_manifest",
    ):
        p = tmp_path / name
        p.write_text(
            json.dumps({"role": "train", "event_uids": ["reserved-id"]})
            if name == "uid_manifest"
            else "{}"
        )
        inputs[name] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    called = []

    def isolation(uids, repo):
        called.append((tuple(uids), repo))
        raise ValueError("Training overlaps authenticated validation exclusions")

    monkeypatch.setattr(loader, "authenticate_train_isolation", isolation)
    monkeypatch.setattr(
        loader,
        "validate_checkpoint_pair",
        lambda *a, **kw: pytest.fail("Checkpoint touched before exclusions"),
    )
    with pytest.raises(ValueError, match="overlaps"):
        loader.load_hierarchy_train_context(**inputs, repo_root=tmp_path)
    assert called == [(("reserved-id",), tmp_path)]
