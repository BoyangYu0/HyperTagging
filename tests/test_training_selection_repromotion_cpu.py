from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat

import pytest

import hypertagging.data.dataset_index as dataset_index_module
import hypertagging.data.selection_repromotion as repromotion
import hypertagging.data.training_selection as training_selection_module
import hypertagging.training.data_module as data_module


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "configs/training_selection/repromotion/train_035k_16key_repromotion.contract.json"
)


def _provenance(contract: dict) -> dict:
    return {
        "repository_head": "1" * 40,
        "implementation_tag": "fixture-clean-tag",
        "implementation_tag_object": "2" * 40,
        "implementation_tag_type": "tag",
        "tracked_worktree_clean": True,
        "base_code_commit": contract["base_code"]["commit"],
        "base_code_tag": contract["base_code"]["tag"],
        "base_tag_is_ancestor": True,
    }


def _package(output_root: Path):
    contract = repromotion.load_repromotion_contract(CONTRACT_PATH)
    return contract, repromotion.build_repromotion_package(
        contract,
        ROOT,
        implementation_provenance=_provenance(contract),
        output_namespace=str(output_root),
    )


def test_real_metadata_repromotion_has_golden_hashes_and_no_source_effects(
    tmp_path, monkeypatch
):
    contract = repromotion.load_repromotion_contract(CONTRACT_PATH)
    expected_paths = [
        ROOT / contract["inputs"][name]["path"]
        for name in ("inventory", "roles", "legacy_selection", "legacy_index")
    ]
    opened: list[Path] = []
    original_open = repromotion.os.open

    def guarded_open(path, flags, *args, **kwargs):
        candidate = Path(path)
        if kwargs.get("dir_fd") is None:
            if candidate not in expected_paths:
                raise AssertionError(f"non-metadata source opened: {candidate}")
            opened.append(candidate)
        return original_open(path, flags, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("path resolution or source effect reached")

    monkeypatch.setattr(repromotion.os, "open", guarded_open)
    monkeypatch.setattr(repromotion.Path, "resolve", bomb)
    package = repromotion.build_repromotion_package(
        contract,
        ROOT,
        implementation_provenance=_provenance(contract),
        output_namespace=str(tmp_path / "inert"),
    )
    assert opened == expected_paths
    assert hashlib.sha256(package.selection_bytes).hexdigest() == (
        "d0516ee4db09d7610a614475057af40282594882e03df1e9106e371a6710a660"
    )
    assert hashlib.sha256(package.index_bytes).hexdigest() == (
        "8d486304ba8018e03d4126271be24b5b43b15dd2c71f0e5dc8c847071b728097"
    )
    assert len(package.selection_bytes) == contract["target"]["selection_file_bytes"]
    assert len(package.index_bytes) == contract["target"]["dataset_index_file_bytes"]
    assert package.selection_payload["manifest_hash"] == (
        "9274e6412a25e8e900efca2c370565346c6d03926e845d7f63a2304a68e87ed4"
    )
    assert package.index_payload["index_hash"] == (
        "45419bb2d4d70f5344072d70e0f7241d6f235d926ef5bf5dc013ebac9569295e"
    )
    assert {
        entry["campaign_config_digest"]
        for entry in package.selection_payload["entries"]
    } == {"fb070c6c9805a6e782dd12708e4f57e71ce5d47b5c55a25c44f436545ac04fd7"}
    assert all(
        set(entry) == training_selection_module.SELECTION_ENTRY_KEYS
        for entry in package.selection_payload["entries"]
    )
    assert package.selection_payload["split_counts"] == {
        "test": 50000,
        "train": 35000,
        "validation": 50000,
    }
    assert package.index_payload["selection_contract"]["included_splits"] == [
        "train",
        "validation",
    ]
    assert set(package.index_payload["source_groups"].values()) == {
        "train",
        "validation",
    }
    assert all(
        value is False
        for key, value in package.receipt_payload["gates"].items()
        if key != "metadata_only"
    )
    assert package.receipt_payload["gates"]["metadata_only"] is True
    assert package.receipt_payload["outputs"][package.selection_name][
        "size_bytes"
    ] == len(package.selection_bytes)
    assert package.receipt_payload["outputs"][package.index_name]["size_bytes"] == len(
        package.index_bytes
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "true"))
def test_contract_requires_exact_closed_false_authorization_set(mutation):
    contract = repromotion.load_repromotion_contract(CONTRACT_PATH)
    if mutation == "missing":
        contract["authorization"].pop("submission_authorized")
    elif mutation == "extra":
        contract["authorization"]["future_authorization"] = False
    else:
        contract["authorization"]["science_authorized"] = True
    with pytest.raises(ValueError, match="authorization must remain false"):
        repromotion._validate_contract(contract)


def test_repromotion_semantic_diff_is_exactly_whitelisted(tmp_path):
    _contract, package = _package(tmp_path / "inert")
    legacy_selection = json.loads(
        (
            ROOT / "configs/training_selection/production_1m_20260812/train_035k.json"
        ).read_text()
    )
    restored_selection = deepcopy(package.selection_payload)
    restored_selection["manifest_hash"] = legacy_selection["manifest_hash"]
    for entry in restored_selection["entries"]:
        entry.pop("campaign_config_digest")
    assert restored_selection == legacy_selection

    legacy_index = json.loads(
        (
            ROOT / "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
            "train_035k.complete_only.index.json"
        ).read_text()
    )
    restored_index = deepcopy(package.index_payload)
    restored_index["index_hash"] = legacy_index["index_hash"]
    restored_index["selection_contract"]["selection_manifest_hash"] = legacy_index[
        "selection_contract"
    ]["selection_manifest_hash"]
    restored_index["selection_contract"]["fingerprint"] = legacy_index[
        "selection_contract"
    ]["fingerprint"]
    assert restored_index == legacy_index


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "projection", "uppercase")
)
def test_inventory_and_role_cross_bindings_fail_closed(mutation):
    contract = repromotion.load_repromotion_contract(CONTRACT_PATH)
    inventory = json.loads((ROOT / contract["inputs"]["inventory"]["path"]).read_text())
    roles = json.loads((ROOT / contract["inputs"]["roles"]["path"]).read_text())
    selection = json.loads(
        (ROOT / contract["inputs"]["legacy_selection"]["path"]).read_text()
    )
    selected_hash = selection["entries"][0]["inventory_entry_hash"]
    selected = next(
        entry
        for entry in inventory["entries"]
        if entry["inventory_entry_hash"] == selected_hash
    )
    if mutation == "missing":
        inventory["entries"].remove(selected)
    elif mutation == "duplicate":
        inventory["entries"].append(deepcopy(selected))
    elif mutation == "projection":
        selected["task_record_hash"] = "0" * 64
    else:
        selected["campaign_config_digest"] = selected["campaign_config_digest"].upper()
    with pytest.raises(ValueError):
        repromotion._promote_selection(inventory, roles, selection, contract["target"])


def test_o_excl_namespace_is_read_only_one_shot_and_does_not_move_inputs(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "inert-package"
    input_paths = [
        ROOT / "configs/training_selection/production_1m_20260812/train_035k.json",
        ROOT / "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
        "train_035k.complete_only.index.json",
    ]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths]
    _contract, package = _package(output_root)
    writes: list[str] = []
    original_write = repromotion._exclusive_write

    def observed_write(directory_fd, name, payload):
        writes.append(name)
        return original_write(directory_fd, name, payload)

    monkeypatch.setattr(repromotion, "_exclusive_write", observed_write)
    paths = repromotion.publish_repromotion_package(package, output_root)
    assert writes == [
        ".inert-package.claim.json",
        f".{package.selection_name}.staging",
        f".{package.index_name}.staging",
        repromotion.OUTPUT_RECEIPT_NAME,
    ]
    assert paths["claim"].is_file()
    assert paths["receipt"].read_bytes() == package.receipt_bytes
    for key in ("selection", "index", "receipt", "claim"):
        metadata = paths[key].stat()
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o555
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
    ] == before
    with pytest.raises(FileExistsError):
        repromotion.publish_repromotion_package(package, output_root)


def test_package_bytes_are_deterministic_for_same_namespace(tmp_path):
    output_root = tmp_path / "repeat"
    _contract, first = _package(output_root)
    _contract, second = _package(output_root)
    assert first.selection_bytes == second.selection_bytes
    assert first.index_bytes == second.index_bytes
    assert first.receipt_bytes == second.receipt_bytes


@pytest.mark.parametrize("collision", ("regular", "symlink"))
def test_atomic_promotion_never_replaces_collision(tmp_path, monkeypatch, collision):
    output_root = tmp_path / f"collision-{collision}"
    _contract, package = _package(output_root)
    original = repromotion._rename_noreplace
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"do-not-touch")
    injected = False

    def inject_collision(directory_fd, source, target):
        nonlocal injected
        if not injected:
            injected = True
            if collision == "regular":
                repromotion._exclusive_write(directory_fd, target, b"stale")
            else:
                repromotion.os.symlink(str(sentinel), target, dir_fd=directory_fd)
        return original(directory_fd, source, target)

    monkeypatch.setattr(repromotion, "_rename_noreplace", inject_collision)
    with pytest.raises(FileExistsError):
        repromotion.publish_repromotion_package(package, output_root)
    assert sentinel.read_bytes() == b"do-not-touch"
    assert not (output_root / repromotion.OUTPUT_RECEIPT_NAME).exists()
    with pytest.raises(FileExistsError):
        repromotion.publish_repromotion_package(package, output_root)


def test_truncated_staging_file_cannot_reach_cas_name_or_receipt(tmp_path, monkeypatch):
    output_root = tmp_path / "truncated"
    _contract, package = _package(output_root)
    original = repromotion._exclusive_write
    corrupted = False

    def truncate_after_write(directory_fd, name, payload):
        nonlocal corrupted
        original(directory_fd, name, payload)
        if name.startswith(".selection.") and name.endswith(".staging"):
            corrupted = True
            repromotion.os.chmod(
                name, 0o644, dir_fd=directory_fd, follow_symlinks=False
            )
            descriptor = repromotion.os.open(
                name,
                repromotion.os.O_WRONLY | repromotion.os.O_TRUNC,
                dir_fd=directory_fd,
            )
            try:
                repromotion.os.write(descriptor, b"truncated")
                repromotion.os.fsync(descriptor)
            finally:
                repromotion.os.close(descriptor)

    monkeypatch.setattr(repromotion, "_exclusive_write", truncate_after_write)
    with pytest.raises(ValueError, match="artifact metadata mismatch"):
        repromotion.publish_repromotion_package(package, output_root)
    assert corrupted is True
    assert not (output_root / package.selection_name).exists()
    assert not (output_root / repromotion.OUTPUT_RECEIPT_NAME).exists()
    with pytest.raises(FileExistsError):
        repromotion.publish_repromotion_package(package, output_root)


def test_partial_publication_is_permanently_fail_closed(tmp_path, monkeypatch):
    output_root = tmp_path / "partial"
    _contract, package = _package(output_root)
    original = repromotion._exclusive_write
    calls = 0

    def fail_after_claim(directory_fd, name, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected selection write failure")
        return original(directory_fd, name, payload)

    monkeypatch.setattr(repromotion, "_exclusive_write", fail_after_claim)
    with pytest.raises(OSError, match="injected"):
        repromotion.publish_repromotion_package(package, output_root)
    assert (tmp_path / ".partial.claim.json").is_file()
    assert output_root.is_dir()
    assert not (output_root / repromotion.OUTPUT_RECEIPT_NAME).exists()
    with pytest.raises(FileExistsError):
        repromotion.publish_repromotion_package(package, output_root)


def test_promoted_outputs_pass_public_preflight_without_payload_open(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "preflight"
    _contract, package = _package(output_root)
    paths = repromotion.publish_repromotion_package(package, output_root)
    expected_resolves = list(package.index_payload["paths"]) + [
        shard["path"] for shard in package.index_payload["shards"]
    ]
    resolved: list[str] = []
    original_resolve = dataset_index_module.Path.resolve

    def observed_resolve(path, *args, **kwargs):
        resolved.append(str(path))
        return original_resolve(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("payload or publication access reached")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", observed_resolve)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(dataset_index_module, "_sha256_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)
    monkeypatch.setattr(training_selection_module.pq, "ParquetFile", bomb)
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    metadata = data_module.preflight_dataset_index_data_binding(
        paths["selection"], paths["index"], required_splits=("train", "validation")
    )
    assert metadata["index_hash"] == package.index_payload["index_hash"]
    assert resolved == expected_resolves
    test_paths = {
        package.selection_payload["data_root"] + "/" + entry["path"]
        for entry in package.selection_payload["entries"]
        if entry["split"] == "test"
    }
    assert test_paths.isdisjoint(resolved)


def test_dirty_or_untagged_provenance_fails_before_metadata_open(tmp_path, monkeypatch):
    contract = repromotion.load_repromotion_contract(CONTRACT_PATH)
    provenance = _provenance(contract)
    provenance["tracked_worktree_clean"] = False
    monkeypatch.setattr(
        repromotion.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata opened before provenance gate")
        ),
    )
    with pytest.raises(ValueError, match="clean tagged"):
        repromotion.build_repromotion_package(
            contract,
            ROOT,
            implementation_provenance=provenance,
            output_namespace=str(tmp_path / "blocked"),
        )
