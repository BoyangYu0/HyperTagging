from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hypertagging.basf2_integration.runtime import (
    BeamSearchReconstructor,
    LEAF_MODE_FIXED,
    ModelOutputs,
    NODE_KIND_OTHER,
    Node,
    OnnxModelBundle,
)
from hypertagging.deployment.export_onnx import (
    ExportConfiguration,
    _validate_destination,
)
from hypertagging.deployment.onnx_contract import (
    file_sha256,
    load_bundle_manifest,
    with_manifest_hash,
)
from tests.helpers.toy_onnx_bundle import (
    MAX_CARDINALITY,
    MAX_NODES,
    N_QUERIES,
    PID_TOKENS,
    ReferenceSession,
    build_toy_onnx_bundle,
    reference_session_factory,
)


def _toy_leaves(*, shared_first_two: bool = False) -> tuple[Node, ...]:
    source_keys = (
        ("shared", "shared", "source:2", "source:3")
        if shared_first_two
        else tuple(f"source:{index}" for index in range(4))
    )
    return tuple(
        Node(
            node_id=index,
            input_token=0,
            current_token=0,
            pdg=22,
            p4=(0.1 * index, 0.0, 0.0, 1.0 + 0.1 * index),
            charge=0.0,
            level=0,
            kind_id=NODE_KIND_OTHER,
            leaf_mode_id=LEAF_MODE_FIXED,
            source_keys=frozenset({source_keys[index]}),
            source_kind="toy",
        )
        for index in range(4)
    )


def test_manifest_and_model_payloads_are_hash_checked(tmp_path: Path) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    manifest = load_bundle_manifest(manifest_path)

    assert manifest["contract"]["levels"] == [1, 2]
    assert manifest["contract"]["pid_tokens"] == list(PID_TOKENS)
    for level, entry in manifest["models"].items():
        assert file_sha256(manifest_path.parent / entry["file"]) == entry["sha256"]
        assert entry["runtime_inputs"] == (["node_mask"] if level == "1" else ["pid_labels"])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["reconstruction_policy"]["beam_width"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload hash mismatch"):
        load_bundle_manifest(manifest_path)


def test_exact_proposal_search_limit_is_enforced(tmp_path: Path) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["reconstruction_policy"]["max_proposals_per_hypothesis"] = 13
    manifest_path.write_text(
        json.dumps(with_manifest_hash(payload)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact-search limit of 12"):
        load_bundle_manifest(manifest_path)
    with pytest.raises(ValueError, match="exact-search limit of 12"):
        ExportConfiguration(max_proposals_per_hypothesis=13).validated()


def test_force_replacement_rejects_unexpected_bundle_entries(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    build_toy_onnx_bundle(bundle)
    (bundle / "unrelated.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="entries outside its validated bundle"):
        _validate_destination(bundle, force=True)
    assert (bundle / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"


def test_bundle_rejects_model_tampering_before_session_creation(tmp_path: Path) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    level_one = manifest_path.parent / "toy_level_1.onnx"
    level_one.write_bytes(level_one.read_bytes() + b"tampered")
    session_calls: list[str] = []

    def session_factory(filename: str, _options: object) -> ReferenceSession:
        session_calls.append(filename)
        return ReferenceSession(filename)

    with pytest.raises(ValueError, match="ONNX model hash mismatch for level 1"):
        OnnxModelBundle(manifest_path, session_factory=session_factory)
    assert session_calls == []


def test_bundle_rejects_release_and_session_contract_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    monkeypatch.setenv("BELLE2_RELEASE", "release-09-00-20")
    with pytest.raises(RuntimeError, match="not declared compatible"):
        OnnxModelBundle(manifest_path, session_factory=reference_session_factory)

    monkeypatch.delenv("BELLE2_RELEASE")

    class WrongInputSession(ReferenceSession):
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="common_features")]

    with pytest.raises(ValueError, match="ONNX inputs differ from manifest"):
        OnnxModelBundle(
            manifest_path,
            session_factory=lambda filename, _options: WrongInputSession(filename),
        )


def test_bundle_validates_runtime_output_shapes(tmp_path: Path) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")

    class WrongShapeSession(ReferenceSession):
        def run(
            self, output_names: list[str], feeds: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            values = super().run(output_names, feeds)
            values[0] = values[0][:, :1]
            return values

    bundle = OnnxModelBundle(
        manifest_path,
        session_factory=lambda filename, _options: WrongShapeSession(filename),
    )
    with pytest.raises(ValueError, match=r"object_logits shape \(1, 1\)"):
        bundle.score(1, {"node_mask": np.ones((1, MAX_NODES), dtype=np.bool_)})


def test_full_depth_onnx_beam_selects_globally_better_non_greedy_tree(
    tmp_path: Path,
) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    scorer = OnnxModelBundle(
        manifest_path, session_factory=reference_session_factory
    )
    reconstructor = BeamSearchReconstructor(scorer)

    result = reconstructor.reconstruct(_toy_leaves())

    assert result.completed
    assert result.stop_reason == "configured_root_reconstructed"
    assert result.levels_processed == 2
    assert result.model_manifest_sha256 == file_sha256(manifest_path)
    assert result.best.score == pytest.approx(0.70 + 0.95, abs=1e-6)
    assert result.best.accepted_by_level == ((4,), (5,))

    by_id = result.best.node_by_id()
    first_mother = by_id[4]
    root = by_id[result.best.completed_root_id]
    assert first_mother.current_token == 3  # K_S0: lower local score, better future
    assert first_mother.daughter_ids == (0, 1)
    assert first_mother.candidate_confidence == pytest.approx(0.70, abs=1e-6)
    assert root.current_token == 1
    assert root.pdg == 300553
    assert root.daughter_ids == (2, 3, 4)
    assert root.source_keys == frozenset(f"source:{index}" for index in range(4))
    assert all(by_id[node_id].parent_id >= 0 for node_id in range(5))

    # Both first-level alternatives survived the beam, and the locally better
    # 0.80 branch is not the selected full-tree result.
    first_level_confidences = sorted(
        hypothesis.node_by_id()[4].candidate_confidence for hypothesis in result.beam
    )
    assert first_level_confidences == pytest.approx([0.70, 0.80], abs=1e-6)
    repeated = reconstructor.reconstruct(_toy_leaves())
    assert repeated.best.fingerprint() == result.best.fingerprint()
    assert repeated.best.score == result.best.score


class _OverlappingSourceScorer:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = deepcopy(manifest)
        self.manifest["contract"]["levels"] = [1]
        self.manifest_sha256 = "scripted-overlapping-source-test"

    def score(self, level: int, inputs: dict[str, np.ndarray]) -> ModelOutputs:
        assert level == 1
        object_logits = np.full((1, N_QUERIES), 12.0, dtype=np.float32)
        type_logits = np.full(
            (1, N_QUERIES, len(PID_TOKENS)), -12.0, dtype=np.float32
        )
        type_logits[0, :, 2] = 12.0
        pointer_logits = np.full(
            (1, N_QUERIES, MAX_NODES), -12.0, dtype=np.float32
        )
        pointer_logits[0, 0, [0, 2]] = 12.0
        pointer_logits[0, 1, [1, 3]] = 12.0
        cardinality_logits = np.full(
            (1, N_QUERIES, MAX_CARDINALITY + 1), -12.0, dtype=np.float32
        )
        cardinality_logits[0, :, 2] = 12.0
        confidence_logits = np.asarray(
            [[np.log(0.8 / 0.2), np.log(0.7 / 0.3)]], dtype=np.float32
        )
        return ModelOutputs(
            object_logits=object_logits,
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=cardinality_logits,
            confidence_logits=confidence_logits,
            leaf_pid_logits=np.zeros(
                (1, MAX_NODES, len(PID_TOKENS)), dtype=np.float32
            ),
            current_p4=np.zeros((1, MAX_NODES, 4), dtype=np.float32),
        )


class _MultipleRootScorer:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = deepcopy(manifest)
        self.manifest["contract"]["levels"] = [1]
        policy = self.manifest["reconstruction_policy"]
        policy["root_requires_all_sources"] = False
        policy["allowed_mother_types_by_level"] = [[1, [1]]]
        self.manifest_sha256 = "scripted-multiple-root-test"

    def score(self, level: int, inputs: dict[str, np.ndarray]) -> ModelOutputs:
        assert level == 1
        object_logits = np.full((1, N_QUERIES), 12.0, dtype=np.float32)
        type_logits = np.full(
            (1, N_QUERIES, len(PID_TOKENS)), -12.0, dtype=np.float32
        )
        type_logits[0, :, 1] = 12.0
        pointer_logits = np.full(
            (1, N_QUERIES, MAX_NODES), -12.0, dtype=np.float32
        )
        pointer_logits[0, 0, [0, 1]] = 12.0
        pointer_logits[0, 1, [2, 3]] = 12.0
        cardinality_logits = np.full(
            (1, N_QUERIES, MAX_CARDINALITY + 1), -12.0, dtype=np.float32
        )
        cardinality_logits[0, :, 2] = 12.0
        confidence_logits = np.asarray(
            [[np.log(0.9 / 0.1), np.log(0.6 / 0.4)]], dtype=np.float32
        )
        return ModelOutputs(
            object_logits=object_logits,
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=cardinality_logits,
            confidence_logits=confidence_logits,
            leaf_pid_logits=np.zeros(
                (1, MAX_NODES, len(PID_TOKENS)), dtype=np.float32
            ),
            current_p4=np.zeros((1, MAX_NODES, 4), dtype=np.float32),
        )


def test_parallel_proposals_cannot_reuse_recursive_detector_source(
    tmp_path: Path,
) -> None:
    manifest_path = build_toy_onnx_bundle(tmp_path / "bundle")
    manifest = load_bundle_manifest(manifest_path)
    result = BeamSearchReconstructor(
        _OverlappingSourceScorer(manifest)
    ).reconstruct(_toy_leaves(shared_first_two=True))

    # Each proposal uses distinct node IDs, but nodes 0 and 1 describe the same
    # detector source.  They therefore cannot be accepted in parallel.
    assert result.best.score == pytest.approx(0.8, abs=1e-6)
    assert len([node for node in result.best.nodes if node.level == 1]) == 1
    for hypothesis in result.beam:
        accepted = [node for node in hypothesis.nodes if node.level == 1]
        for left_index, left in enumerate(accepted):
            for right in accepted[left_index + 1 :]:
                assert left.source_keys.isdisjoint(right.source_keys)


def test_multiple_root_queries_remain_singleton_beam_alternatives(
    tmp_path: Path,
) -> None:
    manifest = load_bundle_manifest(build_toy_onnx_bundle(tmp_path / "bundle"))
    result = BeamSearchReconstructor(_MultipleRootScorer(manifest)).reconstruct(
        _toy_leaves()
    )

    assert result.completed
    assert result.best.score == pytest.approx(0.9, abs=1e-6)
    root = result.best.node_by_id()[result.best.completed_root_id]
    assert root.daughter_ids == (0, 1)
    assert len([node for node in result.best.nodes if node.level == 1]) == 1
    assert all(
        len([node for node in hypothesis.nodes if node.level == 1]) == 1
        for hypothesis in result.beam
    )
