"""Deployment beams preserve future-visible state and committed resources."""

from dataclasses import replace

import numpy as np

from hypertagging.basf2_integration.runtime import BeamSearchReconstructor, Hypothesis, Proposal
from hypertagging.deployment.onnx_contract import load_bundle_manifest
from tests.helpers.toy_onnx_bundle import build_toy_onnx_bundle
from tests.test_onnx_full_decay_beam_cpu import _OverlappingSourceScorer, _toy_leaves


def _committed_state(reconstructor):
    return reconstructor._append_proposals(
        Hypothesis(_toy_leaves(shared_first_two=True)),
        [Proposal(0, 2, (0, 2), 0.9, 0.9, 0.9, 0.8, 0.8)],
        level=1,
    )


def test_deployment_identity_preserves_confidence_and_kinematics(tmp_path):
    manifest = load_bundle_manifest(build_toy_onnx_bundle(tmp_path / "bundle"))
    reconstructor = BeamSearchReconstructor(_OverlappingSourceScorer(manifest))
    state = _committed_state(reconstructor)
    changed = replace(state, nodes=state.nodes[:-1] + (replace(state.nodes[-1], candidate_confidence=0.2),))
    assert state.fingerprint() != changed.fingerprint()
    assert not np.array_equal(
        reconstructor.pack_hypothesis(state, 2)["common_features"],
        reconstructor.pack_hypothesis(changed, 2)["common_features"],
    )
    changed_leaf = replace(state, nodes=(replace(state.nodes[0], p4=(1.0, 0.0, 0.0, 2.0)),) + state.nodes[1:])
    assert state.fingerprint() != changed_leaf.fingerprint()


def test_deployment_rejects_cross_generation_alias_before_graph_and_decode(tmp_path):
    manifest = load_bundle_manifest(build_toy_onnx_bundle(tmp_path / "bundle"))
    scorer = _OverlappingSourceScorer(manifest)
    reconstructor = BeamSearchReconstructor(scorer)
    state = _committed_state(reconstructor)
    packed = reconstructor.pack_hypothesis(state, 2)
    assert not packed["pointer_validity_mask"][0, 1]
    assert packed["pointer_validity_mask"][0, 3]
    assert packed["pointer_validity_mask"][0, 4]
    # The scripted output independently proposes the other alias plus leaf 3;
    # host decoding must reject this even if a graph ignores its input mask.
    outputs = scorer.score(1, packed)
    proposals = reconstructor._decode_proposals(state, outputs, 2)
    assert not proposals


def test_deployment_composite_pointer_confidence_matches_offline_construction(tmp_path):
    import torch
    from hypertagging.models.heterogeneous import composite_physical_features_from_daughters

    manifest = load_bundle_manifest(build_toy_onnx_bundle(tmp_path / "bundle"))
    reconstructor = BeamSearchReconstructor(_OverlappingSourceScorer(manifest))
    state = _committed_state(reconstructor)
    daughters = [state.nodes[index] for index in (0, 2)]
    offline = composite_physical_features_from_daughters(
        daughter_mask=torch.ones(1, 2, dtype=torch.bool),
        p4=torch.tensor([[node.p4 for node in daughters]]),
        charge=torch.tensor([[node.charge for node in daughters]]),
        pid_labels=torch.zeros(1, 2, dtype=torch.long),
        pointer_confidence=torch.full((1, 2), 0.8),
    )
    packed = reconstructor.pack_hypothesis(state, 2)
    np.testing.assert_allclose(packed["composite_features"][0, 4, :9], offline["features"][0, :9].numpy())
