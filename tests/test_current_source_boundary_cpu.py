from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import (
    build_topology_safe_parent_negative_mask,
    build_tree_relation_targets,
    channel_nearest_neighbor_diagnostics,
    collapse_diagnostics,
    hyperbolic_pretraining_loss,
    pool_b_branch_embeddings,
)
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.models.level_autoregressive import (
    LevelReconstructionOutput,
    LevelAutoregressiveReconstructor,
)
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.models.stair_masks import stair_attention_mask
from hypertagging.preprocessing.basf2_mdst import (
    Basf2PreprocessConfig,
    _DirectMdstCollector,
    _select_data_independent_track_fit,
)
from hypertagging.preprocessing.channels import (
    b_root_discovery_diagnostics,
    event_channel_record,
)
from hypertagging.preprocessing.mdst_tree_builder import (
    EventTree,
    FourVector,
    TreeNode,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig,
    batched_free_rollout,
    evaluation_reference_rollout,
)
from hypertagging.training.pretrain_trainer import (
    ContextualPretrainingModel,
    PretrainConfig,
    _add_topology_labels,
    _pretraining_weights,
)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Hypothesis:
    def __init__(self, pdg: int) -> None:
        self.pdg = pdg

    def getPDGCode(self) -> int:
        return self.pdg


class _Fit:
    def __init__(self, p_value: float) -> None:
        self.p_value = p_value

    def getPValue(self) -> float:
        return self.p_value


class _Pair:
    def __init__(self, pdg: int, p_value: float) -> None:
        self.first = _Hypothesis(pdg)
        self.second = _Fit(p_value)


def test_release_specific_track_fit_selection_is_data_independent_and_auditable():
    class Track:
        def getTrackFitResults(self):
            return [_Pair(211, 0.2), _Pair(321, 0.9), _Pair(2212, 0.1)]

    selected = _select_data_independent_track_fit(
        Track(), pion_hypothesis=_Hypothesis(211)
    )
    assert selected.available
    assert selected.fit is not None and selected.fit.getPValue() == 0.9
    assert selected.hypothesis == "kaon"
    assert selected.method == "getTrackFitResults_max_p_value"
    assert selected.fallback_reason is None


def test_track_fit_selector_records_release_fallback_reason():
    pion_fit = _Fit(0.5)

    class Track:
        def getTrackFitResultWithClosestMass(self, hypothesis):
            assert hypothesis == "pion"
            return pion_fit

    selected = _select_data_independent_track_fit(
        Track(), pion_hypothesis="pion"
    )
    assert selected.fit is pion_fit
    assert selected.method == "getTrackFitResultWithClosestMass_pion"
    assert selected.fallback_reason == "getTrackFitResults_unavailable"


def test_pidlikelihood_plural_relation_and_detector_availability_are_recorded(
    tmp_path: Path,
):
    class PidLikelihood:
        def isAvailable(self, detector):
            return detector == "cdc"

        def getLogL(self, hypothesis):
            return -float(hypothesis)

    class Track:
        def getRelatedTo(self, name):
            return PidLikelihood() if name == "PIDLikelihoods" else None

    collector = _DirectMdstCollector(
        Basf2PreprocessConfig(("input.root",), tmp_path / "unused.parquet")
    )
    collector._charged_stable = lambda pdg: pdg
    collector._pid_detector_sets = {"cdc": "cdc", "klm": "klm"}
    values, available, status, detectors = collector._track_pid_likelihoods(
        Track()
    )
    assert all(available.values())
    assert all(value == "valid_likelihood_value" for value in status.values())
    assert detectors == {"cdc": True, "klm": False}
    assert math.isfinite(values["pion"])


def test_pidlikelihood_missing_relation_is_not_fabricated(tmp_path: Path):
    class Track:
        def getRelatedTo(self, _name):
            return None

    collector = _DirectMdstCollector(
        Basf2PreprocessConfig(("input.root",), tmp_path / "unused.parquet")
    )
    collector._charged_stable = lambda pdg: pdg
    collector._pid_detector_sets = {}
    values, available, status, detectors = collector._track_pid_likelihoods(
        Track()
    )
    assert values == {}
    assert not any(available.values())
    assert set(status.values()) == {"relation_missing"}
    assert detectors == {}


def test_klm_cluster_collection_uses_reconstructed_momentum_and_explicit_kind(
    tmp_path: Path,
):
    class Vector:
        def Px(self): return 0.1
        def Py(self): return -0.2
        def Pz(self): return 0.3
        def E(self): return 0.6

    class Position:
        def X(self): return 1.0
        def Y(self): return 2.0
        def Z(self): return 3.0

    class Cluster:
        def getArrayIndex(self): return 7
        def getMomentum(self): return Vector()
        def getClusterPosition(self): return Position()
        def getEnergy(self): return 0.6
        def getMomentumMag(self): return 0.4
        def getTime(self): return 1.2
        def getLayers(self): return 6
        def getInnermostLayer(self): return 2
        def getAssociatedEclClusterFlag(self): return 0
        def getRelatedTo(self, _name): return None

    collector = _DirectMdstCollector(
        Basf2PreprocessConfig(("input.root",), tmp_path / "unused.parquet")
    )
    collector.klm_clusters = [Cluster()]
    records = collector._collect_klm_clusters()
    assert len(records) == 1
    record = records[0]
    assert record.node_kind == "klm_cluster"
    assert record.leaf_kinematics_mode == "klm_cluster"
    assert record.p4.as_tuple() == (0.1, -0.2, 0.3, 0.6)
    assert record.truth_pdg is None
    assert record.klm_features["layers"] == 6.0


def test_b_root_fallback_is_false_when_no_b_candidates_exist():
    tree = EventTree(event_id=1)
    tree.add_node(
        TreeNode(0, 211, 1.0, FourVector(0.1, 0.0, 0.0, 0.2))
    )
    tree.root_ids = [0]
    diagnostics = b_root_discovery_diagnostics(tree)
    channel = event_channel_record(tree)
    assert diagnostics["reason"] == "no_retained_resonance_or_b"
    assert channel["b_root_discovery_valid"] is False
    assert channel["b_root_discovery_fallback"] is False
    assert channel["b1_root_id"] == channel["b2_root_id"] == -1
    assert channel["active_channel_loss_branch_count"] == 0


def test_production_forward_does_not_retain_attention_by_default():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
        n_heads=2,
        n_context_layers=1,
        use_contextual_encoder=True,
        use_relation_bias=True,
        use_hyperbolic_relation_refinement=True,
    )
    production = model(batch, target_level=1)
    diagnostic = model(batch, target_level=1, return_attention=True)
    assert production.attention_weights is None
    assert production.physical_attention_weights is None
    assert production.hyperbolic_attention_weights is None
    assert diagnostic.physical_attention_weights is not None
    assert diagnostic.hyperbolic_attention_weights is not None


class _ScriptedRolloutModel:
    """Deterministic decoder used by both batched and reference rollouts."""

    def __call__(
        self,
        batch,
        *,
        target_level,
        pid_kinematics_mode_override=None,
        pid_temperature_override=None,
        return_attention=False,
    ):
        del pid_kinematics_mode_override, pid_temperature_override, return_attention
        batch_size, node_count = batch["node_mask"].shape
        query_count = 2
        device = batch["node_mask"].device
        context = batch["node_mask"] & (batch["level_ids"] < target_level)
        pointer_logits = torch.full(
            (batch_size, query_count, node_count), -20.0, device=device
        )
        object_logits = torch.full(
            (batch_size, query_count), -20.0, device=device
        )
        type_logits = torch.full(
            (batch_size, query_count, len(PDG_TOKENS)), -20.0, device=device
        )
        cardinality_logits = torch.full(
            (batch_size, query_count, 7), -20.0, device=device
        )
        cardinality_logits[..., 2] = 20.0
        if target_level == 1:
            leaves = batch["node_mask"] & (batch["level_ids"] == 0)
            rank = leaves.long().cumsum(dim=-1) - 1
            first = leaves & (rank < 2)
            second = leaves & (rank >= 2) & (rank < 4)
            pointer_logits[:, 0] = torch.where(
                first, torch.full_like(rank, 20.0), pointer_logits[:, 0]
            )
            pointer_logits[:, 1] = torch.where(
                second, torch.full_like(rank, 20.0), pointer_logits[:, 1]
            )
            object_logits[:, 0] = torch.where(
                leaves.sum(dim=-1) >= 2,
                torch.full_like(object_logits[:, 0], 20.0),
                object_logits[:, 0],
            )
            object_logits[:, 1] = torch.where(
                leaves.sum(dim=-1) >= 4,
                torch.full_like(object_logits[:, 1], 20.0),
                object_logits[:, 1],
            )
            type_logits[..., 4] = 20.0
        elif target_level == 2:
            previous = batch["node_mask"] & (batch["level_ids"] == 1)
            pointer_logits[:, 0] = torch.where(
                previous, torch.full_like(previous, 20.0, dtype=torch.float32),
                pointer_logits[:, 0],
            )
            object_logits[:, 0] = torch.where(
                previous.sum(dim=-1) >= 2,
                torch.full_like(object_logits[:, 0], 20.0),
                object_logits[:, 0],
            )
            type_logits[..., 1] = 20.0
        else:
            type_logits[..., 1] = 20.0
        confidence_logits = torch.full_like(object_logits, 20.0)
        pointer = MotherPointerOutput(
            object_logits=object_logits,
            type_logits=type_logits,
            pointer_logits=pointer_logits,
            cardinality_logits=cardinality_logits,
            confidence_logits=confidence_logits,
        )
        hidden = torch.zeros(batch_size, node_count, 4, device=device)
        hyper = torch.zeros(batch_size, node_count, 2, device=device)
        relation = torch.zeros(batch_size, node_count, node_count, device=device)
        return LevelReconstructionOutput(
            target_level=target_level,
            pointer=pointer,
            node_embeddings=hidden,
            hyperbolic_embeddings=hyper,
            context_mask=context,
            relation_bias=relation,
            attention_weights=None,
            physical_relation_bias=relation,
            physical_attention_weights=None,
            hyperbolic_relation_bias=None,
            hyperbolic_attention_weights=None,
            final_contextual_embeddings=hidden,
            tree_projection=hidden,
            reconstruction_projection=hidden,
            channel_projection=hidden,
        )


def test_full_batched_rollout_matches_independent_reference_and_stops_per_event():
    events = [tiny_level_events()[1], tiny_level_events()[0]]
    full = collate_level_events(events, max_query_slots=4).to_dict()
    model = _ScriptedRolloutModel()
    config = RolloutConfig(
        max_level=3,
        root_types=(1,),
        exclusive_final=True,
        allow_competing=False,
    )
    batched = batched_free_rollout(model, full, config=config)
    assert batched.root_completed_mask.tolist() == [True, False]
    assert batched.levels_completed.tolist() == [2, 1]
    assert batched.stop_code.tolist() == [2, 1]
    for event_index, event in enumerate(events):
        single = collate_level_events([event], max_query_slots=4).to_dict()
        reference = evaluation_reference_rollout(
            model, single, mode="predicted", config=config
        )
        active = batched.batch["node_mask"][event_index]
        for name in ("p4", "pid_labels", "level_ids", "node_ids"):
            torch.testing.assert_close(
                batched.batch[name][event_index, active],
                reference.batch[name][0, reference.batch["node_mask"][0]],
            )
        batched_adjacency = batched.batch["daughter_adjacency"][
            event_index
        ][active][:, active]
        reference_active = reference.batch["node_mask"][0]
        torch.testing.assert_close(
            batched_adjacency,
            reference.batch["daughter_adjacency"][0][reference_active][
                :, reference_active
            ],
        )


def test_full_revised_production_config_activates_modules_and_objectives():
    root = Path(__file__).resolve().parents[1]
    pretrain_args = _load_script("train_hyperbolic_pretrain.py").parse_args(
        ["--config", str(root / "configs/hyperbolic_pretrain.yaml")]
    )
    reconstruction_args = _load_script("train_level_reconstruction.py").parse_args(
        ["--config", str(root / "configs/level_reconstruction.yaml")]
    )
    assert pretrain_args.model_preset == "production_baseline"
    assert pretrain_args.ablation == "full_revised"
    assert pretrain_args.best_metric == "validation_full_training_objective"
    assert all(
        getattr(pretrain_args, name) > 0
        for name in (
            "lca_relation_weight",
            "parent_ranking_weight",
            "exact_tree_distance_weight",
            "radius_depth_weight",
            "channel_weight",
            "variance_weight",
            "covariance_weight",
            "leaf_pid_weight",
            "corruption_class_weight",
            "candidate_correctness_weight",
            "hard_negative_weight",
        )
    )
    assert reconstruction_args.model_preset == "production_baseline"
    assert reconstruction_args.ablation == "full_revised"
    assert reconstruction_args.scheduled_sampling_schedule == "linear"
    assert reconstruction_args.target_policy == "complete_only"
    ablation = ALL_ABLATIONS["full_revised"]
    assert all(
        (
            ablation.heterogeneous_adapters,
            ablation.contextual_euclidean,
            ablation.relation_attention,
            ablation.hyperbolic_relation_attention,
            ablation.lca_relation,
            ablation.parent_ranking,
            ablation.exact_tree_distance,
            ablation.radius_depth,
            ablation.variance_covariance,
            ablation.channel_supervision,
            ablation.leaf_pid,
            ablation.scheduled_sampling,
            ablation.pretrained_encoder_transfer,
        )
    )
    model = ContextualPretrainingModel(
        d_model=16,
        hyper_dim=4,
        n_heads=2,
        n_context_layers=1,
        use_contextual_encoder=ablation.contextual_euclidean,
        use_physical_relations=ablation.relation_attention,
        use_hyperbolic_relations=ablation.hyperbolic_relation_attention,
    )
    events = [
        heterogeneous_from_level_event(tiny_level_events()[1]),
        heterogeneous_from_level_event(tiny_level_events()[1]),
    ]
    batch = collate_heterogeneous_events(events)
    batch["b_side"][:] = torch.tensor(
        [0, 0, 1, 1, 0, 1, -1], dtype=torch.long
    )
    batch["event_ids"] = torch.tensor([1, 2])
    _add_topology_labels(batch)
    mask = stair_attention_mask(batch["level_ids"], batch["node_mask"])
    encoded = model.encoder(batch, attention_mask=mask)
    relation_logits = model.relation_head(encoded.tree_projection)
    targets, pair_mask = build_tree_relation_targets(
        parent_ids=batch["parent_ids"],
        lca_depth=batch["lca_depth"],
        level_ids=batch["level_ids"],
        node_mask=batch["node_mask"],
        b_side=batch["b_side"],
        lca_node_id=batch["lca_node_id"],
        edges_to_lca_from_i=batch["edges_to_lca_from_i"],
        edges_to_lca_from_j=batch["edges_to_lca_from_j"],
    )
    negative_mask = build_topology_safe_parent_negative_mask(
        targets, batch["node_mask"], batch["ancestor_descendant_relation"]
    )
    branches, branch_mask = pool_b_branch_embeddings(
        encoded.channel_projection,
        batch["b_side"],
        batch["node_mask"],
        mode="mean_all",
        level_ids=batch["level_ids"],
    )
    channel_ids = torch.tensor([[101, 202], [101, 202]])
    config = PretrainConfig(
        data="unused",
        output_dir="unused",
        model_preset="production_baseline",
    )
    weights = _pretraining_weights(config)
    assert all(weights[name] > 0 for name in (
        "lca", "parent", "tree_distance", "depth", "channel", "var", "cov"
    ))
    loss = hyperbolic_pretraining_loss(
        z=encoded.hyperbolic_embeddings,
        tree_relation_logits=relation_logits,
        tree_relation_targets=targets,
        tree_relation_mask=pair_mask,
        lca_depth=batch["lca_depth"],
        exact_tree_path_distance=batch["exact_tree_path_distance"],
        parent_negative_mask=negative_mask,
        parent_ids=batch["parent_ids"],
        level_ids=batch["level_ids"],
        node_mask=batch["node_mask"],
        b_side=batch["b_side"],
        node_kind_ids=batch["node_kind_ids"],
        event_ids=batch["event_ids"],
        channel_embeddings=branches,
        channel_mask=branch_mask,
        full_truth_channel_ids=channel_ids,
        reconstructable_channel_ids=channel_ids,
        weights=weights,
        full_event_max_level=batch["full_event_max_level"],
        depth_from_retained_root=batch["depth_from_retained_root"],
        distance_to_nearest_retained_root=batch[
            "distance_to_nearest_retained_root"
        ],
    )
    assert set(("lca", "parent", "tree_distance", "depth", "channel", "var", "cov")) <= set(loss.components)
    for name in (
        "active_denominator_lca",
        "active_denominator_tree_distance",
        "active_denominator_radius",
        "active_denominator_variance",
        "active_denominator_covariance",
        "parent_ranking_accuracy_denominator",
        "channel_active_anchors",
    ):
        assert float(loss.diagnostics[name]) > 0, name
    leaf_mask = batch["node_mask"] & (batch["level_ids"] == 0)
    assert int(leaf_mask.sum()) > 0
    total = loss.total + torch.nn.functional.cross_entropy(
        model.leaf_pid_head(encoded.reconstruction_projection)[leaf_mask],
        batch["truth_pid_labels"][leaf_mask],
    )
    total.backward()
    assert model.encoder.hyper_projection.weight.grad is not None


def test_conditional_collapse_and_channel_neighbor_diagnostics_are_reported():
    torch.manual_seed(17)
    tangent = torch.randn(2, 6, 4) * 0.1
    from hypertagging.models.hyperbolic import expmap0

    z = expmap0(tangent)
    mask = torch.ones(2, 6, dtype=torch.bool)
    diagnostics = collapse_diagnostics(
        z,
        mask,
        level_ids=torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1),
        node_kind_ids=torch.tensor([[1, 1, 3, 3, 3, 3]]).expand(2, -1),
        b_side=torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1),
    )
    assert diagnostics["within_level_effective_rank"] > 0
    assert diagnostics["within_node_kind_effective_rank"] > 0
    assert diagnostics["within_b_side_effective_rank"] > 0
    channels = channel_nearest_neighbor_diagnostics(
        torch.tensor([[[1.0, 0.0], [0.9, 0.1]], [[0.0, 1.0], [0.1, 0.9]]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([[11, 11], [22, 22]]),
    )
    assert channels["channel_nearest_neighbor_anchor_count"] == 4
    assert 0 <= channels["channel_nearest_neighbor_unique_fraction"] <= 1
    assert channels["channel_nearest_neighbor_same_label_fraction"] == 1
