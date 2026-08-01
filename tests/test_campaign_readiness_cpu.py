from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from hypertagging.losses.level_reconstruction import query_proposal_repulsion_loss
from hypertagging.models.heterogeneous import dispatch_node_kind_adapters
from hypertagging.preprocessing.basf2_mdst import (
    Basf2PreprocessConfig,
    TRACK_FIT_POLICY_CANONICAL_PION_V1,
    TRACK_FIT_POLICY_MAX_P_VALUE_V1,
    _select_data_independent_track_fit,
)
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.level_rollout import RolloutConfig
from hypertagging.training.checkpoint_selection import (
    checkpoint_track_decisions,
    initial_track_values,
)
from hypertagging.training.pretrain_trainer import objective_preflight_report
from scripts.train_level_reconstruction import parse_args as parse_reconstruction_args


def test_teacher_loss_and_rollout_f1_select_independent_checkpoint_tracks():
    state = initial_track_values()
    teacher_metrics = {
        "validation_loss_total": 0.4,
        "validation_teacher_forced_terms": 8.0,
        "rollout_validation_events": 0.0,
    }
    state, selected = checkpoint_track_decisions(teacher_metrics, state)
    assert [track.filename for track in selected] == ["best_teacher_forced.pt"]

    rollout_metrics = {
        "validation_loss_total": 0.6,
        "validation_teacher_forced_terms": 8.0,
        "predicted_edge_f1": 0.75,
        "predicted_tree_validity_rate": 1.0,
        "rollout_validation_events": 4.0,
    }
    state, selected = checkpoint_track_decisions(rollout_metrics, state)
    assert {track.filename for track in selected} == {
        "best_rollout_edge_f1.pt",
        "best_rollout_tree_validity.pt",
    }
    assert state["validation_loss_total"] == pytest.approx(0.4)
    assert state["predicted_edge_f1"] == pytest.approx(0.75)


def test_query_repulsion_masks_no_object_overlap_and_is_permutation_invariant():
    logits = torch.tensor(
        [[[2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [2.0, -2.0, 1.0]]],
        requires_grad=True,
    )
    active = torch.tensor([[True, True, False]])
    disjoint = torch.tensor(
        [[[True, False, False], [False, True, False], [False, False, False]]]
    )
    loss = query_proposal_repulsion_loss(
        logits,
        active_query_mask=active,
        matched_pointer_targets=disjoint,
    )
    loss.backward()
    assert loss > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))

    permutation = torch.tensor([1, 0, 2])
    permuted = query_proposal_repulsion_loss(
        logits.detach()[:, permutation],
        active_query_mask=active[:, permutation],
        matched_pointer_targets=disjoint[:, permutation],
    )
    assert permuted == pytest.approx(float(loss.detach()))

    overlapping = disjoint.clone()
    overlapping[:, 1] = overlapping[:, 0]
    ignored = query_proposal_repulsion_loss(
        logits.detach(),
        active_query_mask=active,
        matched_pointer_targets=overlapping,
    )
    assert ignored == 0


def test_node_kind_adapter_dispatch_is_name_based_under_reordered_ids():
    mapping = {
        "unknown": 0,
        "track": 3,
        "ecl_cluster": 1,
        "composite": 2,
        "klm_cluster": 4,
        "other": 5,
    }
    kinds = torch.tensor([[3, 1, 4, 2, 5]])
    values = {
        "track": torch.full((1, 5, 1), 11.0),
        "ecl_cluster": torch.full((1, 5, 1), 22.0),
        "klm_cluster": torch.full((1, 5, 1), 33.0),
        "composite": torch.full((1, 5, 1), 44.0),
        "other": torch.full((1, 5, 1), 55.0),
    }
    output = dispatch_node_kind_adapters(
        kinds, node_kind_to_id=mapping, **values
    )
    assert output.flatten().tolist() == [11.0, 22.0, 33.0, 44.0, 55.0]
    assert NODE_KIND_TO_ID["klm_cluster"] in RolloutConfig().allowed_daughter_node_kinds


def test_track_fit_policy_ablation_is_mc_independent_and_unknown_fails(tmp_path: Path):
    class Fit:
        def __init__(self, pvalue): self.pvalue = pvalue
        def getPValue(self): return self.pvalue

    best, pion = Fit(0.9), Fit(0.1)

    class Track:
        def getTrackFitResults(self): return [("kaon", best), ("pion", pion)]
        def getTrackFitResultWithClosestMass(self, _hypothesis): return pion

    maximum = _select_data_independent_track_fit(
        Track(), pion_hypothesis="pion", policy=TRACK_FIT_POLICY_MAX_P_VALUE_V1
    )
    canonical = _select_data_independent_track_fit(
        Track(), pion_hypothesis="pion", policy=TRACK_FIT_POLICY_CANONICAL_PION_V1
    )
    assert maximum.fit is best and maximum.hypothesis == "kaon"
    assert canonical.fit is pion and canonical.hypothesis == "pion"
    assert Basf2PreprocessConfig(
        ("input.root",), tmp_path / "out.parquet",
        track_fit_policy=TRACK_FIT_POLICY_CANONICAL_PION_V1,
    ).track_fit_policy == TRACK_FIT_POLICY_CANONICAL_PION_V1
    with pytest.raises(ValueError, match="unknown track_fit_policy"):
        Basf2PreprocessConfig(
            ("input.root",), tmp_path / "bad.parquet", track_fit_policy="unknown-v1"
        )


@pytest.mark.parametrize(
    ("filename", "mode"),
    [
        ("soft_pid_expectation.yaml", "soft_expectation"),
        ("temperature_annealed_softmax.yaml", "temperature_softmax"),
        ("straight_through_hard_pid.yaml", "straight_through_hard"),
    ],
)
def test_pid_ablation_yaml_reaches_typed_cli_config(filename: str, mode: str):
    args = parse_reconstruction_args(
        ["--config", str(Path("configs/ablations") / filename)]
    )
    assert args.pid_kinematics_mode == mode


def test_pilot_objective_config_and_preflight_report_fail_on_inactive_objective():
    config = yaml.safe_load(Path("configs/hyperbolic_pretrain_pilot.yaml").read_text())
    assert config["objective_gradient_diagnostics"] is True
    assert config["objective_gradient_diagnostics_every"] == 1
    assert config["pilot_objective_preflight"] is True
    objectives = {"lca": torch.tensor(1.0), "channel": torch.tensor(0.0)}
    gradients = {"gradient_norms": {"shared_encoder": {"lca": 1.0, "channel": 0.0}}}
    with pytest.raises(RuntimeError, match="channel"):
        objective_preflight_report(
            objectives,
            {"lca": 1.0, "channel": 0.2},
            {"lca": 4.0, "channel": 0.0},
            gradients,
            action="fail",
        )


def test_staged_pretraining_configs_only_activate_documented_objectives():
    root = Path("configs/ablations")
    stages = [
        yaml.safe_load((root / name).read_text())
        for name in (
            "pretrain_stage1_topology_parent_anticollapse.yaml",
            "pretrain_stage2_distance_radius.yaml",
            "pretrain_stage3_channel.yaml",
            "pretrain_stage4_candidate_hard_negative.yaml",
        )
    ]
    assert stages[0]["exact_tree_distance_weight"] == 0
    assert stages[1]["exact_tree_distance_weight"] > 0
    assert stages[2]["channel_weight"] > 0
    assert stages[3]["hard_negative_weight"] > 0
