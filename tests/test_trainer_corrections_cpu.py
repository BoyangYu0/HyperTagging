from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from hypertagging.losses.hyperbolic_pretraining import (
    collapse_diagnostics,
    covariance_regularization,
    variance_regularization,
)
from hypertagging.models.hyperbolic import distance, expmap0, logmap0, radius
from hypertagging.models.relations import HyperbolicRelationBias
from hypertagging.training.checkpoint_selection import rollout_checkpoint_eligibility
from hypertagging.training.fixed_validation import select_validation_events
from hypertagging.training.learning_rate import (
    build_warmup_cosine_scheduler,
    learning_rate_schedule_contract,
    lr_multiplier,
    resolve_resume_schedule_contract,
)
from hypertagging.training.model_config import MODEL_PRESETS
from hypertagging.training.pretraining_curriculum import (
    DEFAULT_PRETRAINING_PHASES,
    ProgressivePhaseSchedule,
    default_phase_durations,
)
from hypertagging.training.pretrain_trainer import PretrainConfig, _resolve_phase_schedule
from hypertagging.training.reconstruction_trainer import (
    _require_scientific_capacity_report,
)


def test_progressive_phase_boundaries_and_resume_contract_are_explicit():
    durations = default_phase_durations(20)
    assert durations == (4, 5, 6, 5)
    schedule = ProgressivePhaseSchedule(unit="optimizer_step", durations=durations)
    sequence = [schedule.phase(step=step, events=0).name for step in range(20)]
    assert sequence[:4] == [DEFAULT_PRETRAINING_PHASES[0].name] * 4
    assert sequence[4:9] == [DEFAULT_PRETRAINING_PHASES[1].name] * 5
    assert sequence[9:15] == [DEFAULT_PRETRAINING_PHASES[2].name] * 6
    assert sequence[15:] == [DEFAULT_PRETRAINING_PHASES[3].name] * 5

    payload = {
        "training_state": {
            "curriculum_schedule_contract": schedule.contract(),
            "curriculum_phase_cursor": {
                "completed_optimizer_steps": 11,
                "events_completed": 37,
                "phase_index": 2,
                "final_phase_entered": False,
            },
        }
    }
    resumed = _resolve_phase_schedule(
        PretrainConfig(data="unused", output_dir="unused", max_steps=99), payload
    )
    assert resumed.contract() == schedule.contract()
    assert resumed.phase(step=11, events=37).name == sequence[11]


def test_scientific_short_curriculum_is_rejected_and_legacy_is_named():
    with pytest.raises(ValueError, match="all four phases"):
        _resolve_phase_schedule(
            PretrainConfig(
                data="unused", output_dir="unused", max_steps=3,
                scientific_mode=True,
            ),
            None,
        )
    legacy = _resolve_phase_schedule(
        PretrainConfig(
            data="unused", output_dir="unused",
            curriculum_mode="legacy_alternating_ablation",
        ),
        None,
    )
    assert legacy.mode == "legacy_alternating_ablation"


def test_linear_warmup_cosine_values_and_scheduler_resume_match():
    contract = learning_rate_schedule_contract(
        total_steps=10, warmup_steps=2, base_lrs=(1.0,)
    )
    values = [lr_multiplier(step, contract) for step in range(10)]
    assert values[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert values[-1] == pytest.approx(0.0)

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(optimizer, contract)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    expected = []
    for _ in range(3):
        expected.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=1.0)
    resumed_scheduler = build_warmup_cosine_scheduler(resumed_optimizer, contract)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    actual = []
    for _ in range(3):
        actual.append(resumed_optimizer.param_groups[0]["lr"])
        resumed_optimizer.step()
        resumed_scheduler.step()
    assert actual == pytest.approx(expected)

    with pytest.raises(ValueError, match="legacy checkpoint"):
        resolve_resume_schedule_contract(
            resume_payload={"training_state": {}},
            configured_total_steps=None,
            run_max_steps=10,
            warmup_fraction=0.05,
            warmup_steps=None,
            max_warmup_steps=10_000,
            min_lr_ratio=0.0,
            base_lrs=(1.0,),
        )


def test_geometry_and_vic_are_fp32_under_cpu_autocast_with_finite_gradients():
    source = (torch.randn(2, 4, dtype=torch.bfloat16) * 0.05).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mapped = expmap0(source)
        tangent = logmap0(mapped)
        distances = distance(mapped[0], mapped[1])
        radii = radius(mapped)
        mask = torch.ones(2, dtype=torch.bool)
        vic = variance_regularization(tangent, mask) + covariance_regularization(
            tangent, mask
        )
        diagnostics = collapse_diagnostics(mapped, mask)
        loss = distances + radii.mean() + vic + diagnostics["effective_rank"]
    assert mapped.dtype == tangent.dtype == distances.dtype == radii.dtype == torch.float32
    assert vic.dtype == diagnostics["effective_rank"].dtype == torch.float32
    loss.backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()

    observed_dtypes = []
    relation = HyperbolicRelationBias(hidden_dim=4)
    handle = relation.net[0].register_forward_pre_hook(
        lambda _module, inputs: observed_dtypes.append(inputs[0].dtype)
    )
    relation_source = mapped.detach().reshape(1, 2, 4).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bias = relation(
            z_hyperbolic=relation_source,
            node_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        bias.float().sum().backward()
    handle.remove()
    assert observed_dtypes == [torch.float32]
    assert relation_source.grad is not None
    assert torch.isfinite(relation_source.grad).all()


def test_corrected_scientific_configs_and_small_candidate_contract():
    pilot = yaml.safe_load(
        Path("configs/hyperbolic_pretrain_pilot.yaml").read_text(encoding="utf-8")
    )
    reconstruction = yaml.safe_load(
        Path("configs/level_reconstruction.yaml").read_text(encoding="utf-8")
    )
    slurm_scientific = yaml.safe_load(
        Path("configs/slurm/pretrain_035k_scientific.yaml").read_text(
            encoding="utf-8"
        )
    )
    slurm_diagnostic = yaml.safe_load(
        Path("configs/slurm/pretrain_diagnostic.yaml").read_text(encoding="utf-8")
    )
    small_candidate_diagnostic = yaml.safe_load(
        Path("configs/slurm/pretrain_diagnostic_small_candidate.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert pilot["model_preset"] == "gpu_debug"
    assert pilot["objective_dominance_ratio"] == 20.0
    assert pilot["pilot_objective_violation_action"] == "fail"
    assert pilot["validation_events"] == 256
    assert slurm_scientific["model_preset"] == "small_candidate"
    assert slurm_scientific["validation_events"] == 2000
    assert slurm_diagnostic["model_preset"] == "gpu_debug"
    assert slurm_diagnostic["validation_events"] == 32
    assert small_candidate_diagnostic["model_preset"] == "small_candidate"
    assert small_candidate_diagnostic["max_steps"] == 4
    assert small_candidate_diagnostic["curriculum_phase_steps"] == [1, 1, 1, 1]
    assert small_candidate_diagnostic["validation_batches"] == 1
    assert small_candidate_diagnostic["validation_events"] <= 32
    assert reconstruction["max_validation_events"] == 2000
    assert reconstruction["rollout_validation_events"] == 1000
    assert reconstruction["best_metric"] == "predicted_edge_f1"
    assert reconstruction["best_mode"] == "max"

    candidate = MODEL_PRESETS["small_candidate"]
    assert (candidate.d_model, candidate.hyper_dim) == (128, 32)
    assert (candidate.n_heads, candidate.n_context_layers, candidate.ffn_dim) == (
        4, 4, 512
    )
    assert candidate.capacity_report_required
    with pytest.raises(ValueError, match="capacity-report-required"):
        _require_scientific_capacity_report(
            candidate, SimpleNamespace(dataset_index=None), scientific_mode=True
        )
    _require_scientific_capacity_report(
        candidate, SimpleNamespace(dataset_index={"index_hash": "checked"}),
        scientific_mode=True,
    )


def test_rollout_gates_reject_ineligible_primary_metrics():
    gates = {
        "minimum_tree_validity": 0.999,
        "minimum_p4_closure": 1.0,
        "maximum_recursive_source_conflicts": 0,
        "required_denominators": [
            "rollout_validation_events",
            "predicted_edge_denominator",
            "predicted_p4_closure_denominator",
        ],
    }
    eligible = {
        "rollout_validation_events": 100,
        "predicted_edge_denominator": 200,
        "predicted_p4_closure_denominator": 50,
        "predicted_tree_validity_rate": 0.999,
        "predicted_p4_closure_rate": 1.0,
        "predicted_recursive_source_conflicts": 0,
    }
    assert rollout_checkpoint_eligibility(eligible, gates)["eligible"]
    invalid = dict(eligible, predicted_tree_validity_rate=0.998)
    result = rollout_checkpoint_eligibility(invalid, gates)
    assert not result["eligible"]
    assert "minimum:predicted_tree_validity_rate" in result["failures"]
    zero_denominator = dict(eligible, predicted_edge_denominator=0)
    assert not rollout_checkpoint_eligibility(zero_denominator, gates)["eligible"]


@dataclass(frozen=True)
class _Event:
    event_uid: str


def test_scientific_fixed_validation_is_order_independent_and_restorable():
    events = [_Event(f"validation:{index}") for index in range(20)]
    selected, uids, contract = select_validation_events(
        events,
        limit=5,
        scientific_mode=True,
        selection_manifest_hash="manifest-hash",
        seed=17,
    )
    reversed_selected, reversed_uids, _ = select_validation_events(
        reversed(events),
        limit=5,
        scientific_mode=True,
        selection_manifest_hash="manifest-hash",
        seed=17,
    )
    assert [event.event_uid for event in selected] == list(uids)
    assert [event.event_uid for event in reversed_selected] == list(reversed_uids)
    assert reversed_uids == uids
    assert contract["mode"] == "manifest_validation_role_uid_hash"
    restored, restored_uids, _ = select_validation_events(
        reversed(events),
        limit=5,
        scientific_mode=True,
        selection_manifest_hash="manifest-hash",
        seed=999,
        restored_event_uids=uids,
    )
    assert [event.event_uid for event in restored] == list(uids)
    assert restored_uids == uids
