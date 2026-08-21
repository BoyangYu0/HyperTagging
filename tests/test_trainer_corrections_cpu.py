from dataclasses import dataclass
import json
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
from hypertagging.models.hyperbolic import (
    bound_tangent_norm,
    distance,
    expmap0,
    logmap0,
    radius,
)
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
from hypertagging.training.pretrain_trainer import (
    PretrainConfig,
    _allow_empty_channel_memory_expansion,
    _leaf_pid_training_weight,
    _nonfinite_gradient_report,
    _objective_diagnostics_due,
    _resolve_phase_schedule,
    _resolve_amp_dtype,
    _tensor_gradient_norm,
    objective_preflight_report,
)
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


def test_empty_channel_memory_expansion_requires_exact_channel_phase_boundary():
    schedule = ProgressivePhaseSchedule(
        unit="optimizer_step", durations=(1094, 1094, 1094, 1094)
    )
    config = PretrainConfig(
        data="unused",
        output_dir="unused",
        channel_memory_size=4096,
    )
    payload = {
        "step": 2188,
        "config": {"channel_memory_size": 0},
    }
    assert _allow_empty_channel_memory_expansion(config, schedule, payload)
    with pytest.raises(ValueError, match="exact channel-phase boundary step 2188"):
        _allow_empty_channel_memory_expansion(
            config,
            schedule,
            {**payload, "step": 2189},
        )


def test_scientific_pretraining_rejects_missing_channel_support_before_data_access(
    tmp_path,
):
    from hypertagging.training.pretrain_trainer import train_hyperbolic_pretraining

    with pytest.raises(ValueError, match="requires positive channel_memory_size"):
        train_hyperbolic_pretraining(
            PretrainConfig(
                data=str(tmp_path / "not-read.parquet"),
                output_dir=str(tmp_path / "output"),
                scientific_mode=True,
                channel_memory_size=0,
                channel_zero_positive_action="fail",
            )
        )
    with pytest.raises(ValueError, match="requires channel_zero_positive_action=fail"):
        train_hyperbolic_pretraining(
            PretrainConfig(
                data=str(tmp_path / "not-read.parquet"),
                output_dir=str(tmp_path / "output"),
                scientific_mode=True,
                channel_memory_size=4096,
                channel_zero_positive_action="warn",
            )
        )


def test_leaf_pid_phase_weights_preserve_early_training_and_taper_late_phases():
    config = PretrainConfig(
        data="unused",
        output_dir="unused",
        leaf_pid_weight=1.0,
        leaf_pid_phase_weights=(1.0, 1.0, 0.5, 0.5),
    )
    assert [
        _leaf_pid_training_weight(config, phase_index=index) for index in range(4)
    ] == [1.0, 1.0, 0.5, 0.5]
    fallback = PretrainConfig(
        data="unused", output_dir="unused", leaf_pid_weight=0.75
    )
    assert _leaf_pid_training_weight(fallback, phase_index=3) == 0.75


def test_late_leaf_pid_weight_has_margin_for_measured_h100_dominance():
    objectives = {"lca": torch.tensor(0.47), "leaf_pid": torch.tensor(0.74)}
    denominators = {"lca": 1.0, "leaf_pid": 1.0}
    gradients = {
        "gradient_norms": {
            "shared_encoder": {"lca": 1.0, "leaf_pid": 20.1182},
            "tree_projection": {"lca": 1.0, "leaf_pid": 1.0},
            "hyperbolic_projection": {"lca": 1.0, "leaf_pid": 1.0},
        }
    }
    with pytest.raises(RuntimeError, match="leaf_pid/lca=20.1182"):
        objective_preflight_report(
            objectives,
            {"lca": 1.0, "leaf_pid": 1.0},
            denominators,
            gradients,
            dominance_ratio=20.0,
            action="fail",
        )
    report = objective_preflight_report(
        objectives,
        {"lca": 1.0, "leaf_pid": 0.5},
        denominators,
        gradients,
        dominance_ratio=20.0,
        action="fail",
    )
    assert report["pass"]
    assert report["weighted_dominance_ratio"] == pytest.approx(10.0591)
    assert report["dominance_threshold"] / report["weighted_dominance_ratio"] > 1.98


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


def test_gradient_norm_uses_fp64_accumulation_and_amp_scale_is_serialized():
    gradient = torch.tensor([1.0e20, -1.0e20], dtype=torch.float32)
    assert _tensor_gradient_norm((gradient,)) == pytest.approx(2.0**0.5 * 1.0e20)
    config = PretrainConfig(data="unused", output_dir="unused")
    assert config.amp_init_scale == 4096.0


def test_h100_bfloat16_policy_is_explicit_and_scaler_safe_without_gpu_access():
    assert _resolve_amp_dtype(
        device=torch.device("cpu"), mixed_precision=True, amp_dtype="bfloat16"
    ) is None
    assert _resolve_amp_dtype(
        device=torch.device("cuda"),
        mixed_precision=True,
        amp_dtype="bfloat16",
        cuda_bf16_supported=True,
    ) is torch.bfloat16
    assert _resolve_amp_dtype(
        device=torch.device("cuda"),
        mixed_precision=True,
        amp_dtype="float16",
    ) is torch.float16
    with pytest.raises(RuntimeError, match="not supported"):
        _resolve_amp_dtype(
            device=torch.device("cuda"),
            mixed_precision=True,
            amp_dtype="bfloat16",
            cuda_bf16_supported=False,
        )


def test_smooth_tangent_bound_prevents_boundary_saturation_with_finite_gradient():
    source = (torch.randn(8, 32) * 20.0).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bounded = bound_tangent_norm(source, maximum=1.5)
        mapped = expmap0(bounded)
        loss = logmap0(mapped).square().mean()
    assert bounded.dtype == mapped.dtype == torch.float32
    assert float(torch.linalg.vector_norm(bounded, dim=-1).max().detach()) == pytest.approx(
        1.5, abs=5e-7
    )
    assert float(torch.linalg.vector_norm(mapped, dim=-1).max().detach()) < 0.95
    loss.backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_nonfinite_gradient_report_names_first_parameter_and_counts_values():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.tensor([[float("nan"), 1.0], [float("inf"), 2.0]])
    parameters[1].grad = torch.tensor([float("-inf"), 0.0])
    report = _nonfinite_gradient_report(model.named_parameters())
    assert report["first_offending_parameter"] == "0.weight"
    assert report["offending_parameter_count"] == 2
    assert report["nonfinite_elements"] == 3
    assert report["offending_parameters"][0]["nan_elements"] == 1
    assert report["offending_parameters"][0]["positive_inf_elements"] == 1


def test_objective_preflight_runs_at_each_active_phase_entry():
    common = {
        "enabled": True,
        "start_step": 0,
        "max_steps": 4376,
        "cadence": 547,
        "preflight_enabled": True,
        "preflight_retry_pending": False,
    }
    assert _objective_diagnostics_due(
        completed_step=1095,
        phase_entry=True,
        **common,
    )
    assert not _objective_diagnostics_due(
        completed_step=1096,
        phase_entry=False,
        **common,
    )
    assert _objective_diagnostics_due(
        completed_step=1641,
        phase_entry=False,
        **common,
    )
    assert not _objective_diagnostics_due(
        completed_step=1095,
        phase_entry=True,
        **{**common, "preflight_enabled": False},
    )
    assert _objective_diagnostics_due(
        completed_step=1096,
        phase_entry=False,
        **{**common, "preflight_retry_pending": True},
    )


def test_rerun_radius_weight_satisfies_shared_encoder_dominance_gate():
    names = ("lca", "tree_distance", "radius", "covariance", "leaf_pid")
    objectives = {name: torch.tensor(1.0) for name in names}
    denominators = {name: 1.0 for name in names}
    denominators["covariance"] = 2.0
    gradient_report = {
        "gradient_norms": {
            "shared_encoder": {
                "lca": 0.6073,
                "tree_distance": 1.6169,
                "radius": 69.4427,
                "covariance": 0.0549,
                "leaf_pid": 1.1201,
            }
        }
    }
    weights = {
        "lca": 1.0,
        "tree_distance": 1.0,
        "radius": 0.02,
        "covariance": 0.01,
        "leaf_pid": 1.0,
    }
    report = objective_preflight_report(
        objectives,
        weights,
        denominators,
        gradient_report,
        dominance_ratio=20.0,
        action="fail",
    )
    assert report["pass"]
    assert report["weighted_dominance_ratio"] == pytest.approx(1.6169 / 0.6073)
    assert report["objectives"]["radius"][
        "weighted_shared_encoder_gradient_norm"
    ] == pytest.approx(69.4427 * 0.02)
    with pytest.raises(RuntimeError, match="radius/lca"):
        objective_preflight_report(
            objectives,
            {**weights, "radius": 0.2},
            denominators,
            gradient_report,
            dominance_ratio=20.0,
            action="fail",
        )


@pytest.mark.parametrize(
    ("name", "denominator"),
    [
        ("lca", 0.0),
        ("parent", 0.0),
        ("tree_distance", 0.0),
        ("radius", 0.0),
        ("channel", 0.0),
        ("variance", 1.0),
        ("covariance", 1.0),
        ("leaf_pid", 0.0),
        ("corruption_class", 0.0),
        ("candidate_correctness", 0.0),
        ("hard_negative", 0.0),
    ],
)
def test_objective_preflight_skips_active_objectives_without_batch_support(
    name: str, denominator: float
):
    report = objective_preflight_report(
        {name: torch.tensor(0.0)},
        {name: 1.0},
        {name: denominator},
        {
            "gradient_norms": {
                group: {name: 0.0}
                for group in (
                    "shared_encoder",
                    "tree_projection",
                    "hyperbolic_projection",
                )
            }
        },
        action="fail",
    )
    row = report["objectives"][name]
    assert report["pass"]
    assert report["evaluation_status"] == "passed_with_skips"
    assert report["not_evaluable_objectives"] == {
        name: "insufficient_support"
    }
    assert row["support_status"] == "insufficient_support"
    assert row["has_sufficient_support"] is False
    assert row["evaluation_status"] == "not_evaluable"
    assert row["skipped"] is True
    assert row["skip_reason"] == "insufficient_support"
    json.dumps(report, allow_nan=False)


def test_objective_preflight_fails_supported_nonzero_loss_with_zero_shared_gradient():
    with pytest.raises(RuntimeError, match="channel:zero_gradient"):
        objective_preflight_report(
            {"channel": torch.tensor(0.75)},
            {"channel": 0.2},
            {"channel": 3.0},
            {
                "gradient_norms": {
                    "shared_encoder": {"channel": 0.0},
                    "tree_projection": {"channel": 2.0},
                    "hyperbolic_projection": {"channel": 1.0},
                }
            },
            action="fail",
        )


@pytest.mark.parametrize(
    ("raw_loss", "weight"),
    [(2.8675262910837773e-6, 0.01), (1.0, 1e-7)],
)
def test_objective_preflight_accepts_supported_loss_satisfied_within_tolerance(
    raw_loss: float,
    weight: float,
):
    report = objective_preflight_report(
        {"covariance": torch.tensor(raw_loss, dtype=torch.float64)},
        {"covariance": weight},
        {"covariance": 16.0},
        {
            "gradient_norms": {
                "shared_encoder": {"covariance": 0.0},
                "tree_projection": {"covariance": 0.0},
                "hyperbolic_projection": {"covariance": 0.0},
            }
        },
        weighted_loss_tolerance=1e-7,
        action="fail",
    )
    row = report["objectives"]["covariance"]
    assert report["pass"]
    assert report["weighted_loss_tolerance"] == 1e-7
    assert report["satisfied_within_tolerance_objectives"] == ["covariance"]
    assert row["evaluation_status"] == "satisfied_within_tolerance"
    assert row["weighted_loss_tolerance"] == 1e-7
    assert row["meaningful_nonzero_loss"] is False
    assert row["shared_encoder_gradient_norm"] == 0.0
    json.dumps(report, allow_nan=False)


def test_objective_preflight_rejects_supported_loss_above_tolerance_with_zero_gradient():
    with pytest.raises(RuntimeError, match="covariance:zero_gradient"):
        objective_preflight_report(
            {"covariance": torch.tensor(1.0001e-5, dtype=torch.float64)},
            {"covariance": 0.01},
            {"covariance": 16.0},
            {
                "gradient_norms": {
                    "shared_encoder": {"covariance": 0.0},
                    "tree_projection": {"covariance": 0.0},
                    "hyperbolic_projection": {"covariance": 0.0},
                }
            },
            weighted_loss_tolerance=1e-7,
            action="fail",
        )


@pytest.mark.parametrize(
    ("loss", "denominator", "shared_gradient", "violation"),
    [
        (1.0, 0.0, 0.0, "loss_without_support"),
        (6e-8, 0.0, 0.0, "loss_without_support"),
        (0.0, 0.0, 1.0, "gradient_without_support"),
        (1.0, -1.0, 1.0, "invalid_denominator"),
        (1.0, 1.0, float("nan"), "non_finite"),
    ],
)
def test_objective_preflight_rejects_inconsistent_or_broken_supported_objective(
    loss: float,
    denominator: float,
    shared_gradient: float,
    violation: str,
):
    with pytest.raises(RuntimeError, match=violation):
        objective_preflight_report(
            {"hard_negative": torch.tensor(loss)},
            {"hard_negative": 0.1},
            {"hard_negative": denominator},
            {
                "gradient_norms": {
                    "shared_encoder": {"hard_negative": shared_gradient},
                    "tree_projection": {"hard_negative": 0.0},
                    "hyperbolic_projection": {"hard_negative": 0.0},
                }
            },
            action="fail",
        )


def test_corrected_scientific_configs_and_small_candidate_contract():
    from scripts.train_hyperbolic_pretrain import parse_args

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
    h100_rerun = yaml.safe_load(
        Path("configs/slurm/pretrain_035k_h100_rerun_20260815.yaml").read_text(
            encoding="utf-8"
        )
    )
    fullscale = yaml.safe_load(
        Path("configs/slurm/pretrain_1m_h100_20260821.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert pilot["model_preset"] == "gpu_debug"
    assert pilot["objective_dominance_ratio"] == 20.0
    assert pilot["pilot_objective_violation_action"] == "fail"
    assert pilot["validation_events"] == 256
    assert slurm_scientific["model_preset"] == "small_candidate"
    assert slurm_scientific["log_every"] == 1
    assert slurm_scientific["amp_init_scale"] == 4096.0
    assert slurm_scientific["validation_events"] == 2000
    assert slurm_diagnostic["model_preset"] == "gpu_debug"
    assert slurm_diagnostic["validation_events"] == 32
    assert small_candidate_diagnostic["model_preset"] == "small_candidate"
    assert small_candidate_diagnostic["max_steps"] == 4
    assert small_candidate_diagnostic["curriculum_phase_steps"] == [1, 1, 1, 1]
    assert small_candidate_diagnostic["validation_batches"] == 1
    assert small_candidate_diagnostic["validation_events"] <= 32
    assert h100_rerun["amp_dtype"] == "bfloat16"
    assert h100_rerun["max_tangent_norm"] == 1.5
    assert h100_rerun["batch_size"] == 16
    assert h100_rerun["max_steps"] * h100_rerun["batch_size"] == 70_016
    assert sum(h100_rerun["curriculum_phase_steps"]) == h100_rerun["max_steps"]
    assert h100_rerun["validate_every"] == h100_rerun["checkpoint_every"] == 547
    assert h100_rerun["validation_events"] == 512
    assert h100_rerun["num_workers"] == 0
    assert h100_rerun["learning_rate"] == 5e-4
    assert h100_rerun["warmup_fraction"] == 0.1
    assert h100_rerun["min_lr_ratio"] == 0.1
    assert h100_rerun["gradient_clip"] == 0.5
    assert h100_rerun["radius_depth_weight"] == 0.02
    assert h100_rerun["pilot_objective_preflight"] is True
    assert h100_rerun["pilot_objective_violation_action"] == "fail"
    assert h100_rerun["objective_dominance_ratio"] == 20.0
    assert h100_rerun["objective_weighted_loss_tolerance"] == 1e-7
    assert h100_rerun["channel_memory_size"] == 4096
    assert h100_rerun["leaf_pid_phase_weights"] == [1.0, 1.0, 0.5, 0.5]
    assert fullscale["data"].endswith("train_865k.json")
    assert fullscale["dataset_index"].endswith("train_865k.complete_only.index.json")
    assert fullscale["max_steps"] == 108128
    assert fullscale["max_steps"] * fullscale["batch_size"] == 1_730_048
    assert fullscale["max_steps"] * fullscale["batch_size"] - 2 * 865_000 == 48
    assert fullscale["curriculum_phase_steps"] == [27032] * 4
    assert fullscale["checkpoint_every"] == fullscale["validate_every"] == 13516
    assert fullscale["validation_events"] == 5000
    assert fullscale["validation_batches"] == 313
    assert fullscale["amp_dtype"] == "bfloat16"
    assert fullscale["channel_memory_size"] == 4096
    assert fullscale["channel_zero_positive_action"] == "fail"
    parsed_fullscale = parse_args(
        ["--config", "configs/slurm/pretrain_1m_h100_20260821.yaml"]
    )
    assert parsed_fullscale.max_steps == fullscale["max_steps"]
    assert parsed_fullscale.validation_events == fullscale["validation_events"]
    assert parsed_fullscale.validation_batches == fullscale["validation_batches"]
    rerun_schedule = learning_rate_schedule_contract(
        total_steps=h100_rerun["lr_schedule_total_steps"],
        warmup_fraction=h100_rerun["warmup_fraction"],
        max_warmup_steps=h100_rerun["max_warmup_steps"],
        min_lr_ratio=h100_rerun["min_lr_ratio"],
        base_lrs=[h100_rerun["learning_rate"]],
    )
    assert rerun_schedule["warmup_steps"] == 438
    assert rerun_schedule["warmup_steps"] * h100_rerun["batch_size"] == 7_008
    assert rerun_schedule["min_lr_ratio"] * h100_rerun["learning_rate"] == 5e-5
    parsed_rerun = parse_args(
        ["--config", "configs/slurm/pretrain_035k_h100_rerun_20260815.yaml"]
    )
    for name in (
        "learning_rate",
        "warmup_fraction",
        "min_lr_ratio",
        "gradient_clip",
        "radius_depth_weight",
        "pilot_objective_preflight",
        "objective_weighted_loss_tolerance",
        "pilot_objective_violation_action",
        "validation_events",
        "leaf_pid_phase_weights",
    ):
        assert getattr(parsed_rerun, name) == h100_rerun[name]
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
