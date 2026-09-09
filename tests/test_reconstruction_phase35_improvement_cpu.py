from __future__ import annotations

import ast
import hashlib
import json
import shlex
from pathlib import Path

import pytest

from hypertagging.training.fixed_validation import excluded_event_uids_contract
from scripts import run_reconstruction_phase35 as runner
from scripts.run_reconstruction_phase35 import (
    _training_config,
    _validation_selection_audit,
    replay_slot_audit,
)
from scripts.slurm import render_reconstruction_phase35_job as render
from scripts.slurm import submit_reconstruction_phase35_campaign as submit
from scripts.slurm import verify_reconstruction_phase35_contract as verify

ROOT = Path(__file__).resolve().parents[1]


def test_phase35_system_python_entrypoints_avoid_zip_strict() -> None:
    """Cluster /usr/bin/python3 is 3.9; zip(strict=...) arrived in 3.10."""

    system_python_entrypoints = (
        "scripts/slurm/finalize_reconstruction_fullscale_receipt.py",
        "scripts/slurm/monitor_gpu_telemetry.py",
        "scripts/slurm/render_reconstruction_phase35_job.py",
        "scripts/slurm/submit_reconstruction_phase35_campaign.py",
        "scripts/slurm/verify_reconstruction_phase35_contract.py",
    )
    for relative in system_python_entrypoints:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        unsupported = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "zip"
            and any(keyword.arg == "strict" for keyword in node.keywords)
        ]
        assert not unsupported, relative


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregistration() -> dict:
    return json.loads((ROOT / verify.PREREGISTRATION).read_text())


@pytest.mark.campaign_artifacts('artifacts/evaluation/full_decay_phase34_81096_best_validation_100_both_20260904.json')
def test_phase35_preregistration_binds_failure_evidence_and_sealed_data() -> None:
    prereg = preregistration()
    assert prereg["study_id"] == verify.PARENT_STUDY_ID
    assert (
        prereg["study_classification"]
        == "exploratory_validation_performance_screen"
    )
    report = prereg["diagnostic_basis"]["full_decay_report"]
    assert digest(ROOT / report["path"]) == report["sha256"]
    assert report["diagnosis"] == "one_level_two_daughter_mode_collapse"
    assert report["configured_root_completion"] == 0.0
    assert report["empty_level_count"] == 705
    assert report["accepted_composite_levels"] == {"1": 98}
    assert report["accepted_composite_cardinalities"] == {"2": 98}
    lineage = prereg["diagnostic_basis"]["lineage_receipt"]
    assert digest(ROOT / lineage["path"]) == lineage["sha256"]
    data = prereg["data_binding"]
    assert data["split_counts"] == {
        "train": 35000,
        "validation": 50000,
        "test": 0,
    }
    assert data["sealed_test_role_access"] == "forbidden"
    assert digest(ROOT / data["selection_manifest"]) == data[
        "selection_manifest_sha256"
    ]
    assert digest(ROOT / data["dataset_index"]) == data["dataset_index_sha256"]
    source = prereg["source_checkpoint"]
    assert source["step"] == 81096
    assert source["pretraining_success_gate_passed"] is True
    assert digest(ROOT / source["path"]) == source["sha256"]
    runtime = prereg["runtime_binding"]
    assert digest(Path(runtime["gpu_environment"]) / "bin/python") == runtime[
        "python_sha256"
    ]
    assert digest(Path(runtime["gpu_environment"]) / "pyvenv.cfg") == runtime[
        "pyvenv_cfg_sha256"
    ]
    assert digest(ROOT / runtime["runtime_contract"]) == runtime[
        "runtime_contract_sha256"
    ]
    assert digest(ROOT / runtime["requirements_lock"]) == runtime[
        "requirements_lock_sha256"
    ]


@pytest.mark.campaign_artifacts('runtime_inputs/reconstruction_phase34_20260904/train_035k.repromoted.json')
def test_phase35_repair_preregistration_is_exactly_two_clean_restart_arms() -> None:
    parent = preregistration()
    repair = verify.load_repair_preregistration()
    assert repair["study_id"] == verify.STUDY_ID
    assert repair["parent_study_id"] == verify.PARENT_STUDY_ID
    assert repair["eligible_arm_roles"] == list(verify.PHASE35_ARM_ROLES) == [
        "depth_balanced_masked_aux_encoder_adapt",
        "depth_balanced_masked_aux_frozen",
    ]
    assert repair["excluded_completed_arm_role"] == (
        "depth_balanced_fallback_frozen"
    )
    assert repair["excluded_completed_arm_rerun_authorized"] is False
    assert repair["clean_restart_required"] is True
    assert repair["resume_from_failed_attempt_authorized"] is False
    assert repair["reuse_failed_output_authorized"] is False
    assert repair["restart_source_checkpoint"] == {
        "path": parent["source_checkpoint"]["path"],
        "step": parent["source_checkpoint"]["step"],
        "sha256": parent["source_checkpoint"]["sha256"],
    }
    assert repair["execution_policy"] == {
        **parent["execution_policy"],
        "task_count": 2,
        "global_concurrency": 2,
    }
    assert repair["hierarchy_supervision"] == (
        verify.PHASE35_HIERARCHY_SUPERVISION
    )
    hierarchy = repair["hierarchy_supervision"]
    assert hierarchy["target_levels"] == [1, 2, 3, 4, 5, 6]
    assert hierarchy["eligible_mother_counts_by_level"] == {
        "1": 75460,
        "2": 32095,
        "3": 20559,
        "4": 11666,
        "5": 6568,
        "6": 885,
    }
    assert hierarchy["upsilon4s_root_count"] == 10000
    assert sum(hierarchy["upsilon4s_roots_by_level"].values()) == 10000
    assert hierarchy["continuum_forest_roots"] == {
        "root_count": 37200,
        "event_count": 17796,
        "continuum_event_count": 25000,
    }
    assert hierarchy["artificial_continuum_resonance_token"] is False
    assert hierarchy["inference_seed"] == "detector_fsps_only"
    assert hierarchy["truth_targets"] == (
        "eligible_complete_mothers_all_levels"
    )
    historical = {
        item["arm_role"]: item for item in repair["historical_attempts"]
    }
    assert historical["depth_balanced_fallback_frozen"]["rerun_authorized"] is False
    assert historical["depth_balanced_masked_aux_encoder_adapt"][
        "rerun_authorized"
    ] is True
    assert historical["depth_balanced_masked_aux_frozen"][
        "rerun_authorized"
    ] is True
    for item in historical.values():
        assert digest(ROOT / item["contract"]["path"]) == item["contract"][
            "sha256"
        ]
        assert digest(ROOT / item["attempt_receipt"]["path"]) == item[
            "attempt_receipt"
        ]["sha256"]
        assert digest(ROOT / item["stderr"]["path"]) == item["stderr"][
            "sha256"
        ]
    diagnostics = repair["step500_diagnostics"]
    assert diagnostics["decision"] == "clean_restart_both_failed_arms"
    assert len(diagnostics["reports"]) == 4
    for report in diagnostics["reports"]:
        assert report["arm_role"] in verify.PHASE35_ARM_ROLES
        assert report["checkpoint_step"] == 500
        assert digest(ROOT / report["path"]) == report["sha256"]


def test_phase35_repair_identity_cannot_schedule_completed_fallback() -> None:
    for arm_role in verify.PHASE35_ARM_ROLES:
        assert verify.phase35_task_id(arm_role) == (
            f"phase35-repair-{arm_role}-20260904"
        )
        assert verify.phase35_contract_relative_path(arm_role) == (
            f"artifacts/codex/reconstruction_phase35_repair_{arm_role}_"
            "contract_20260904.json"
        )
        config = verify.resolved_arm_config(preregistration(), arm_role)
        assert config["unrepresentable_target_policy"] == (
            "masked_representable_only"
        )
        assert config["auxiliary_teacher_weight"] == 0.25
    with pytest.raises(RuntimeError, match="arm set changed"):
        verify.phase35_task_id("depth_balanced_fallback_frozen")
    with pytest.raises(RuntimeError, match="arm set changed"):
        verify.phase35_contract_relative_path("depth_balanced_fallback_frozen")


@pytest.mark.campaign_artifacts('artifacts/runs/ht-reconstruction-phase34-checkpoint-comparison-20260904/resume_source_baseline/16251359/training/best.pt')
def test_phase35_excludes_every_previously_accessed_validation_event() -> None:
    prereg = preregistration()
    binding = prereg["validation_exclusion"]
    manifest_path = ROOT / binding["manifest"]
    assert digest(manifest_path) == binding["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sealed_test_role_access"] == "forbidden"
    assert len(manifest["event_uids"]) == len(set(manifest["event_uids"])) == 3956
    assert verify.exclusion_identity(manifest["event_uids"]) == {
        "event_uid_count": binding["event_uid_count"],
        "event_uids_sha256": binding["event_uids_sha256"],
        "event_uids_hash_scheme": binding["event_uids_hash_scheme"],
    }
    assert verify.load_validation_exclusions(prereg) == tuple(
        manifest["event_uids"]
    )
    assert [
        item["event_uid_count"] for item in manifest["pairwise_source_overlaps"]
    ] == [22, 22, 0]
    for source in manifest["sources"]:
        assert digest(ROOT / source["path"]) == source["sha256"]


@pytest.mark.campaign_artifacts('runtime_inputs/reconstruction_phase34_20260904/train_035k.complete_only.repromoted.index.json')
def test_phase35_capacity_and_target_policy_decisions_match_index() -> None:
    prereg = preregistration()
    index = json.loads(
        (ROOT / prereg["data_binding"]["dataset_index"]).read_text()
    )
    policy = index["policy_capacity_statistics"]
    assert policy["complete_only"] == policy["reconstructable_partial"]
    assert (
        policy["complete_only"]["eligible_targets"]
        == prereg["diagnostic_basis"]["target_distribution"][
            "eligible_target_count"
        ]
        == 331960
    )
    common = prereg["common_training_contract"]
    query_capacity = dict(common["n_queries_by_level"])
    cardinality_capacity = dict(common["max_cardinality_by_level"])
    for level, histogram in index["mother_count_histograms_by_level"].items():
        assert query_capacity[int(level)] > max(map(int, histogram))
    for level, histogram in index[
        "daughter_cardinality_histograms_by_level"
    ].items():
        assert cardinality_capacity[int(level)] >= max(map(int, histogram))


def test_phase35_arms_form_a_controlled_balanced_replay_campaign() -> None:
    prereg = preregistration()
    common = prereg["common_training_contract"]
    assert common["max_steps"] == 2188
    assert common["replay_slot_budget"] == 140032
    assert common["replay_slot_counts_by_level"] == {
        "1": 23339,
        "2": 23339,
        "3": 23339,
        "4": 23339,
        "5": 23338,
        "6": 23338,
    }
    replay_contract = common["balanced_level_replay_contract"]
    assert replay_contract["materialized_train_event_count"] == 35000
    assert replay_contract["planned_schedule"] == {
        "start_slot": 0,
        "slot_count": common["replay_slot_budget"],
        "end_slot_exclusive": common["replay_slot_budget"],
        "level_counts": common["replay_slot_counts_by_level"],
        "max_minus_min_level_count": 1,
    }
    assert replay_contract["eligible_pool_counts_by_level"] == {
        "1": 27620,
        "2": 16160,
        "3": 12961,
        "4": 10001,
        "5": 6542,
        "6": 885,
    }
    assert common["training_budget_semantics"] == (
        "four_train_size_equivalent_level_conditioned_replay_slots_not_four_"
        "literal_dataset_epochs"
    )
    assert "planned_training_epochs" not in common
    assert "actual_event_presentations_target" not in common
    assert "nominal_full_batch_slots" not in common
    assert common["learning_rate"] == 0.0003
    assert common["min_lr_ratio"] == 0.05
    assert common["scheduled_sampling_probability"] == 0.5
    assert common["level_sampling_mode"] == "balanced_level_replay"
    assert common["rollout_continue_through_empty_levels"] is True
    assert common["type_conditioned_daughter_relation_bias"] is True
    assert common["query_repulsion_weight"] == 0.01
    assert common["early_stopping_patience"] is None
    arm_a = verify.resolved_arm_config(
        prereg, "depth_balanced_fallback_frozen"
    )
    arm_b = verify.resolved_arm_config(
        prereg, "depth_balanced_masked_aux_frozen"
    )
    arm_c = verify.resolved_arm_config(
        prereg, "depth_balanced_masked_aux_encoder_adapt"
    )
    differences_ab = {
        key for key in arm_a if arm_a[key] != arm_b[key]
    }
    differences_bc = {
        key for key in arm_b if arm_b[key] != arm_c[key]
    }
    assert differences_ab == {
        "unrepresentable_target_policy",
        "auxiliary_teacher_weight",
    }
    assert differences_bc == {"freeze_pretrained_encoder_steps"}
    assert arm_a["freeze_pretrained_encoder_steps"] == 2188
    assert arm_b["unrepresentable_target_policy"] == "masked_representable_only"
    assert arm_c["freeze_pretrained_encoder_steps"] == 1094
    assert arm_c["encoder_lr_multiplier"] == 0.05


def test_phase35_runner_maps_every_intervention() -> None:
    prereg = preregistration()
    config = verify.resolved_arm_config(
        prereg, "depth_balanced_masked_aux_encoder_adapt"
    )
    runtime = {
        "selection_manifest": str(
            ROOT / prereg["data_binding"]["selection_manifest"]
        ),
        "dataset_index": str(ROOT / prereg["data_binding"]["dataset_index"]),
        "checkpoint": str(ROOT / prereg["source_checkpoint"]["path"]),
    }
    mapped = _training_config(
        config=config,
        runtime=runtime,
        training_output=ROOT / "unused-phase35-test-output",
        validation_exclusions=("validation:a", "validation:b"),
    )
    assert mapped.max_steps == 2188
    assert mapped.validation_excluded_event_uids == (
        "validation:a",
        "validation:b",
    )
    assert mapped.rollout_continue_through_empty_levels is True
    assert mapped.level_sampling_mode == "balanced_level_replay"
    assert mapped.unrepresentable_target_policy == "masked_representable_only"
    assert mapped.auxiliary_teacher_weight == 0.25
    assert mapped.freeze_pretrained_encoder_steps == 1094
    assert mapped.encoder_lr_multiplier == 0.05
    assert mapped.type_conditioned_daughter_relation_bias is True
    assert mapped.query_repulsion_weight == 0.01
    assert dict(mapped.n_queries_by_level)[6] == 2
    assert mapped.minimum_encoder_transfer_coverage == 1.0
    assert mapped.allow_low_encoder_transfer_coverage is False
    assert mapped.require_exact_leaf_pid_transfer is True
    assert mapped.rollout_max_level == 6
    assert mapped.rollout_root_types == (1,)
    assert mapped.rollout_exclusive_final is True
    assert mapped.rollout_use_learned_confidence is True


def _phase35_replay_contract() -> dict[str, object]:
    return {
        "version": "balanced-level-replay-v1",
        "levels": [1, 2, 3, 4, 5, 6],
        "seed": 20260904,
        "target_policy": "complete_only",
        "min_daughters": 2,
        "materialized_train_event_count": 35000,
        "materialized_train_uid_sha256": "a" * 64,
        "eligible_pool_counts_by_level": {
            str(level): 10 + level for level in range(1, 7)
        },
        "eligible_pool_uid_sha256_by_level": {
            str(level): f"{level:x}" * 64 for level in range(1, 7)
        },
        "uid_hash_scheme": "sha256-u64be-length-prefixed-utf8-v1",
        "level_schedule": "global_slot_round_robin",
        "pool_order": "uid_lexicographic_before_permutation",
        "pool_permutation": "sha256_ranked_by_seed_level_cycle_uid",
        "replacement_policy": "new_seeded_permutation_on_pool_exhaustion",
        "planned_schedule": {
            "start_slot": 0,
            "slot_count": 140032,
            "end_slot_exclusive": 140032,
            "level_counts": {
                "1": 23339,
                "2": 23339,
                "3": 23339,
                "4": 23339,
                "5": 23338,
                "6": 23338,
            },
            "max_minus_min_level_count": 1,
        },
    }


def _phase35_replay_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    levels = (1, 2, 3, 4, 5, 6)
    for step in range(1, 2189):
        start = (step - 1) * 64
        counts = {
            level: sum(
                1 for slot in range(start, start + 64) if slot % 6 == level - 1
            )
            for level in levels
        }
        row: dict[str, object] = {
            "step": step,
            "target_levels": list(levels),
            "balanced_replay_slot_start": float(start),
            "balanced_replay_slot_end_exclusive": float(start + 64),
            "optimized_event_level_count": 64.0,
        }
        for level in levels:
            row[f"selected_event_level_{level}_count"] = float(counts[level])
            row[f"optimized_event_level_{level}_count"] = float(counts[level])
        rows.append(row)
    return rows


def _write_metric_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_phase35_runner_audits_every_balanced_replay_slot(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metric_rows(metrics, _phase35_replay_rows())
    config = {
        "batch_size": 64,
        "max_steps": 2188,
        "replay_slot_budget": 140032,
        "replay_slot_counts_by_level": {
            str(level): count
            for level, count in runner.PHASE35_REPLAY_SLOT_COUNTS.items()
        },
        "level_sampling_mode": "balanced_level_replay",
        "seed": 20260904,
        "target_policy": "complete_only",
        "balanced_level_replay_contract": _phase35_replay_contract(),
    }

    audit = replay_slot_audit(
        metrics,
        config=config,
        trainer_contract=_phase35_replay_contract(),
    )

    assert audit["optimizer_steps"] == 2188
    assert audit["slot_count"] == 140032
    assert audit["slot_end_exclusive"] == 140032
    assert audit["selected_slot_counts_by_level"] == {
        str(level): count
        for level, count in runner.PHASE35_REPLAY_SLOT_COUNTS.items()
    }
    assert audit["optimized_slot_counts_by_level"] == audit[
        "selected_slot_counts_by_level"
    ]
    assert audit["trainer_contract_match"] is True
    assert audit["preregistered_contract_match"] is True

    changed_contract = _phase35_replay_contract()
    changed_contract["materialized_train_event_count"] = 34999
    with pytest.raises(RuntimeError, match="preregistered real-data contract"):
        replay_slot_audit(
            metrics,
            config=config,
            trainer_contract=changed_contract,
        )


@pytest.mark.parametrize("corruption", ["slot_gap", "optimization_mismatch"])
def test_phase35_replay_audit_fails_closed_on_runtime_corruption(
    tmp_path: Path, corruption: str
) -> None:
    rows = _phase35_replay_rows()
    if corruption == "slot_gap":
        rows[100]["balanced_replay_slot_start"] = 6401.0
    else:
        rows[0]["optimized_event_level_1_count"] = 10.0
    metrics = tmp_path / f"{corruption}.jsonl"
    _write_metric_rows(metrics, rows)
    config = {
        "batch_size": 64,
        "max_steps": 2188,
        "replay_slot_budget": 140032,
        "replay_slot_counts_by_level": {
            str(level): count
            for level, count in runner.PHASE35_REPLAY_SLOT_COUNTS.items()
        },
        "level_sampling_mode": "balanced_level_replay",
        "seed": 20260904,
        "target_policy": "complete_only",
        "balanced_level_replay_contract": _phase35_replay_contract(),
    }

    with pytest.raises(RuntimeError, match="replay slot range|selected/optimized"):
        replay_slot_audit(
            metrics,
            config=config,
            trainer_contract=_phase35_replay_contract(),
        )


def test_phase35_validation_audit_requires_exact_manifest_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tuple(f"validation:{index:04d}" for index in range(2000))
    rollout = expected[:1000]
    exclusions = ("validation:prior-a", "validation:prior-b")
    selection = {
        "split": "validation",
        "event_uids": list(expected),
        "rollout_event_uids": list(rollout),
        "rollout_was_run": True,
        "deterministic": True,
        "scientific_mode": True,
        "strategy": "manifest_validation_role_uid_hash",
        **excluded_event_uids_contract(exclusions),
    }
    payload = {"validation_selection": selection}
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: payload)

    audit = _validation_selection_audit(
        Path("unused.pt"),
        exclusions=exclusions,
        expected_event_uids=expected,
        expected_rollout_event_uids=rollout,
    )
    assert audit["event_count"] == 2000
    assert audit["rollout_event_count"] == 1000
    assert audit["event_uids_sha256"] == verify.uid_sequence_sha256(
        list(expected)
    )
    assert audit["ordered_event_uids_match"] is True
    assert audit["ordered_rollout_prefix_match"] is True

    selection["event_uids"][1000:1002] = reversed(
        selection["event_uids"][1000:1002]
    )
    with pytest.raises(RuntimeError, match="ordered checkpoint-selection"):
        _validation_selection_audit(
            Path("unused.pt"),
            exclusions=exclusions,
            expected_event_uids=expected,
            expected_rollout_event_uids=rollout,
        )
    selection["event_uids"] = list(expected)
    selection["rollout_event_uids"][:2] = reversed(
        selection["rollout_event_uids"][:2]
    )
    with pytest.raises(RuntimeError, match="ordered checkpoint-selection prefix"):
        _validation_selection_audit(
            Path("unused.pt"),
            exclusions=exclusions,
            expected_event_uids=expected,
            expected_rollout_event_uids=rollout,
        )


def test_phase35_live_slurm_contract_is_exact_and_no_requeue(tmp_path: Path) -> None:
    task_id = "phase35-repair-depth_balanced_masked_aux_frozen-20260904"
    contract_path = tmp_path / "contract.json"
    contract = {"task_id": task_id, "contract_sha256": "a" * 64}
    record = (
        f"JobId=123 JobName={task_id} Partition=inter Account=others "
        "Requeue=0 Restarts=0 NumCPUs=8 NumNodes=1 NumTasks=1 CPUs/Task=8 "
        "TimeLimit=1-00:00:00 MinMemoryNode=64G "
        f"Comment=phase35:{contract['contract_sha256']} "
        "ReqTRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,gres/gpu:h100nvl=1 "
        "TresPerNode=gres:gpu:h100nvl:1 "
        f"Command={ROOT / verify.WRAPPER} WorkDir={ROOT}"
    )
    verify.verify_live_slurm_record(
        record, contract=contract, contract_path=contract_path
    )
    verify.verify_live_slurm_record(
        record.replace("NumNodes=1", "NumNodes=1-1"),
        contract=contract,
        contract_path=contract_path,
    )
    with pytest.raises(RuntimeError, match="NumNodes"):
        verify.verify_live_slurm_record(
            record.replace("NumNodes=1", "NumNodes=1-2"),
            contract=contract,
            contract_path=contract_path,
        )
    with pytest.raises(RuntimeError, match="live Slurm allocation"):
        verify.verify_live_slurm_record(
            record.replace("Requeue=0", "Requeue=1"),
            contract=contract,
            contract_path=contract_path,
        )
    with pytest.raises(RuntimeError, match="gpu_exact"):
        verify.verify_live_slurm_record(
            record.replace(
                "gres/gpu=1,gres/gpu:h100nvl=1",
                "gres/gpu=2,gres/gpu:h100nvl=2",
            ),
            contract=contract,
            contract_path=contract_path,
        )


@pytest.mark.campaign_artifacts('runtime_inputs/reconstruction_phase34_20260904/train_035k.repromoted.json')
def test_phase35_wrapper_renderer_and_receipt_are_fail_closed() -> None:
    prereg = preregistration()
    assert prereg["execution_policy"] == {
        "task_count": 3,
        "gres": "gpu:h100nvl:1",
        "cpus_per_task": 8,
        "memory": "64G",
        "time": "24:00:00",
        "requeue": False,
        "maximum_restarts": 0,
        "global_concurrency": 3,
    }
    assert verify.load_repair_preregistration()["execution_policy"] == {
        **prereg["execution_policy"],
        "task_count": 2,
        "global_concurrency": 2,
    }
    wrapper = (ROOT / verify.WRAPPER).read_text()
    assert "#SBATCH --no-requeue" in wrapper
    assert '[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]' in wrapper
    assert wrapper.index("trap finalize EXIT") < wrapper.index(
        'verify_reconstruction_phase35_contract.py "${job_contract}" '
        '--shell-output "${runtime_file}"'
    )
    assert "--max-restarts 0" in wrapper
    renderer = (
        ROOT / "scripts/slurm/render_reconstruction_phase35_job.py"
    ).read_text()
    assert '"training_authorized": True' in renderer
    assert '"sealed_test_access_authorized": False' in renderer
    assert '"promotion_authorized": False' in renderer
    synthetic_contract = {
        "resources": prereg["execution_policy"],
        "task_id": "phase35-repair-synthetic",
        "contract_sha256": "a" * 64,
    }
    command = submit.submission_command(
        synthetic_contract, ROOT / "synthetic-contract.json"
    )
    assert "--parsable" in command
    assert "--hold" in command
    assert "--no-requeue" in command
    assert "--signal=B:USR1@300" not in command
    assert "--requeue" not in command
    submit.verify_submission_line(
        " ".join(command),
        command,
    )
    with pytest.raises(RuntimeError, match="authorized argv"):
        submit.verify_submission_line(
            " ".join(command).replace("--no-requeue", "--requeue"),
            command,
        )
    submitter = (
        ROOT / "scripts/slurm/submit_reconstruction_phase35_campaign.py"
    ).read_text()
    release_status = submitter.index('status = "release_in_progress"')
    persisted_release = submitter.index(
        "atomic_json(output, receipt(status))", release_status
    )
    assert release_status < persisted_release < submitter.index(
        "run(release_argv)", persisted_release
    )
    finalizer = (
        ROOT / "scripts/slurm/finalize_reconstruction_fullscale_receipt.py"
    ).read_text()
    assert '"best_rollout_edge_f1"' in finalizer
    assert "PHASE35_CONTRACT_VERSION" in finalizer
    assert (
        "hypertagging-reconstruction-phase35-improvement-contract-v1"
        in finalizer
    )
    assert "return 0 if terminal_success or not phase35 else 1" in finalizer
    assert "terminal evidence finalization failed" in wrapper
    assert '[[ -s "${receipt_file}" ]]' in wrapper
    for source in render.SOURCE_FILES:
        assert (ROOT / source).is_file(), source


def _campaign_receipt_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, status: str
) -> tuple[dict, str, Path]:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    git_sha = "a" * 40
    test_receipt = {"path": verify.PHASE35_TEST_RECEIPT, "sha256": "b" * 64}
    contracts: list[tuple[Path, dict]] = []
    for arm_role in verify.PHASE35_ARM_ROLES:
        relative = verify.phase35_contract_relative_path(arm_role)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contract = {
            "contract_version": verify.CONTRACT_VERSION,
            "study_id": verify.STUDY_ID,
            "task_id": verify.phase35_task_id(arm_role),
            "arm_role": arm_role,
            "expected_git_sha": git_sha,
            "expected_git_tag": verify.IMPLEMENTATION_TAG,
            "test_receipt": test_receipt,
            "campaign_submission_receipt": verify.CAMPAIGN_SUBMISSION_RECEIPT,
            "resources": {
                "gres": "gpu:h100nvl:1",
                "cpus_per_task": 8,
                "memory": "64G",
                "time": "24:00:00",
            },
        }
        contract["contract_sha256"] = verify.canonical_contract_hash(contract)
        path.write_text(json.dumps(contract), encoding="utf-8")
        contracts.append((path, contract))

    planned_jobs = [
        {
            "arm_role": contract["arm_role"],
            "task_id": contract["task_id"],
            "contract": str(path.relative_to(tmp_path)),
            "contract_sha256": contract["contract_sha256"],
            "scheduler_comment": f"phase35:{contract['contract_sha256']}",
        }
        for path, contract in contracts
    ]
    jobs: list[dict] = []
    release_records: dict[str, str] = {}
    current_records: dict[str, str] = {}
    for index, (path, contract) in enumerate(contracts, start=101):
        job_id = str(index)

        def record(*, state: str, reason: str) -> str:
            return (
                f"JobId={job_id} JobName={contract['task_id']} "
                "Partition=inter Account=others Requeue=0 Restarts=0 "
                "NumCPUs=8 NumNodes=1 NumTasks=1 CPUs/Task=8 "
                "TimeLimit=1-00:00:00 MinMemoryNode=64G "
                f"Comment=phase35:{contract['contract_sha256']} "
                "ReqTRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,"
                "gres/gpu:h100nvl=1 TresPerNode=gres:gpu:h100nvl:1 "
                f"Command={tmp_path / verify.WRAPPER} WorkDir={tmp_path} "
                f"JobState={state} Reason={reason}"
            )

        command = verify.phase35_submission_command(contract, path)
        held_record = record(state="PENDING", reason="JobHeldUser")
        current_records[job_id] = record(state="PENDING", reason="Resources")
        job = {
            "arm_role": contract["arm_role"],
            "task_id": contract["task_id"],
            "job_id": job_id,
            "contract": str(path.relative_to(tmp_path)),
            "contract_sha256": contract["contract_sha256"],
            "submission_argv": command,
            "scheduler_submit_line": shlex.join(command),
            "scheduler_record_while_held": held_record,
            "scheduler_state_while_held": "PENDING",
            "requeue": 0,
            "restarts": 0,
        }
        if status == "submitted":
            released_record = current_records[job_id]
            job["scheduler_record_after_release"] = released_record
            job["scheduler_state_after_release"] = "PENDING"
            release_records[job_id] = released_record
        jobs.append(job)
    job_ids = [item["job_id"] for item in jobs]
    receipt = submit._with_receipt_hash(
        {
            "receipt_version": (
                verify.CAMPAIGN_SUBMISSION_RECEIPT_VERSION
            ),
            "created_at": "2026-09-04T00:00:00+00:00",
            "updated_at": "2026-09-04T00:00:01+00:00",
            "status": status,
            "study_id": verify.STUDY_ID,
            "git_sha": git_sha,
            "git_tag": verify.IMPLEMENTATION_TAG,
            "test_receipt": test_receipt,
            "permissions": dict(verify.PHASE35_CAMPAIGN_PERMISSIONS),
            "sealed_test_accessed": False,
            "automatic_promotion": False,
            "atomic_campaign_release": True,
            "jobs_initially_submitted_held": True,
            "planned_jobs": planned_jobs,
            "active_submission_arm": None,
            "jobs": jobs,
            "submitted_job_ids": job_ids,
            "release_argv": [
                "/opt/slurm/bin/scontrol",
                "release",
                ",".join(job_ids),
            ],
            "scheduler_records_after_release": release_records,
            "cancellation_states": {},
            "error": None,
        }
    )
    receipt_path = tmp_path / verify.CAMPAIGN_SUBMISSION_RECEIPT
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(verify, "slurm_job", current_records.__getitem__)
    return contracts[0][1], job_ids[0], receipt_path


@pytest.mark.parametrize("status", ["release_in_progress", "submitted"])
def test_phase35_campaign_receipt_exactly_binds_atomic_two_arm_repair_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    contract, job_id, receipt_path = _campaign_receipt_fixture(
        tmp_path, monkeypatch, status=status
    )
    validated = verify.verify_campaign_submission_receipt(
        job_id=job_id,
        contract=contract,
    )
    assert validated["status"] == status

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["permissions"]["promotion_authorized"] = True
    receipt_path.write_text(
        json.dumps(submit._with_receipt_hash(tampered)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="atomic campaign"):
        verify.verify_campaign_submission_receipt(
            job_id=job_id,
            contract=contract,
        )


def test_phase35_campaign_receipt_rejects_pre_release_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, job_id, receipt_path = _campaign_receipt_fixture(
        tmp_path, monkeypatch, status="release_in_progress"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "prepared_held"
    receipt_path.write_text(
        json.dumps(submit._with_receipt_hash(receipt)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="atomic campaign"):
        verify.verify_campaign_submission_receipt(
            job_id=job_id,
            contract=contract,
        )


def test_phase35_campaign_receipt_rejects_missing_repair_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, job_id, receipt_path = _campaign_receipt_fixture(
        tmp_path, monkeypatch, status="submitted"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    removed_id = receipt["jobs"].pop()["job_id"]
    receipt["submitted_job_ids"].remove(removed_id)
    receipt["scheduler_records_after_release"].pop(removed_id)
    receipt["release_argv"][-1] = ",".join(receipt["submitted_job_ids"])
    receipt_path.write_text(
        json.dumps(submit._with_receipt_hash(receipt)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="exactly two jobs"):
        verify.verify_campaign_submission_receipt(
            job_id=job_id,
            contract=contract,
        )


def test_phase35_release_intent_rejects_any_job_still_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, job_id, receipt_path = _campaign_receipt_fixture(
        tmp_path, monkeypatch, status="release_in_progress"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    current_records = {
        item["job_id"]: item["scheduler_record_while_held"].replace(
            "Reason=JobHeldUser", "Reason=Resources"
        )
        for item in receipt["jobs"]
    }
    current_records[job_id] = receipt["jobs"][0][
        "scheduler_record_while_held"
    ]
    monkeypatch.setattr(verify, "slurm_job", current_records.__getitem__)
    with pytest.raises(RuntimeError, match="remains held"):
        verify.verify_campaign_submission_receipt(
            job_id=job_id,
            contract=contract,
        )
