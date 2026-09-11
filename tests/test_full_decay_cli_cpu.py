from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def _script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_full_decay.py"
    spec = importlib.util.spec_from_file_location("evaluate_full_decay_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_args() -> list[str]:
    return [
        "--pretraining-checkpoint",
        "pretrain.pt",
        "--reconstruction-checkpoint",
        "reconstruction.pt",
        "--data",
        "selection.json",
        "--dataset-index",
        "index.json",
        "--output",
        "report.json",
    ]


def test_full_decay_cli_defaults_to_cpu_both_scopes(monkeypatch):
    module = _script_module()
    args = module.parse_args(_required_args())
    assert args.scope == "both"
    assert args.split == "validation"
    assert args.data == ["selection.json"]
    assert args.source_category is None
    assert args.event_selection == "auto"
    assert args.use_learned_confidence is None
    assert not args.beam_search
    assert args.beam_width == 4
    assert args.threads == 1
    assert args.deterministic_algorithms is False
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    policy = module.rollout_policy_identity(continue_through_empty_levels=True)
    assert module.REPORT_VERSION == "hypertagging-offline-full-decay-evaluation-v3"
    assert module.BEAM_REPORT_VERSION == "hypertagging-offline-full-decay-evaluation-v4"
    assert policy["empty_level_policy"] == "continue_to_max_level"
    assert len(str(policy["sha256"])) == 64


def test_full_decay_cli_rejects_invalid_threshold():
    module = _script_module()
    with pytest.raises(SystemExit):
        module.parse_args(_required_args() + ["--object-threshold", "1.5"])


@pytest.mark.parametrize(
    "option,value",
    [
        ("--beam-width", "0"),
        ("--beam-max-candidates-per-query", "0"),
        ("--beam-max-proposals-per-level", "-1"),
        ("--beam-max-daughter-options", "0"),
        ("--beam-max-type-options", "0"),
        ("--beam-max-cardinality-options", "0"),
        ("--beam-max-candidate-expansions-per-query", "0"),
        ("--beam-max-nodes-per-hypothesis", "0"),
        ("--beam-score-length-normalization", "nan"),
        ("--beam-empty-level-penalty", "-1"),
        ("--beam-empty-level-penalty", "inf"),
        ("--beam-oracle-k", "0"),
        ("--beam-oracle-k", "5"),
    ],
)
def test_full_decay_cli_rejects_invalid_beam_configuration(option, value):
    with pytest.raises(SystemExit):
        _script_module().parse_args(_required_args() + [option, value])


def test_full_decay_cli_resolves_explicit_beam_config():
    module = _script_module()
    args = module.parse_args(
        _required_args()
        + [
            "--beam-search",
            "--beam-width",
            "3",
            "--beam-max-candidates-per-query",
            "2",
            "--beam-max-proposals-per-level",
            "8",
            "--beam-score-length-normalization",
            "0.5",
            "--beam-empty-level-penalty",
            "0.2",
            "--beam-oracle-k",
            "1",
            "--beam-oracle-k",
            "3",
        ]
    )
    config = module._beam_config_from_args(args)
    assert args.beam_search
    assert config.beam_width == 3 and config.max_candidates_per_query == 2
    assert config.max_proposals_per_level == 8
    assert (
        config.score_length_normalization == 0.5 and config.empty_level_penalty == 0.2
    )
    assert args.beam_oracle_k == [1, 3]


def test_example_beam_config_matches_validated_defaults():
    from dataclasses import asdict
    from hypertagging.reconstruction.beam_search import BeamSearchConfig

    path = (
        Path(__file__).resolve().parents[1]
        / "configs/reconstruction/full_depth_beam.json"
    )
    assert asdict(BeamSearchConfig(**json.loads(path.read_text()))) == asdict(
        BeamSearchConfig()
    )


def test_cli_report_keeps_greedy_results_and_adds_parallel_beam_metrics(
    tmp_path, monkeypatch
):
    import torch
    from test_hierarchical_inference_cpu import (
        _normalized_native_v4_batch,
        _TwoBThenUpsilonModel,
        _rollout_config,
    )

    class ScriptedCPUModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scripted = _TwoBThenUpsilonModel()

        def forward(self, batch, **kwargs):
            return self.scripted(batch, **kwargs)

    module = _script_module()
    source = _normalized_native_v4_batch()
    # Declare an Upsilon truth root for an eligible full-scope report.
    for name in ("pid_labels", "pid_target_labels", "truth_pid_labels"):
        if name in source:
            source[name][0, 6] = 1
    model = ScriptedCPUModel().eval()
    context = SimpleNamespace(
        model=model,
        constraint_policy=_rollout_config().constraint_policy,
        checkpoint={"config": {}, "confidence_head_trained": False},
        rollout_pid_kinematics_mode="soft_decision_hard_construction",
        report_metadata={},
        events=[
            SimpleNamespace(
                event_uid="beam-fixture", source_category="mixed", source_file="fixture"
            )
        ],
        collated_event_batch=lambda *args, **kwargs: {
            key: value.clone() for key, value in source.items()
        },
    )
    monkeypatch.setattr(
        module,
        "validate_checkpoint_pair",
        lambda *a, **kw: SimpleNamespace(as_dict=lambda: {}),
    )
    monkeypatch.setattr(module, "load_trained_evaluation_context", lambda **kw: context)
    monkeypatch.setattr(module, "_evaluator_code_provenance", lambda: {})
    paths = [tmp_path / "greedy.json", tmp_path / "beam.json", tmp_path / "ranking.json"]
    for path, extra in zip(
        paths, ([], ["--beam-search", "--beam-width", "2"], ["--beam-width", "2"])
    ):
        assert (
            module.main(
                _required_args()
                + [
                    "--output",
                    str(path),
                    "--max-level",
                    "3",
                    "--scope",
                    "full",
                    "--omit-trees",
                ]
                + extra
            )
            == 0
        )
    greedy, beam, ranking = [json.loads(path.read_text()) for path in paths]
    for report in (greedy, beam, ranking):
        retained = report["retained_tree_checks"]
        assert retained["version"] == "retained-direct-tree-checks-v1"
        assert retained["summaries"]["full/greedy"]["unavailable_unit_count"] == 0
        assert not retained["replaces_preregistered_metrics"]
        assert report["events"][0]["scopes"]["full"]["retained_tree_metrics"]["available"]
    retained_beam = beam["events"][0]["scopes"]["full"]["retained_tree_beam"]
    assert retained_beam["candidate_count"] == len(beam["events"][0]["scopes"]["full"]["beam"]["candidates"])
    for candidate in retained_beam["candidates"]:
        assert candidate["metrics"]["available"]
    for candidate in ranking["beam_search"]["events"][0]["candidates"]:
        assert candidate["retained_tree_metrics_by_scope"]["full"]["available"]
    assert "full/proposal_beam/normalized_joint_log_probability" in ranking["retained_tree_checks"]["summaries"]
    assert ranking["configuration"]["beam_search"]["algorithm"] == "diagnostic_proposal_set_beam"
    assert ranking["beam_search"]["event_count"] == 1
    assert ranking["beam_search"]["events"][0]["candidate_count"] > 0
    assert ranking["summaries"]["full"]["decay_metrics"] == greedy["summaries"]["full"]["decay_metrics"]
    assert greedy["report_version"] == module.REPORT_VERSION
    assert (
        not {
            "beam_search_enabled",
            "beam_search",
            "beam_oracle_ks",
        }
        & greedy["configuration"].keys()
    )
    assert beam["report_version"] == module.BEAM_REPORT_VERSION
    assert beam["configuration"]["beam_search_enabled"] is True
    assert (
        beam["summaries"]["full"]["decay_metrics"]
        == greedy["summaries"]["full"]["decay_metrics"]
    )
    assert (
        beam["summaries"]["full"]["inference"]
        == greedy["summaries"]["full"]["inference"]
    )
    assert beam["configuration"]["beam_search"]["beam_width"] == 2
    comparison = beam["events"][0]["scopes"]["full"]["beam"]
    assert comparison["candidates"][0]["rank"] == 1
    assert "top1_inference" in comparison
    assert "reconstructed_tree" not in comparison["candidates"][0]
    assert (
        beam["summaries_by_source_category"]["mixed"]["full"]["beam"]["event_count"]
        == 1
    )
    shapes = beam["summaries_by_target_shape"]["full"]
    assert shapes
    for shape in shapes.values():
        assert shape["beam"]["aggregation_unit"] == "truth_decay_unit"
        assert "coherent_event_perfect_lcag" not in shape["beam"]["oracle_at_k"]["2"]


def test_full_decay_cli_accepts_repeatable_source_categories():
    module = _script_module()
    args = module.parse_args(
        _required_args()
        + ["--source-category", "mixed", "--source-category", "charged"]
    )
    assert args.source_category == ["mixed", "charged"]


def test_full_decay_cli_allows_explicit_confidence_diagnostic_override():
    module = _script_module()
    args = module.parse_args(_required_args() + ["--disable-learned-confidence"])
    assert args.use_learned_confidence is False


def test_full_decay_cli_accepts_deterministic_algorithm_requirement():
    module = _script_module()
    args = module.parse_args(
        _required_args() + ["--threads", "1", "--deterministic-algorithms"]
    )
    assert args.threads == 1
    assert args.deterministic_algorithms is True


def test_half_tree_beam_oracle_uses_aggregate_lcag_metrics(monkeypatch):
    module = _script_module()

    class HalfEvaluation:
        halves = ()

    monkeypatch.setattr(
        module,
        "summarize_decay_evaluations",
        lambda _rows: {
            "perfect_lcag": {"value": 0.2},
            "lcag_pair_accuracy": {"value": 0.3},
            "mother_pid_coverage": {"value": 0.4},
            "source_recall": {"value": 0.5},
            "source_precision": {"value": 0.6},
        },
    )
    assert module._beam_oracle_key(HalfEvaluation()) == (0.2, 0.3, 0.4, 0.5, 0.6)


def test_full_decay_explicit_uid_manifest_is_hashed_and_validation_only(
    tmp_path,
):
    module = _script_module()
    uids = ["validation:a", "validation:b"]
    manifest = tmp_path / "cohort.json"
    payload = {
        "manifest_version": "hypertagging-reconstruction-evaluation-cohort-v1",
        "role": "validation",
        "sealed_test_role_access": "forbidden",
        "event_uid_count": len(uids),
        "event_uids_sha256": module._uid_sequence_sha256(uids),
        "event_uids": uids,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    loaded = module._load_event_uid_manifest(manifest, len(uids))

    assert loaded["event_uids"] == uids
    assert len(loaded["sha256"]) == 64
    payload["role"] = "test"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is invalid"):
        module._load_event_uid_manifest(manifest, len(uids))


def test_output_path_cannot_alias_direct_or_manifest_referenced_input(tmp_path):
    module = _script_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    alias = tmp_path / "checkpoint-alias.pt"
    alias.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="must not alias"):
        module._validated_output_path(
            alias,
            direct_inputs=(checkpoint,),
        )
    hardlink = tmp_path / "checkpoint-hardlink.pt"
    hardlink.hardlink_to(checkpoint)
    with pytest.raises(ValueError, match="must not alias"):
        module._validated_output_path(
            hardlink,
            direct_inputs=(checkpoint,),
        )

    shard = tmp_path / "events.parquet"
    shard.write_bytes(b"events")
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "data_root": str(tmp_path),
                "entries": [{"path": shard.name}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not alias"):
        module._validated_output_path(
            shard,
            direct_inputs=(manifest,),
            data_arguments=[str(manifest)],
        )


def test_output_path_allows_existing_report_overwrite(tmp_path):
    module = _script_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")

    assert (
        module._validated_output_path(
            report,
            direct_inputs=(checkpoint,),
        )
        == report.resolve()
    )


def test_atomic_report_publication_does_not_touch_legacy_temp_aliases(tmp_path):
    module = _script_module()
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel")
    report = tmp_path / "report.json"
    legacy_symlink = tmp_path / f".{report.name}.{os.getpid()}.tmp"
    legacy_hardlink = tmp_path / f".{report.name}.{os.getpid() + 1}.tmp"
    legacy_symlink.symlink_to(sentinel)
    legacy_hardlink.hardlink_to(sentinel)

    module._atomic_write_json(report, {"status": "valid"})

    assert json.loads(report.read_text()) == {"status": "valid"}
    assert sentinel.read_bytes() == b"sentinel"
    assert legacy_symlink.is_symlink()
    assert legacy_symlink.read_bytes() == b"sentinel"
    assert os.path.samefile(legacy_hardlink, sentinel)


def test_code_provenance_lists_untracked_names_without_reading_payload(
    tmp_path, monkeypatch
):
    module = _script_module()
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n")
    subprocess.run(("git", "add", "tracked.py"), cwd=tmp_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=tmp_path,
        check=True,
    )
    restricted = tmp_path / "restricted.parquet"
    restricted.write_bytes(b"must-not-be-read")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    def bomb_read_bytes(_path):
        raise AssertionError("untracked payload content was read")

    monkeypatch.setattr(module.Path, "read_bytes", bomb_read_bytes)
    report = module._evaluator_code_provenance()

    assert report["index_matches_worktree"]
    assert not report["provenance_complete"]
    assert report["untracked_paths"] == ["restricted.parquet"]
    assert report["worktree_patch_sha256"] != "unknown"

    tracked.write_text("value = 2\n")
    changed = module._evaluator_code_provenance()
    assert not changed["index_matches_worktree"]
    assert not changed["provenance_complete"]


def test_relative_manifest_root_binds_shard_sidecar_and_marker_aliases(tmp_path):
    module = _script_module()
    manifest_dir = tmp_path / "manifests"
    data_root = tmp_path / "published"
    manifest_dir.mkdir()
    data_root.mkdir()
    shard = data_root / "events.parquet"
    sidecar = data_root / "events.parquet.metadata.json"
    marker = data_root / "events.parquet.complete"
    for path in (shard, sidecar, marker):
        path.write_bytes(path.name.encode())
    manifest = manifest_dir / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "data_root": "../published",
                "entries": [
                    {
                        "path": shard.name,
                        "sidecar_path": sidecar.name,
                        "completion_marker_path": marker.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module._manifest_referenced_paths(manifest) == [
        shard.resolve(),
        sidecar.resolve(),
        marker.resolve(),
    ]
    for destination in (
        sidecar,
        tmp_path / "marker-symlink",
        tmp_path / "marker-hardlink",
    ):
        if destination.name == "marker-symlink":
            destination.symlink_to(marker)
        elif destination.name == "marker-hardlink":
            destination.hardlink_to(marker)
        with pytest.raises(ValueError, match="must not alias"):
            module._validated_output_path(
                destination,
                direct_inputs=(manifest,),
                data_arguments=[str(manifest)],
            )
