from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

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
        "--pretraining-checkpoint", "pretrain.pt",
        "--reconstruction-checkpoint", "reconstruction.pt",
        "--data", "selection.json",
        "--dataset-index", "index.json",
        "--output", "report.json",
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
    assert args.threads == 1
    assert args.deterministic_algorithms is False
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    policy = module.rollout_policy_identity(continue_through_empty_levels=True)
    assert module.REPORT_VERSION == "hypertagging-offline-full-decay-evaluation-v3"
    assert policy["empty_level_policy"] == "continue_to_max_level"
    assert len(str(policy["sha256"])) == 64


def test_full_decay_cli_rejects_invalid_threshold():
    module = _script_module()
    with pytest.raises(SystemExit):
        module.parse_args(_required_args() + ["--object-threshold", "1.5"])


def test_full_decay_cli_accepts_repeatable_source_categories():
    module = _script_module()
    args = module.parse_args(
        _required_args()
        + ["--source-category", "mixed", "--source-category", "charged"]
    )
    assert args.source_category == ["mixed", "charged"]


def test_full_decay_cli_allows_explicit_confidence_diagnostic_override():
    module = _script_module()
    args = module.parse_args(
        _required_args() + ["--disable-learned-confidence"]
    )
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

    assert module._validated_output_path(
        report,
        direct_inputs=(checkpoint,),
    ) == report.resolve()


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
        shard.resolve(), sidecar.resolve(), marker.resolve()
    ]
    for destination in (sidecar, tmp_path / "marker-symlink", tmp_path / "marker-hardlink"):
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
