"""CPU-only train-calibrated radial diagnostics; no strict/test selection or updates."""

from pathlib import Path
import sys, json, hashlib, argparse
from dataclasses import replace
import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from hypertagging.training.pretrain_trainer import PretrainConfig, _add_topology_labels
from hypertagging.evaluation.pretraining_validation import (
    _build_model,
    _validate_checkpoint_contract,
)
from hypertagging.training.data_module import build_real_data_module
from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.training.pretraining_curriculum import (
    DEFAULT_PRETRAINING_PHASES,
    build_curriculum_batch,
)
from hypertagging.training.pretraining_diagnostics import radial_cap_diagnostics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_same(x, y)
    else:
        assert a == b


def measure(model, dm, events, seed, corruption_objective):
    model.eval()
    results = {}
    with torch.inference_mode():
        for phase in DEFAULT_PRETRAINING_PHASES:
            tangents = []
            levels = []
            for start in range(0, len(events), 8):
                batch = dm.normalize_batch(
                    collate_heterogeneous_events(events[start : start + 8])
                )
                _add_topology_labels(batch)
                c = build_curriculum_batch(
                    batch,
                    phase.view,
                    seed=seed + start,
                    corruption_objective=corruption_objective,
                    truth_guided_structural_relation_inputs=False,
                )
                encoded, _, runtime = model.encode_runtime(
                    c.batch, attention_mask=c.batch["curriculum_attention_mask"]
                )
                t = model.encoder.tangent_scale(
                    model.encoder.hyper_projection(encoded.tree_projection.float())
                )
                mask = runtime["node_mask"]
                tangents.append(t[mask])
                levels.append(runtime["level_ids"][mask])
            t = torch.cat(tangents)
            l = torch.cat(levels)
            m = torch.ones_like(l, dtype=torch.bool)
            results[phase.name] = {
                "node_count": len(l),
                **radial_cap_diagnostics(t, m, l, model.encoder.max_tangent_norm),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    closeout = json.loads(
        (
            ROOT / "artifacts/codex/reconstruction_phase63_closeout_20260930.json"
        ).read_text()
    )
    checkpoint = (
        Path(closeout["arms"]["corrected_recovery"]["source_root"])
        / "runtime_inputs/reconstruction_phase62_20260928/pretraining-control-step2188.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = replace(
        PretrainConfig(**payload["config"]),
        device="cpu",
        mixed_precision=False,
        num_workers=0,
    )
    data = (
        ROOT
        / "configs/training_selection/phase60_validation_expansion_20260925/train_070k.json"
    )
    index = (
        ROOT
        / "artifacts/experiment_readiness/reconstruction_phase60_20260925/train_070k.complete_only.index.json"
    )
    dm = build_real_data_module(
        data,
        seed=config.seed,
        allow_legacy_conflated=False,
        normalization_state=payload["normalizer_state"],
        required_splits=("train", "validation"),
        num_workers=0,
        persistent_workers=False,
        dataset_index=index,
        rescan_dataset=False,
        target_policy="complete_only",
        scientific_mode=True,
    )
    _validate_checkpoint_contract(payload, dm)
    selection = json.loads(data.read_text())
    paths = {
        Path(s["path"]).name: Path(s["path"])
        for s in json.loads(index.read_text())["shards"]
    }
    train_uids = []
    for entry in selection["entries"]:
        if entry["split"] == "train":
            train_uids.extend(
                pq.read_table(paths[Path(entry["path"]).name], columns=["event_uid"])[
                    "event_uid"
                ].to_pylist()
            )
    assert len(train_uids) == len(set(train_uids)) == 70000
    chosen = sorted(
        train_uids,
        key=lambda uid: hashlib.sha256(
            ("phase64-radial-calibration:" + uid).encode()
        ).hexdigest(),
    )[:128]
    cohort = json.loads(
        (
            ROOT
            / "configs/reconstruction/ht_reconstruction_phase63_validation_cohort_20260928.json"
        ).read_text()
    )
    development = cohort["checkpoint_selection_event_uids"][:32]
    assert not set(chosen) & set(development) and not set(development) & set(
        cohort["event_uids"]
    )
    train = list(dm.iter_events("train", event_uids=chosen))
    dev = list(dm.iter_events("validation", event_uids=development))
    assert {e.event_uid for e in train} == set(chosen) and {
        e.event_uid for e in dev
    } == set(development)
    train.sort(key=lambda e: chosen.index(e.event_uid))
    dev.sort(key=lambda e: development.index(e.event_uid))
    model = _build_model(payload, config, dm, torch.device("cpu"))
    result = {
        "version": "phase67-radial-diagnostic-v1",
        "sealed_test_accessed": False,
        "strict_events_accessed": False,
        "train_calibration_event_uids": chosen,
        "development_event_uids": development,
        "source_checkpoint_sha256": digest(checkpoint),
        "models": {},
    }
    source = Path(
        "/project/agkuhr/users/boyang/data/HyperTagging_artifacts/phase66_review_20261004/training-source"
    )
    for arm, job in (
        ("scheduled_mixed", "16810677"),
        ("teacher_only", "16810678"),
    ):
        run = source / f"artifacts/runs/ht-reconstruction-phase67-20261004/{arm}/{job}"
        native = json.loads((run / "result.json").read_text())
        bindings = {
            "refined": {
                "path": native["pretraining_refinement"]["checkpoint"],
                "sha256": native["pretraining_refinement"]["checkpoint_sha256"],
            },
            "selected": native["checkpoints"]["best"],
            "final": native["checkpoints"]["final"],
        }
        for stage, binding in bindings.items():
            assert digest(binding["path"]) == binding["sha256"]
            cp = torch.load(binding["path"], map_location="cpu", weights_only=False)
            assert_same(cp["normalizer_state"], payload["normalizer_state"])
            model = _build_model(payload, config, dm, torch.device("cpu"))
            model.encoder.load_state_dict(cp["encoder_state_dict"], strict=True)
            for sample, events in (("train", train), ("development", dev)):
                key = arm + "_" + stage + "_" + sample
                result["models"][key] = measure(
                    model, dm, events, 20260930, config.corruption_objective
                )
                print(key + " complete", flush=True)
            result.setdefault("checkpoint_bindings", {})[arm + "_" + stage] = binding[
                "sha256"
            ]
            args.output.with_suffix(".progress.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("post-training radial diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
