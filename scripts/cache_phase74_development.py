"""Cache authenticated detector inputs and separate supervision under CPU admission."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import read, binding, write  # noqa: E402


def cache(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler allocation required")
    import torch
    from hypertagging.data.heterogeneous import collate_heterogeneous_events
    from hypertagging.evaluation.trained_context import load_trained_evaluation_context
    from hypertagging.evaluation.full_decay_metrics import _tree_view, _truth_b_roots
    from hypertagging.reconstruction.hierarchical_inference import (
        project_schema_v4_fsps,
    )
    from hypertagging.reconstruction.level_rollout import (
        _constrained_rollout_model_batch,
    )
    from hypertagging.reconstruction.beam_search import _truth_free_model_view

    torch.set_num_threads(1)
    admission = json.loads((output / "data-admission.json").read_text())
    train_design, dev_design = (
        read(admission["training"]),
        read(admission["development"]),
    )
    inputs = admission["inputs"]
    for key in ("selection", "index", "reconstruction-checkpoint"):
        read(inputs[key]) if key != "reconstruction-checkpoint" else None
    # Historical model is loaded solely to validate data/normalization/PID contracts;
    # no weight, prediction, embedding or optimizer transfers into a fresh model.
    from scripts.prepare_phase74_development_data import sha

    if (
        sha(inputs["reconstruction-checkpoint"]["path"])
        != inputs["reconstruction-checkpoint"]["sha256"]
    ):
        raise ValueError("Normalizer provenance checkpoint changed")
    ctx = load_trained_evaluation_context(
        checkpoint=inputs["reconstruction-checkpoint"]["path"],
        data=inputs["selection"]["path"],
        dataset_index=inputs["index"]["path"],
        split="validation",
        device="cpu",
        max_events=600,
        diagnostic_allow_external_independent_sample=True,
        event_selection="explicit_uids",
        explicit_event_uids=dev_design["event_uids"],
    )
    normalizer = ctx.model.runtime_feature_normalizer

    def encode(event):
        full = ctx.data_module.normalize_batch(collate_heterogeneous_events([event]))
        projection = project_schema_v4_fsps(full)
        detector = _truth_free_model_view(
            _constrained_rollout_model_batch(
                projection.batch, target_level=1, policy=ctx.constraint_policy
            )
        )
        keys = projection.evaluation_leaf_source_keys[0].tolist()
        view = _tree_view(full, 0, truth=True)
        mapping = {key: i for i, key in enumerate(keys)}
        # All truth topology stays in a separate supervision object.
        node_sets = []
        for node in range(len(view.active)):
            source_set = (
                view.source_set(node) if bool(view.active[node]) else frozenset()
            )
            node_sets.append(
                frozenset(mapping[key] for key in source_set)
                if source_set <= mapping.keys()
                else frozenset()
            )
        parents = full["parent_ids"][0].tolist()
        roots = (
            _truth_b_roots(view)
            if event.source_category in ("charged", "mixed")
            else []
        )
        groups = [node_sets[root] for root in roots]
        target = torch.zeros(len(keys), dtype=torch.long)
        if event.source_category in ("charged", "mixed") and (
            len(groups) != 2 or any(not g for g in groups)
        ):
            target.fill_(-1)
            groups = []
        else:
            for slot, group in enumerate(groups, 1):
                for position in group:
                    if target[position] != 0:
                        raise ValueError("Overlapping truth B groups")
                    target[position] = slot
        if not 1 <= len(keys) <= 256:
            raise ValueError("FSP capacity exceeded; no silent truncation")
        return {
            "uid": event.event_uid,
            "category": event.source_category,
            "full": full,
            "detector": detector,
            "sources": projection.batch["recursive_leaf_source_mask"][0].clone(),
            "charge": projection.batch["charge"][0].clone(),
            "targets": target,
            "supervision": {
                "node_sets": node_sets,
                "parents": parents,
                "b_groups": groups,
                "deep_sets": [
                    node_sets[i]
                    for i in range(len(node_sets))
                    if int(full["level_ids"][0, i]) >= 2 and node_sets[i]
                ],
            },
        }

    train = [
        encode(event)
        for event in ctx.data_module.iter_events(
            "train", event_uids=train_design["event_uids"]
        )
    ]
    dev = [encode(event) for event in ctx.events]
    train.sort(key=lambda x: x["uid"])
    dev.sort(key=lambda x: x["uid"])
    if {r["uid"] for r in train} != set(train_design["event_uids"]) or {
        r["uid"] for r in dev
    } != set(dev_design["event_uids"]):
        raise ValueError("Cached identities differ from designated cohort")
    payload = {
        "train": train,
        "development": dev,
        "runtime_normalizer": normalizer,
        "normalizer_state": ctx.checkpoint["normalizer_state"],
        "source_feature_contract": ctx.checkpoint["feature_contract"],
        "historical_parameters_transferred": False,
    }
    path = output / "data-cache.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    write(
        output / "cache-admission.json",
        {
            "status": "PASS",
            "cache": binding(path),
            "data_admission": binding(output / "data-admission.json"),
            "train": len(train),
            "development": len(dev),
            "max_fsp": max(len(row["targets"]) for row in train + dev),
            "max_full_nodes": max(
                row["full"]["node_mask"].shape[1] for row in train + dev
            ),
            "normalization": "inherited_authenticated_70000_train_fitted_only_unchanged",
            "historical_parameters_transferred": False,
            "capacity_dropped_events": 0,
            "target_unavailable": {
                role: sum(bool((r["targets"] < 0).all()) for r in rows)
                for role, rows in [("train", train), ("development", dev)]
            },
        },
    )
    print("cache complete", len(train), len(dev), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    cache(p.parse_args().output.resolve())
