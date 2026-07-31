from __future__ import annotations

import torch

from hypertagging.data.dataset_index import build_dataset_index, load_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.evaluation import load_trained_evaluation_context
from hypertagging.models.ablation import build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import (
    ParquetEventWriter, feature_spec_v4, iter_event_records_v4,
)
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import save_training_checkpoint
from hypertagging.training.data_module import build_real_data_module
from hypertagging.training.model_config import MODEL_PRESETS


def test_trained_evaluation_restores_normalization_and_heldout_contract(tmp_path):
    source = write_notebook_fixture_v4(tmp_path / "source.parquet", row_group_size=1)
    shards = []
    for index, record in enumerate(iter_event_records_v4(source)):
        record["source_file"] = f"fixture-{index}.root"
        shard = tmp_path / f"events-{index}.parquet"
        with ParquetEventWriter(shard, event_buffer_size=1) as writer:
            writer.write_event(record)
        shards.append(shard)
    split_config = SourceAwareSplitConfig(
        train_fraction=0.5,
        validation_fraction=0.5,
        test_fraction=0.0,
        group_by_source_file=False,
        seed=1,
    )
    # Choose a deterministic seed that places the two fixture UIDs on both sides.
    for seed in range(1, 1000):
        candidate = SourceAwareSplitConfig(
            train_fraction=0.5, validation_fraction=0.5, test_fraction=0.0,
            group_by_source_file=False, seed=seed,
        )
        from hypertagging.data.splitting import stable_split_name
        assigned = {
            stable_split_name(record, candidate)
                for shard in shards for record in iter_event_records_v4(shard)
        }
        if assigned == {"train", "validation"}:
            split_config = candidate
            break
    index_path = build_dataset_index(
        shards, tmp_path / "index.json", split_config=split_config
    )
    module = build_real_data_module(
        shards, dataset_index=index_path, split_config=split_config,
        seed=split_config.seed, required_splits=("train", "validation"),
    )
    architecture = MODEL_PRESETS["tiny_cpu"]
    model = build_ablation_model(
        "full_revised", n_features=12, n_types=len(PDG_TOKENS),
        hidden_dim=architecture.d_model, hyper_dim=architecture.hyper_dim,
        n_queries=architecture.n_queries, max_cardinality=architecture.max_cardinality,
        n_heads=architecture.n_heads, n_context_layers=architecture.n_context_layers,
        curvature=architecture.curvature, ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
    )
    policy = ReconstructionConstraintPolicy(initial_state_policy="upsilon4s")
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint.pt", model=model, encoder=model.encoder,
        config={
            "ablation": "full_revised", "seed": split_config.seed,
            "target_policy": "complete_only", "max_events": None,
            "rollout_pid_kinematics_mode": "hard",
        },
        normalizer_state=module.normalization_state(),
        split_manifest_hash=module.split_manifest_hash,
        feature_contract={
            "feature_spec_hash": feature_spec_v4()["feature_spec_hash"],
            "model_feature_contract_hash": feature_spec_v4()["model_feature_contract_hash"],
            "reconstruction_constraint_policy": policy.to_dict(),
        },
        data_order_contract={
            "dataset_index_hash": load_dataset_index(index_path)["index_hash"]
        },
        architecture=architecture.to_dict(),
    )
    context = load_trained_evaluation_context(
        checkpoint=checkpoint, data=shards, dataset_index=index_path,
        split="validation", max_events=2,
    )
    assert context.events
    assert context.report_metadata["evaluation_split"] == "validation"
    assert context.report_metadata["evaluated_event_uids"] == [
        event.event_uid for event in context.events
    ]
    assert context.constraint_policy.initial_state_policy == "upsilon4s"
    assert context.rollout_pid_kinematics_mode == "hard"
    assert torch.equal(
        context.model.runtime_feature_normalizer.common_mean,
        module.normalizers["common"].mean,
    )
    batch = context.collated_batch()
    assert batch["runtime_features_are_raw"]
    assert torch.isfinite(batch["track_features"]).all()
