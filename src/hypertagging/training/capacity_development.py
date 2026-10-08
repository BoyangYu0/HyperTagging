"""Matched fresh-history baseline and assembly objectives for Phase74 only."""

from __future__ import annotations
import torch
from torch.nn import functional as F
from hypertagging.training.pretrain_trainer import (
    ContextualPretrainingModel,
    PretrainConfig,
    _add_topology_labels,
    _leaf_pid_loss,
    _pretraining_weights,
    _hard_negative_tree_loss,
    valid_b_root_channel_mask,
)
from hypertagging.training.pretraining_curriculum import (
    DEFAULT_PRETRAINING_PHASES,
    build_curriculum_batch,
)
from hypertagging.losses.hyperbolic_pretraining import (
    build_tree_relation_targets,
    build_topology_safe_parent_negative_mask,
    pool_b_branch_embeddings,
    hyperbolic_pretraining_loss,
)
from hypertagging.models.assembly_development import (
    AssemblyMembershipDecoder,
    assembly_relation_loss,
)
from hypertagging.models.direct_membership import direct_membership_loss


SEED = 202610081
ARMS = ("128-existing", "128-assembly", "256-existing", "256-assembly")
ARCHITECTURE = {
    "hyper_dim": 32,
    "curvature": 1.0,
    "n_heads": 4,
    "n_context_layers": 4,
    "dropout": 0.0,
    "hyper_projection_init_scale": 0.05,
    "tangent_scale_mode": "fixed",
    "max_tangent_norm": 4.0,
    "hyperbolic_level_encoding": "learned_euclidean",
}


def fresh_model(width, normalizer):
    if width not in (128, 256):
        raise ValueError("Unknown contextual width")
    torch.manual_seed(SEED)
    model = ContextualPretrainingModel(
        d_model=width, ffn_dim=4 * width, channel_memory_size=512, **ARCHITECTURE
    )
    model.set_runtime_feature_normalizer(normalizer)
    return model


def fresh_decoder():
    torch.manual_seed(SEED + 1)
    return AssemblyMembershipDecoder()


def detector_features(model, detector):
    mask = detector["node_mask"]
    attention = mask[:, :, None] & mask[:, None, :]
    encoded, pid_logits, runtime = model.encode_runtime(
        detector, attention_mask=attention
    )
    return encoded.node_embeddings[0], pid_logits, runtime


def membership_loss(result, targets):
    mask = torch.ones_like(targets[None], dtype=torch.bool)
    # Both proposal and refinement are trained; proposal information is also
    # consumed differentiably by refinement, not merely an auxiliary output.
    return (
        direct_membership_loss(result["logits"], result["objects"], targets[None], mask)
        + direct_membership_loss(
            result["proposal_logits"], result["proposal_objects"], targets[None], mask
        )
    ) / 2


def assembly_loss(model, decoder, row):
    h, _, _ = detector_features(model, row["detector"])
    result = decoder(h, row["sources"])
    member = membership_loss(result, row["targets"])
    relation, counts = assembly_relation_loss(result["states"], row["supervision"])
    return (
        member + relation,
        {"membership": member, "within_b_relation": relation},
        counts,
    )


def baseline_loss(model, batch, step, total_steps=1000):
    """Existing native progressive pretraining objectives, unchanged coefficients.

    The existing teacher/corruption curriculum remains a BASELINE training view;
    assembly_loss independently uses projected detector input and model-generated
    partitions. Teacher/corrupted composites are never called generated states.
    """
    config = PretrainConfig(
        data="",
        output_dir="",
        tangent_variance_target=0.05,
        radius_target_mode="generation_height_radius",
    )
    fraction = step / total_steps
    phase = DEFAULT_PRETRAINING_PHASES[
        0 if fraction < 0.2 else 1 if fraction < 0.45 else 2 if fraction < 0.75 else 3
    ]
    batch = dict(batch)
    _add_topology_labels(batch)
    curriculum = build_curriculum_batch(
        batch,
        phase.view,
        seed=SEED + step,
        corruption_objective="invalid_candidate",
        truth_guided_structural_relation_inputs=False,
    )
    encoded, pid_logits, train = model.encode_runtime(
        curriculum.batch, attention_mask=curriculum.batch["curriculum_attention_mask"]
    )
    structural = curriculum.structural_positive_mask
    relation_logits = model.relation_head(encoded.tree_projection)
    targets, mask = build_tree_relation_targets(
        parent_ids=train["parent_ids"],
        lca_depth=train["lca_depth"],
        level_ids=train["level_ids"],
        node_mask=structural,
        b_side=train["b_side"],
        lca_node_id=train["lca_node_id"],
        edges_to_lca_from_i=train["edges_to_lca_from_i"],
        edges_to_lca_from_j=train["edges_to_lca_from_j"],
    )
    negative = build_topology_safe_parent_negative_mask(
        targets, structural, train["ancestor_descendant_relation"]
    )
    branches, branch_mask = pool_b_branch_embeddings(
        encoded.channel_projection,
        train["b_side"],
        train["node_mask"],
        mode=model.channel_pooling,
        level_ids=train["level_ids"],
    )
    branch_mask &= valid_b_root_channel_mask(
        train, corrupted_node_mask=curriculum.corrupted_node_mask
    )[:, None]
    memory, full_ids, reco_ids = model.channel_memory.contents()
    structural_features = torch.cat(
        [
            train["b_channel_count_arrays"],
            train["b_depth_pid_count_arrays"].flatten(start_dim=2),
            train["b_branch_multiplicity_summaries"],
            train["b_intermediate_count_arrays"],
        ],
        -1,
    )
    output = hyperbolic_pretraining_loss(
        z=encoded.hyperbolic_embeddings,
        tree_relation_logits=relation_logits,
        tree_relation_targets=targets,
        tree_relation_mask=mask,
        lca_depth=train["lca_depth"],
        exact_tree_path_distance=train["exact_tree_path_distance"],
        parent_negative_mask=negative,
        parent_ids=train["parent_ids"],
        level_ids=train["level_ids"],
        node_mask=structural,
        b_side=train["b_side"],
        node_kind_ids=train["node_kind_ids"],
        event_ids=train["event_ids"],
        channel_embeddings=branches,
        channel_mask=branch_mask,
        full_truth_channel_ids=torch.stack(
            [train["b1_full_truth_channel_ids"], train["b2_full_truth_channel_ids"]], -1
        ),
        reconstructable_channel_ids=torch.stack(
            [
                train["b1_reconstructable_channel_ids"],
                train["b2_reconstructable_channel_ids"],
            ],
            -1,
        ),
        channel_branch_count_arrays=structural_features,
        channel_memory_embeddings=memory,
        channel_memory_full_truth_ids=full_ids,
        channel_memory_reconstructable_ids=reco_ids,
        weights=_pretraining_weights(config, phase=phase),
        curvature=1.0,
        full_event_max_level=train.get("full_event_max_level"),
        tangent_variance_target=0.05,
        radius_target_mode="generation_height_radius",
        depth_from_retained_root=train["depth_from_retained_root"],
        distance_to_nearest_retained_root=train["distance_to_nearest_retained_root"],
    )
    nodes = train["node_mask"] & (train["level_ids"] > 0)
    zero = encoded.node_embeddings.sum() * 0
    pid = _leaf_pid_loss(pid_logits, train)
    corruption = (
        F.cross_entropy(
            model.corruption_type_head(encoded.node_embeddings)[nodes],
            curriculum.corruption_code[nodes],
        )
        if nodes.any()
        else zero
    )
    correct = (
        F.binary_cross_entropy_with_logits(
            model.candidate_correctness_head(encoded.node_embeddings).squeeze(-1)[
                nodes
            ],
            (~curriculum.invalid_candidate_mask).float()[nodes],
        )
        if nodes.any()
        else zero
    )
    hard = _hard_negative_tree_loss(
        encoded.hyperbolic_embeddings, curriculum.hard_negative_pairs, curvature=1.0
    )
    loss = (
        output.total
        + pid * ("leaf_pid" in phase.objectives)
        + 0.1 * corruption * ("corruption" in phase.objectives)
        + 0.1 * correct * ("candidate_correctness" in phase.objectives)
        + 0.1 * hard * ("hard_negative" in phase.objectives)
    )
    enqueue = (
        branch_mask
        & train.get("b_root_discovery_valid", torch.ones_like(branch_mask[:, 0]))[
            :, None
        ]
    )
    if "channel" not in phase.objectives or phase.view.value == "corrupted_composites":
        enqueue = torch.zeros_like(enqueue)
    model.channel_memory.enqueue(
        branches,
        enqueue,
        torch.stack(
            [train["b1_full_truth_channel_ids"], train["b2_full_truth_channel_ids"]], -1
        ),
        torch.stack(
            [
                train["b1_reconstructable_channel_ids"],
                train["b2_reconstructable_channel_ids"],
            ],
            -1,
        ),
    )
    return loss, {
        **output.components,
        "pid": pid,
        "corruption": corruption,
        "correctness": correct,
        "hard_negative": hard,
    }
