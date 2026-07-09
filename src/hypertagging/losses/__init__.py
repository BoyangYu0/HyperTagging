"""Loss functions migrated from historical HyperTagging repositories."""

from hypertagging.losses.embedding_losses import (
    CompositeLoss,
    build_angle_matrix,
    build_distance_matrix,
    colab_intra_loss,
    colab_radius_loss,
    connection_loss_from_embeddings,
    connection_loss_from_predictions,
    grafei_inter_loss,
    grafei_intra_loss,
    grafei_radius_loss,
    toy_mc_inter_loss,
    toy_mc_radius_loss,
    vicreg_loss,
)
from hypertagging.losses.gpt_losses import distance as gpt_distance
from hypertagging.losses.gpt_losses import radius_loss as gpt_radius_loss
from hypertagging.losses.hyperbolic_pretraining import hyperbolic_pretraining_loss, radius_depth_loss
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.losses.link_losses import link_cross_entropy, link_metrics, transfer_link_metrics
from hypertagging.losses.physics import invariant_mass, p4_sum_consistency_loss, soft_daughter_sum_p4
from hypertagging.losses.reconstruction_losses import (
    embedding_cosine_distance,
    embedding_mse_distance,
    get_class_weight,
    momentum_metrics,
    pdg_metrics,
    plain_momentum_metrics,
    recover_pdg,
)
from hypertagging.losses.set_matching import hungarian_or_greedy, matching_cost

__all__ = [
    "CompositeLoss",
    "build_angle_matrix",
    "build_distance_matrix",
    "colab_intra_loss",
    "colab_radius_loss",
    "connection_loss_from_embeddings",
    "connection_loss_from_predictions",
    "embedding_cosine_distance",
    "embedding_mse_distance",
    "get_class_weight",
    "gpt_distance",
    "gpt_radius_loss",
    "hungarian_or_greedy",
    "hyperbolic_pretraining_loss",
    "invariant_mass",
    "level_reconstruction_loss",
    "grafei_inter_loss",
    "grafei_intra_loss",
    "grafei_radius_loss",
    "link_metrics",
    "link_cross_entropy",
    "matching_cost",
    "momentum_metrics",
    "p4_sum_consistency_loss",
    "pdg_metrics",
    "plain_momentum_metrics",
    "radius_depth_loss",
    "recover_pdg",
    "soft_daughter_sum_p4",
    "toy_mc_inter_loss",
    "toy_mc_radius_loss",
    "transfer_link_metrics",
    "vicreg_loss",
]
