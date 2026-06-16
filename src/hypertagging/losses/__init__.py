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
from hypertagging.losses.link_losses import link_cross_entropy, link_metrics, transfer_link_metrics
from hypertagging.losses.reconstruction_losses import (
    embedding_cosine_distance,
    embedding_mse_distance,
    get_class_weight,
    momentum_metrics,
    pdg_metrics,
    plain_momentum_metrics,
    recover_pdg,
)

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
    "grafei_inter_loss",
    "grafei_intra_loss",
    "grafei_radius_loss",
    "link_metrics",
    "link_cross_entropy",
    "momentum_metrics",
    "pdg_metrics",
    "plain_momentum_metrics",
    "recover_pdg",
    "toy_mc_inter_loss",
    "toy_mc_radius_loss",
    "transfer_link_metrics",
    "vicreg_loss",
]
