import torch
import torch.nn.functional as F

from hypertagging.losses.embedding_losses import (
    EPSILON,
    CompositeLoss,
    build_angle_matrix,
    build_distance_matrix,
    colab_intra_loss,
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
from hypertagging.losses.link_losses import link_metrics, transfer_link_metrics
from hypertagging.losses.reconstruction_losses import (
    embedding_cosine_distance,
    embedding_mse_distance,
    get_class_weight,
    momentum_metrics,
    pdg_metrics,
    plain_momentum_metrics,
    recover_pdg,
)


def _vectors():
    return torch.tensor(
        [[0.10, 0.05, 0.02], [0.20, 0.03, 0.04], [0.05, 0.12, 0.01]],
        dtype=torch.float32,
    )


def test_embedding_angle_distance_and_radius_match_legacy_formulas():
    vectors = _vectors()
    dataset = {
        "mass": torch.tensor([1, 5, 10], dtype=torch.int64),
        "E_Rec": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    }

    angle_ref = torch.exp(
        F.cosine_similarity(
            vectors.unsqueeze(1).repeat(1, len(vectors), 1),
            vectors.unsqueeze(0).repeat(len(vectors), 1, 1),
            dim=-1,
            eps=EPSILON,
        )
    )
    scale = 1 / (1 - torch.norm(vectors, dim=1) ** 2)
    dist = torch.norm(vectors[:, None] - vectors, dim=2) ** 2
    distance_ref = torch.exp(-torch.acosh(1 + 2 * scale * dist * scale[:, None] + EPSILON))
    r_euclidean = torch.norm(vectors, dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - EPSILON))
    grafei_radius_ref = F.mse_loss(r_poincare, 0.6 * torch.sqrt(1 - dataset["mass"].float() / 100) + 0.3)
    toy_radius_ref = F.mse_loss(r_poincare, 0.6 * torch.sqrt(1 - dataset["E_Rec"].float()) + 0.3)

    torch.testing.assert_close(build_angle_matrix(vectors), angle_ref)
    torch.testing.assert_close(build_distance_matrix(vectors), distance_ref)
    torch.testing.assert_close(grafei_radius_loss(vectors, dataset), grafei_radius_ref)
    torch.testing.assert_close(toy_mc_radius_loss(vectors, dataset), toy_radius_ref)


def test_embedding_intra_inter_and_vicreg_match_legacy_formulas():
    vectors = _vectors()
    dataset = {
        "evtNums": torch.tensor([1, 1, 2], dtype=torch.int64),
        "evtNum": torch.tensor([1, 1, 2], dtype=torch.int64),
        "channel": torch.tensor([0, 0, 1], dtype=torch.int64),
        "pattern": torch.tensor(
            [[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
            dtype=torch.float32,
        ),
    }

    a_nurf, a_buff = build_angle_matrix(vectors, amplifier=1)
    same_evt = dataset["evtNums"][:, None] == dataset["evtNums"]
    intra_ref = (-torch.log(a_buff / (torch.einsum("ij,ij->j", a_nurf, (~same_evt).float()) + EPSILON))[same_evt]).mean()
    pattern = dataset["pattern"]
    pattern_sim = F.cosine_similarity(
        pattern.unsqueeze(1).repeat(1, len(pattern), 1),
        pattern.unsqueeze(0).repeat(len(pattern), 1, 1),
        dim=-1,
        eps=EPSILON,
    )
    angle = build_angle_matrix(vectors)
    distance = build_distance_matrix(vectors)
    inter_ref = (pattern_sim * (-torch.log(angle / angle.sum(dim=-1)) - torch.log(distance / distance.sum(dim=-1)))).mean()

    torch.testing.assert_close(grafei_intra_loss(vectors, dataset), intra_ref)
    torch.testing.assert_close(grafei_inter_loss(vectors, dataset), inter_ref)
    assert torch.isfinite(vicreg_loss(vectors, dataset))


def test_toy_mc_inter_and_colab_intra_match_legacy_formulas():
    vectors = _vectors()
    toy_dataset = {"channel": torch.tensor([0, 0, 1], dtype=torch.int64)}
    mask = toy_dataset["channel"][:, None] == toy_dataset["channel"]
    angle = build_angle_matrix(vectors)
    distance = build_distance_matrix(vectors)
    ref = (
        -torch.log(angle / (torch.einsum("ij,ij->j", angle, (~mask).float()) + EPSILON))
        - torch.log(distance / (torch.einsum("ij,ij->j", distance, (~mask).float()) + EPSILON))
    )[mask].mean()

    batched_vectors = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]], [[2.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    colab_dataset = {"padding_mask": torch.tensor([[True, True, False], [True, False, False]])}
    m = colab_dataset["padding_mask"]
    masked = batched_vectors * m.unsqueeze(-1)
    weight = m.shape[-1] / m.sum(dim=-1)
    colab_ref = ((masked - masked.mean(dim=-2, keepdim=True) * weight[:, None, None]) ** 2 * m.unsqueeze(-1) * weight[:, None, None]).mean()

    torch.testing.assert_close(toy_mc_inter_loss(vectors, toy_dataset), ref)
    torch.testing.assert_close(colab_intra_loss(batched_vectors, colab_dataset), colab_ref)


def test_connection_losses_match_legacy_formulas():
    vectors = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    dataset = {
        "padding_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "links": torch.tensor([[0, 0, 1], [0, 1, -1]], dtype=torch.int64),
    }
    left = vectors.unsqueeze(2).repeat(1, 1, vectors.shape[1], 1)
    right = vectors.unsqueeze(1).repeat(1, vectors.shape[1], 1, 1)
    predictions = (F.cosine_similarity(left, right, dim=-1) + 1) / 2

    torch.testing.assert_close(
        connection_loss_from_embeddings(vectors, dataset)[0],
        connection_loss_from_predictions(predictions, dataset, already_scaled=True)[0],
    )


def test_link_and_reconstruction_metrics_match_legacy_formulas():
    pred = torch.tensor([[[0.1, 0.9], [2.0, 0.5]], [[0.8, 0.2], [0.4, 1.0]]])
    goal = torch.tensor([[1, 0], [0, 1]])
    mask = torch.tensor([[True, True], [True, False]])

    loss, acc = link_metrics(pred, goal, mask)
    torch.testing.assert_close(loss, F.cross_entropy(pred[mask], goal[mask]))
    torch.testing.assert_close(acc, (torch.argmax(pred, dim=-1)[mask] == goal[mask]).float().mean())

    transfer_loss, transfer_acc = transfer_link_metrics(pred, pred + 0.1, mask)
    torch.testing.assert_close(transfer_loss, F.mse_loss(pred[mask], (pred + 0.1)[mask]))
    torch.testing.assert_close(
        transfer_acc,
        (torch.argmax(pred, dim=-1)[mask] == torch.argmax(pred + 0.1, dim=-1)[mask]).float().mean(),
    )

    class_weight = get_class_weight(goal, mask, pad_lenth=3)
    assert class_weight.shape == (3,)
    torch.testing.assert_close(recover_pdg(pred), torch.argmax(pred, dim=-1))


def test_pdg_momentum_and_embedding_distances_match_legacy_formulas():
    pdg_pred = torch.tensor([[[0.1, 0.9], [2.0, 0.5]], [[0.8, 0.2], [0.4, 1.0]]])
    pdg_goal = torch.tensor([[1, 0], [0, 1]])
    mask = torch.tensor([[True, True], [True, False]])
    pdg_loss, pdg_acc = pdg_metrics(pdg_pred, pdg_goal, mask)
    torch.testing.assert_close(pdg_loss, F.cross_entropy(pdg_pred[mask], pdg_goal[mask]))
    torch.testing.assert_close(pdg_acc, (torch.argmax(pdg_pred, dim=-1)[mask] == pdg_goal[mask]).float().mean())

    pred = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 20
    goal = pred + 0.1
    mse, mae = momentum_metrics(pred, goal, mask, spatial_weight=3)
    reduced_mse, _ = momentum_metrics(pred, goal, mask, spatial_weight=1)
    plain_mse, plain_mae = plain_momentum_metrics(pred, goal, mask)
    torch.testing.assert_close(mse, F.mse_loss(pred[mask][..., :3], goal[mask][..., :3]) * 3 + F.mse_loss(pred[mask][..., 3], goal[mask][..., 3]))
    torch.testing.assert_close(reduced_mse, F.mse_loss(pred[mask][..., :3], goal[mask][..., :3]) + F.mse_loss(pred[mask][..., 3], goal[mask][..., 3]))
    torch.testing.assert_close(mae, F.l1_loss(pred[mask], goal[mask]))
    torch.testing.assert_close(plain_mse, F.mse_loss(pred[mask], goal[mask]))
    torch.testing.assert_close(plain_mae, F.l1_loss(pred[mask], goal[mask]))
    torch.testing.assert_close(embedding_mse_distance(pred[mask], goal[mask]), F.mse_loss(pred[mask], goal[mask]))
    torch.testing.assert_close(embedding_cosine_distance(pred[mask], goal[mask]), (1 - F.cosine_similarity(pred[mask], goal[mask])).mean())


def test_gpt_losses_match_legacy_formulas():
    pred = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 100
    goal = pred + 0.05
    goal[0, 3] = 0
    level_mask = torch.tensor([[True, True, False, True], [True, False, True, False]])
    robust = F.mse_loss(pred[level_mask], goal[level_mask])
    particle_mask = goal[level_mask].sum(dim=-1) != 0
    accurate = F.l1_loss(pred[level_mask][particle_mask], goal[level_mask][particle_mask])
    torch.testing.assert_close(gpt_distance(pred, goal, level_mask), 10 * robust + accurate)

    mass = torch.tensor([[1, 2, 0, 0], [5, 0, 10, 0]], dtype=torch.float32)
    link_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    r_euclidean = torch.norm(pred[link_mask], dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - EPSILON))
    ref = F.l1_loss(r_poincare, 0.9 * torch.sqrt(1 - mass[link_mask] / 100) + 0.1)
    torch.testing.assert_close(gpt_radius_loss(pred, mass, link_mask), ref)


def test_composite_loss_matches_historical_weighted_sum():
    losses = CompositeLoss()
    losses.add_loss("a", lambda output, batch: output["a"] + batch["b"], weight=2.0)
    losses.add_loss("c", lambda output, batch: output["c"], weight=0.5)

    total, parts = losses({"a": torch.tensor(1.0), "c": torch.tensor(4.0)}, {"b": torch.tensor(3.0)})

    torch.testing.assert_close(total, torch.tensor(10.0))
    assert set(parts) == {"a", "c"}
