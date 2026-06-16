import torch
import torch.nn.functional as F

from hypertagging.losses.link_losses import link_metrics, transfer_link_metrics
from hypertagging.losses.reconstruction_losses import (
    embedding_cosine_distance,
    embedding_mse_distance,
    momentum_metrics,
    pdg_metrics,
    plain_momentum_metrics,
    recover_pdg,
)
from hypertagging.reconstruction import (
    build_reconstructed_batch,
    build_reconstructed_link_batch,
    single_level_reconstruction_step,
    sort_energy,
)


class TinyGenerator(torch.nn.Module):
    def __init__(self, pdg_logits, feature):
        super().__init__()
        self.pdg_logits = torch.nn.Parameter(pdg_logits.clone())
        self.feature = torch.nn.Parameter(feature.clone())

    def forward(self, batch):
        return self.pdg_logits, self.feature


class TinyEmbedder(torch.nn.Module):
    def forward(self, batch):
        pdg = batch["pdg"].float()
        feat = batch["feature"]
        masked = feat * batch["padding_mask"][..., None]
        return torch.stack(
            [
                pdg.sum(dim=1) / 10,
                masked[..., 0].sum(dim=1),
                masked[..., -1].sum(dim=1),
            ],
            dim=-1,
        )


class TinyLinker(torch.nn.Module):
    def forward(self, batch):
        score = batch["feature_x"][..., :2] @ batch["feature_y"][..., :2].transpose(1, 2)
        pdg_term = (batch["pdg_x"].float().unsqueeze(-1) - batch["pdg_y"].float().unsqueeze(1)).abs()
        return score - 0.01 * pdg_term


def _batch():
    return {
        "pdg_x": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "pdg_y": torch.tensor([[2, 1, 0], [4, 3, 5]], dtype=torch.long),
        "feature_x": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 20,
        "feature_y": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4) / 20,
        "padding_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "links": torch.tensor([[0, 0, -1], [1, 1, 2]], dtype=torch.long),
        "emb": torch.tensor([[0.2, 1.0, 2.0], [1.2, 3.0, 4.0]], dtype=torch.float32),
        "emb_y": torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]], dtype=torch.float32),
    }


def _outputs():
    pdg_logits = torch.tensor(
        [
            [[0.1, 0.2, 1.0, 0.0, 0.0, 0.0], [0.1, 1.2, 0.2, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0, 0.0, 1.2, 0.0], [0.1, 0.0, 0.0, 1.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 1.3]],
        ],
        dtype=torch.float32,
    )
    feature = torch.arange(48, 72, dtype=torch.float32).reshape(2, 3, 4) / 20
    return pdg_logits, feature


def test_sort_energy_matches_legacy_formula():
    pdg_logits, feature = _outputs()
    mask = _batch()["padding_mask"]

    sorted_pdg, sorted_feature = sort_energy(pdg_logits, feature, mask, recover=False)

    masked_feat = torch.where(mask[..., None].repeat((1, 1, 4)), feature, torch.tensor(float("-inf")))
    indices = torch.argsort(masked_feat[:, :, -1], descending=True)
    expected_feat = torch.gather(masked_feat, 1, indices.unsqueeze(-1).expand(-1, -1, 4))
    expected_pdg = torch.gather(pdg_logits, 1, indices.unsqueeze(-1).expand(-1, -1, pdg_logits.size(-1)))
    torch.testing.assert_close(sorted_pdg, expected_pdg)
    torch.testing.assert_close(sorted_feature, torch.nan_to_num(expected_feat, neginf=0))


def test_reduced_single_level_step_matches_legacy_inline_formulas():
    batch = _batch()
    pdg_logits, feature = _outputs()
    model = TinyGenerator(pdg_logits, feature)
    embedder = TinyEmbedder()
    linker = TinyLinker()

    result = single_level_reconstruction_step(
        model,
        batch,
        variant="grafei_reduced",
        embedding_model=embedder,
        link_model=linker,
    )

    reconstructed = {
        "pdg": recover_pdg(pdg_logits) * batch["padding_mask"],
        "feature": feature * batch["padding_mask"][..., None],
        "padding_mask": batch["padding_mask"],
    }
    goal = {"pdg": batch["pdg_y"], "feature": batch["feature_y"], "padding_mask": batch["padding_mask"]}
    reconstructed_link = {
        "pdg_x": batch["pdg_x"],
        "pdg_y": reconstructed["pdg"],
        "feature_x": batch["feature_x"],
        "feature_y": reconstructed["feature"],
        "padding_mask": batch["padding_mask"],
    }
    pdg_loss, pdg_acc = pdg_metrics(pdg_logits, batch["pdg_y"], batch["padding_mask"])
    feat_loss, feat_err = momentum_metrics(feature, batch["feature_y"], batch["padding_mask"], spatial_weight=1)
    emb_loss = embedding_cosine_distance(embedder(reconstructed), embedder(goal))
    link_loss, link_acc = link_metrics(linker(reconstructed_link), batch["links"], batch["padding_mask"])

    torch.testing.assert_close(result.losses["pdg"], pdg_loss)
    torch.testing.assert_close(result.losses["feature"], feat_loss)
    torch.testing.assert_close(result.losses["embedding"], emb_loss)
    torch.testing.assert_close(result.losses["link"], link_loss)
    torch.testing.assert_close(result.total_loss, pdg_loss + feat_loss + emb_loss + link_loss)
    torch.testing.assert_close(result.metrics["pdg_acc"], pdg_acc)
    torch.testing.assert_close(result.metrics["feature_err"], feat_err)
    torch.testing.assert_close(result.metrics["link_acc"], link_acc)


def test_full_grafei_single_level_step_uses_recovered_pdg_mask_and_transfer_link():
    batch = _batch()
    pdg_logits, feature = _outputs()
    model = TinyGenerator(pdg_logits, feature)
    embedder = TinyEmbedder()
    linker = TinyLinker()

    result = single_level_reconstruction_step(
        model,
        batch,
        variant="grafei",
        embedding_model=embedder,
        link_model=linker,
        weights=(1.0, 1.0, 1.0, 1.0),
    )

    out_pdg = recover_pdg(pdg_logits)
    out_feat = feature * (out_pdg > 0)[..., None]
    reconstructed = {"pdg": out_pdg, "feature": out_feat, "padding_mask": batch["padding_mask"]}
    reconstructed_link = build_reconstructed_link_batch(batch, reconstructed)
    pdg_loss, _ = pdg_metrics(pdg_logits, batch["pdg_y"], batch["padding_mask"])
    feat_loss, _ = momentum_metrics(feature, batch["feature_y"], batch["padding_mask"], spatial_weight=3)
    emb_loss = embedding_cosine_distance(embedder(reconstructed), batch["emb"])
    link_loss, _ = transfer_link_metrics(linker(reconstructed_link), linker(batch), batch["padding_mask"])

    torch.testing.assert_close(result.recovered_pdg, out_pdg)
    torch.testing.assert_close(result.reconstructed["feature"], out_feat)
    torch.testing.assert_close(result.losses["feature"], feat_loss)
    torch.testing.assert_close(result.total_loss, pdg_loss + feat_loss + emb_loss + link_loss)


def test_toy_mc_single_level_step_uses_plain_momentum_and_embedding_mse():
    batch = _batch()
    pdg_logits, feature = _outputs()
    model = TinyGenerator(pdg_logits, feature)
    embedder = TinyEmbedder()

    result = single_level_reconstruction_step(
        model,
        batch,
        variant="toy_mc",
        embedding_model=embedder,
        link_model=None,
    )

    reconstructed = build_reconstructed_batch(pdg_logits, feature, batch["padding_mask"], variant="toy_mc")
    pdg_loss, _ = pdg_metrics(pdg_logits, batch["pdg_y"], batch["padding_mask"])
    feat_loss, _ = plain_momentum_metrics(feature, batch["feature_y"], batch["padding_mask"])
    emb_loss = embedding_mse_distance(embedder(reconstructed), batch["emb_y"])

    torch.testing.assert_close(result.losses["pdg"], pdg_loss)
    torch.testing.assert_close(result.losses["feature"], feat_loss)
    torch.testing.assert_close(result.losses["embedding"], emb_loss)
    torch.testing.assert_close(result.losses["link"], torch.tensor(0.0))
    torch.testing.assert_close(result.total_loss, pdg_loss + feat_loss + emb_loss)
