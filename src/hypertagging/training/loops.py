"""Small training-loop building blocks with CPU dry-run support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hypertagging.losses.embedding_losses import grafei_radius_loss
from hypertagging.losses.gpt_losses import distance as gpt_distance
from hypertagging.losses.gpt_losses import radius_loss as gpt_radius_loss
from hypertagging.losses.link_losses import link_metrics
from hypertagging.losses.reconstruction_losses import momentum_metrics, pdg_metrics
from hypertagging.models import GPTReconstructor, HyperEmbedder, MultiGPT, Reconstructor, linearLinker
from hypertagging.training.dry_run import (
    embedding_batch,
    gpt_batch,
    link_batch,
    reconstruction_batch,
)


TrainingStage = Literal["embedding", "link", "reconstruction", "gpt"]
GptDryRunVariant = Literal["single", "multi"]


@dataclass(frozen=True)
class DryRunSummary:
    """Summary of a one-batch training dry run."""

    stage: TrainingStage
    device: str
    model_class: str
    optimizer_class: str
    loss: float
    output_shapes: tuple[tuple[int, ...], ...]
    parameter_count: int
    backward_ran: bool


def build_optimizer(
    model: torch.nn.Module,
    *,
    stage: TrainingStage,
    lr: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build the optimizer family used by the historical stage."""

    params = filter(lambda p: p.requires_grad, model.parameters())
    if stage == "embedding":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def run_one_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    backward: bool = True,
) -> torch.Tensor:
    """Run the one-batch train step used by dry-run tests."""

    if backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.detach()


def run_embedding_dry_run(
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run the embedding training stage."""

    device = torch.device(device)
    model = HyperEmbedder(
        n_features=4,
        tr_width=8,
        tr_n_head=1,
        tr_n=1,
        tr_hidden_size=16,
        pdg_emb=2,
        dim_hyper=3,
        num_pdg=8,
        device=device,
    )
    batch = embedding_batch(device)
    output = model(batch)
    loss = grafei_radius_loss(output, batch)
    optimizer = build_optimizer(model, stage="embedding")
    detached = run_one_batch(model, batch, loss, optimizer, backward=backward)
    return _summary("embedding", device, model, optimizer, detached, (tuple(output.shape),), backward)


def run_link_dry_run(
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run the link-prediction training stage."""

    device = torch.device(device)
    model = linearLinker(
        n_features=4,
        link_width=8,
        link_n_head=1,
        link_n_layers=1,
        link_fc=16,
        pdg_emb=2,
        num_pdg=8,
        device=device,
    )
    batch = link_batch(device)
    output = model(batch)
    loss, _acc = link_metrics(output, batch["links"], batch["padding_mask"])
    optimizer = build_optimizer(model, stage="link")
    detached = run_one_batch(model, batch, loss, optimizer, backward=backward)
    return _summary("link", device, model, optimizer, detached, (tuple(output.shape),), backward)


def run_reconstruction_dry_run(
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run the reconstruction training stage."""

    device = torch.device(device)
    model = Reconstructor(
        n_features=4,
        gen_tr_width=8,
        gen_encoder_n_head=1,
        gen_encoder_n_layers=1,
        gen_encoder_fc=16,
        gen_decoder_n_head=1,
        gen_decoder_n_layers=1,
        gen_decoder_fc=16,
        pdg_emb=2,
        dim_hyper=3,
        num_pdg=8,
        device=device,
    )
    batch = reconstruction_batch(device)
    pdg_out, feat_out = model(batch)
    pdg_loss, _pdg_acc = pdg_metrics(pdg_out, batch["pdg_y"], batch["padding_mask"])
    feat_loss, _feat_mae = momentum_metrics(feat_out, batch["feature_y"], batch["padding_mask"], spatial_weight=1)
    loss = pdg_loss + feat_loss
    optimizer = build_optimizer(model, stage="reconstruction")
    detached = run_one_batch(model, batch, loss, optimizer, backward=backward)
    return _summary(
        "reconstruction",
        device,
        model,
        optimizer,
        detached,
        (tuple(pdg_out.shape), tuple(feat_out.shape)),
        backward,
    )


def run_gpt_dry_run(
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run the GPT-like reconstruction stage."""

    return run_gpt_variant_dry_run(variant="single", device=device, backward=backward)


def run_multi_gpt_dry_run(
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run the combined GPT-like reconstruction/link stage."""

    return run_gpt_variant_dry_run(variant="multi", device=device, backward=backward)


def run_gpt_variant_dry_run(
    *,
    variant: GptDryRunVariant = "single",
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Build and dry-run a GPT-like stage variant."""

    device = torch.device(device)
    model = _build_gpt_model(variant, device)
    batch = gpt_batch(device)
    outputs = model(batch)
    out_rec, out_link = outputs if variant == "multi" else (outputs, None)
    level_mask = batch["lvl_code"].bool()
    link_mask = batch["links"] >= 0
    rec_loss = gpt_distance(out_rec, batch["target"], level_mask)
    r_loss = gpt_radius_loss(out_rec, batch["mass"], link_mask)
    loss = rec_loss + r_loss
    output_shapes = (tuple(out_rec.shape),)
    if out_link is not None:
        link_loss, _acc = link_metrics(out_link, batch["links"], link_mask)
        loss = loss + link_loss
        output_shapes = (tuple(out_rec.shape), tuple(out_link.shape))
    optimizer = build_optimizer(model, stage="gpt")
    detached = run_one_batch(model, batch, loss, optimizer, backward=backward)
    return _summary("gpt", device, model, optimizer, detached, output_shapes, backward)


def _build_gpt_model(variant: GptDryRunVariant, device: torch.device) -> torch.nn.Module:
    if variant == "single":
        return GPTReconstructor(
            tr_width=8,
            tr_n_head=1,
            tr_n=1,
            tr_hidden_size=16,
            dim_hyper=4,
            device=device,
        )
    if variant == "multi":
        return MultiGPT(
            rec_width=8,
            rec_n_head=1,
            rec_n=1,
            rec_hidden_size=16,
            link_width=8,
            link_n_head=1,
            link_n=1,
            link_hidden_size=16,
            dim_hyper=4,
            device=device,
        )
    raise ValueError(f"Unknown GPT dry-run variant: {variant}")


def run_stage_dry_run(
    stage: TrainingStage,
    *,
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> DryRunSummary:
    """Dispatch a one-batch dry run by stage name."""

    if stage == "embedding":
        return run_embedding_dry_run(device=device, backward=backward)
    if stage == "link":
        return run_link_dry_run(device=device, backward=backward)
    if stage == "reconstruction":
        return run_reconstruction_dry_run(device=device, backward=backward)
    if stage == "gpt":
        return run_gpt_variant_dry_run(variant="single", device=device, backward=backward)
    raise ValueError(f"Unknown training stage: {stage}")


def _summary(
    stage: TrainingStage,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    output_shapes: tuple[tuple[int, ...], ...],
    backward: bool,
) -> DryRunSummary:
    return DryRunSummary(
        stage=stage,
        device=device.type,
        model_class=model.__class__.__name__,
        optimizer_class=optimizer.__class__.__name__,
        loss=float(loss.cpu()),
        output_shapes=output_shapes,
        parameter_count=sum(p.numel() for p in model.parameters()),
        backward_ran=backward,
    )
