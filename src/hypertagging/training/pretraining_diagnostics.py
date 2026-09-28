"""Support-aware development diagnostics; never inference inputs."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def channel_retrieval_metrics(
    embeddings, channel_ids, event_ids, *, view_ids=None, source_ids=None
):
    """Micro retrieval with all views/branches of the query event excluded.

    Identity is the fixed validation UID index, not a minibatch-local row.
    Queries without another-event same-channel peer are unavailable.
    """
    result = {}
    embeddings = F.normalize(embeddings.float(), dim=-1)
    allowed = event_ids[:, None] != event_ids[None, :]
    similarities = (embeddings @ embeddings.T).masked_fill(~allowed, -torch.inf)
    supported = ((channel_ids[:, None] == channel_ids[None, :]) & allowed).any(-1)
    nearest = similarities.argmax(-1)
    correct = channel_ids[nearest] == channel_ids
    groups = [("validation_channel_retrieval", torch.ones_like(supported))]
    if view_ids is not None:
        groups += [
            (f"validation_view_{int(v)}_channel_retrieval", view_ids == v)
            for v in view_ids.unique()
        ]
    for name, selection in groups:
        mask = supported & selection
        denominator = int(mask.sum())
        numerator = int((correct & mask).sum())
        result[name + "_queries"] = float(denominator)
        result[name + "_correct"] = float(numerator)
        if denominator:
            result[name + "_accuracy"] = numerator / denominator
    if source_ids is not None:
        cross = (
            allowed
            & (source_ids[:, None] != source_ids[None, :])
            & (source_ids[:, None] >= 0)
            & (source_ids[None, :] >= 0)
        )
        support = ((channel_ids[:, None] == channel_ids[None, :]) & cross).any(-1)
        nearest = similarities.masked_fill(~cross, -torch.inf).argmax(-1)
        numerator = int(((channel_ids[nearest] == channel_ids) & support).sum())
        denominator = int(support.sum())
        result["validation_cross_source_channel_retrieval_queries"] = float(denominator)
        result["validation_cross_source_channel_retrieval_correct"] = float(numerator)
        if denominator:
            result["validation_cross_source_channel_retrieval_accuracy"] = (
                numerator / denominator
            )
    result["validation_channel_retrieval_contract_version"] = 2.0
    return result


def radial_cap_diagnostics(tangent, mask, levels, maximum):
    norms = tangent.float().norm(dim=-1)
    selected = norms[mask]
    if not selected.numel():
        return {}
    out = {
        "precap_norm_mean": float(selected.mean()),
        "precap_norm_p50": float(selected.quantile(0.5)),
        "precap_norm_p95": float(selected.quantile(0.95)),
    }
    bounded = norms
    if maximum is not None:
        ratio = selected / maximum
        derivative = 1 - ratio.tanh().square()
        out.update(
            radial_derivative_mean=float(derivative.mean()),
            radial_derivative_p05=float(derivative.quantile(0.05)),
            radial_saturated_fraction=float((derivative < 1e-4).float().mean()),
        )
        bounded = maximum * (norms / maximum).tanh()
    out["postcap_norm_p95"] = float(bounded[mask].quantile(0.95))
    for level in levels[mask].unique():
        values = 2 * bounded[mask & (levels == level)]
        out[f"level_{int(level)}_radius_variance"] = float(values.var(unbiased=False))
    return out
