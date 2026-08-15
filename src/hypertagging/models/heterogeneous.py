"""Type-specific detector/composite frontends with one shared latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.hyperbolic import (
    BoundedTangentScale,
    bound_tangent_norm,
    expmap0,
    initialize_hyper_projection,
)
from hypertagging.models.relation_attention import RelationAwareSetTransformer
from hypertagging.models.relations import HyperbolicRelationBias, PhysicalRelationBias
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID, NODE_KINDS
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES as CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES as COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES as COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES as TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, validate_pid_tokens
from hypertagging.preprocessing.schema_v4 import (
    CATEGORICAL_COMMON_FEATURE_NAMES,
    KLM_FEATURE_NAMES,
    KLM_MODEL_INPUT_SCALES,
    MODEL_COMPOSITE_FEATURE_NAMES,
    adapt_model_composite_features,
)


def masked_mean_pool(
    embeddings: torch.Tensor,
    daughter_adjacency: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant daughter pooling for every possible mother."""

    weights = daughter_adjacency.to(embeddings.dtype)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("bmn,bnd->bmd", weights, embeddings) / denominator


def dispatch_node_kind_adapters(
    kinds: torch.Tensor,
    *,
    track: torch.Tensor,
    ecl_cluster: torch.Tensor,
    klm_cluster: torch.Tensor,
    composite: torch.Tensor,
    other: torch.Tensor,
    node_kind_to_id: dict[str, int] | None = None,
) -> torch.Tensor:
    """Dispatch by vocabulary name rather than incidental declaration order."""

    mapping = NODE_KIND_TO_ID if node_kind_to_id is None else node_kind_to_id
    output = other
    for name, values in (
        ("track", track),
        ("ecl_cluster", ecl_cluster),
        ("klm_cluster", klm_cluster),
        ("composite", composite),
    ):
        output = torch.where(
            (kinds == mapping[name]).unsqueeze(-1), values, output
        )
    return output


class _MaskedBlockEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.projection = nn.Sequential(
            nn.Linear(2 * n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if values.shape != availability.shape:
            raise ValueError("feature values and availability masks must have identical shape")
        if values.shape[-1] > self.n_features:
            raise ValueError(
                f"feature block has width {values.shape[-1]}, expected at most {self.n_features}"
            )
        if values.shape[-1] < self.n_features:
            padding = self.n_features - values.shape[-1]
            values = torch.nn.functional.pad(values, (0, padding))
            availability = torch.nn.functional.pad(availability, (0, padding), value=False)
        clean = torch.nan_to_num(values)
        masked = torch.where(availability, clean, torch.zeros_like(clean))
        return self.projection(torch.cat([masked, availability.to(clean.dtype)], dim=-1))


class CommonNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(COMMON_FEATURE_NAMES), d_model)


class TrackNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(TRACK_FEATURE_NAMES), d_model)


class ClusterNodeEncoder(_MaskedBlockEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__(len(CLUSTER_FEATURE_NAMES), d_model)


class KlmNodeEncoder(_MaskedBlockEncoder):
    """Masked, fixed-scale adapter for schema-v4 reconstructed KLM inputs."""

    def __init__(self, d_model: int) -> None:
        super().__init__(len(KLM_FEATURE_NAMES), d_model)
        self.register_buffer(
            "input_scales", torch.tensor(KLM_MODEL_INPUT_SCALES, dtype=torch.float32)
        )

    def forward(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        scales = self.input_scales.to(dtype=values.dtype, device=values.device)
        return super().forward(values / scales, availability)


class CompositeNodeEncoder(nn.Module):
    """Encode reco-derived structure together with a pooled daughter summary."""

    def __init__(self, d_model: int, n_pid: int = len(PDG_TOKENS)) -> None:
        super().__init__()
        self.structural = _MaskedBlockEncoder(len(MODEL_COMPOSITE_FEATURE_NAMES), d_model)
        self.pid_histogram = nn.Sequential(
            nn.Linear(n_pid + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.combine = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        daughter_summary: torch.Tensor,
        daughter_pid_histogram: torch.Tensor,
        histogram_available: torch.Tensor,
    ) -> torch.Tensor:
        features, availability = adapt_model_composite_features(features, availability)
        structural = self.structural(features, availability)
        histogram = torch.nan_to_num(daughter_pid_histogram)
        histogram = torch.where(
            histogram_available.unsqueeze(-1),
            histogram,
            torch.zeros_like(histogram),
        )
        histogram = histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1.0)
        histogram_h = self.pid_histogram(
            torch.cat([histogram, histogram_available.unsqueeze(-1).to(histogram.dtype)], dim=-1)
        )
        return self.combine(torch.cat([structural, daughter_summary, histogram_h], dim=-1))


@dataclass(frozen=True)
class HeterogeneousEncoderOutput:
    adapter_embeddings: torch.Tensor
    node_embeddings: torch.Tensor
    hyperbolic_embeddings: torch.Tensor
    tree_projection: torch.Tensor
    reconstruction_projection: torch.Tensor
    channel_projection: torch.Tensor
    daughter_summary: torch.Tensor
    physical_relation_bias: torch.Tensor
    physical_attention_weights: torch.Tensor | None
    hyperbolic_relation_bias: torch.Tensor
    hyperbolic_attention_weights: torch.Tensor | None
    final_contextual_embeddings: torch.Tensor

    @property
    def attention_weights(self) -> torch.Tensor | None:
        """Compatibility view of the final active stage, never a bias-sum softmax."""

        weights = (
            self.hyperbolic_attention_weights
            if self.hyperbolic_attention_weights is not None
            else self.physical_attention_weights
        )
        return weights


class HeterogeneousNodeEncoder(nn.Module):
    """Different frontends, shared normalization/context space and Poincare ball."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        hyper_dim: int = 16,
        n_pid: int = len(PDG_TOKENS),
        max_level: int = 32,
        curvature: float = 1.0,
        n_heads: int = 4,
        n_context_layers: int = 2,
        use_contextual_encoder: bool = True,
        use_physical_context: bool = True,
        use_hyperbolic_refinement: bool = False,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        hyper_projection_init_scale: float = 0.05,
        tangent_scale_mode: str = "fixed",
        max_tangent_norm: float | None = None,
        hyperbolic_level_encoding: str = "learned_euclidean",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hyper_dim = hyper_dim
        self.n_heads = n_heads
        self.n_context_layers = n_context_layers
        self.ffn_dim = ffn_dim or 2 * d_model
        self.dropout = float(dropout)
        self.curvature = curvature
        if hyperbolic_level_encoding not in {
            "learned_euclidean", "bounded_tangent_level_embedding", "none"
        }:
            raise ValueError("unknown hyperbolic_level_encoding")
        self.hyperbolic_level_encoding = hyperbolic_level_encoding
        self.use_contextual_encoder = use_contextual_encoder
        self.use_physical_context = use_physical_context
        self.use_hyperbolic_refinement = use_hyperbolic_refinement
        if n_pid != len(PDG_TOKENS):
            n_pid = len(PDG_TOKENS)
        self.common_encoder = CommonNodeEncoder(d_model)
        self.track_encoder = TrackNodeEncoder(d_model)
        self.cluster_encoder = ClusterNodeEncoder(d_model)
        self.klm_encoder = KlmNodeEncoder(d_model)
        self.composite_encoder = CompositeNodeEncoder(d_model)
        self.other_encoder = nn.Parameter(torch.zeros(d_model))
        self.pid_embedding = nn.Embedding(n_pid, d_model)
        self.node_kind_embedding = nn.Embedding(len(NODE_KINDS), d_model)
        self.level_embedding = nn.Embedding(max_level + 2, d_model)
        if hyperbolic_level_encoding == "bounded_tangent_level_embedding":
            self.tangent_level_directions = nn.Parameter(
                torch.randn(max_level + 2, d_model) * 0.02
            )
            self.tangent_level_gaps = nn.Parameter(torch.zeros(max_level + 2))
        else:
            self.register_parameter("tangent_level_directions", None)
            self.register_parameter("tangent_level_gaps", None)
        self.active_embedding = nn.Embedding(2, d_model)
        self.copied_embedding = nn.Embedding(2, d_model)
        availability_width = (
            len(COMMON_FEATURE_NAMES)
            + len(TRACK_FEATURE_NAMES)
            + len(CLUSTER_FEATURE_NAMES)
            + len(KLM_FEATURE_NAMES)
            + len(MODEL_COMPOSITE_FEATURE_NAMES)
            + 1
        )
        self.availability_encoder = nn.Linear(availability_width, d_model)
        self.shared_norm = nn.LayerNorm(d_model)
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.physical_relation_bias = PhysicalRelationBias(
            d_model,
            enabled=use_physical_context,
        )
        self.physical_contextualizer = RelationAwareSetTransformer(
            d_model,
            n_heads=n_heads,
            n_layers=n_context_layers,
            feedforward_dim=ffn_dim,
            dropout=dropout,
        )
        self.hyperbolic_relation_bias = HyperbolicRelationBias(
            d_model,
            enabled=use_hyperbolic_refinement,
            curvature=curvature,
        )
        self.hyperbolic_contextualizer = RelationAwareSetTransformer(
            d_model,
            n_heads=n_heads,
            n_layers=1,
            feedforward_dim=ffn_dim,
            dropout=dropout,
        )
        self.tree_head = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, d_model)
        self.channel_head = nn.Linear(d_model, d_model)
        self.hyper_projection = nn.Linear(d_model, hyper_dim)
        initialize_hyper_projection(
            self.hyper_projection, output_std=hyper_projection_init_scale
        )
        self.tangent_scale = BoundedTangentScale(mode=tangent_scale_mode)
        if max_tangent_norm is not None and max_tangent_norm <= 0:
            raise ValueError("max_tangent_norm must be positive when supplied")
        self.max_tangent_norm = max_tangent_norm

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> HeterogeneousEncoderOutput:
        common_availability = batch["common_availability"].clone()
        for name in CATEGORICAL_COMMON_FEATURE_NAMES:
            common_availability[..., COMMON_FEATURE_NAMES.index(name)] = False
        common = self.common_encoder(batch["common_features"], common_availability)
        track = self.track_encoder(batch["track_features"], batch["track_availability"])
        cluster = self.cluster_encoder(batch["cluster_features"], batch["cluster_availability"])
        klm = self.klm_encoder(batch["klm_features"], batch["klm_availability"])
        kinds = batch["node_kind_ids"]
        active_kinds = kinds[batch["node_mask"]]
        if active_kinds.numel() and (
            int(active_kinds.min()) < 0 or int(active_kinds.max()) >= len(NODE_KINDS)
        ):
            raise ValueError("node_kind_ids contain an invalid explicit node kind")
        kinds = kinds.clamp_min(0)
        pid = batch["pid_labels"]
        validate_pid_tokens(pid[batch["node_mask"]], name="encoder PID labels")
        if self.pid_embedding.num_embeddings != len(PDG_TOKENS):
            raise ValueError(
                "PID embedding size must equal the reduced PID vocabulary; "
                f"got {self.pid_embedding.num_embeddings}, expected {len(PDG_TOKENS)}"
            )
        levels = batch["level_ids"].clamp(0, self.level_embedding.num_embeddings - 1)
        if self.hyperbolic_level_encoding == "learned_euclidean":
            level_features = self.level_embedding(levels)
        elif self.hyperbolic_level_encoding == "bounded_tangent_level_embedding":
            assert self.tangent_level_directions is not None
            assert self.tangent_level_gaps is not None
            directions = torch.nn.functional.normalize(
                self.tangent_level_directions, dim=-1, eps=1e-8
            )
            gaps = torch.nn.functional.softplus(self.tangent_level_gaps)
            radii = torch.flip(
                torch.cumsum(torch.flip(gaps, dims=(0,)), dim=0), dims=(0,)
            )
            radii = 0.5 * radii / radii[0].clamp_min(1e-8)
            level_features = (directions * radii[:, None])[levels]
        else:
            level_features = self.level_embedding(levels) * 0.0
        histogram_available = batch.get(
            "daughter_input_pid_histogram_available",
            batch["daughter_pid_histogram_available"],
        )
        model_composite_features, model_composite_availability = adapt_model_composite_features(
            batch["composite_features"], batch["composite_availability"]
        )
        availability = torch.cat(
            [
                common_availability,
                batch["track_availability"],
                batch["cluster_availability"],
                batch["klm_availability"],
                model_composite_availability,
                histogram_available.unsqueeze(-1),
            ],
            dim=-1,
        ).to(common.dtype)

        other = self.other_encoder.view(1, 1, -1).expand_as(common).clone()
        # Composite inputs need the pooled pre-context daughter summary, so the
        # first dispatch intentionally leaves composites on the neutral
        # ``other`` adapter.  The completed composite adapter is dispatched in
        # the second pass below.
        specific = dispatch_node_kind_adapters(
            kinds,
            track=track,
            ecl_cluster=cluster,
            klm_cluster=klm,
            composite=other,
            other=other,
        )
        if "current_pid_probabilities" in batch:
            probabilities = batch["current_pid_probabilities"]
            if probabilities.shape != (*pid.shape, len(PDG_TOKENS)):
                raise ValueError("current_pid_probabilities has an invalid shape")
            pid_h = probabilities.to(self.pid_embedding.weight.dtype) @ self.pid_embedding.weight
        else:
            pid_h = self.pid_embedding(pid)
        pre_composite = self.shared_norm(
            common
            + specific
            + pid_h
            + self.node_kind_embedding(kinds)
            + level_features
            + self.active_embedding(batch["active"].long())
            + self.copied_embedding(batch["copied"].long())
            + self.availability_encoder(availability)
        )
        daughter_summary = masked_mean_pool(pre_composite, batch["daughter_adjacency"])
        composite = self.composite_encoder(
            model_composite_features,
            model_composite_availability,
            daughter_summary,
            batch.get(
                "daughter_input_pid_histogram",
                batch["daughter_pid_histogram"],
            ),
            histogram_available,
        )
        specific = dispatch_node_kind_adapters(
            kinds,
            track=track,
            ecl_cluster=cluster,
            klm_cluster=klm,
            composite=composite,
            other=other,
        )
        specific = dispatch_node_kind_adapters(
            kinds,
            track=track,
            ecl_cluster=cluster,
            klm_cluster=klm,
            composite=composite,
            other=other,
        )
        h0 = self.shared_norm(
            common
            + specific
            + pid_h
            + self.node_kind_embedding(kinds)
            + level_features
            + self.active_embedding(batch["active"].long())
            + self.copied_embedding(batch["copied"].long())
            + self.availability_encoder(availability)
        )
        adapter_h = self.shared_norm(h0 + self.shared_mlp(h0))
        adapter_h = adapter_h * batch["node_mask"].unsqueeze(-1)
        if attention_mask is None:
            attention_mask = batch["node_mask"][:, :, None] & batch["node_mask"][:, None, :]
        physical_bias = self.physical_relation_bias(
            p4=batch["p4"],
            charge=batch["charge"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
            node_kind_ids=batch.get("node_kind_ids"),
            copied=batch.get("copied"),
            source_node_ids=batch.get("source_node_ids"),
            recursive_leaf_source_mask=batch.get("recursive_leaf_source_mask"),
            # Truth topology remains available to losses under its original
            # names.  Context consumes only this explicit, stage-built view
            # of links among nodes that already exist at inference time.
            parent_ids=None,
            ancestor_descendant_relation=batch.get(
                "current_reconstructed_ancestor_descendant_relation"
            ),
            reco_ids=batch.get("reco_ids"),
        )
        if self.use_contextual_encoder:
            h, physical_attention_weights = self.physical_contextualizer(
                adapter_h,
                relation_bias=physical_bias,
                attention_mask=attention_mask,
                node_mask=batch["node_mask"],
                return_attention=return_attention,
            )
        else:
            h = adapter_h
            physical_attention_weights = None
        preliminary_tree = self.tree_head(h)
        preliminary_z = expmap0(
            bound_tangent_norm(
                self.tangent_scale(self.hyper_projection(preliminary_tree)),
                maximum=self.max_tangent_norm,
            ),
            curvature=self.curvature,
        )
        hyper_bias = self.hyperbolic_relation_bias(
            z_hyperbolic=preliminary_z,
            node_mask=batch["node_mask"],
        )
        if self.use_hyperbolic_refinement:
            h, hyperbolic_attention_weights = self.hyperbolic_contextualizer(
                h,
                relation_bias=hyper_bias,
                attention_mask=attention_mask,
                node_mask=batch["node_mask"],
                return_attention=return_attention,
            )
        else:
            hyperbolic_attention_weights = None
        tree = self.tree_head(h)
        reconstruction = self.reconstruction_head(h)
        channel = self.channel_head(h)
        z = expmap0(
            bound_tangent_norm(
                self.tangent_scale(self.hyper_projection(tree)),
                maximum=self.max_tangent_norm,
            ),
            curvature=self.curvature,
        )
        z = z * batch["node_mask"].unsqueeze(-1)
        return HeterogeneousEncoderOutput(
            adapter_h,
            h,
            z,
            tree,
            reconstruction,
            channel,
            daughter_summary,
            physical_bias,
            physical_attention_weights,
            hyper_bias,
            hyperbolic_attention_weights,
            h,
        )


def composite_physical_features_from_daughters(
    *,
    daughter_mask: torch.Tensor,
    p4: torch.Tensor,
    charge: torch.Tensor,
    pid_labels: torch.Tensor,
    pid_probabilities: torch.Tensor | None = None,
    pointer_confidence: torch.Tensor | None = None,
    copied: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build persistent daughter-derived composite state without context."""

    weights = daughter_mask.to(p4.dtype)
    summed_p4 = torch.einsum("bn,bnf->bf", weights, p4)
    summed_charge = torch.einsum("bn,bn->b", weights, charge)
    count = weights.sum(dim=-1)
    confidence = (
        torch.where(daughter_mask, pointer_confidence, torch.ones_like(pointer_confidence))
        if pointer_confidence is not None
        else None
    )
    confidence_mean = (
        torch.einsum("bn,bn->b", weights, pointer_confidence) / count.clamp_min(1)
        if pointer_confidence is not None
        else torch.zeros_like(count)
    )
    confidence_min = (
        confidence.min(dim=-1).values if confidence is not None else torch.zeros_like(count)
    )
    copied_fraction = (
        torch.einsum("bn,bn->b", weights, copied.to(weights.dtype)) / count.clamp_min(1)
        if copied is not None
        else torch.zeros_like(count)
    )
    features = torch.stack(
        [
            summed_p4[:, 0],
            summed_p4[:, 1],
            summed_p4[:, 2],
            summed_p4[:, 3],
            summed_charge,
            count,
            confidence_mean,
            confidence_min,
            copied_fraction,
        ],
        dim=-1,
    )
    availability = torch.ones_like(features, dtype=torch.bool)
    if pointer_confidence is None:
        availability[:, 6:8] = False
    if pid_probabilities is not None:
        if pid_probabilities.shape != (*pid_labels.shape, len(PDG_TOKENS)):
            raise ValueError("daughter PID probabilities have an invalid shape")
        histogram = torch.einsum(
            "bn,bnc->bc", weights, pid_probabilities.to(weights.dtype)
        )
    else:
        histogram = torch.zeros(
            (*pid_labels.shape[:-1], len(PDG_TOKENS)),
            dtype=p4.dtype,
            device=p4.device,
        )
        validate_pid_tokens(pid_labels, name="composite daughter PID labels")
        histogram.scatter_add_(-1, pid_labels, weights)
    return {
        "p4": summed_p4,
        "charge": summed_charge,
        "features": features,
        "availability": availability,
        "daughter_pid_histogram": histogram,
        "daughter_pid_histogram_available": count > 0,
    }


def composite_token_from_daughters(
    *,
    daughter_mask: torch.Tensor,
    p4: torch.Tensor,
    charge: torch.Tensor,
    pid_labels: torch.Tensor,
    pid_probabilities: torch.Tensor | None = None,
    daughter_embeddings: torch.Tensor,
    pointer_confidence: torch.Tensor | None = None,
    copied: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build persistent state plus a transient pooled contextual summary."""

    result = composite_physical_features_from_daughters(
        daughter_mask=daughter_mask,
        p4=p4,
        charge=charge,
        pid_labels=pid_labels,
        pid_probabilities=pid_probabilities,
        pointer_confidence=pointer_confidence,
        copied=copied,
    )
    weights = daughter_mask.to(daughter_embeddings.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1)
    result["daughter_summary"] = (
        torch.einsum("bn,bnd->bd", weights, daughter_embeddings) / count
    )
    return result


__all__ = [
    "ClusterNodeEncoder",
    "CommonNodeEncoder",
    "CompositeNodeEncoder",
    "HeterogeneousEncoderOutput",
    "HeterogeneousNodeEncoder",
    "TrackNodeEncoder",
    "composite_physical_features_from_daughters",
    "composite_token_from_daughters",
    "dispatch_node_kind_adapters",
    "masked_mean_pool",
]
