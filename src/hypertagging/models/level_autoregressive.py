"""Level-autoregressive set reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.hyperbolic import HyperbolicNodeEncoder
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.mother_pointer import MotherPointerDecoder, MotherPointerOutput
from hypertagging.models.relation_attention import RelationAwareSetTransformer
from hypertagging.models.relations import RelationBias
from hypertagging.models.stair_masks import context_mask_for_level, stair_attention_mask
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES as CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES as COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES as COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES as TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, validate_pid_tokens
from hypertagging.reconstruction.pid_state import rebuild_runtime_pid_state
from hypertagging.data.streaming import RuntimeFeatureNormalizer


@dataclass(frozen=True)
class LevelReconstructionOutput:
    target_level: int
    pointer: MotherPointerOutput
    node_embeddings: torch.Tensor
    hyperbolic_embeddings: torch.Tensor
    context_mask: torch.Tensor
    relation_bias: torch.Tensor
    attention_weights: torch.Tensor
    physical_relation_bias: torch.Tensor
    physical_attention_weights: torch.Tensor | None
    hyperbolic_relation_bias: torch.Tensor | None
    hyperbolic_attention_weights: torch.Tensor | None
    final_contextual_embeddings: torch.Tensor
    tree_projection: torch.Tensor
    reconstruction_projection: torch.Tensor
    channel_projection: torch.Tensor
    leaf_pid_logits: torch.Tensor | None = None
    current_pid_probabilities: torch.Tensor | None = None
    current_pid_tokens: torch.Tensor | None = None
    current_p4: torch.Tensor | None = None
    second_pass_common_features: torch.Tensor | None = None
    second_pass_common_availability: torch.Tensor | None = None
    relation_pid_kinematics_mode: str = "input"
    decision_pid_kinematics_mode: str = "input"


class LevelAutoregressiveReconstructor(nn.Module):
    """Tree-biased encoder plus pointer-based mother set decoder."""

    def __init__(
        self,
        *,
        n_features: int,
        n_types: int,
        hidden_dim: int = 32,
        hyper_dim: int = 16,
        n_queries: int = 8,
        n_heads: int = 4,
        n_context_layers: int = 2,
        encoder_mode: str = "heterogeneous",
        use_contextual_encoder: bool = True,
        use_relation_bias: bool = True,
        use_hyperbolic_relation_refinement: bool = False,
        canonical_pion_first_level: bool = False,
        curvature: float = 1.0,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        max_cardinality: int = 6,
        n_queries_by_level: tuple[tuple[int, int], ...] = (),
        max_cardinality_by_level: tuple[tuple[int, int], ...] = (),
        pid_kinematics_mode: str = "soft_expectation",
        pid_temperature: float = 1.0,
        hyper_projection_init_scale: float = 0.05,
        tangent_scale_mode: str = "fixed",
        hyperbolic_level_encoding: str = "learned_euclidean",
        type_conditioned_daughter_relation_bias: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_mode = encoder_mode
        self.use_contextual_encoder = use_contextual_encoder
        self.canonical_pion_first_level = bool(canonical_pion_first_level)
        self.pid_kinematics_mode = str(pid_kinematics_mode)
        self.pid_temperature = float(pid_temperature)
        self.type_conditioned_daughter_relation_bias = bool(
            type_conditioned_daughter_relation_bias
        )
        # The public argument is retained for source compatibility, but the
        # scientific contract has exactly one model vocabulary.
        n_types = len(PDG_TOKENS)
        if encoder_mode == "flat":
            self.encoder: nn.Module = HyperbolicNodeEncoder(
                n_features=n_features,
                n_pid=len(PDG_TOKENS),
                hidden_dim=hidden_dim,
                hyper_dim=hyper_dim,
                hyper_projection_init_scale=hyper_projection_init_scale,
                tangent_scale_mode=tangent_scale_mode,
            )
        elif encoder_mode == "heterogeneous":
            self.encoder = HeterogeneousNodeEncoder(
                d_model=hidden_dim,
                hyper_dim=hyper_dim,
                n_pid=len(PDG_TOKENS),
                n_heads=n_heads,
                n_context_layers=n_context_layers,
                use_contextual_encoder=use_contextual_encoder,
                use_physical_context=use_relation_bias,
                use_hyperbolic_refinement=use_hyperbolic_relation_refinement,
                curvature=curvature,
                ffn_dim=ffn_dim,
                dropout=dropout,
                hyper_projection_init_scale=hyper_projection_init_scale,
                tangent_scale_mode=tangent_scale_mode,
                hyperbolic_level_encoding=hyperbolic_level_encoding,
            )
        else:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}")
        self.flat_relation_bias = RelationBias(hidden_dim=hidden_dim, enabled=use_relation_bias)
        self.flat_contextualizer = RelationAwareSetTransformer(
            hidden_dim,
            n_heads=n_heads,
            n_layers=n_context_layers,
        )
        self.decoder = MotherPointerDecoder(
            hidden_dim=hidden_dim, n_types=n_types, n_queries=n_queries,
            max_cardinality=max_cardinality,
            type_conditioned_daughter_relation_bias=(
                type_conditioned_daughter_relation_bias
            ),
        )
        query_map = dict(n_queries_by_level)
        cardinality_map = dict(max_cardinality_by_level)
        self.level_decoders = nn.ModuleDict(
            {
                str(level): MotherPointerDecoder(
                    hidden_dim=hidden_dim,
                    n_types=n_types,
                    n_queries=query_map.get(level, n_queries),
                    max_cardinality=cardinality_map.get(level, max_cardinality),
                    type_conditioned_daughter_relation_bias=(
                        type_conditioned_daughter_relation_bias
                    ),
                )
                for level in sorted(set(query_map) | set(cardinality_map))
            }
        )
        self.leaf_pid_head = nn.Linear(hidden_dim, len(PDG_TOKENS))
        self.runtime_feature_normalizer = RuntimeFeatureNormalizer.identity(
            len(COMMON_FEATURE_NAMES), len(COMPOSITE_FEATURE_NAMES)
        )

    def set_runtime_feature_normalizer(
        self, normalizer: RuntimeFeatureNormalizer
    ) -> None:
        self.runtime_feature_normalizer = normalizer

    @property
    def relation_bias(self) -> nn.Module:
        """Compatibility handle for gradient/ablation inspection."""

        if self.encoder_mode == "heterogeneous":
            return self.encoder.physical_relation_bias  # type: ignore[attr-defined]
        return self.flat_relation_bias

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        target_level: int = 1,
        pid_kinematics_mode_override: str | None = None,
        pid_temperature_override: float | None = None,
    ) -> LevelReconstructionOutput:
        pid_mode = (
            self.pid_kinematics_mode
            if pid_kinematics_mode_override is None
            else str(pid_kinematics_mode_override)
        )
        pid_temperature = (
            self.pid_temperature
            if pid_temperature_override is None
            else float(pid_temperature_override)
        )
        if self.encoder_mode == "heterogeneous":
            batch = _upgrade_flat_batch(batch)
            _assert_truth_free_model_inputs(batch)
            first_pass_batch = _normalize_runtime_feature_blocks(
                batch, self.runtime_feature_normalizer
            )
            first_pass = self.encoder(
                first_pass_batch,
                attention_mask=stair_attention_mask(
                    first_pass_batch["level_ids"], first_pass_batch["node_mask"]
                ),
            )
            leaf_pid_logits = self.leaf_pid_head(first_pass.reconstruction_projection)
            runtime = rebuild_runtime_pid_state(
                batch,
                leaf_pid_logits,
                mode=pid_mode,
                temperature=pid_temperature,
            )
            reconstruction_batch = _runtime_reconstruction_batch(
                batch,
                runtime,
                normalizer=self.runtime_feature_normalizer,
                canonical_batch=first_pass_batch,
                use_canonical=(
                    self.canonical_pion_first_level and target_level == 1
                ),
            )
            encoded = self.encoder(
                reconstruction_batch,
                attention_mask=stair_attention_mask(
                    reconstruction_batch["level_ids"],
                    reconstruction_batch["node_mask"],
                ),
            )
            h = encoded.node_embeddings
            reconstruction_h = encoded.reconstruction_projection
            z = encoded.hyperbolic_embeddings
            tree_projection = encoded.tree_projection
            reconstruction_projection = encoded.reconstruction_projection
            channel_projection = encoded.channel_projection
            relation_bias = encoded.physical_relation_bias + (
                encoded.hyperbolic_relation_bias
                if self.encoder.use_hyperbolic_refinement  # type: ignore[attr-defined]
                else torch.zeros_like(encoded.physical_relation_bias)
            )
            attention_weights = encoded.attention_weights
            physical_relation_bias = encoded.physical_relation_bias
            physical_attention_weights = encoded.physical_attention_weights
            hyperbolic_relation_bias = encoded.hyperbolic_relation_bias
            hyperbolic_attention_weights = encoded.hyperbolic_attention_weights
            current_probabilities = runtime.probabilities
            current_tokens = runtime.current_tokens
            current_p4 = runtime.p4
        else:
            h, z = self.encoder(
                batch["node_features"],
                batch["pid_labels"],
                batch["level_ids"],
                batch["charge"],
            )
            tree_projection = h
            reconstruction_projection = h
            channel_projection = h
            relation_bias = self.flat_relation_bias(
                p4=batch["p4"],
                charge=batch["charge"],
                level_ids=batch["level_ids"],
                z_hyperbolic=z,
                node_mask=batch["node_mask"],
                node_kind_ids=batch.get("node_kind_ids"),
                copied=batch.get("copied"),
                source_node_ids=batch.get("source_node_ids"),
                recursive_leaf_source_mask=batch.get("recursive_leaf_source_mask"),
                parent_ids=batch.get("parent_ids"),
                ancestor_descendant_relation=batch.get("ancestor_descendant_relation"),
                reco_ids=batch.get("reco_ids"),
            )
            if self.use_contextual_encoder:
                h, attention_weights = self.flat_contextualizer(
                    reconstruction_projection,
                    relation_bias=relation_bias,
                    attention_mask=stair_attention_mask(batch["level_ids"], batch["node_mask"]),
                    node_mask=batch["node_mask"],
                )
            else:
                h = reconstruction_projection
                attention_weights = relation_bias.new_zeros(
                    (*relation_bias.shape[:1], 1, *relation_bias.shape[-2:])
                )
            physical_relation_bias = relation_bias
            physical_attention_weights = (
                attention_weights if self.use_contextual_encoder else None
            )
            hyperbolic_relation_bias = None
            hyperbolic_attention_weights = None
            reconstruction_h = reconstruction_projection
            leaf_pid_logits = self.leaf_pid_head(reconstruction_h)
            current_probabilities = None
            current_tokens = batch["pid_labels"]
            current_p4 = batch["p4"]
        context_mask = context_mask_for_level(batch["level_ids"], batch["node_mask"], target_level)
        decoder = (
            self.level_decoders[str(target_level)]
            if str(target_level) in self.level_decoders
            else self.decoder
        )
        pointer_validity = batch.get("pointer_validity_mask")
        if pointer_validity is not None and pointer_validity.ndim == 2:
            pointer_validity = pointer_validity[:, None, :].expand(
                -1, decoder.n_queries, -1
            )
        pointer = decoder(
            reconstruction_h,
            context_mask,
            target_level=target_level,
            allowed_type_mask=batch.get("allowed_type_mask"),
            type_logit_bias=batch.get("type_logit_bias"),
            pointer_validity_mask=pointer_validity,
            node_pid_probabilities=(
                current_probabilities
                if current_probabilities is not None
                else torch.nn.functional.one_hot(
                    current_tokens, num_classes=len(PDG_TOKENS)
                ).to(reconstruction_h.dtype)
            ),
            node_charge=batch["charge"],
            node_kind_ids=batch.get(
                "node_kind_ids", torch.zeros_like(batch["level_ids"])
            ),
            node_level_ids=batch["level_ids"],
            node_relation_summary=(
                relation_bias.masked_fill(
                    ~batch["node_mask"][:, None, :], 0.0
                ).sum(dim=-1)
                / batch["node_mask"].sum(dim=-1, keepdim=True).clamp_min(1)
            ),
        )
        return LevelReconstructionOutput(
            target_level=target_level,
            pointer=pointer,
            node_embeddings=h,
            hyperbolic_embeddings=z,
            context_mask=context_mask,
            relation_bias=relation_bias,
            attention_weights=attention_weights,
            physical_relation_bias=physical_relation_bias,
            physical_attention_weights=physical_attention_weights,
            hyperbolic_relation_bias=hyperbolic_relation_bias,
            hyperbolic_attention_weights=hyperbolic_attention_weights,
            final_contextual_embeddings=h,
            tree_projection=tree_projection,
            reconstruction_projection=reconstruction_projection,
            channel_projection=channel_projection,
            leaf_pid_logits=leaf_pid_logits,
            current_pid_probabilities=current_probabilities,
            current_pid_tokens=current_tokens,
            current_p4=current_p4,
            second_pass_common_features=(
                reconstruction_batch["common_features"]
                if self.encoder_mode == "heterogeneous"
                else None
            ),
            second_pass_common_availability=(
                reconstruction_batch["common_availability"]
                if self.encoder_mode == "heterogeneous"
                else None
            ),
            relation_pid_kinematics_mode=(
                pid_mode if self.encoder_mode == "heterogeneous" else "input"
            ),
            decision_pid_kinematics_mode=(
                pid_mode if self.encoder_mode == "heterogeneous" else "input"
            ),
        )


@torch.no_grad()
def compare_pid_kinematics_modes(
    model: LevelAutoregressiveReconstructor,
    batch: dict[str, torch.Tensor],
    *,
    target_level: int = 1,
    temperature: float = 0.5,
) -> dict[str, float]:
    """Measure the train/rollout PID-kinematics mismatch on one bounded batch."""

    soft = model(
        batch,
        target_level=target_level,
        pid_kinematics_mode_override="soft_expectation",
        pid_temperature_override=1.0,
    )
    hard = model(
        batch,
        target_level=target_level,
        pid_kinematics_mode_override="hard",
    )
    annealed = model(
        batch,
        target_level=target_level,
        pid_kinematics_mode_override="temperature_softmax",
        pid_temperature_override=float(temperature),
    )
    node_mask = batch["node_mask"]
    composite = node_mask & (batch["level_ids"] > 0)
    soft_p4 = soft.current_p4 if soft.current_p4 is not None else batch["p4"]
    hard_p4 = hard.current_p4 if hard.current_p4 is not None else batch["p4"]
    soft_mass = (
        soft_p4[..., 3].square() - soft_p4[..., :3].square().sum(dim=-1)
    ).clamp_min(0).sqrt()
    hard_mass = (
        hard_p4[..., 3].square() - hard_p4[..., :3].square().sum(dim=-1)
    ).clamp_min(0).sqrt()
    probabilities = soft.current_pid_probabilities
    entropy = 0.0
    if probabilities is not None:
        raw = node_mask & (batch["level_ids"] == 0) & (batch["charge"] != 0)
        if raw.any():
            values = probabilities[raw].clamp_min(1e-12)
            entropy = float((-(values * values.log()).sum(dim=-1)).mean())
    return {
        "soft_hard_energy_difference": float((soft_p4[..., 3] - hard_p4[..., 3])[node_mask].abs().mean()),
        "soft_hard_mother_mass_difference": float((soft_mass - hard_mass)[composite].abs().mean()) if composite.any() else 0.0,
        "pid_entropy": entropy,
        "soft_hard_relation_bias_change": float((soft.relation_bias - hard.relation_bias).abs().mean()),
        "soft_hard_pointer_logit_change": float((soft.pointer.pointer_logits - hard.pointer.pointer_logits).abs().mean()),
        "annealed_soft_pointer_logit_change": float((soft.pointer.pointer_logits - annealed.pointer.pointer_logits).abs().mean()),
    }


def _runtime_reconstruction_batch(
    batch: dict[str, torch.Tensor],
    runtime,
    *,
    normalizer: RuntimeFeatureNormalizer,
    canonical_batch: dict[str, torch.Tensor],
    use_canonical: bool,
) -> dict[str, torch.Tensor]:
    if use_canonical:
        return canonical_batch
    output = dict(batch)
    output["current_pid_probabilities"] = runtime.probabilities
    output["current_pid_tokens"] = runtime.current_tokens
    output["current_pid_available"] = runtime.available
    output["p4"] = runtime.p4
    output["daughter_input_pid_histogram"] = runtime.daughter_input_histograms
    output["daughter_input_pid_histogram_available"] = (
        runtime.daughter_histogram_available
    )
    output["daughter_pid_histogram"] = runtime.daughter_input_histograms
    output["daughter_pid_histogram_available"] = runtime.daughter_histogram_available
    common = output["common_features"].clone()
    common[..., :4] = runtime.p4
    mass2 = runtime.p4[..., 3].square() - runtime.p4[..., :3].square().sum(dim=-1)
    common[..., 4] = mass2.clamp_min(0).sqrt()
    output["common_features"] = common
    composite = output["composite_features"].clone()
    if composite.shape[-1] >= 4:
        composite[..., :4] = torch.einsum(
            "bmn,bnf->bmf",
            output["daughter_adjacency"].to(runtime.p4.dtype),
            runtime.p4,
        )
    output["composite_features"] = composite
    (
        output["common_features"],
        output["common_availability"],
        output["composite_features"],
        output["composite_availability"],
    ) = normalizer.normalize_runtime(
        output["common_features"],
        output["common_availability"],
        output["composite_features"],
        output["composite_availability"],
    )
    return output


def _normalize_runtime_feature_blocks(
    batch: dict[str, torch.Tensor],
    normalizer: RuntimeFeatureNormalizer,
) -> dict[str, torch.Tensor]:
    output = dict(batch)
    (
        output["common_features"],
        output["common_availability"],
        output["composite_features"],
        output["composite_availability"],
    ) = normalizer.normalize_runtime(
        batch["common_features"],
        batch["common_availability"],
        batch["composite_features"],
        batch["composite_availability"],
    )
    output["node_features"] = output["common_features"]
    return output


def _assert_truth_free_model_inputs(batch: dict[str, torch.Tensor]) -> None:
    """Fail fast if the explicit raw-track input contract is violated."""

    from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID

    raw = batch["node_mask"] & (
        batch["leaf_kinematics_mode_ids"]
        == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    )
    if raw.any() and (batch["pid_labels"][raw] != 0).any():
        raise ValueError(
            "raw_track_predicted_pid nodes must enter the model with unknown input token 0"
        )
    if "daughter_input_pid_histogram" not in batch:
        raise ValueError("explicit daughter_input_pid_histogram is required")
    required_provenance = {
        "model_input_source_ids",
        "daughter_input_pid_source_ids",
        "truth_supervision_source_ids",
        "daughter_truth_pid_source_ids",
    }
    missing = sorted(required_provenance - set(batch))
    if missing:
        raise ValueError(
            "explicit feature provenance is required: " + ", ".join(missing)
        )
    if (
        batch["daughter_input_pid_source_ids"]
        == batch["daughter_truth_pid_source_ids"]
    )[batch["node_mask"]].any():
        raise ValueError(
            "input and truth daughter PID histograms have the same declared provenance"
        )


def construct_mother_p4(pointer_logits: torch.Tensor, p4: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
    """Construct mother p4 from daughter pointers, never from MC mother p4."""

    weights = (pointer_logits > 0).float() if hard else torch.sigmoid(pointer_logits)
    return torch.einsum("bqn,bnf->bqf", weights, p4)


def _upgrade_flat_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upgrade legacy/tiny batches without claiming detector-specific values."""

    if "common_features" in batch:
        from hypertagging.data.heterogeneous import (
            MODEL_INPUT_SOURCE_TO_ID,
            TRUTH_SUPERVISION_SOURCE_TO_ID,
        )

        batch = dict(batch)
        input_sources = torch.full_like(
            batch["level_ids"],
            MODEL_INPUT_SOURCE_TO_ID["versioned_compatibility_adapter"],
        )
        truth_sources = torch.full_like(
            batch["level_ids"],
            TRUTH_SUPERVISION_SOURCE_TO_ID["retained_mc_truth"],
        )
        batch.setdefault("model_input_source_ids", input_sources)
        batch.setdefault("daughter_input_pid_source_ids", input_sources.clone())
        batch.setdefault("truth_supervision_source_ids", truth_sources)
        batch.setdefault("daughter_truth_pid_source_ids", truth_sources.clone())
        if "runtime_composite_type_source_ids" not in batch:
            from hypertagging.reconstruction.pid_state import (
                COMPOSITE_TYPE_SOURCE_TO_ID,
            )

            batch = dict(batch)
            sources = torch.full_like(
                batch["level_ids"], COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"]
            )
            sources[batch["level_ids"] > 0] = COMPOSITE_TYPE_SOURCE_TO_ID[
                "truth_teacher_forced"
            ]
            batch["runtime_composite_type_source_ids"] = sources
        return batch
    p4 = batch["p4"]
    charge = batch["charge"]
    levels = batch["level_ids"]
    active = batch["node_mask"].bool()
    copied = batch.get("copied", torch.zeros_like(active))
    adjacency = batch["daughter_adjacency"]
    mass2 = p4[..., 3].square() - p4[..., :3].square().sum(dim=-1)
    common = torch.stack(
        [
            p4[..., 0],
            p4[..., 1],
            p4[..., 2],
            p4[..., 3],
            mass2.clamp_min(0).sqrt(),
            charge,
            batch["pid_labels"].float(),
            levels.float().clamp_min(0),
            active.float(),
            copied.float(),
            adjacency.sum(dim=-1).float(),
            torch.zeros_like(charge),
        ],
        dim=-1,
    )
    common_availability = active.unsqueeze(-1).expand_as(common).clone()
    common_availability[..., -1] = False
    from hypertagging.preprocessing.schema_v4 import CATEGORICAL_COMMON_FEATURE_NAMES
    for name in CATEGORICAL_COMMON_FEATURE_NAMES:
        common_availability[..., COMMON_FEATURE_NAMES.index(name)] = False
    batch = dict(batch)
    from hypertagging.data.heterogeneous import (
        MODEL_INPUT_SOURCE_TO_ID,
        TRUTH_SUPERVISION_SOURCE_TO_ID,
    )

    input_sources = torch.full_like(
        levels, MODEL_INPUT_SOURCE_TO_ID["versioned_compatibility_adapter"]
    )
    truth_sources = torch.full_like(
        levels, TRUTH_SUPERVISION_SOURCE_TO_ID["retained_mc_truth"]
    )
    batch["model_input_source_ids"] = input_sources
    batch["daughter_input_pid_source_ids"] = input_sources.clone()
    batch["truth_supervision_source_ids"] = truth_sources
    batch["daughter_truth_pid_source_ids"] = truth_sources.clone()
    batch["common_features"] = common
    batch["common_availability"] = common_availability
    shape = (*active.shape, len(TRACK_FEATURE_NAMES))
    batch["track_features"] = p4.new_zeros(shape)
    batch["track_availability"] = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    shape = (*active.shape, len(CLUSTER_FEATURE_NAMES))
    batch["cluster_features"] = p4.new_zeros(shape)
    batch["cluster_availability"] = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    shape = (*active.shape, len(COMPOSITE_FEATURE_NAMES))
    composite = p4.new_zeros(shape)
    composite[..., :4] = torch.einsum("bmn,bnf->bmf", adjacency.float(), p4)
    composite[..., 4] = torch.einsum("bmn,bn->bm", adjacency.float(), charge)
    composite[..., 5] = adjacency.sum(dim=-1)
    composite[..., 8] = (
        torch.einsum("bmn,bn->bm", adjacency.float(), copied.float())
        / adjacency.sum(dim=-1).clamp_min(1)
    )
    composite_availability = torch.zeros(shape, dtype=torch.bool, device=p4.device)
    has_daughters = adjacency.any(dim=-1)
    for index in (0, 1, 2, 3, 4, 5, 8):
        composite_availability[..., index] = has_daughters
    batch["composite_features"] = composite
    batch["composite_availability"] = composite_availability
    histogram = p4.new_zeros((*active.shape, len(PDG_TOKENS)))
    daughter_tokens = (
        batch["pid_labels"][:, None, :]
        .expand(-1, active.shape[1], -1)
    )
    validate_pid_tokens(batch["pid_labels"][active], name="upgraded batch PID labels")
    histogram.scatter_add_(-1, daughter_tokens, adjacency.float())
    batch["daughter_input_pid_histogram"] = histogram
    batch["daughter_input_pid_histogram_available"] = has_daughters
    batch["daughter_truth_pid_histogram"] = histogram.clone()
    batch["daughter_truth_pid_histogram_available"] = has_daughters.clone()
    batch["daughter_pid_histogram"] = histogram
    batch["daughter_pid_histogram_available"] = has_daughters
    if "node_kind_ids" in batch:
        kinds = batch["node_kind_ids"]
    else:
        kinds = torch.full_like(levels, NODE_KIND_TO_ID["unknown"])
        kinds[levels > 0] = NODE_KIND_TO_ID["composite"]
        kinds[(levels == 0) & (charge != 0)] = NODE_KIND_TO_ID["track"]
    batch["node_kind_ids"] = kinds
    # Historical flat CPU fixtures predate the separated target field. Their
    # reduced composite labels are topology targets, so expose that fact
    # explicitly instead of leaving teacher-forced composites unknown.
    batch.setdefault("pid_target_labels", batch["pid_labels"].clone())
    default_ids = torch.arange(
        active.shape[1],
        device=p4.device,
        dtype=torch.long,
    )[None].expand_as(levels)
    batch.setdefault("node_ids", default_ids)
    batch.setdefault("reco_ids", torch.full_like(levels, -1))
    batch.setdefault("source_node_ids", default_ids)
    batch.setdefault("copied_from", torch.full_like(levels, -1))
    batch.setdefault("active", active)
    batch.setdefault("b_side", torch.full_like(levels, -1))
    if "leaf_kinematics_mode_ids" not in batch:
        from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID

        modes = torch.full_like(levels, LEAF_MODE_TO_ID["truth_topology_only"])
        modes[levels > 0] = LEAF_MODE_TO_ID["composite"]
        modes[(levels == 0) & (kinds == NODE_KIND_TO_ID["track"])] = (
            LEAF_MODE_TO_ID["fixed_hypothesis_candidate"]
        )
        modes[(levels == 0) & (kinds == NODE_KIND_TO_ID["ecl_cluster"])] = (
            LEAF_MODE_TO_ID["ecl_cluster"]
        )
        batch["leaf_kinematics_mode_ids"] = modes
    if "runtime_composite_type_source_ids" not in batch:
        from hypertagging.reconstruction.pid_state import (
            COMPOSITE_TYPE_SOURCE_TO_ID,
        )

        sources = torch.full_like(
            levels, COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"]
        )
        sources[levels > 0] = COMPOSITE_TYPE_SOURCE_TO_ID[
            "truth_teacher_forced"
        ]
        batch["runtime_composite_type_source_ids"] = sources
    daughter_count = adjacency.sum(dim=-1).long()
    batch.setdefault("full_truth_daughter_count", daughter_count.clone())
    batch.setdefault(
        "retained_truth_daughter_count_expected", daughter_count.clone()
    )
    batch.setdefault("retained_daughter_count", daughter_count.clone())
    batch.setdefault("reconstructed_daughter_count", daughter_count.clone())
    batch.setdefault("complete_truth_decay", active.clone())
    batch.setdefault("complete_reconstructable_decay", active.clone())
    batch.setdefault("recursive_reconstructable_complete", active.clone())
    batch.setdefault("partial_missing_daughters", torch.zeros_like(active))
    batch.setdefault("contracted_intermediate", torch.zeros_like(active))
    batch.setdefault("valid_reconstruction_target", daughter_count >= 2)
    maximum_level = torch.where(active, levels, torch.zeros_like(levels)).max(
        dim=-1, keepdim=True
    ).values
    batch.setdefault("truth_root_distance", maximum_level - levels.clamp_min(0))
    batch.setdefault("full_event_max_level", maximum_level.expand_as(levels))
    return batch
