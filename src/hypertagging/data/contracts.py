"""Data contracts for historical HyperTagging batch dictionaries.

These contracts document observed batch keys, shapes, and dtypes without
implementing historical preprocessing. Ambiguous fields are kept provisional
until equivalence tests are added in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


DTypeKind = Literal["bool", "float", "int"]
ShapeDim = int | str | None


@dataclass(frozen=True)
class FieldSpec:
    """One expected field inside a historical batch dictionary."""

    name: str
    dtype: DTypeKind
    ndim: int
    shape: tuple[ShapeDim, ...]
    description: str
    provisional: bool = False


@dataclass(frozen=True)
class BatchContract:
    """A named set of required and optional batch fields."""

    name: str
    purpose: str
    source_files: tuple[str, ...]
    required: tuple[FieldSpec, ...]
    optional: tuple[FieldSpec, ...] = ()
    notes: tuple[str, ...] = ()

    def validate(self, batch: Mapping[str, Any]) -> None:
        """Validate keys, ranks, dtype families, and symbolic shape equality."""

        symbols: dict[str, int] = {}
        for field in self.required:
            if field.name not in batch:
                raise KeyError(f"{self.name}: missing required field {field.name!r}.")
            _validate_field(self.name, field, batch[field.name], symbols)

        for field in self.optional:
            if field.name in batch:
                _validate_field(self.name, field, batch[field.name], symbols)


def validate_batch(contract: BatchContract | str, batch: Mapping[str, Any]) -> None:
    """Validate ``batch`` against a named contract or ``BatchContract``."""

    if isinstance(contract, str):
        contract = get_contract(contract)
    contract.validate(batch)


def get_contract(name: str) -> BatchContract:
    """Return a registered Phase 3 data contract by name."""

    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown data contract {name!r}.") from exc


def _validate_field(
    contract_name: str,
    field: FieldSpec,
    value: Any,
    symbols: dict[str, int],
) -> None:
    shape = _shape(value)
    if len(shape) != field.ndim:
        raise ValueError(
            f"{contract_name}.{field.name}: expected ndim {field.ndim}, got {len(shape)}."
        )
    _validate_dtype(contract_name, field, value)
    for expected, actual in zip(field.shape, shape, strict=True):
        if expected is None:
            continue
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError(
                    f"{contract_name}.{field.name}: expected dim {expected}, got {actual}."
                )
            continue
        previous = symbols.setdefault(expected, actual)
        if previous != actual:
            raise ValueError(
                f"{contract_name}.{field.name}: symbol {expected!r} expected {previous}, "
                f"got {actual}."
            )


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return tuple(np.asarray(value).shape)


def _validate_dtype(contract_name: str, field: FieldSpec, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if field.dtype == "bool" and value.dtype is torch.bool:
            return
        if field.dtype == "float" and torch.is_floating_point(value):
            return
        if field.dtype == "int" and not torch.is_floating_point(value) and value.dtype is not torch.bool:
            return
    else:
        dtype = np.asarray(value).dtype
        if field.dtype == "bool" and np.issubdtype(dtype, np.bool_):
            return
        if field.dtype == "float" and np.issubdtype(dtype, np.floating):
            return
        if field.dtype == "int" and np.issubdtype(dtype, np.integer):
            return
    raise TypeError(
        f"{contract_name}.{field.name}: expected {field.dtype} dtype family, "
        f"got {getattr(value, 'dtype', np.asarray(value).dtype)}."
    )


TOY_MC_BATCH = BatchContract(
    name="toy_mc",
    purpose="Toy-MC/HyperTagging event batch used by historical utility datasets.",
    source_files=(
        "HyperTagging/ak/utils.py",
        "HyperTagging/reduced/utils.py",
        "HyperTagging/noR/SampleEmbedding.py",
        "HyperTagging/noR/Reconstruction.py",
        "graFEI_gpt/utils.py",
    ),
    required=(
        FieldSpec("pdg", "int", 2, ("B", "P"), "Per-particle PDG codes."),
        FieldSpec("feature", "float", 3, ("B", "P", "F"), "Per-particle features excluding PDG."),
        FieldSpec("padding_mask", "bool", 2, ("B", "P"), "True for non-padding particles."),
        FieldSpec("motherPDG", "int", 2, ("B", "P"), "Legacy mother PDG labels.", provisional=True),
        FieldSpec("motherIndex", "int", 2, ("B", "P"), "Legacy mother index labels.", provisional=True),
        FieldSpec("channel", "int", 1, ("B",), "Channel identifier."),
        FieldSpec("evtNum", "int", 1, ("B",), "Event number."),
        FieldSpec("depth", "int", 1, ("B",), "Depth label."),
        FieldSpec("E_Rec", "float", 1, ("B",), "Reconstructed energy label."),
    ),
    optional=(
        FieldSpec("arrayIndex", "int", 2, ("B", "P"), "Present in graFEI_gpt utility variant.", True),
    ),
    notes=(
        "The noR reconstruction script uses pdg_x/pdg_y and feature_x/feature_y instead of this compact toy-MC embedding form.",
        "arrayIndex is observed only in the graFEI_gpt utility variant and remains provisional.",
    ),
)

GRAFEI_PAIR_BATCH = BatchContract(
    name="grafei_pairs",
    purpose="GraFEI pair batch used by historical link-prediction scripts.",
    source_files=(
        "graFEI/Link.py",
        "graFEI_reduced/Link.py",
    ),
    required=(
        FieldSpec("pdg_x", "int", 2, ("B", "P"), "Left particle PDG codes."),
        FieldSpec("pdg_y", "int", 2, ("B", "P"), "Right particle PDG codes."),
        FieldSpec("feature_x", "float", 3, ("B", "P", "F"), "Left particle features."),
        FieldSpec("feature_y", "float", 3, ("B", "P", "F"), "Right particle features."),
        FieldSpec("padding_mask", "bool", 2, ("B", "P"), "True for non-padding left particles."),
        FieldSpec("mass", "int", 1, ("B",), "Mass-category target from preprocessing."),
        FieldSpec("pattern", "int", 2, ("B", "K"), "Decay-pattern target from preprocessing."),
        FieldSpec("links", "int", 2, ("B", "P"), "Link labels padded with -1."),
    ),
)

GRAFEI_COMBINED_BATCH = BatchContract(
    name="grafei_combined",
    purpose="GraFEI reconstruction batch combining pair data with pretrained embeddings.",
    source_files=(
        "graFEI/Reconstruction.py",
        "graFEI_reduced/Reconstruction.py",
    ),
    required=(
        FieldSpec("pdg_x", "int", 2, ("B", "P"), "Left particle PDG codes."),
        FieldSpec("pdg_y", "int", 2, ("B", "P"), "Right particle PDG codes."),
        FieldSpec("feature_x", "float", 3, ("B", "P", "F"), "Left particle features."),
        FieldSpec("feature_y", "float", 3, ("B", "P", "F"), "Right particle features."),
        FieldSpec("padding_mask", "bool", 2, ("B", "P"), "True for non-padding left particles."),
        FieldSpec("mass", "int", 1, ("B",), "Mass-category target from preprocessing."),
        FieldSpec("links", "int", 2, ("B", "P"), "Link labels padded with -1."),
        FieldSpec("channels", "int", 1, ("B",), "Channel identifier."),
        FieldSpec("evtNums", "int", 1, ("B",), "Event number."),
        FieldSpec("emb", "float", 2, ("B", "E"), "Precomputed event embedding."),
    ),
    notes=(
        "SampleEmbedding.py uses a related non-combined batch with pdg/feature/pattern/channels/evtNums.",
    ),
)

GPT_LINK_FLATTENED_BATCH = BatchContract(
    name="gpt_link_flattened",
    purpose="GPT-like flattened adjacent-level batch for autoregressive link prediction.",
    source_files=("graFEI_gpt/Link.py",),
    required=(
        FieldSpec("emb_x", "float", 3, ("B", "P", "E"), "Source level embeddings."),
        FieldSpec("emb_y", "float", 3, ("B", "P", "E"), "Next level embeddings."),
        FieldSpec("links", "int", 2, ("B", "P"), "Flattened link labels padded with -1."),
        FieldSpec("padding_mask", "bool", 2, ("B", "P"), "True for non-padding source particles."),
    ),
)

GPT_RECONSTRUCTION_FLATTENED_BATCH = BatchContract(
    name="gpt_reconstruction_flattened",
    purpose="GPT-like flattened full-event batch for autoregressive reconstruction.",
    source_files=("graFEI_gpt/Reconstruction.py",),
    required=(
        FieldSpec("emb", "float", 3, ("B", "T", "E"), "Flattened event embeddings."),
        FieldSpec("target", "float", 3, ("B", "T", "E"), "Shifted target embeddings."),
        FieldSpec("src_mask", "float", 3, ("B", "T", "T"), "Autoregressive level mask."),
        FieldSpec("links", "int", 2, ("B", "T"), "Flattened link labels padded with -1."),
        FieldSpec("mass", "int", 2, ("B", "T"), "Mass-category targets."),
        FieldSpec("lvl_code", "float", 2, ("B", "T"), "Level encoding."),
    ),
)

CONTRACTS: dict[str, BatchContract] = {
    contract.name: contract
    for contract in (
        TOY_MC_BATCH,
        GRAFEI_PAIR_BATCH,
        GRAFEI_COMBINED_BATCH,
        GPT_LINK_FLATTENED_BATCH,
        GPT_RECONSTRUCTION_FLATTENED_BATCH,
    )
}
