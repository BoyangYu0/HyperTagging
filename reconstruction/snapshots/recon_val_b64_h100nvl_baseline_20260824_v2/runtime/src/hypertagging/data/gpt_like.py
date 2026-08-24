"""GPT-like GraFEI preprocessing and flattened-batch adapters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Iterable, Literal

import numpy as np
import torch

from hypertagging.data.contracts import validate_batch
from hypertagging.data.preprocessing import (
    DEFAULT_GRAFEI_INPUT_ROOT,
    DryRunResult,
    prepare_legacy_dataset,
)


GptBatchTask = Literal["reconstruction", "link"]


@lru_cache(maxsize=256)
def get_level_mask(size: int, particles_per_level: int = 2) -> np.ndarray:
    """Return the historical block-autoregressive level mask."""

    mask = np.zeros((size, size), dtype=np.float32)
    for index in range(particles_per_level, size + 1, particles_per_level):
        mask[index - particles_per_level : index, index:] = float("-inf")
    return mask


def collate_gpt_reconstruction(batch: Iterable[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Collate flattened GPT-like event examples using historical formulas."""

    examples = list(batch)
    max_num_particles = max(int(example["shape"][0]) for example in examples)
    emb_dim = examples[0]["emb"].shape[-1]
    out = {
        "emb": np.zeros((len(examples), max_num_particles, emb_dim), dtype=np.float32),
        "target": np.zeros((len(examples), max_num_particles, emb_dim), dtype=np.float32),
        "src_mask": np.full((len(examples), max_num_particles, max_num_particles), "-inf", dtype=np.float32),
        "links": np.full((len(examples), max_num_particles), -1, dtype=np.int_),
        "mass": np.zeros((len(examples), max_num_particles), dtype=np.int_),
        "lvl_code": np.zeros((len(examples), max_num_particles), dtype=np.float32),
    }
    for index, example in enumerate(examples):
        event_len, level_len = int(example["shape"][0]), int(example["shape"][1])
        n_levels = int(event_len / level_len)
        out["emb"][index, :event_len] = example["emb"]
        out["target"][index, : event_len - level_len] = example["emb"][level_len:]
        out["src_mask"][index, :event_len, :event_len] = get_level_mask(event_len, level_len)
        out["src_mask"][index, event_len:] = 0
        links = example["links"][: event_len - level_len]
        out["links"][index, : event_len - level_len] = np.where(
            links >= 0,
            links + level_len * np.arange(n_levels - 1).repeat(level_len),
            -1,
        )
        out["mass"][index, : event_len - level_len] = example["mass"][level_len:]
        out["lvl_code"][index, :event_len] = np.exp(-np.arange(n_levels)).repeat(level_len)
    return {key: torch.from_numpy(value) for key, value in out.items()}


def collate_gpt_link(batch: Iterable[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Collate flattened embedding-link examples using historical padding."""

    examples = list(batch)
    max_num_particles = max(len(example["emb_x"]) for example in examples)
    emb_dim = examples[0]["emb_x"].shape[-1]
    out = {
        "emb_x": np.zeros((len(examples), max_num_particles, emb_dim), dtype=np.float32),
        "emb_y": np.zeros((len(examples), max_num_particles, emb_dim), dtype=np.float32),
        "links": np.full((len(examples), max_num_particles), -1, dtype=np.int_),
        "padding_mask": np.zeros((len(examples), max_num_particles), dtype=np.bool_),
    }
    for index, example in enumerate(examples):
        for key in out:
            array = example[key]
            out[key][index, : len(array)] = array
    return {key: torch.from_numpy(value) for key, value in out.items()}


def validate_gpt_batch(
    batch: dict[str, torch.Tensor],
    *,
    task: GptBatchTask,
) -> dict[str, torch.Tensor]:
    """Validate a GPT-like batch by task."""

    contracts = {
        "reconstruction": "gpt_reconstruction_flattened",
        "link": "gpt_link_flattened",
    }
    validate_batch(contracts[task], batch)
    return batch


def validate_gpt_reconstruction_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate a GPT reconstruction batch against the Phase 3 contract."""

    return validate_gpt_batch(batch, task="reconstruction")


def validate_gpt_link_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate a GPT link batch against the Phase 3 contract."""

    return validate_gpt_batch(batch, task="link")


def prepare_gpt_like(
    chunk_index: int,
    *,
    input_root: Path | str = DEFAULT_GRAFEI_INPUT_ROOT,
    output_root: Path | str | None = None,
    dry_run: bool = True,
) -> DryRunResult | subprocess.CompletedProcess[str]:
    """Prepare or run the legacy GPT-like GraFEI preprocessing script."""

    return prepare_legacy_dataset(
        "gpt_like",
        chunk_index,
        input_root=input_root,
        output_root=output_root,
        dry_run=dry_run,
    )


__all__ = [
    "collate_gpt_link",
    "collate_gpt_reconstruction",
    "get_level_mask",
    "prepare_gpt_like",
    "validate_gpt_batch",
    "validate_gpt_link_batch",
    "validate_gpt_reconstruction_batch",
]
