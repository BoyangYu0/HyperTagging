"""Tiny synthetic fixture batches for CPU-only contract tests."""

from __future__ import annotations

import numpy as np


def tiny_toy_mc_batch() -> dict[str, np.ndarray]:
    """Return a minimal Toy-MC/HyperTagging-style batch."""

    pdg = np.array([[11, -11, 0], [13, 22, 211]], dtype=np.int32)
    return {
        "pdg": pdg,
        "feature": np.array(
            [
                [[1.0, 0.1], [2.0, 0.2], [0.0, 0.0]],
                [[1.5, 0.3], [0.7, 0.4], [2.1, 0.5]],
            ],
            dtype=np.float32,
        ),
        "padding_mask": pdg > 0,
        "motherPDG": np.array([[443, 443, 0], [511, 511, 511]], dtype=np.int32),
        "motherIndex": np.array([[0, 0, -1], [0, 1, 1]], dtype=np.int32),
        "channel": np.array([0, 1], dtype=np.int32),
        "evtNum": np.array([1001, 1002], dtype=np.int32),
        "depth": np.array([2, 3], dtype=np.int32),
        "E_Rec": np.array([5.28, 5.27], dtype=np.float32),
    }


def tiny_grafei_pair_batch() -> dict[str, np.ndarray]:
    """Return a minimal GraFEI link-prediction pair batch."""

    pdg_x = np.array([[11, 22, 0], [13, 211, 321]], dtype=np.int32)
    pdg_y = np.array([[443, 443, 0], [511, 511, 511]], dtype=np.int32)
    return {
        "pdg_x": pdg_x,
        "pdg_y": pdg_y,
        "feature_x": np.ones((2, 3, 4), dtype=np.float32),
        "feature_y": np.full((2, 3, 4), 2.0, dtype=np.float32),
        "padding_mask": pdg_x > 0,
        "mass": np.array([1, 2], dtype=np.int32),
        "pattern": np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.int32),
        "links": np.array([[0, 0, -1], [1, 1, 2]], dtype=np.int64),
    }


def tiny_grafei_combined_batch() -> dict[str, np.ndarray]:
    """Return a minimal GraFEI reconstruction batch with embeddings."""

    batch = tiny_grafei_pair_batch()
    batch.pop("pattern")
    batch.update(
        {
            "channels": np.array([2, 2], dtype=np.int32),
            "evtNums": np.array([2001, 2002], dtype=np.int32),
            "emb": np.arange(12, dtype=np.float32).reshape(2, 6),
        }
    )
    return batch


def tiny_gpt_link_flattened_batch() -> dict[str, np.ndarray]:
    """Return a minimal GPT-like flattened link batch."""

    return {
        "emb_x": np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        "emb_y": np.arange(48, 96, dtype=np.float32).reshape(2, 3, 8),
        "links": np.array([[0, 1, -1], [1, 2, 2]], dtype=np.int64),
        "padding_mask": np.array([[True, True, False], [True, True, True]], dtype=np.bool_),
    }


def tiny_gpt_reconstruction_flattened_batch() -> dict[str, np.ndarray]:
    """Return a minimal GPT-like flattened reconstruction batch."""

    src_mask = np.zeros((2, 4, 4), dtype=np.float32)
    src_mask[:, np.triu_indices(4, k=1)[0], np.triu_indices(4, k=1)[1]] = -np.inf
    return {
        "emb": np.arange(64, dtype=np.float32).reshape(2, 4, 8),
        "target": np.arange(64, 128, dtype=np.float32).reshape(2, 4, 8),
        "src_mask": src_mask,
        "links": np.array([[0, 1, -1, -1], [1, 2, 2, -1]], dtype=np.int64),
        "mass": np.array([[1, 2, 0, 0], [2, 3, 4, 0]], dtype=np.int64),
        "lvl_code": np.array([[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]], dtype=np.float32),
    }


def all_tiny_batches() -> dict[str, dict[str, np.ndarray]]:
    """Return all Phase 3 fixture batches keyed by contract name."""

    return {
        "toy_mc": tiny_toy_mc_batch(),
        "grafei_pairs": tiny_grafei_pair_batch(),
        "grafei_combined": tiny_grafei_combined_batch(),
        "gpt_link_flattened": tiny_gpt_link_flattened_batch(),
        "gpt_reconstruction_flattened": tiny_gpt_reconstruction_flattened_batch(),
    }
