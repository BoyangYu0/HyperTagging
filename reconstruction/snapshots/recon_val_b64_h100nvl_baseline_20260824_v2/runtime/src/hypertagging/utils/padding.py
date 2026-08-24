"""Legacy padding helpers migrated without scientific reinterpretation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import awkward as ak
import numpy as np


def pack_with_evt_num(array: ak.Array, field: str = "evtNum") -> ak.Array:
    """Pack a flat awkward array into event groups using legacy evtNum counts.

    Historical repositories used ``np.unique(..., return_counts=True)`` and
    ``ak.unflatten``. That preserves counts but assumes records are already
    ordered by event number.
    """

    _, counts = np.unique(array[field].to_numpy(), return_counts=True)
    return ak.unflatten(array, counts)


def pack_with_evtNum(array: ak.Array) -> ak.Array:
    """Compatibility alias for the historical camel-case helper name."""

    return pack_with_evt_num(array)


def pad_to(
    array: ak.Array,
    fill_value: Any = 0,
    pad_length: int | None = None,
    dtype: np.dtype | type = np.float32,
    v_kwargs: Mapping[str, tuple[int, Any, np.dtype | type]] | None = None,
) -> np.ndarray | dict[str, np.ndarray] | ak.Array:
    """Pad a jagged awkward array using the historical HyperTagging behavior.

    This combines the variants found in the historical repositories:
    ``HyperTagging`` used automatic maximum length padding; GraFEI variants
    added ``pad_length``; ``graFEI_gpt`` added ``v_kwargs`` splitting. The
    ``v_kwargs`` branch intentionally preserves the legacy slice behavior.
    """

    if array.fields:
        raise TypeError("pad_to does not support awkward records; select a field first.")

    if v_kwargs is not None and len(array) <= 1:
        return array

    lengths = ak.num(array)
    if pad_length is None:
        pad_length = int(ak.max(lengths))

    shape: list[int] = [len(array), int(pad_length)]
    if array.ndim > 2:
        shape += list(array[0].to_numpy().shape[1:])

    if v_kwargs is None:
        out = np.ones(shape, dtype=dtype) * fill_value
    else:
        out = np.ones(shape) * fill_value
        layouts: list[int] = []
        for layout, section_fill, _section_dtype in v_kwargs.values():
            out[:, :, layout:] = section_fill
            layouts.append(layout)

    flat_array = ak.flatten(array, axis=1).to_numpy()
    count = 0
    for i, length in enumerate(lengths):
        length = int(length)
        out[i, :length] = flat_array[count : count + length]
        count += length

    if v_kwargs is None:
        return out.astype(dtype, copy=False)

    layouts.append(shape[-1])
    slices = [slice(layouts[i], layouts[i + 1]) for i in range(len(layouts) - 1)]
    return {
        key: out[:, slices[i]].astype(section_dtype)
        for i, (key, (_layout, _section_fill, section_dtype)) in enumerate(v_kwargs.items())
    }
