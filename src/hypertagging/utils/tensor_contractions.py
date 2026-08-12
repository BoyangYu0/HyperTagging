"""CUDA-safe tensor contractions for Boolean membership relations."""

from __future__ import annotations

import torch


def boolean_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return whether each matrix-product reduction has any shared member.

    Inputs are interpreted as membership masks, so every nonzero value is one
    member. The FP32 reduction avoids unsupported CUDA integer kernels and is
    kept out of autocast. Only positivity is consumed, so loss of exactness in
    very wide nonnegative counts cannot change the Boolean result.
    """

    left_membership = left.ne(0).to(torch.float32)
    right_membership = right.ne(0).to(torch.float32)
    with torch.autocast(device_type=left.device.type, enabled=False):
        if left_membership.ndim == 3 and right_membership.ndim == 3:
            product = torch.bmm(left_membership, right_membership)
        else:
            product = torch.matmul(left_membership, right_membership)
    return product.gt(0)
