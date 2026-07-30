import builtins

import pytest
import torch

from hypertagging.losses.set_matching import hungarian_assignment


def test_nontrivial_hungarian_and_truth_order_invariance():
    cost = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [2.0, 2.0, 0.0]])
    result = hungarian_assignment(cost, production=False, allow_bruteforce=True)
    assert sorted(result) == [(0, 1), (1, 0), (2, 2)]
    permuted = hungarian_assignment(cost[:, [2, 0, 1]], production=False, allow_bruteforce=True)
    mapped = sorted((row, [2, 0, 1][column]) for row, column in permuted)
    assert mapped == sorted(result)


def test_production_fails_without_scipy(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="SciPy is required"):
        hungarian_assignment(torch.eye(2), production=True)
    assert hungarian_assignment(
        torch.eye(2),
        production=False,
        allow_bruteforce=True,
    )
