import numpy as np
import torch
from scripts.diagnose_phase79_probe_optimization import risk_gradient, prepare


def test_convex_risk_gradient_and_label_reversal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3))
    y = np.arange(20) % 2
    w = np.full(20, 0.05)
    t = rng.normal(size=4)
    loss, g = risk_gradient(t, x, y, w)
    for i in range(4):
        e = np.eye(4)[i] * 1e-5
        numeric = (
            risk_gradient(t + e, x, y, w)[0] - risk_gradient(t - e, x, y, w)[0]
        ) / 2e-5
        assert abs(numeric - g[i]) < 1e-8
    flip, fg = risk_gradient(-t, x, 1 - y, w)
    assert abs(flip - loss) < 1e-12
    np.testing.assert_allclose(fg, -g, atol=1e-12)


def test_event_balanced_risk_excludes_assessment_and_unknowns():
    r = {
        "uid": "a",
        "partition": "fit",
        "encoder": torch.randn(5, 2),
        "target": torch.tensor([1, 1, 2, 2, -1]),
        "ij": torch.triu_indices(5, 5, 1),
    }
    ref = {"mean": torch.zeros(4), "scale": torch.ones(4)}
    other = {
        **r,
        "uid": "b",
        "partition": "assessment",
        "encoder": torch.full((5, 2), float("nan")),
    }
    x, y, w, ids = prepare([r, other], ref)
    assert ids == ["a"] and len(x) == 6 and np.isfinite(x).all()
    assert np.isclose(w.sum(), 1) and np.isclose(w[y == 0].sum(), 0.5)
    r2 = {**r, "target": torch.where(r["target"] > 0, 3 - r["target"], r["target"])}
    xx, yy, ww, _ = prepare([r2], ref)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    np.testing.assert_array_equal(w, ww)
