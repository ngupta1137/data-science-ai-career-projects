import numpy as np
import pytest

from claims_pricing import metrics


def test_gini_flat_and_hand_computed():
    # Flat predictions -> Gini close to 0
    exposure = np.array([1.0, 1.0, 1.0, 1.0])
    loss = np.array([10.0, 20.0, 15.0, 25.0])
    flat_pred = np.array([17.5, 17.5, 17.5, 17.5])
    g_flat = metrics.gini_coefficient(flat_pred, loss, exposure)
    assert abs(g_flat) < 0.05

    # Hand-computed 2-policy case
    # Policy 1: pred=1, exp=1, loss=0
    # Policy 2: pred=2, exp=1, loss=10
    # cum_exp: [0, 0.5, 1.0], cum_loss: [0, 0, 1.0]
    # trapezoid area: (0.5 * 0)/2 + (0.5 * 1)/2 = 0.25
    # Gini = 1 - 2 * 0.25 = 0.50
    exp2 = np.array([1.0, 1.0])
    loss2 = np.array([0.0, 10.0])
    pred2 = np.array([1.0, 2.0])
    assert metrics.gini_coefficient(pred2, loss2, exp2) == pytest.approx(0.50)


def test_loss_ratio_hand_computed():
    losses = np.array([30.0, 40.0])
    prems = np.array([50.0, 50.0])
    assert metrics.loss_ratio(losses, prems) == pytest.approx(0.70)
    assert metrics.loss_ratio(np.array([0.0]), np.array([0.0])) == 0.0


def test_decile_loss_ratios_structure(rng):
    n = 200
    actual = rng.gamma(2.0, 50.0, n)
    charged = rng.uniform(50.0, 150.0, n)
    rank = rng.normal(100.0, 20.0, n)
    exp = rng.uniform(0.1, 1.0, n)

    deciles = metrics.decile_loss_ratios(actual, charged, rank, exp)
    assert len(deciles) == 10
    assert sum(d["n_policies"] for d in deciles) == n
    for d in deciles:
        assert d["loss_ratio"] >= 0.0
        assert d["exposure"] > 0.0


def test_rmse():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    # diff = [0, 0, -2], sq = [0, 0, 4], mean = 4/3, sqrt = 1.15470
    assert metrics.rmse(a, b) == pytest.approx(np.sqrt(4.0 / 3.0))
