import numpy as np
import pytest

from marketing_mix import metrics


def test_r2_score_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0])
    # mean = 20.0, ss_tot = (-10)^2 + 0 + 10^2 = 200.0
    y_pred_perfect = np.array([10.0, 20.0, 30.0])
    assert metrics.r2_score(y_true, y_pred_perfect) == pytest.approx(1.0)

    y_pred_mean = np.array([20.0, 20.0, 20.0])
    assert metrics.r2_score(y_true, y_pred_mean) == pytest.approx(0.0)

    y_pred_err = np.array([12.0, 20.0, 28.0])
    # ss_res = 2^2 + 0 + (-2)^2 = 8.0 -> R^2 = 1 - 8/200 = 0.96
    assert metrics.r2_score(y_true, y_pred_err) == pytest.approx(0.96)


def test_rmse_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 17.0, 34.0])
    # diffs = [2, -3, 4], sq = [4, 9, 16] -> mean = 29/3 -> sqrt(29/3)
    assert metrics.rmse(y_true, y_pred) == pytest.approx(np.sqrt(29.0 / 3.0))


def test_mape_hand_computed():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # errors = 10/100 = 0.10, 20/200 = 0.10 -> mean = 0.10
    assert metrics.mape(y_true, y_pred) == pytest.approx(0.10)


def test_spearman_rank_correlation():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert metrics.spearman_rank_correlation(x, y) == pytest.approx(1.0)

    y_rev = [40.0, 30.0, 20.0, 10.0]
    assert metrics.spearman_rank_correlation(x, y_rev) == pytest.approx(-1.0)


def test_budget_optimization_constraints():
    total_budget = 60_000.0
    min_spend = 5_000.0
    channel_params = {
        "tv": {"half_sat": 40_000.0, "max_lift": 50_000.0},
        "search": {"half_sat": 18_000.0, "max_lift": 28_000.0},
        "social": {"half_sat": 25_000.0, "max_lift": 35_000.0},
    }
    alloc = metrics.optimize_budget_mmm(channel_params, total_budget, min_spend)
    assert sum(alloc.values()) == pytest.approx(total_budget, rel=1e-4)
    for spend in alloc.values():
        assert spend >= min_spend - 1e-4

    # Test linear budget optimization constraints
    slopes = {"tv": 0.35, "search": -0.05, "social": 0.25}
    alloc_lin = metrics.optimize_budget_linear(slopes, total_budget, min_spend)
    assert sum(alloc_lin.values()) == pytest.approx(total_budget, rel=1e-4)
    # Channel with negative slope gets exactly minimum spend
    assert alloc_lin["search"] == pytest.approx(min_spend, rel=1e-4)
