import numpy as np
import pytest

from src import metrics

# Hand-computed example.
Y = [3.0, -0.5, 2.0, 7.0]
P = [2.5, 0.0, 2.0, 8.0]
# Errors: 0.5, -0.5, 0, -1. Squared: 0.25, 0.25, 0, 1 -> SSE 1.5, MSE 0.375.
# Mean of Y is 2.875. SST = 0.125^2 + 3.375^2 + 0.875^2 + 4.125^2 = 29.1875.


def test_rmse_matches_hand_value():
    assert metrics.rmse(Y, P) == pytest.approx(np.sqrt(0.375))


def test_mae_matches_hand_value():
    assert metrics.mae(Y, P) == pytest.approx(0.5)


def test_r2_matches_hand_value():
    assert metrics.r2(Y, P) == pytest.approx(1 - 1.5 / 29.1875)


def test_bias_sign_means_overprediction():
    assert metrics.bias([1.0, 1.0], [2.0, 3.0]) == pytest.approx(1.5)


def test_r2_is_nan_when_truth_is_constant():
    assert np.isnan(metrics.r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


def test_perfect_predictions_give_zero_interval():
    y = np.arange(20, dtype=float)
    groups = np.repeat(np.arange(5), 4)
    low, high = metrics.cluster_bootstrap_ci(y, y, groups, metrics.rmse, n_boot=50, seed=1)
    assert (low, high) == (0.0, 0.0)


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    y = rng.normal(size=60)
    p = y + rng.normal(scale=0.5, size=60)
    groups = np.repeat(np.arange(12), 5)
    a = metrics.cluster_bootstrap_ci(y, p, groups, metrics.rmse, n_boot=100, seed=7)
    b = metrics.cluster_bootstrap_ci(y, p, groups, metrics.rmse, n_boot=100, seed=7)
    assert a == b


def test_interval_contains_point_estimate_in_a_typical_case():
    rng = np.random.default_rng(5)
    y = rng.normal(size=400)
    p = y + rng.normal(scale=0.4, size=400)
    groups = np.repeat(np.arange(40), 10)
    low, high = metrics.cluster_bootstrap_ci(y, p, groups, metrics.rmse, n_boot=300, seed=2)
    assert low <= metrics.rmse(y, p) <= high


def test_target_band_puts_capped_rows_in_their_own_band():
    labels = metrics.target_band([1.0, 2.0, 3.0, 4.0, 5.0, 5.00001])
    assert labels.tolist() == [
        "0_to_1.5",
        "1.5_to_2.5",
        "2.5_to_3.5",
        "3.5_to_cap",
        "censored_at_cap",
        "censored_at_cap",
    ]


def test_segment_table_counts_and_errors():
    table = metrics.segment_table([1.0, 1.0, 2.0], [2.0, 2.0, 2.0], ["a", "a", "b"])
    assert table["a"]["n"] == 2
    assert table["a"]["rmse"] == pytest.approx(1.0)
    assert table["b"]["rmse"] == pytest.approx(0.0)
