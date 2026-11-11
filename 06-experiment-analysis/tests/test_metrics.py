import numpy as np
import pytest

from experiment_analysis.metrics import mean_ci, n_per_arm, variance_ratio_ci, wilson_ci


def test_wilson_zero_successes_hand_value():
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(1.959964**2 / (10 + 1.959964**2), abs=1e-5)  # 0.2775


def test_wilson_is_symmetric_at_half():
    low, high = wilson_ci(50, 100)
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-9)
    assert 0.40 < low < 0.41 and 0.59 < high < 0.60


def test_wilson_rejects_empty():
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


def test_n_per_arm_hand_value():
    # 2 * (1.959964 + 0.841621)^2 / 0.01 = 1569.8, rounded up
    assert n_per_arm(0.1) == 1570


def test_n_per_arm_scales_with_variance_and_effect():
    assert n_per_arm(0.1, variance_factor=0.64) == 1005
    assert n_per_arm(0.2) == pytest.approx(n_per_arm(0.1) / 4, abs=1)
    with pytest.raises(ValueError):
        n_per_arm(0.0)


def test_mean_ci_contains_mean_and_is_ordered():
    values = np.arange(100, dtype=float)
    low, high = mean_ci(values)
    assert low < values.mean() < high


def test_variance_ratio_ci_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    den = rng.normal(0, 1, 500)
    num = 0.5 * den
    point, low, high = variance_ratio_ci(num, den, n_boot=200, seed=1)
    assert point == pytest.approx(0.25)
    assert low == pytest.approx(0.25) and high == pytest.approx(0.25)  # exactly paired
    assert variance_ratio_ci(num, den, 200, 1) == (point, low, high)
