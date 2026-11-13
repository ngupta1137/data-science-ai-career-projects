import numpy as np
import pytest

from credit_score import config, metrics


def test_profit_terms_hand_computed():
    approve = np.array([1, 1, 0])
    y = np.array([0, 1, 1])
    assert metrics.profit_terms(approve, y).tolist() == pytest.approx(
        [config.MARGIN, -config.LOSS, 0.0]
    )


def test_threshold_is_break_even():
    p = config.APPROVE_BELOW
    assert (1 - p) * config.MARGIN - p * config.LOSS == pytest.approx(0.0)


def test_gini_and_calibration():
    y = np.array([0, 0, 1, 1])
    assert metrics.gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.calibration_ratio(np.array([0.25, 0.25, 0.25, 0.25]), y) == pytest.approx(0.5)


def test_boot_ci_of_constant(rng):
    assert metrics.boot_ci(np.ones(5), rng, 20)["ci95"] == [1.0, 1.0]
