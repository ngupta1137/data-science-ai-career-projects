import numpy as np
import pytest

from uplift_eval import metrics


def test_qini_curve_hand_example():
    score = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 0])
    x, q = metrics.qini_curve(score, t, y)
    assert x.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    # one treated buyer, no control buyers, so the cumulative gain is 1 buyer, i.e. 0.25 of 4 customers
    assert q.tolist() == pytest.approx([0.0, 0.25, 0.25, 0.25, 0.25])
    assert metrics.auuc(x, q) == pytest.approx(0.21875)
    # random targeting is the straight line to the same end point: area 0.5 * 0.25
    assert metrics.qini_coefficient(score, t, y) == pytest.approx(0.21875 - 0.125)


def test_qini_uses_the_treated_to_control_ratio():
    score = np.array([3.0, 2.0, 1.0])
    t = np.array([1, 1, 0])
    y = np.array([1, 1, 1])
    _, q = metrics.qini_curve(score, t, y)
    # at k=3: treated buyers 2, control buyers 1 scaled by n_t / n_c = 2 -> 2 - 2 = 0
    assert q[-1] == pytest.approx(0.0)


def test_a_perfect_ranking_beats_a_reversed_one():
    rng = np.random.default_rng(0)
    n = 20000
    tau = rng.uniform(-0.2, 0.4, n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + t * tau).astype(int)
    good = metrics.qini_coefficient(tau, t, y)
    bad = metrics.qini_coefficient(-tau, t, y)
    assert good > 0.01 > -0.01 > bad


def test_uninformative_score_has_a_qini_near_zero():
    rng = np.random.default_rng(1)
    n = 20000
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + 0.1 * t).astype(int)
    assert abs(metrics.qini_coefficient(rng.random(n), t, y)) < 0.005


def test_ipw_value_hand_example():
    policy = np.array([1, 1, 0, 0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 1, 1])
    terms = metrics.ipw_value(policy, t, y, np.full(4, 0.5), value=10.0, cost=1.0)
    # row 0: 10 * 1 * (1 / 0.5) - 1 = 19; row 1: 10 * 0 * (-1 / 0.5) - 1 = -1; others are untreated
    assert terms.tolist() == pytest.approx([19.0, -1.0, 0.0, 0.0])
    assert terms.mean() == pytest.approx(4.5)


def test_ipw_is_unbiased_in_a_simulated_trial():
    rng = np.random.default_rng(2)
    n = 400000
    tau = rng.uniform(-0.1, 0.3, n)
    base = 0.2
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < base + t * tau).astype(int)
    policy = tau > 0.1
    est = metrics.ipw_value(policy, t, y, np.full(n, 0.5), 50.0, 2.0).mean()
    assert est == pytest.approx(metrics.true_value(policy, tau, 50.0, 2.0), abs=0.1)


def test_true_value():
    assert metrics.true_value(
        np.array([1, 0, 1]), np.array([0.1, 0.5, 0.0]), 50.0, 2.0
    ) == pytest.approx((3.0 + -2.0) / 3)


def test_top_share():
    pol = metrics.top_share(np.array([0.1, 0.9, 0.5, 0.3]), 0.5)
    assert pol.tolist() == [False, True, True, False]


def test_boot_ci_brackets_the_mean():
    x = np.random.default_rng(0).standard_normal(500) + 2
    lo, hi = metrics.boot_mean_ci(x, np.random.default_rng(1), 200)
    assert lo < x.mean() < hi
    lo2, hi2 = metrics.boot_ci(lambda i: x[i].mean(), len(x), np.random.default_rng(1), 200)
    assert lo2 < x.mean() < hi2
