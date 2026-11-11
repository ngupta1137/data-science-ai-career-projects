import numpy as np
import pytest

from llm_judge_eval import metrics


def test_wilson_matches_hand_value():
    lo, hi = metrics.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_empty_is_nan():
    lo, hi = metrics.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_kappa_hand_example():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    # observed 3/4, chance 0.5 * 0.25 + 0.5 * 0.75 = 0.5, so kappa = 0.25 / 0.5
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.5)


def test_kappa_perfect_and_constant():
    x = np.array([1, 0, 1, 0])
    assert metrics.cohen_kappa(x, x) == 1.0
    assert np.isnan(metrics.cohen_kappa(np.ones(4), np.ones(4)))


def test_agreement():
    assert metrics.agreement(np.array([1, 0, 1, 1]), np.array([1, 1, 1, 0])) == 0.5


def test_paired_bootstrap_of_identical_arrays_is_zero(rng):
    x = rng.random(200) < 0.5
    assert metrics.boot_diff_ci(x, x, rng, 50) == (0.0, 0.0)


def test_mean_ci_contains_mean(rng):
    x = (rng.random(500) < 0.7).astype(float)
    lo, hi = metrics.boot_mean_ci(x, rng, 200)
    assert lo < x.mean() < hi


def test_fit_bias_recovers_planted_coefficients(rng):
    n = 40000
    signal = rng.standard_normal(n)
    dl = rng.standard_normal(n)
    z = 0.4 + 1.5 * signal + 0.5 * dl
    pick = rng.random(n) < 1 / (1 + np.exp(-z))
    fit = metrics.fit_bias(pick, signal, dl)
    assert fit["position"] == pytest.approx(0.4, abs=0.06)
    assert fit["quality_slope"] == pytest.approx(1.5, abs=0.08)
    assert fit["length_slope"] == pytest.approx(0.5, abs=0.06)


def test_cluster_bias_ci_brackets_point(rng):
    n = 800
    sig = rng.standard_normal(n)
    dl = rng.standard_normal(n)
    pick0 = rng.random(n) < 1 / (1 + np.exp(-(0.3 + sig + 0.5 * dl)))
    pick1 = rng.random(n) < 1 / (1 + np.exp(-(0.3 - sig - 0.5 * dl)))
    out = metrics.cluster_bias_ci(pick0, pick1, sig, dl, rng, 40)
    for block in out.values():
        assert block["ci95"][0] <= block["value"] <= block["ci95"][1]
