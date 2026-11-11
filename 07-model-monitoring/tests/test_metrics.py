import numpy as np
import pytest

from drift_monitor import data, metrics


def test_psi_is_zero_for_identical_distributions():
    p = np.full(10, 0.1)
    assert metrics.psi(p, p) == pytest.approx(0.0)


def test_psi_hand_value():
    ref = np.array([0.5, 0.5])
    cur = np.array([0.8, 0.2])
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert metrics.psi(ref, cur) == pytest.approx(expected)
    assert expected == pytest.approx(0.4159, abs=1e-4)  # 0.1410 + 0.2749


def test_psi_floors_empty_bins_instead_of_returning_infinity():
    value = metrics.psi(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert np.isfinite(value) and value > 1.0


def test_psi_is_symmetric():
    a, b = np.array([0.6, 0.4]), np.array([0.3, 0.7])
    assert metrics.psi(a, b) == pytest.approx(metrics.psi(b, a))


def test_ks_hand_example():
    ref = np.arange(1, 101) / 100.0  # uniform grid
    batch = np.sort(np.array([[0.9, 0.95, 0.99, 1.0]]))
    # ref CDF at those points is 0.9..1.0; batch ECDF is 0.25..1.0. Largest gap is at 0.9.
    # D = max(i/n - F_ref) = max(0.25-0.9, 0.5-0.95, 0.75-0.99, 1-1.0) is negative;
    # D = max(F_ref - (i-1)/n) = max(0.9-0, 0.95-0.25, 0.99-0.5, 1.0-0.75) = 0.9
    assert metrics.ks_statistic(ref, batch)[0] == pytest.approx(0.9)


def test_ks_matches_scipy_on_random_data(rng):
    from scipy import stats

    ref = np.sort(rng.standard_normal(5000))
    batch = np.sort(rng.standard_normal((1, 300)) + 0.3, axis=1)
    ours = metrics.ks_statistic(ref, batch)[0]
    assert ours == pytest.approx(stats.ks_2samp(batch[0], ref).statistic, abs=2e-3)


def test_wilson_matches_hand_value():
    low, high = metrics.wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.2775, abs=1e-3)
    with pytest.raises(ValueError):
        metrics.wilson_ci(0, 0)


def test_null_stream_statistics_are_small(ref, rng):
    st = metrics.stream_stats(ref, data.draw_stream(rng, "null"))
    assert np.nanmax(st["psi_max"]) < 0.15
    assert np.nanmax(st["ks_max"]) < 0.15
    assert abs(np.nanmean(st["auc_drop"])) < 0.02


def test_auc_drop_is_unavailable_in_the_last_lag_weeks(ref, rng):
    st = metrics.stream_stats(ref, data.draw_stream(rng, "null"), lag=4)
    assert np.isnan(st["auc_drop"][-4:]).all() and not np.isnan(st["auc_drop"][:-4]).any()
    assert not np.isnan(
        metrics.stream_stats(ref, data.draw_stream(rng, "null"), lag=0)["auc_drop"]
    ).any()
