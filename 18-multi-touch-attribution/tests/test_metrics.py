import pytest

from touch_attribution import config, data, metrics


def test_share_mae_hand_computed():
    est = {"display": 0.20, "social": 0.30, "search": 0.40, "email": 0.10}
    truth = {"display": 0.25, "social": 0.25, "search": 0.35, "email": 0.15}
    # Errors: |0.20-0.25|=0.05, |0.30-0.25|=0.05, |0.40-0.35|=0.05, |0.10-0.15|=0.05 -> mean 0.05
    mae = metrics.channel_shares_mae(est, truth)
    assert mae == pytest.approx(0.05, rel=1e-6)


def test_spearman_correlation_ordering():
    truth = {"display": 0.10, "social": 0.20, "search": 0.30, "email": 0.40}
    perfect = {"display": 0.15, "social": 0.22, "search": 0.28, "email": 0.35}
    assert metrics.channel_shares_spearman(perfect, truth) == pytest.approx(1.0, rel=1e-5)

    reversed_order = {"display": 0.40, "social": 0.30, "search": 0.20, "email": 0.10}
    assert metrics.channel_shares_spearman(reversed_order, truth) == pytest.approx(-1.0, rel=1e-5)


def test_budget_reallocation_structure():
    ds = data.generate_journeys(n_journeys=100, seed=42)
    equal_shares = dict.fromkeys(config.CHANNELS, 0.25)
    res = metrics.evaluate_budget_reallocation(equal_shares, ds, total_budget=10_000.0)

    assert res["total_budget"] == 10_000.0
    assert res["incremental_conversions"] > 0.0
    for ch in config.CHANNELS:
        assert res["channel_spend"][ch] == pytest.approx(2500.0, rel=1e-3)
        assert res["channel_conversions"][ch] >= 0.0
