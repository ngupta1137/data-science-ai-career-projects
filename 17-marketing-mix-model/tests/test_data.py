import numpy as np
import pytest

from marketing_mix import config, data


def test_generate_shapes_and_types(rng):
    n_weeks = 52
    d = data.generate(rng, n_weeks)
    assert len(d) == n_weeks
    assert len(d.revenue) == n_weeks
    assert len(d.true_base) == n_weeks
    assert len(d.true_incremental) == n_weeks

    for ch in config.CHANNELS:
        assert ch in d.spend
        assert len(d.spend[ch]) == n_weeks
        assert np.all(d.spend[ch] > 0)
        assert ch in d.true_effects
        assert len(d.true_effects[ch]) == n_weeks
        assert np.all(d.true_effects[ch] > 0)

    assert np.all(d.revenue > 0)


def test_adstock_weights_sum_to_one():
    for alpha in [0.1, 0.5, 0.8]:
        w = np.array([alpha**lag for lag in range(13)])
        w_norm = w / np.sum(w)
        assert np.sum(w_norm) == pytest.approx(1.0)


def test_hill_saturation_properties():
    half_sat = 20_000.0
    max_lift = 30_000.0
    # At zero spend, return is 0
    assert data.hill_saturation(np.array([0.0]), half_sat, max_lift)[0] == 0.0
    # At half_sat spend, return is 0.5 * max_lift
    assert data.hill_saturation(np.array([half_sat]), half_sat, max_lift)[0] == pytest.approx(
        0.5 * max_lift
    )
    # Strictly concave (diminishing marginal returns)
    x1, x2, x3 = 10_000.0, 20_000.0, 30_000.0
    y = data.hill_saturation(np.array([x1, x2, x3]), half_sat, max_lift)
    assert (y[1] - y[0]) > (y[2] - y[1])


def test_tv_search_spend_correlation(rng):
    d = data.generate(rng, 104)
    corr = np.corrcoef(d.spend["tv"], d.spend["search"])[0, 1]
    assert 0.40 <= corr <= 0.85


def test_validate_invariants(rng):
    d = data.generate(rng, 52)
    data.validate(d)

    # Negative spend raises error
    d_bad = data.MarketingData(
        weeks=d.weeks,
        spend={**d.spend, "tv": -d.spend["tv"]},
        revenue=d.revenue,
        true_base=d.true_base,
        true_effects=d.true_effects,
        true_incremental=d.true_incremental,
    )
    with pytest.raises(ValueError, match="Negative spend"):
        data.validate(d_bad)

    # NaN revenue raises error
    bad_rev = d.revenue.copy()
    bad_rev[0] = np.nan
    d_nan = data.MarketingData(
        weeks=d.weeks,
        spend=d.spend,
        revenue=bad_rev,
        true_base=d.true_base,
        true_effects=d.true_effects,
        true_incremental=d.true_incremental,
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        data.validate(d_nan)


def test_fingerprint_deterministic():
    fp1 = data.fingerprint(seed=42)
    fp2 = data.fingerprint(seed=42)
    assert fp1 == fp2
    assert len(fp1) == 64
