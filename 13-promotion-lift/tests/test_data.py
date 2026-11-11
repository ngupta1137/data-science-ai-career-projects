import numpy as np
import pytest

from promo_lift import config, data


def test_shapes_and_validation(rng):
    c = data.generate(rng, 50)
    data.validate(c)
    assert c.item.shape == (50, config.WEEKS)
    assert c.treated.sum() == 25


def test_parts_sum_to_true_profit(rng):
    c = data.generate(rng, 30)
    total = sum(c.true_parts.values())
    assert np.allclose(total, c.true_profit)


def test_promotion_lifts_units_but_loses_money_in_this_setup(rng):
    c = data.generate(rng, 400)
    assert c.true_parts["extra_units_sold"].mean() > 0
    assert c.true_profit.mean() < 0


def test_season_peaks_in_promo_weeks():
    s = data.season()
    assert s[data.weeks("promo")].mean() > s[data.weeks("pre")].mean()


def test_picked_stores_have_higher_trend_sales(rng):
    c = data.generate(rng, 400, "picked")
    late, early = c.item[:, -8:].sum(1), c.item[:, :8].sum(1)
    growth = late / np.maximum(early, 1)
    assert growth[c.treated == 1].mean() > growth[c.treated == 0].mean()


def test_validate_rejects_one_arm(rng):
    c = data.generate(rng, 20)
    c.treated[:] = 1
    with pytest.raises(data.DataValidationError):
        data.validate(c)


def test_fingerprint_is_stable():
    assert data.fingerprint() == data.fingerprint()
