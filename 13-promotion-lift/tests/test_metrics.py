import numpy as np
import pytest

from promo_lift import config, data, metrics


def test_welch_matches_hand_computation():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0])
    r = metrics.welch(a, b)
    assert r["value"] == pytest.approx(1.0)
    assert r["se"] == pytest.approx(np.sqrt(1 / 3 + 2 / 2))


def test_call_and_covers():
    assert metrics.call([1, 2]) == "run it"
    assert metrics.call([-2, -1]) == "stop it"
    assert metrics.call([-1, 1]) == "unclear"
    assert metrics.covers([-1, 1], 0.5) and not metrics.covers([-1, 1], 2)


def _fixed_campaign(item_value=10, sib_value=5):
    item = np.full((4, config.WEEKS), item_value)
    sib = np.full((4, config.WEEKS), sib_value)
    return data.Campaign(item, sib, np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.zeros(4), {})


def test_store_profit_charges_discount_only_to_promoted_stores_in_promo_weeks():
    c = _fixed_campaign()
    promo = data.weeks("promo")
    p = metrics.store_profit(c, promo, include_sibling=False)
    n = len(promo)
    assert p[0] == pytest.approx(n * 10 * (config.MARGIN_ITEM - config.DISCOUNT))
    assert p[2] == pytest.approx(n * 10 * config.MARGIN_ITEM)


def test_full_did_on_flat_sales_equals_the_discount_cost():
    # Flat sales, no lift: the only difference is the discount paid by promoted stores.
    c = _fixed_campaign()
    e = metrics.estimate(c, "full_did")
    assert e["value"] == pytest.approx(-config.DISCOUNT * 10 * config.PROMO_WEEKS)
