"""Store-level profit, three estimators of incremental profit, and the call they support."""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config
from .data import Campaign, weeks


def store_profit(c: Campaign, window: np.ndarray, include_sibling: bool) -> np.ndarray:
    """Observed profit per store over the given weeks. Promoted stores pay the discount in promo weeks."""
    promo = np.isin(window, weeks("promo"))
    unit_margin = config.MARGIN_ITEM - config.DISCOUNT * (
        promo[None, :] & (c.treated[:, None] == 1)
    )
    total = (unit_margin * c.item[:, window]).sum(1)
    if include_sibling:
        total = total + config.MARGIN_SIBLING * c.sibling[:, window].sum(1)
    return total


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference in means a minus b with a Welch 95% interval."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    denom = va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)
    q = float(stats.t.ppf(0.975, (va + vb) ** 2 / denom)) if denom > 0 else 1.959963984540054
    est = float(a.mean() - b.mean())
    return {"value": est, "se": se, "ci95": [est - q * se, est + q * se]}


def one_sample(a: np.ndarray) -> dict:
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    q = float(stats.t.ppf(0.975, len(a) - 1))
    est = float(a.mean())
    return {"value": est, "se": se, "ci95": [est - q * se, est + q * se]}


def estimate(c: Campaign, name: str) -> dict:
    """Incremental profit per promoted store over the campaign."""
    pre, promo, post = weeks("pre"), weeks("promo"), weeks("post")
    t, ctl = c.treated == 1, c.treated == 0
    if name == "pre_post":
        # Promoted stores only, promo weeks against the weeks before, promoted item only.
        d = store_profit(c, promo, False) - len(promo) / len(pre) * store_profit(c, pre, False)
        return one_sample(d[t])
    if name == "window_did":
        # Control stores remove the season, but only the promoted item in promo weeks is counted.
        d = store_profit(c, promo, False) - len(promo) / len(pre) * store_profit(c, pre, False)
        return welch(d[t], d[ctl])
    if name == "full_did":
        window = np.concatenate([promo, post])
        d = store_profit(c, window, True) - len(window) / len(pre) * store_profit(c, pre, True)
        return welch(d[t], d[ctl])
    raise ValueError(name)


def call(ci: list[float]) -> str:
    """'run it' if the interval is above zero, 'stop it' if below, 'unclear' otherwise."""
    if ci[0] > 0:
        return "run it"
    if ci[1] < 0:
        return "stop it"
    return "unclear"


def covers(ci: list[float], truth: float) -> bool:
    return bool(ci[0] <= truth <= ci[1])
