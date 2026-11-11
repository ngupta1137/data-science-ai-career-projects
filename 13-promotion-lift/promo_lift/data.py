"""A simulated grocery chain where the effect of a promotion is known exactly.

Each store sells a promoted item and a sibling item every week. Sales follow a store size, a
store trend, a shared season and noise. The promotion lifts the promoted item, takes some sales
from the sibling, and pulls some future sales forward. Because the code knows the expected sales
with and without the promotion, it knows the true incremental profit of every store.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


@dataclass
class Campaign:
    item: np.ndarray  # (stores, weeks) units sold of the promoted item
    sibling: np.ndarray  # (stores, weeks) units sold of the sibling item
    treated: np.ndarray  # (stores,) 1 if the store ran the promotion
    urban: np.ndarray  # (stores,) 1 for urban stores
    true_profit: np.ndarray  # (stores,) true incremental profit if the store is treated
    true_parts: dict  # name -> (stores,) parts of true_profit, which sum to it

    def __len__(self) -> int:
        return len(self.treated)


def weeks(name: str) -> np.ndarray:
    s, p = config.PROMO_START, config.PROMO_WEEKS
    if name == "pre":
        return np.arange(s - config.PRE_WEEKS, s)
    if name == "promo":
        return np.arange(s, s + p)
    if name == "post":
        return np.arange(s + p, s + p + config.POST_WEEKS)
    raise ValueError(name)


def season() -> np.ndarray:
    w = np.arange(config.WEEKS)
    phase = 2 * np.pi * (w - config.SEASON_PEAK_WEEK) / config.WEEKS
    return 1.0 + config.SEASON_AMPLITUDE * np.cos(phase)


def generate(rng: np.random.Generator, n_stores: int, assignment: str = "random") -> Campaign:
    """One campaign. assignment is 'random' (a trial) or 'picked' (a manager picks growing stores)."""
    w = np.arange(config.WEEKS)
    urban = (rng.random(n_stores) < config.URBAN_SHARE).astype(int)
    base = np.exp(rng.normal(config.BASE_LOG_MEAN, config.BASE_LOG_SD, n_stores))
    trend = rng.normal(config.TREND_MEAN, config.TREND_SD, n_stores)
    sib_scale = config.SIBLING_RATIO * np.exp(rng.normal(0, 0.2, n_stores))
    mu_item0 = base[:, None] * season()[None, :] * (1 + trend[:, None]) ** w[None, :]
    mu_sib0 = mu_item0 * sib_scale[:, None]

    if assignment == "random":
        treated = np.zeros(n_stores, int)
        treated[rng.permutation(n_stores)[: n_stores // 2]] = 1
    elif assignment == "picked":
        score = trend + rng.normal(0, config.TREND_SD / 2, n_stores)
        treated = (score >= np.median(score)).astype(int)
    else:
        raise ValueError(assignment)

    promo = np.isin(w, weeks("promo")).astype(float)
    post = np.isin(w, weeks("post")).astype(float)
    lift = np.where(urban == 1, config.LIFT["urban"], config.LIFT["rural"])
    mu_item1 = mu_item0 * (1 + lift[:, None] * promo) * (1 - config.PULL_FORWARD * post)
    mu_sib1 = mu_sib0 * (1 - config.CANNIBALISATION * promo)

    profit1 = ((config.MARGIN_ITEM - config.DISCOUNT * promo) * mu_item1).sum(1) + (
        config.MARGIN_SIBLING * mu_sib1
    ).sum(1)
    profit0 = (config.MARGIN_ITEM * mu_item0).sum(1) + (config.MARGIN_SIBLING * mu_sib0).sum(1)

    promo_units0 = (mu_item0 * promo).sum(1)
    parts = {
        "extra_units_sold": config.MARGIN_ITEM * ((mu_item1 - mu_item0) * promo).sum(1),
        "discount_on_extra_units": -config.DISCOUNT * ((mu_item1 - mu_item0) * promo).sum(1),
        "discount_on_units_that_would_have_sold": -config.DISCOUNT * promo_units0,
        "pull_forward": config.MARGIN_ITEM * ((mu_item1 - mu_item0) * post).sum(1),
        "cannibalisation": config.MARGIN_SIBLING * (mu_sib1 - mu_sib0).sum(1),
    }

    t = treated[:, None] == 1
    mu_item = np.where(t, mu_item1, mu_item0)
    mu_sib = np.where(t, mu_sib1, mu_sib0)
    k = 1.0 / config.NOISE_CV**2
    item = rng.poisson(mu_item * rng.gamma(k, 1 / k, mu_item.shape))
    sibling = rng.poisson(mu_sib * rng.gamma(k, 1 / k, mu_sib.shape))
    return Campaign(item, sibling, treated, urban, profit1 - profit0, parts)


def validate(c: Campaign) -> None:
    problems = []
    if c.item.shape != c.sibling.shape or c.item.shape[0] != len(c.treated):
        problems.append("shapes do not match")
    if c.item.shape[1] != config.WEEKS:
        problems.append(f"expected {config.WEEKS} weeks")
    if (c.item < 0).any() or (c.sibling < 0).any():
        problems.append("negative sales")
    if set(np.unique(c.treated)) != {0, 1}:
        problems.append("need both promoted and control stores")
    if problems:
        raise DataValidationError("; ".join(problems))


def fingerprint(seed: int = 0) -> str:
    c = generate(np.random.default_rng(seed), 20)
    blob = np.concatenate(
        [c.item.ravel(), c.sibling.ravel(), c.treated, np.round(c.true_profit, 6)]
    )
    return hashlib.sha256(blob.astype(float).tobytes()).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    if path.exists():
        expected = path.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"generator fingerprint mismatch: expected {expected}, got {digest}"
            )
    return digest
