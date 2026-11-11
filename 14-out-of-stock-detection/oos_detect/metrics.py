"""Net value of a visit policy and store-level bootstrap intervals."""

from __future__ import annotations

import numpy as np

from . import config


def net_value_per_store_day(
    base_lost: np.ndarray, out: dict, store: np.ndarray, n_stores: int
) -> np.ndarray:
    """Margin saved minus visit cost, per store per evaluation day."""
    per_shelf = (base_lost - out["lost_margin"]) - config.VISIT_COST * out["visits"]
    return np.bincount(store, per_shelf, minlength=n_stores) / config.EVAL_DAYS


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"value": float(x.mean()), "ci95": [float(lo), float(hi)]}


def precision(out: dict) -> float:
    v = out["visits"].sum()
    return float(out["useful_visits"].sum() / v) if v else float("nan")


def recall(out: dict) -> float:
    e = out["episodes"].sum()
    return float(out["fixed"].sum() / e) if e else float("nan")
