"""Profit of an approval rule, ranking quality, calibration and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config


def profit_terms(approve: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Profit per applicant: margin on repaid approved loans, loss on defaulted ones, zero if declined."""
    return approve * np.where(y == 1, -config.LOSS, config.MARGIN)


def gini(y: np.ndarray, score: np.ndarray) -> float:
    return float(2 * roc_auc_score(y, score) - 1)


def calibration_ratio(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean predicted default chance over observed default rate."""
    return float(pred.mean() / y.mean()) if y.mean() > 0 else float("nan")


def boot_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"value": float(x.mean()), "ci95": [float(lo), float(hi)]}
