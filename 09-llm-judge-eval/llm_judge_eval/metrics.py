"""Agreement, kappa, bootstrap intervals, and the bias regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

UNPENALIZED = 1e10


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def agreement(decision: np.ndarray, label: np.ndarray) -> float:
    return float((decision == label).mean())


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters."""
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def _indices(rng: np.random.Generator, n: int, boots: int) -> np.ndarray:
    return rng.integers(0, n, size=(boots, n))


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator, boots: int) -> tuple[float, float]:
    idx = _indices(rng, len(x), boots)
    means = x.astype(float)[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, boots: int
) -> tuple[float, float]:
    """Paired bootstrap interval for mean(a) - mean(b) over the same rows."""
    idx = _indices(rng, len(a), boots)
    diffs = (a.astype(float) - b.astype(float))[idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def fit_bias(pick_first: np.ndarray, signal: np.ndarray, dl: np.ndarray) -> dict:
    """Logistic fit of 'the judge picked the first slot'.

    logit P(first) = position + quality_slope * signal + length_slope * dl, where signal and dl
    are first-slot minus second-slot. The intercept is the slot bias; it is zero for a judge that
    ignores slot. Effectively no penalty (C is huge), so the coefficients are the plain maximum likelihood fit.
    """
    x = np.column_stack([signal, dl])
    model = LogisticRegression(C=UNPENALIZED, max_iter=1000).fit(x, pick_first.astype(int))
    return {
        "position": float(model.intercept_[0]),
        "quality_slope": float(model.coef_[0][0]),
        "length_slope": float(model.coef_[0][1]),
    }


def cluster_bias_ci(
    pick0: np.ndarray,
    pick1: np.ndarray,
    signal0: np.ndarray,
    dl0: np.ndarray,
    rng: np.random.Generator,
    boots: int,
) -> dict:
    """Fit on both orders of every pair, interval by resampling pairs (the two calls travel together).

    Order 1 shows Y first, so its signal and length difference are the negatives of order 0.
    """
    n = len(pick0)
    pick = np.concatenate([pick0, pick1])
    sig = np.concatenate([signal0, -signal0])
    dl = np.concatenate([dl0, -dl0])
    point = fit_bias(pick, sig, dl)
    draws = {k: [] for k in point}
    for _ in range(boots):
        i = rng.integers(0, n, n)
        if len(np.unique(np.concatenate([pick0[i], pick1[i]]))) < 2:
            continue  # a resample with one outcome cannot be fitted
        fit = fit_bias(
            np.concatenate([pick0[i], pick1[i]]),
            np.concatenate([signal0[i], -signal0[i]]),
            np.concatenate([dl0[i], -dl0[i]]),
        )
        for k, v in fit.items():
            draws[k].append(v)
    return {
        k: {
            "value": point[k],
            "ci95": [float(np.quantile(draws[k], 0.025)), float(np.quantile(draws[k], 0.975))],
        }
        for k in point
    }
