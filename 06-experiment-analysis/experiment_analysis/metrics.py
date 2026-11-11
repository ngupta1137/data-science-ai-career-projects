"""Uncertainty and design formulas."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal interval for the mean of simulation outputs."""
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return m - z * se, m + z * se


def variance_ratio_ci(
    num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Var(num) / Var(den) across experiments, with a bootstrap interval over experiments.

    Resampling keeps num and den paired, because both come from the same experiment.
    """
    rng = np.random.default_rng(seed)
    n = len(num)
    point = float(np.var(num, ddof=1) / np.var(den, ddof=1))
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ratios[b] = np.var(num[idx], ddof=1) / np.var(den[idx], ddof=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return point, float(low), float(high)


def n_per_arm(effect: float, power: float = 0.8, alpha: float = 0.05, variance_factor: float = 1.0):
    """Sample size per arm for a two-sided z test, outcome variance 1.

    n = 2 * (z_{1-alpha/2} + z_{power})^2 * variance_factor / effect^2.
    CUPED multiplies the outcome variance by (1 - rho^2), so pass that as variance_factor.
    """
    if effect <= 0:
        raise ValueError("effect must be positive")
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2.0 * (z_a + z_b) ** 2 * variance_factor / effect**2))
