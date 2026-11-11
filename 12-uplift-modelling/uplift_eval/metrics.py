"""Uplift curve, Qini coefficient, inverse-probability-weighted profit, and true policy value."""

from __future__ import annotations

import numpy as np


def qini_curve(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental buyers when customers are treated in order of score, best first.

    Q(k) = buyers among treated in the top k minus buyers among controls in the top k, scaled by
    the ratio of treated to control counts. Returns (fraction treated, Q / n) including (0, 0).
    """
    order = np.argsort(-score, kind="stable")
    t_o, y_o = t[order], y[order]
    n_t = np.cumsum(t_o)
    n_c = np.cumsum(1 - t_o)
    y_t = np.cumsum(y_o * t_o)
    y_c = np.cumsum(y_o * (1 - t_o))
    ratio = np.divide(n_t, n_c, out=np.zeros(len(t_o)), where=n_c > 0)
    q = y_t - y_c * ratio
    n = len(score)
    return np.concatenate([[0.0], np.arange(1, n + 1) / n]), np.concatenate([[0.0], q / n])


def auuc(x: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (q[1:] + q[:-1]) / 2.0))


def qini_coefficient(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Area under the uplift curve minus the area under the straight line of random targeting."""
    x, q = qini_curve(score, t, y)
    return auuc(x, q) - 0.5 * q[-1]


def ipw_value(
    policy: np.ndarray, t: np.ndarray, y: np.ndarray, propensity, value: float, cost: float
) -> np.ndarray:
    """Per-customer terms whose mean is the incremental profit of the policy over treating nobody.

    Valid when propensity is the true chance of being offered (a randomised trial).
    """
    pi = np.asarray(policy, float)
    w = t / propensity - (1 - t) / (1 - propensity)
    return pi * (value * y * w - cost)


def true_value(policy: np.ndarray, tau: np.ndarray, value: float, cost: float) -> float:
    """Exact incremental profit per customer, from the simulator's true effects."""
    return float(np.mean(np.asarray(policy, float) * (value * tau - cost)))


def top_share(score: np.ndarray, share: float) -> np.ndarray:
    """Policy that treats the top `share` of customers by score."""
    k = round(share * len(score))
    pol = np.zeros(len(score), bool)
    pol[np.argsort(-score, kind="stable")[:k]] = True
    return pol


def boot_ci(stat, n: int, rng: np.random.Generator, boots: int) -> tuple[float, float]:
    """Percentile interval of stat(index) over bootstrap resamples of row indices."""
    vals = [stat(rng.integers(0, n, n)) for _ in range(boots)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator, boots: int) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(boots, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
