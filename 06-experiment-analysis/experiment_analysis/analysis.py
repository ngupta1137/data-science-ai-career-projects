"""Estimators. All take arrays of shape (experiments, users) and return one value per experiment."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diff_and_se(y_t: np.ndarray, y_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Difference in means and its Welch standard error."""
    n_t, n_c = y_t.shape[-1], y_c.shape[-1]
    diff = y_t.mean(-1) - y_c.mean(-1)
    se = np.sqrt(y_t.var(-1, ddof=1) / n_t + y_c.var(-1, ddof=1) / n_c)
    return diff, se


def cuped_adjust(
    x_t: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, y_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CUPED: y - theta * (x - mean(x)), theta = cov(x, y) / var(x) pooled over both arms.

    Valid only when x is fixed before treatment starts.
    """
    x = np.concatenate([x_t, x_c], axis=-1)
    y = np.concatenate([y_t, y_c], axis=-1)
    xm = x.mean(-1, keepdims=True)
    ym = y.mean(-1, keepdims=True)
    theta = ((x - xm) * (y - ym)).sum(-1, keepdims=True) / ((x - xm) ** 2).sum(-1, keepdims=True)
    return y_t - theta * (x_t - xm), y_c - theta * (x_c - xm)


def p_value(diff: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Two-sided p from a normal approximation."""
    return 2.0 * stats.norm.sf(np.abs(diff / se))


def z_crit(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def looks_z(y_t: np.ndarray, y_c: np.ndarray, n_looks: int) -> np.ndarray:
    """z statistic at `n_looks` equally spaced interim looks. Shape (experiments, looks).

    Look k uses the first k/n_looks of the users in each arm.
    """
    n = y_t.shape[-1]
    if n % n_looks:
        raise ValueError("n must be divisible by n_looks")
    idx = np.arange(1, n_looks + 1) * (n // n_looks)
    out = []
    for m in idx:
        diff, se = diff_and_se(y_t[..., :m], y_c[..., :m])
        out.append(diff / se)
    return np.stack(out, axis=-1)


def holm_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down. p has shape (experiments, metrics). Returns a boolean mask."""
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    sorted_p = np.take_along_axis(p, order, axis=-1)
    passes = sorted_p <= alpha / (m - np.arange(m))
    leading = np.cumprod(passes, axis=-1).astype(bool)
    reject = np.zeros_like(leading)
    np.put_along_axis(reject, order, leading, axis=-1)
    return reject


def bonferroni_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    return p <= alpha / p.shape[-1]
