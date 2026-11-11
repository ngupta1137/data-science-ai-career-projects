"""Drift statistics, per-batch, and the interval helper."""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config
from .data import N_BINS, Reference

EPS = 1e-4


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def psi(ref_props: np.ndarray, props: np.ndarray) -> np.ndarray:
    """Population stability index: sum (p - q) * ln(p / q). Shares are floored at 1e-4."""
    q = np.clip(ref_props, EPS, None)
    p = np.clip(props, EPS, None)
    return ((p - q) * np.log(p / q)).sum(axis=-1)


def ks_statistic(sorted_ref: np.ndarray, batch_sorted: np.ndarray) -> np.ndarray:
    """Two-sample Kolmogorov-Smirnov statistic. batch_sorted: (weeks, size), sorted per row."""
    n = batch_sorted.shape[-1]
    f_ref = np.searchsorted(sorted_ref, batch_sorted, side="right") / len(sorted_ref)
    upper = np.arange(1, n + 1) / n
    return np.maximum((upper - f_ref).max(-1), (f_ref - (upper - 1.0 / n)).max(-1))


def stream_stats(ref: Reference, stream: dict, lag: int = config.LABEL_LAG) -> dict:
    """One value per week for each monitor. NaN where a statistic is not available.

    auc_drop uses labels, which arrive `lag` weeks late, so batches in the last `lag`
    weeks have no observable value.
    """
    x, cat, y = stream["x"], stream["cat"], stream["y"]
    weeks, size, _ = x.shape
    psis, kss = [], []
    bins = np.arange(N_BINS)
    for j in range(config.N_NUM):
        idx = np.searchsorted(ref.edges[j], x[:, :, j])
        counts = (idx[:, :, None] == bins).sum(1) / size
        psis.append(psi(ref.props[j], counts))
        kss.append(ks_statistic(ref.sorted_ref[j], np.sort(x[:, :, j], axis=1)))
    cat_counts = (cat[:, :, None] == np.arange(3)).sum(1) / size
    psis.append(psi(ref.cat_props, cat_counts))

    score = ref.score(x, cat)
    sidx = np.searchsorted(ref.score_edges, score)
    score_counts = (sidx[:, :, None] == bins).sum(1) / size
    n_pos = y.sum(-1)
    n_neg = size - n_pos
    ranks = stats.rankdata(score, axis=-1)
    auc = ((ranks * y).sum(-1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc_drop = ref.ref_auc - auc
    if lag > 0:
        auc_drop = auc_drop.copy()
        auc_drop[weeks - lag :] = np.nan
    return {
        "psi_max": np.max(psis, axis=0),
        "ks_max": np.max(kss, axis=0),
        "score_psi": psi(ref.score_props, score_counts),
        "auc_drop": auc_drop,
    }
