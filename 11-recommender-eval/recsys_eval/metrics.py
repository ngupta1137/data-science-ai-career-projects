"""Ranking metrics on a dense score matrix, true-value metrics, and exposure statistics.

scores: (users, items) floats. seen: bool matrix of items to exclude (already clicked).
relevant: bool matrix of held-out clicks. avail: bool vector of items that exist in the window.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TIE_NOISE = 1e-10


def topk(
    scores: np.ndarray,
    seen: np.ndarray,
    avail: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices of the top k unseen, available items per user, best first. Exact ties break at random."""
    s = scores + TIE_NOISE * rng.random(scores.shape)
    s = np.where(seen | ~avail[None, :], -np.inf, s)
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    order = np.argsort(-s[rows, part], axis=1)
    return part[rows, order]


def _gains(top: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    return relevant[np.arange(len(top))[:, None], top].astype(float)


def ndcg_at_k(top: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Per-user NDCG. Binary relevance, log2 discount. Users with no relevant item get nan."""
    k = top.shape[1]
    disc = 1.0 / np.log2(np.arange(k) + 2)
    dcg = (_gains(top, relevant) * disc).sum(axis=1)
    n_rel = relevant.sum(axis=1)
    ideal = np.cumsum(disc)[np.clip(n_rel, 1, k) - 1]
    return np.where(n_rel > 0, dcg / ideal, np.nan)


def recall_at_k(top: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    n_rel = relevant.sum(axis=1)
    return np.where(n_rel > 0, _gains(top, relevant).sum(axis=1) / np.maximum(n_rel, 1), np.nan)


def hit_rate_at_k(top: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    n_rel = relevant.sum(axis=1)
    return np.where(n_rel > 0, (_gains(top, relevant).sum(axis=1) > 0).astype(float), np.nan)


def true_like_rate(top: np.ndarray, true_prob: np.ndarray) -> np.ndarray:
    """Mean true probability that the user likes each recommended item. Needs the simulator."""
    return true_prob[np.arange(len(top))[:, None], top].mean(axis=1)


def sampled_hit_rate(
    scores: np.ndarray,
    relevant: np.ndarray,
    seen: np.ndarray,
    avail: np.ndarray,
    k: int,
    n_neg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hit rate at k when each held-out click is ranked against n_neg random items (per user mean).

    This is the popular shortcut. It is cheap and it inflates the metric and can reorder models.
    """
    users, items = np.nonzero(relevant)
    pool = np.nonzero(avail)[0]
    negs = pool[rng.integers(0, len(pool), size=(len(users), n_neg))]
    blocked = seen[users[:, None], negs] | relevant[users[:, None], negs]
    s = scores + TIE_NOISE * rng.random(scores.shape)
    pos = s[users, items]
    neg = np.where(blocked, -np.inf, s[users[:, None], negs])
    rank = (neg > pos[:, None]).sum(axis=1)
    hit = (rank < k).astype(float)
    out = np.full(relevant.shape[0], np.nan)
    for u in np.unique(users):
        out[u] = hit[users == u].mean()
    return out


def coverage(top: np.ndarray, n_avail: int) -> float:
    """Share of available items recommended to at least one user."""
    return float(len(np.unique(top)) / n_avail)


def gini(top: np.ndarray, n_items: int) -> float:
    """Inequality of how often items are recommended. 0 is even, 1 is one item for everyone."""
    counts = np.sort(np.bincount(top.ravel(), minlength=n_items).astype(float))
    if counts.sum() == 0:
        return float("nan")
    n = len(counts)
    cum = np.cumsum(counts)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator, boots: int) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(boots, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, boots: int):
    """Paired bootstrap over users for mean(a) - mean(b)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    idx = rng.integers(0, len(d), size=(boots, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
