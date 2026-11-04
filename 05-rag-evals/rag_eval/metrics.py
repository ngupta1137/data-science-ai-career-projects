"""Retrieval and answer metrics, plus bootstrap uncertainty."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def recall_at_k(ranked_ids: list[str], gold: list[str], k: int) -> float:
    """Share of gold documents found in the top k."""
    return len(set(ranked_ids[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranked_ids: list[str], gold: list[str]) -> float:
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold: list[str], k: int) -> float:
    """Binary relevance. 1.0 means every gold document is at the top."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked_ids[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_items: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling questions. stat_fn takes an index array.

    The question set is small and generated from templates, so the interval describes this
    question set only, not questions in general.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(rng.integers(0, n_items, size=n_items))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
