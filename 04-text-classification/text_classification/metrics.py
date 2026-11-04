"""Classification metrics, confusion analysis, and bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of per-class F1. Classes with no predictions score zero."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class(y_true, y_pred, classes) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        str(c): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }


def confusion(y_true, y_pred, classes) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=classes)


def top_confused_pairs(matrix: np.ndarray, classes, k: int = 10) -> list[dict]:
    """Largest off-diagonal cells: documents of class `true` predicted as class `predicted`."""
    off = matrix.copy().astype(int)
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    rows = []
    for index in flat:
        i, j = np.unravel_index(index, off.shape)
        if off[i, j] > 0:
            rows.append(
                {"true": str(classes[i]), "predicted": str(classes[j]), "count": int(off[i, j])}
            )
    return rows


def word_counts(texts) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts])


def length_buckets(counts: np.ndarray, edges) -> np.ndarray:
    names = np.array(["short", "medium", "long"])
    return names[np.digitize(counts, edges)]


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_rows: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling documents. stat_fn takes an index array.

    Documents are treated as independent. Crossposted and quoted posts are not, so the
    interval is somewhat too narrow.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(rng.integers(0, n_rows, size=n_rows))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def top_features(pipeline, classes, k: int = 8) -> dict:
    """Highest-weight tokens per class for a fitted TF-IDF pipeline."""
    vocab = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    clf = pipeline.named_steps["clf"]
    weights = clf.feature_log_prob_ if hasattr(clf, "feature_log_prob_") else clf.coef_
    if weights.shape[0] == 1 and len(classes) == 2:
        weights = np.vstack([-weights[0], weights[0]])
    out = {}
    for i, c in enumerate(classes):
        out[str(c)] = vocab[np.argsort(weights[i])[::-1][:k]].tolist()
    return out
