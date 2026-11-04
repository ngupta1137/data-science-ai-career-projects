"""Error metrics, uncertainty, and segment breakdowns."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import TARGET_CAP


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst == 0.0:
        return float("nan")
    return 1.0 - sse / sst


def bias(y_true, y_pred) -> float:
    """Mean of prediction minus truth. Positive means the model over-predicts."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def cluster_bootstrap_ci(
    y_true,
    y_pred,
    groups,
    metric: Callable,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval that resamples whole spatial cells, not single rows.

    Rows in one cell are correlated. Resampling rows would treat them as independent
    and give an interval that is too narrow.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[i] for i in picks])
        stats[b] = metric(y_true[idx], y_pred[idx])
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def target_band(y_true) -> np.ndarray:
    """Price bands in units of $100,000. Capped rows get their own band."""
    y_true = np.asarray(y_true, dtype=float)
    return np.select(
        [y_true >= TARGET_CAP, y_true > 3.5, y_true > 2.5, y_true > 1.5],
        ["censored_at_cap", "3.5_to_cap", "2.5_to_3.5", "1.5_to_2.5"],
        default="0_to_1.5",
    )


def segment_table(y_true, y_pred, labels, min_n: int = 1) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    labels = np.asarray(labels)
    table = {}
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        n = int(mask.sum())
        if n < min_n:
            continue
        table[str(name)] = {
            "n": n,
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mae": mae(y_true[mask], y_pred[mask]),
            "bias": bias(y_true[mask], y_pred[mask]),
        }
    return table
