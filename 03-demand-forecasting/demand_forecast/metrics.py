"""Forecast error metrics, backtest origins, and block-bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(diff**2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent. Undefined where both values are zero, which are skipped."""
    a = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    denom = (np.abs(a) + np.abs(f)) / 2
    keep = denom > 0
    return float(np.mean(np.abs(a - f)[keep] / denom[keep]) * 100) if keep.any() else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean of forecast minus actual. Positive means over-forecasting."""
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def seasonal_naive_scale(history, period: int) -> float:
    """In-sample MAE of the seasonal naive forecast. The denominator of MASE."""
    history = np.asarray(history, float)
    if len(history) <= period:
        raise ValueError("history must be longer than the period")
    return float(np.mean(np.abs(history[period:] - history[:-period])))


def mase(y_true, y_pred, scale: float) -> float:
    """MAE divided by the in-sample seasonal naive MAE. Below 1 beats that benchmark."""
    return mae(y_true, y_pred) / scale


def rolling_origins(n_rows: int, initial: int, step: int, horizon: int) -> list[int]:
    """Forecast origins. The block y[o : o + horizon] is forecast from y[:o]."""
    return list(range(initial, n_rows - horizon + 1, step))


def bootstrap_blocks_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_blocks: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling whole forecast blocks.

    Days inside one 7 day block are correlated, so blocks are the unit of resampling.
    stat_fn takes an array of block indices.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(rng.integers(0, n_blocks, size=n_blocks))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def segment_table(y_true, y_pred, labels) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    labels = np.asarray(labels)
    table = {}
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        table[str(name)] = {
            "n": int(mask.sum()),
            "mae": mae(y_true[mask], y_pred[mask]),
            "bias": bias(y_true[mask], y_pred[mask]),
        }
    return table
