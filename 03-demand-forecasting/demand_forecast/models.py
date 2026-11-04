"""Baselines and models. Baselines come first, on purpose.

Baselines need only the history, so they are plain functions of (history, feature block).
Learned models are scikit-learn estimators fitted on all rows before the forecast origin.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import HORIZON

BASELINES = ("baseline_naive_last", "baseline_seasonal_naive_7", "baseline_mean_28")
TUNED = ("ridge", "hist_gradient_boosting")


def baseline_forecast(name: str, history, block_features) -> np.ndarray:
    """Forecast one block of HORIZON days. history is the target up to the origin."""
    history = np.asarray(history, float)
    if name == "baseline_naive_last":
        return np.full(HORIZON, history[-1])
    if name == "baseline_seasonal_naive_7":
        return block_features[f"lag_{HORIZON}"].to_numpy(dtype=float)
    if name == "baseline_mean_28":
        return np.full(HORIZON, history[-28:].mean())
    raise KeyError(name)


def make_model(name: str, seed: int, params: dict):
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(**params))])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise KeyError(name)


def param_grid(name: str, fast: bool) -> list[dict]:
    if name == "ridge":
        return [{"alpha": a} for a in ([1.0] if fast else [0.1, 1.0, 10.0, 100.0])]
    if name == "hist_gradient_boosting":
        if fast:
            return [{"learning_rate": 0.1, "max_leaf_nodes": 8, "max_iter": 100}]
        return [
            {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 200, "min_samples_leaf": 10}
            for lr in (0.05, 0.1)
            for leaves in (4, 8)
        ]
    raise KeyError(name)
