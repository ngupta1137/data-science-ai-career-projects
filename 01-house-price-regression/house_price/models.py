"""Models and their search grids. Baselines come first, on purpose."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ("baseline_mean", "baseline_median")
TUNED = ("ridge", "hist_gradient_boosting")


def make_models(seed: int) -> dict:
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "baseline_median": DummyRegressor(strategy="median"),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }


def param_grid(name: str, fast: bool) -> dict:
    if name == "ridge":
        return {"model__alpha": [1.0] if fast else [0.1, 1.0, 10.0, 100.0]}
    if name == "hist_gradient_boosting":
        if fast:
            return {"learning_rate": [0.1], "max_leaf_nodes": [15], "l2_regularization": [0.0]}
        return {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0],
        }
    raise KeyError(name)
