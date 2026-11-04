"""Models and search grids. The baseline comes first, on purpose.

No class weighting or resampling is used. Both distort predicted probabilities, and the
cost policy in this project needs probabilities that mean something.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE = "baseline_prior"
TUNED = ("logistic", "hist_gradient_boosting")


def make_models(seed: int) -> dict:
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "logistic": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def param_grid(name: str, fast: bool) -> dict:
    if name == "logistic":
        return {"model__C": [1.0] if fast else [0.01, 0.1, 1.0, 10.0]}
    if name == "hist_gradient_boosting":
        if fast:
            return {"learning_rate": [0.1], "max_leaf_nodes": [15], "l2_regularization": [0.0]}
        return {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
        }
    raise KeyError(name)
