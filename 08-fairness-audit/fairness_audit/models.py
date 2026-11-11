"""Scoring models and decision rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config


def fit_scorer(df: pd.DataFrame, features: list[str] | tuple[str, ...], seed: int = config.SEED):
    """Logistic regression on the given columns. Returns a function df -> probability."""
    cols = list(features)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(df[cols], df["y"])
    return lambda frame: model.predict_proba(frame[cols])[:, 1]


def utility(y: np.ndarray, approve: np.ndarray, cost_bad: float = config.COST_BAD) -> float:
    """Mean utility per applicant: +benefit for an approved good, -cost for an approved bad."""
    return float(
        (config.BENEFIT_GOOD * (approve & (y == 1)) - cost_bad * (approve & (y == 0))).mean()
    )


def best_threshold(
    score: np.ndarray, y: np.ndarray, cost_bad: float = config.COST_BAD, n_grid: int = 200
) -> float:
    """Single threshold that maximizes utility on the given rows. Ties take the lower value."""
    grid = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_grid)))
    values = [utility(y, score >= t, cost_bad) for t in grid]
    return float(grid[int(np.argmax(values))])


def group_thresholds(score: np.ndarray, y: np.ndarray, g: np.ndarray, target_tpr: float) -> dict:
    """Per-group thresholds so that each group's true positive rate is at least `target_tpr`.

    Equal opportunity post-processing. The threshold is the lower quantile of the group's
    positive scores, chosen on the rows passed in (use dev, never test).
    """
    out = {}
    for gval in (0, 1):
        pos = score[(g == gval) & (y == 1)]
        out[gval] = float(np.quantile(pos, 1.0 - target_tpr, method="lower"))
    return out


def apply_thresholds(score: np.ndarray, g: np.ndarray, thresholds: dict) -> np.ndarray:
    t = np.where(g == 1, thresholds[1], thresholds[0])
    return score >= t
