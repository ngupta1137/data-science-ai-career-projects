"""Ways to turn raw judge calls into a verdict, plus reference deciders.

All deciders return True where they say X is better. v0 and v1 are the verdicts of the two
calls (X first, Y first) as "X is better".
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def swap_score(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """-1, 0 or +1: both calls say Y, the calls split, both calls say X."""
    return v0.astype(int) + v1.astype(int) - 1


def tie_rate(v0: np.ndarray, v1: np.ndarray) -> float:
    """Share of pairs where the verdict flips when the answers swap slots."""
    return float((v0 != v1).mean())


def decide_single(rng: np.random.Generator, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """One call per pair, with the order picked at random. The common shortcut."""
    first_x = rng.random(len(v0)) < 0.5
    return np.where(first_x, v0, v1)


def decide_swap(rng: np.random.Generator, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Call both orders. A split verdict is broken by a coin, which counts a tie as half a point.

    Warning: with independent calls this has the same expected agreement as one call in a random
    order, because the coin discards the information that the calls split. Swapping pays off only
    when the split is used, as an abstention (`decide_triage`) or as a feature (`decide_calibrated`).
    """
    s = swap_score(v0, v1)
    coin = rng.random(len(v0)) < 0.5
    return np.where(s == 0, coin, s > 0)


def fit_calibrator(
    v0: np.ndarray, v1: np.ndarray, dl: np.ndarray, label: np.ndarray
) -> LogisticRegression:
    """Learn from human labels how much to trust the swap score versus the length difference.

    No intercept: relabelling X and Y flips the features and the label, so the model must be odd.
    """
    x = np.column_stack([swap_score(v0, v1), dl])
    return LogisticRegression(C=1e10, fit_intercept=False, max_iter=1000).fit(x, label.astype(int))


def decide_calibrated(
    model: LogisticRegression, v0: np.ndarray, v1: np.ndarray, dl: np.ndarray
) -> np.ndarray:
    x = np.column_stack([swap_score(v0, v1), dl])
    return model.decision_function(x) > 0


def decide_triage(v0: np.ndarray, v1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verdict only where both orders agree. Returns (verdict, decided); split pairs go to a human."""
    return v0, v0 == v1


def decide_length_only(dl: np.ndarray) -> np.ndarray:
    return dl > 0


def decide_oracle(dq: np.ndarray) -> np.ndarray:
    return dq > 0


def decide_coin(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.random(n) < 0.5
