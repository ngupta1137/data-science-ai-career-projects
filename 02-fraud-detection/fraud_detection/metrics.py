"""Ranking, calibration, and cost metrics, plus bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision. A constant scorer gets the fraud rate."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def brier(y_true, probs) -> float:
    return float(brier_score_loss(y_true, probs))


def recall_at_top_fraction(y_true, scores, fraction: float) -> float:
    """Share of all fraud caught when the top `fraction` of transactions by score is flagged."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    k = int(np.ceil(fraction * len(scores)))
    order = np.argsort(-scores, kind="stable")[:k]
    total = y_true.sum()
    return float(y_true[order].sum() / total) if total else float("nan")


def policy_cost(y_true, amount, flag, review_cost: float) -> float:
    """Cost of a flagging policy.

    A flagged transaction costs the review cost whether or not it is fraud.
    An unflagged fraud costs its full amount. An unflagged legitimate one costs nothing.
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    missed = float(np.sum(amount[(y_true == 1) & (~flag)]))
    return missed + float(review_cost) * int(flag.sum())


def best_global_threshold(scores, y_true, amount, review_cost: float):
    """Exact cost-minimising threshold for 'flag if score >= t'. Returns (t, cost).

    t is +inf when flagging nothing is cheapest. Only thresholds that fall between
    distinct scores are considered, so tied scores are never split.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    fraud_amount = (amount * (y_true == 1))[order]
    total = float(fraud_amount.sum())
    caught = np.concatenate([[0.0], np.cumsum(fraud_amount)])
    k = np.arange(len(s) + 1)
    cost = total - caught + float(review_cost) * k
    valid = np.ones(len(s) + 1, dtype=bool)
    if len(s) > 1:
        valid[1:-1] = s[:-1] > s[1:]
    cost = np.where(valid, cost, np.inf)
    best = int(np.argmin(cost))
    threshold = float("inf") if best == 0 else float(s[best - 1])
    return threshold, float(cost[best])


def expected_cost_flag(probs, amount, review_cost: float) -> np.ndarray:
    """Per-transaction rule: flag when expected loss from not flagging is at least the review cost.

    This is only as good as the probabilities, which is why calibration matters here.
    """
    return np.asarray(probs, dtype=float) * np.asarray(amount, dtype=float) >= float(review_cost)


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(scores, y_true):
    """Two-parameter sigmoid calibration on logit(score). Returns a callable."""
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(_logit(scores).reshape(-1, 1), np.asarray(y_true))

    def apply(new_scores):
        return model.predict_proba(_logit(new_scores).reshape(-1, 1))[:, 1]

    return apply


def fit_isotonic(scores, y_true):
    """Monotone step calibration. Needs many positives; reported as an alternative only."""
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(scores, dtype=float), np.asarray(y_true, dtype=float))
    return lambda new_scores: model.predict(np.asarray(new_scores, dtype=float))


def calibration_ratio(y_true, probs) -> float:
    """Sum of predicted probabilities over the number of fraud rows. 1.0 is calibrated in the large."""
    total = float(np.sum(y_true))
    return float(np.sum(probs) / total) if total else float("nan")


def reliability_table(y_true, probs, n_bins: int = 10) -> list[dict]:
    """Equal-count bins by predicted probability. Equal-width bins are empty at fraud rates."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "n": len(chunk),
                "mean_predicted": float(probs[chunk].mean()),
                "observed_rate": float(y_true[chunk].mean()),
                "fraud_count": int(y_true[chunk].sum()),
            }
        )
    return rows


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_rows: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling rows. stat_fn takes an index array.

    Rows are treated as independent. Transactions on one card are not, so the
    interval is narrower than the true uncertainty. The model card says so.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        try:
            stats[b] = stat_fn(idx)
        except ValueError:
            stats[b] = np.nan
    low, high = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
