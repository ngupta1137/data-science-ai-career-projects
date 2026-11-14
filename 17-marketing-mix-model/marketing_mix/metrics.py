"""Actuarial and econometric metrics for marketing mix modeling and budget optimization."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from . import config, data


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (0 to 1)."""
    mask = y_true != 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def spearman_rank_correlation(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def true_channel_steady_state_lift(channel: str, spend: float) -> float:
    """Compute true steady-state weekly incremental revenue under known Hill saturation."""
    p = config.TRUE_CHANNEL_PARAMS[channel]
    return float(data.hill_saturation(np.array([spend]), p["half_sat"], p["max_lift"])[0])


def evaluate_allocation_truth(allocation: dict[str, float]) -> float:
    """Evaluate true weekly incremental revenue across all channels under known truth."""
    return float(sum(true_channel_steady_state_lift(ch, spend) for ch, spend in allocation.items()))


def optimize_budget_mmm(
    channel_params: dict[str, dict[str, float]],
    total_budget: float,
    min_spend: float = 5_000.0,
) -> dict[str, float]:
    """Find budget allocation maximizing estimated steady-state incremental revenue.

    For each channel, estimated response is:
    f_c(w_c) = max_lift_c * (w_c / (w_c + K_c)).
    Subject to sum(w_c) == total_budget and w_c >= min_spend.
    """
    channels = list(channel_params.keys())
    n_ch = len(channels)

    def neg_return(w: np.ndarray) -> float:
        total = 0.0
        for i, ch in enumerate(channels):
            k = channel_params[ch]["half_sat"]
            m = channel_params[ch]["max_lift"]
            spend = w[i]
            total += m * (spend / (spend + k))
        return -total

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - total_budget}
    bnds = [(min_spend, total_budget - (n_ch - 1) * min_spend)] * n_ch
    init_w = np.array([total_budget / n_ch] * n_ch)

    res = minimize(neg_return, init_w, method="SLSQP", bounds=bnds, constraints=cons)
    return {ch: float(res.x[i]) for i, ch in enumerate(channels)}


def optimize_budget_linear(
    channel_slopes: dict[str, float],
    total_budget: float,
    min_spend: float = 5_000.0,
) -> dict[str, float]:
    """Naive linear allocation: allocates budget proportionally to positive linear slopes.

    If a slope is negative or zero (e.g. from multicollinearity), naive allocation sets
    its spend to the minimum bound and distributes the remainder to positive channels.
    """
    channels = list(channel_slopes.keys())
    pos_slopes = {ch: max(float(channel_slopes[ch]), 0.0) for ch in channels}
    sum_pos = sum(pos_slopes.values())

    if sum_pos == 0.0:
        equal_share = total_budget / len(channels)
        return dict.fromkeys(channels, equal_share)

    # Allocate discretionary budget above minimum spend
    discretionary = total_budget - len(channels) * min_spend
    allocation = {}
    for ch in channels:
        share = pos_slopes[ch] / sum_pos
        allocation[ch] = min_spend + discretionary * share
    return allocation


def bootstrap_holdout_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = config.N_BOOT,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap interval for out-of-sample R^2.

    Returns (point_estimate, ci_low, ci_high).
    """
    point = r2_score(y_true, y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(r2_score(y_true[idx], y_pred[idx]))

    ci_low = float(np.percentile(boot_scores, 2.5))
    ci_high = float(np.percentile(boot_scores, 97.5))
    return point, ci_low, ci_high


def bootstrap_relative_gain(
    y_true: np.ndarray,
    pred_headline: np.ndarray,
    pred_naive: np.ndarray,
    n_boot: int = config.N_BOOT,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Bootstrap interval for paired RMSE reduction over naive baseline.

    Returns (point_reduction, ci_low, ci_high) where reduction = (rmse_naive - rmse_headline) / rmse_naive.
    """
    base_naive = rmse(y_true, pred_naive)
    base_head = rmse(y_true, pred_headline)
    point = (base_naive - base_head) / base_naive if base_naive > 0 else 0.0

    n = len(y_true)
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rn = rmse(y_true[idx], pred_naive[idx])
        rh = rmse(y_true[idx], pred_headline[idx])
        diff = (rn - rh) / rn if rn > 0 else 0.0
        boot_diffs.append(diff)

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    return point, ci_low, ci_high
