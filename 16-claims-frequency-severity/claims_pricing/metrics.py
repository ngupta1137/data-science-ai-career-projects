"""Actuarial pricing metrics: Gini index, loss ratios, and decile analysis."""

from __future__ import annotations

import numpy as np


def gini_coefficient(
    pred_rate: np.ndarray, actual_losses: np.ndarray, exposure: np.ndarray
) -> float:
    """Actuarial Gini coefficient (concentration index) of pure premium.

    Orders policies by predicted rate ascending, then plots cumulative exposure against
    cumulative actual loss. G = 1 - 2 * Area Under Lorenz Curve.
    A flat rating has G = 0. A model with good risk discrimination has G > 0.
    """
    total_loss = float(np.sum(actual_losses))
    total_exposure = float(np.sum(exposure))
    if total_loss <= 0.0 or total_exposure <= 0.0:
        return 0.0

    if float(np.ptp(pred_rate)) == 0.0:
        return 0.0

    order = np.argsort(pred_rate)
    ordered_exp = exposure[order]
    ordered_loss = actual_losses[order]

    cum_exp = np.insert(np.cumsum(ordered_exp) / total_exposure, 0, 0.0)
    cum_loss = np.insert(np.cumsum(ordered_loss) / total_loss, 0, 0.0)

    # Trapezoid rule area under Lorenz curve
    area = np.sum((cum_exp[1:] - cum_exp[:-1]) * (cum_loss[1:] + cum_loss[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)


def loss_ratio(actual_losses: np.ndarray, charged_premium: np.ndarray) -> float:
    total_premium = float(np.sum(charged_premium))
    if total_premium <= 0.0:
        return 0.0
    return float(np.sum(actual_losses) / total_premium)


def decile_loss_ratios(
    actual_losses: np.ndarray,
    charged_premium: np.ndarray,
    rank_by: np.ndarray,
    exposure: np.ndarray,
) -> list[dict]:
    """Segment policies into 10 risk deciles by predicted risk and compute loss ratios."""
    quantiles = np.percentile(rank_by, np.linspace(0, 100, 11))
    # Ensure strictly monotonic bin edges
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-6

    deciles = []
    for d in range(10):
        low, high = quantiles[d], quantiles[d + 1]
        if d == 9:
            mask = (rank_by >= low) & (rank_by <= high)
        else:
            mask = (rank_by >= low) & (rank_by < high)

        d_loss = float(np.sum(actual_losses[mask]))
        d_prem = float(np.sum(charged_premium[mask]))
        d_exp = float(np.sum(exposure[mask]))
        d_lr = d_loss / max(d_prem, 1e-6)

        deciles.append(
            {
                "decile": d + 1,
                "n_policies": int(np.sum(mask)),
                "exposure": round(d_exp, 2),
                "actual_losses": round(d_loss, 2),
                "charged_premium": round(d_prem, 2),
                "loss_ratio": round(d_lr, 4),
            }
        )
    return deciles


def bootstrap_gini_and_diff(
    pred_headline: np.ndarray,
    pred_competitor: np.ndarray,
    actual_losses: np.ndarray,
    exposure: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap 95% confidence intervals for headline Gini and paired difference."""
    rng = np.random.default_rng(seed)
    n = len(actual_losses)
    boot_headline = np.empty(n_boot)
    boot_diff = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        g_head = gini_coefficient(pred_headline[idx], actual_losses[idx], exposure[idx])
        g_comp = gini_coefficient(pred_competitor[idx], actual_losses[idx], exposure[idx])
        boot_headline[i] = g_head
        boot_diff[i] = g_head - g_comp

    ci_headline = (
        float(np.percentile(boot_headline, 2.5)),
        float(np.percentile(boot_headline, 97.5)),
    )
    ci_diff = (
        float(np.percentile(boot_diff, 2.5)),
        float(np.percentile(boot_diff, 97.5)),
    )
    return ci_headline, ci_diff


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))
