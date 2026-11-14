"""Evaluation metrics: credit share MAE, Spearman correlation, and budget reallocation yield."""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config, data


def channel_shares_mae(est: dict[str, float], truth: dict[str, float]) -> float:
    """Mean absolute error across channel credit shares (in proportion scale, 0 to 1)."""
    errors = [abs(est[ch] - truth[ch]) for ch in config.CHANNELS]
    return float(np.mean(errors))


def channel_shares_spearman(est: dict[str, float], truth: dict[str, float]) -> float:
    """Spearman rank correlation between estimated and true credit shares."""
    est_vals = [est[ch] for ch in config.CHANNELS]
    true_vals = [truth[ch] for ch in config.CHANNELS]
    res = stats.spearmanr(est_vals, true_vals)
    return float(res.statistic if not np.isnan(res.statistic) else 0.0)


def evaluate_budget_reallocation(
    channel_shares: dict[str, float],
    test_dataset: data.JourneyDataset,
    total_budget: float = config.REALLOCATION_BUDGET,
) -> dict[str, float]:
    """Calculate the true incremental conversions delivered by allocating budget via channel shares.

    Budget allocated to channel c is B_c = total_budget * w_c.
    Touches purchased: N_c = B_c / unit_cost_c.
    Average true incremental conversion lift per touch is measured empirically from test_dataset.
    A mild logarithmic saturation factor models diminishing returns at scale.
    """
    # Compute average true incremental lift per touch on test cohort
    channel_touch_counts: dict[str, int] = dict.fromkeys(config.CHANNELS, 0)
    channel_total_lifts: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)

    for j in test_dataset.journeys:
        for ch in j.path:
            channel_touch_counts[ch] += 1
            channel_total_lifts[ch] += j.true_channel_lifts.get(ch, 0.0)

    avg_lift_per_touch: dict[str, float] = {}
    for ch in config.CHANNELS:
        count = channel_touch_counts[ch]
        avg_lift_per_touch[ch] = (channel_total_lifts[ch] / count) if count > 0 else 0.0

    # Calculate allocated spend, touches, and true incremental conversions
    allocated_spend: dict[str, float] = {}
    channel_conversions: dict[str, float] = {}

    for ch in config.CHANNELS:
        spend = total_budget * channel_shares[ch]
        allocated_spend[ch] = spend
        unit_cost = config.CHANNEL_COSTS[ch]
        touches = spend / unit_cost

        # Diminishing returns scaling: touches / (1 + touches / 20_000)
        effective_touches = touches / (1.0 + touches / 25_000.0)
        convs = effective_touches * avg_lift_per_touch[ch]
        channel_conversions[ch] = float(convs)

    total_incremental_conversions = sum(channel_conversions.values())
    return {
        "total_budget": total_budget,
        "incremental_conversions": round(total_incremental_conversions, 1),
        "channel_spend": {ch: round(allocated_spend[ch], 1) for ch in config.CHANNELS},
        "channel_conversions": {ch: round(channel_conversions[ch], 1) for ch in config.CHANNELS},
    }


def bootstrap_share_mae(
    test_dataset: data.JourneyDataset,
    est_shares: dict[str, float],
    n_resamples: int = 500,
    seed: int = config.SEED,
) -> tuple[float, list[float]]:
    """Bootstrap 95% confidence interval on channel credit share MAE over resampled test journeys."""
    rng = np.random.default_rng(seed)
    n = len(test_dataset.journeys)
    maes: list[float] = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample_converters: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)

        for idx in indices:
            j = test_dataset.journeys[idx]
            if j.converted:
                for ch in config.CHANNELS:
                    sample_converters[ch] += j.true_channel_lifts.get(ch, 0.0)

        sample_total = sum(sample_converters.values())
        if sample_total > 0:
            sample_true_shares = {
                ch: sample_converters[ch] / sample_total for ch in config.CHANNELS
            }
        else:
            sample_true_shares = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))

        mae = channel_shares_mae(est_shares, sample_true_shares)
        maes.append(mae)

    point_estimate = float(np.mean(maes))
    lower = float(np.percentile(maes, 2.5))
    upper = float(np.percentile(maes, 97.5))
    return round(point_estimate, 4), [round(lower, 4), round(upper, 4)]
