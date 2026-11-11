"""Rate estimates, shelf scores and the day-by-day simulation of a visit policy."""

from __future__ import annotations

import numpy as np

from . import config
from .data import World


def zero_runs(sales: np.ndarray) -> np.ndarray:
    """Length of the zero-sales run ending on each day, per shelf."""
    out = np.zeros_like(sales)
    run = np.zeros(sales.shape[0], dtype=sales.dtype)
    for d in range(sales.shape[1]):
        run = np.where(sales[:, d] == 0, run + 1, 0)
        out[:, d] = run
    return out


def rate_naive(history: np.ndarray) -> np.ndarray:
    """Mean sales per day, counting empty-shelf days as zero demand."""
    return history.mean(1)


def rate_clean(history: np.ndarray, min_run: int = config.CLEAN_RUN_MIN) -> np.ndarray:
    """Mean sales per day, leaving out days inside zero runs of at least min_run days."""
    runs = zero_runs(history)
    # mark every day of a qualifying run, not just its tail
    drop = np.zeros(history.shape, bool)
    for d in range(history.shape[1] - 1, -1, -1):
        end = runs[:, d] >= min_run
        if d + 1 < history.shape[1]:
            end |= drop[:, d + 1] & (history[:, d] == 0)
        drop[:, d] = end
    kept = (~drop).sum(1)
    mean = np.where(kept > 0, (history * ~drop).sum(1) / np.maximum(kept, 1), 0.0)
    floor = 1.0 / (2 * history.shape[1])  # a shelf with no sales in history still gets a small rate
    return np.maximum(mean, floor)


def posterior_empty(
    rate: np.ndarray, run: np.ndarray, prior: float = config.PRIOR_EMPTY
) -> np.ndarray:
    """Chance the shelf is empty given `run` days with no sales and a rate, by Bayes' rule.

    In stock, a run of r zero days has chance exp(-rate * r). Empty, it has chance one. The prior
    ignores how long the shelf has been empty, which is a simplification.
    """
    like_in_stock = np.exp(-rate * run)
    return prior / (prior + (1 - prior) * like_in_stock)


def score(
    policy: str, run: np.ndarray, r_clean: np.ndarray, r_naive: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(score, eligible) for each shelf. Higher score is visited first among eligible shelves."""
    if policy == "zero_run_rule":
        return run.astype(float), run >= config.ZERO_RUN_RULE
    if policy == "poisson_alarm":
        s = r_clean * run  # minus log chance of the run if in stock
        return s, s >= -np.log(0.01)
    if policy in ("expected_value", "expected_value_naive_rate"):
        rate = r_clean if policy == "expected_value" else np.maximum(r_naive, 1e-6)
        value = posterior_empty(rate, run) * rate * config.MARGIN * config.EXPECTED_REMAINING_DAYS
        return value, (value > config.VISIT_COST) & (run > 0)
    raise ValueError(policy)


def simulate(w: World, policy: str, capacity: int) -> dict:
    """Run the world with a visit policy on the evaluation days. Returns per-shelf outcomes."""
    n, h = len(w), config.HISTORY_DAYS
    empty = np.zeros(n, bool)
    ends = np.zeros(n, int)
    started = np.full(n, -1)
    sales = np.zeros((n, w.days), dtype=w.demand.dtype)
    run = np.zeros(n, int)
    lost = np.zeros(n)
    empty_days = 0
    visits = np.zeros(n, int)
    useful = np.zeros(n, int)
    episodes = np.zeros(n, int)  # evaluation-period episodes
    fixed = np.zeros(n, int)
    delays: list[int] = []
    flagged: list[int] = []
    calib: list[tuple[float, bool]] = []
    r_clean = r_naive = None
    for d in range(w.days):
        new = ~empty & (w.start_u[:, d] < config.OOS_START_RATE)
        empty |= new
        ends = np.where(new, d + w.duration[:, d], ends)
        started = np.where(new, d, started)
        if d >= h:
            episodes += new
        sales[:, d] = np.where(empty, 0, w.demand[:, d])
        run = np.where(sales[:, d] == 0, run + 1, 0)
        if d == h - 1:
            r_clean, r_naive = rate_clean(sales[:, :h]), rate_naive(sales[:, :h])
        if d >= h:
            lost += np.where(empty, w.demand[:, d] * config.MARGIN, 0.0)
            empty_days += int(empty.sum())
            if policy == "oracle":
                s, ok = (
                    w.rate * empty,
                    empty
                    & (w.rate * config.MARGIN * config.EXPECTED_REMAINING_DAYS > config.VISIT_COST),
                )
            elif policy != "no_visits":
                s, ok = score(policy, run, r_clean, r_naive)
            if policy != "no_visits":
                idx = np.flatnonzero(ok)
                flagged.append(len(idx))
                pick = idx[np.argsort(-s[idx], kind="stable")[:capacity]]
                visits[pick] += 1
                if policy in ("expected_value", "expected_value_naive_rate"):
                    rate = r_clean if policy == "expected_value" else np.maximum(r_naive, 1e-6)
                    post = posterior_empty(rate[pick], run[pick])
                    calib.extend(zip(post.tolist(), empty[pick].tolist(), strict=True))
                hit = pick[empty[pick]]
                useful[hit] += 1
                fixed[hit] += started[hit] >= h
                delays.extend((d - started[hit][started[hit] >= h]).tolist())
                empty[hit] = False
                run[pick] = 0
        empty &= d + 1 < ends
    return {
        "lost_margin": lost,
        "visits": visits,
        "useful_visits": useful,
        "episodes": episodes,
        "fixed": fixed,
        "delays": np.array(delays, dtype=float),
        "flagged_per_day": float(np.mean(flagged)) if flagged else 0.0,
        "empty_share": empty_days / (n * (w.days - h)),
        "visit_calibration": np.array(calib, dtype=float).reshape(-1, 2),
        "rate_clean": r_clean,
        "rate_naive": r_naive,
    }
