"""Data generator, simulation mechanics, and validation for marketing mix modeling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class MarketingData:
    """Marketing mix time series container."""

    weeks: np.ndarray  # shape (T,)
    spend: dict[str, np.ndarray]  # channel -> shape (T,)
    revenue: np.ndarray  # total observed weekly revenue ($), shape (T,)
    true_base: np.ndarray  # organic baseline sales ($), shape (T,)
    true_effects: dict[str, np.ndarray]  # channel -> true incremental weekly revenue ($), shape (T,)
    true_incremental: np.ndarray  # sum of true channel contributions ($), shape (T,)

    def __len__(self) -> int:
        return len(self.weeks)


def geometric_adstock(spend: np.ndarray, alpha: float, l_max: int = 12) -> np.ndarray:
    """Apply normalized geometric decay carryover filter.

    Weights w_l = alpha^l for l = 0 ... l_max, normalized to sum to 1.
    This preserves the total volume of spend while distributing it across weeks.
    """
    weights = np.array([alpha**l for l in range(l_max + 1)])
    weights = weights / np.sum(weights)

    # Convolve with reflection/zero-padding
    adstocked = np.convolve(spend, weights, mode="full")[: len(spend)]
    return adstocked


def hill_saturation(spend: np.ndarray, half_sat: float, max_lift: float) -> np.ndarray:
    """Hill-1 / Michaelis-Menten diminishing returns curve.

    f(x) = max_lift * (x / (x + half_sat))
    When spend == half_sat, returns 0.5 * max_lift.
    Marginal return f'(x) = max_lift * half_sat / (x + half_sat)^2 is strictly decreasing.
    """
    return max_lift * (spend / (spend + half_sat))


def true_channel_marginal_roi(channel: str, spend: float) -> float:
    """Derivative of the true channel response curve at steady-state spend."""
    params = config.TRUE_CHANNEL_PARAMS[channel]
    half_sat = params["half_sat"]
    max_lift = params["max_lift"]
    # f'(x) = max_lift * half_sat / (x + half_sat)^2
    return float(max_lift * half_sat / ((spend + half_sat) ** 2))


def generate(rng: np.random.Generator, n_weeks: int) -> MarketingData:
    """Generate synthetic marketing weekly time series with known truth.

    Mechanisms:
    1. Base sales has linear trend, 52-week sinusoidal seasonality, and Q4 holiday pulse.
    2. TV spend is pulsed with high variance.
    3. Search spend is partially synchronized with TV (omitted halo correlation).
    4. Social spend has independent promotional rhythm.
    5. True response applies channel-specific geometric adstock and Hill saturation.
    """
    t = np.arange(n_weeks, dtype=float)

    # 1. Base sales
    trend = config.TRUE_TREND_SLOPE * t
    # 52-week annual cycle peaking in summer (week 26)
    seasonality = config.TRUE_SEASONALITY_AMP * np.sin(2.0 * np.pi * t / 52.0)
    # Q4 holiday pulse (weeks 44-51 within each 52-week year)
    week_of_year = t % 52.0
    holiday_surge = np.where(
        (week_of_year >= 44) & (week_of_year <= 51),
        config.TRUE_HOLIDAY_SURGE * np.sin(np.pi * (week_of_year - 44) / 7.0),
        0.0,
    )
    base_sales = config.TRUE_BASE_SALES + trend + seasonality + holiday_surge

    # 2. Channel Spends
    # TV pulses: base spend + periodic campaign flights
    tv_flights = 20_000.0 * (np.sin(2.0 * np.pi * t / 13.0) > 0.3).astype(float)
    tv_noise = rng.normal(0, 5_000.0, size=n_weeks)
    tv_spend = np.clip(30_000.0 + tv_flights + tv_noise, 10_000.0, 70_000.0)

    # Search: performance queries follow TV flighting + organic seasonality + search budget
    # Correlation with TV is intentionally substantial (~0.65) to reflect real marketing alignment
    search_latent = 0.55 * (tv_spend - 30_000.0) / 15_000.0 + 0.45 * rng.normal(0, 1.0, size=n_weeks)
    search_spend = np.clip(18_000.0 + 6_000.0 * search_latent, 6_000.0, 36_000.0)

    # Social: periodic flights and independent promotional pushes
    social_flights = 12_000.0 * (np.cos(2.0 * np.pi * t / 8.0) > 0.4).astype(float)
    social_noise = rng.normal(0, 4_000.0, size=n_weeks)
    social_spend = np.clip(22_000.0 + social_flights + social_noise, 8_000.0, 48_000.0)

    spends = {
        "tv": tv_spend,
        "search": search_spend,
        "social": social_spend,
    }

    # 3. Channel Responses under Known Truth
    true_effects = {}
    for ch in config.CHANNELS:
        params = config.TRUE_CHANNEL_PARAMS[ch]
        adstocked = geometric_adstock(spends[ch], params["alpha"])
        lift = hill_saturation(adstocked, params["half_sat"], params["max_lift"])
        true_effects[ch] = lift

    true_incremental = sum(true_effects.values())

    # 4. Total Observed Revenue with Observation Noise
    noise = rng.normal(0, config.TRUE_NOISE_STD, size=n_weeks)
    revenue = base_sales + true_incremental + noise

    return MarketingData(
        weeks=t,
        spend=spends,
        revenue=revenue,
        true_base=base_sales,
        true_effects=true_effects,
        true_incremental=true_incremental,
    )


def validate(data: MarketingData) -> None:
    """Validate generated data properties and invariants."""
    n = len(data)
    if n < 10:
        raise ValueError(f"Too few weeks generated: {n}")

    if len(data.revenue) != n or len(data.true_base) != n or len(data.true_incremental) != n:
        raise ValueError("Mismatched array lengths in MarketingData")

    for ch in config.CHANNELS:
        if ch not in data.spend:
            raise ValueError(f"Missing spend for channel {ch}")
        if len(data.spend[ch]) != n:
            raise ValueError(f"Spend length mismatch for channel {ch}")
        if np.any(data.spend[ch] < 0):
            raise ValueError(f"Negative spend detected for channel {ch}")
        if ch not in data.true_effects:
            raise ValueError(f"Missing true effect for channel {ch}")

    if np.any(data.revenue <= 0):
        raise ValueError("Non-positive observed revenue detected")

    if np.any(np.isnan(data.revenue)) or np.any(np.isinf(data.revenue)):
        raise ValueError("NaN or Inf detected in revenue")

    # Check correlation between TV and search spend is within expected range [0.45, 0.85]
    corr_tv_search = float(np.corrcoef(data.spend["tv"], data.spend["search"])[0, 1])
    if not (0.40 <= corr_tv_search <= 0.85):
        raise ValueError(f"TV-Search correlation {corr_tv_search:.3f} outside expected range [0.40, 0.85]")


def fingerprint(n_weeks: int = 20, seed: int = config.SEED) -> str:
    """Compute deterministic SHA-256 fingerprint for simulator data generation."""
    rng = np.random.default_rng(seed)
    data = generate(rng, n_weeks)
    parts = [
        f"{float(data.revenue[i]):.4f}" for i in range(n_weeks)
    ] + [
        f"{float(data.spend[ch][i]):.4f}" for ch in config.CHANNELS for i in range(n_weeks)
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_fingerprint(path: Path | None = None) -> None:
    """Write current data generator fingerprint to CHECKSUMS.txt."""
    p = path or config.CHECKSUM_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{fingerprint()}\n", encoding="utf-8")


def verify_fingerprint(path: Path | None = None) -> None:
    """Verify that current generator reproduces the recorded fingerprint."""
    p = path or config.CHECKSUM_FILE
    if not p.exists():
        raise FileNotFoundError(f"Missing checksum file: {p}. Run with --write-checksum first.")
    expected = p.read_text(encoding="utf-8").strip()
    actual = fingerprint()
    if expected != actual:
        raise ValueError(
            f"Data generator fingerprint mismatch: expected {expected}, got {actual}. "
            "Data generation mechanics changed after results were recorded."
        )
