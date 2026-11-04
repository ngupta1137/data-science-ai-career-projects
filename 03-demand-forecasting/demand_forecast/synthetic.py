"""Synthetic daily series with the Bike Sharing schema. Used by tests and smoke runs only.

Nothing measured on this data says anything about real-data performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_demand(n: int = 731, seed: int = 0) -> pd.DataFrame:
    """Trend, weekly and yearly seasonality, noise. casual + registered equals cnt."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2011-01-01", periods=n, freq="D")
    t = np.arange(n)
    weekday = ((days.dayofweek + 1) % 7).to_numpy()  # 0 is Sunday, as in the source data
    holiday = (rng.random(n) < 0.03).astype(int)
    workingday = ((weekday >= 1) & (weekday <= 5) & (holiday == 0)).astype(int)
    yearly = 1.0 + 0.4 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 80) / 365)
    weekly = np.where(workingday == 1, 1.1, 0.85)
    level = (1500 + 3.0 * t) * yearly * weekly
    cnt = np.maximum(50, level + rng.normal(0, 150, n)).round().astype(int)
    casual = (cnt * rng.uniform(0.1, 0.4, n)).round().astype(int)
    return pd.DataFrame(
        {
            "dteday": days.strftime("%Y-%m-%d"),
            "season": (days.month % 12 // 3 + 1).to_numpy(),
            "mnth": days.month.to_numpy(),
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "casual": casual,
            "registered": cnt - casual,
            "cnt": cnt,
        }
    )
