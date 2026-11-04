"""Feature construction under a forecast horizon.

A feature is allowed only if it would be known HORIZON days before the day being forecast.
Everything built from the target is shifted by at least HORIZON. Calendar facts (weekday,
month, holiday, working day) are known in advance. Weather is not, so it is left out.

Two leaky variants exist on purpose, so the traps can be measured:
  same_day_components : adds casual and registered for the same day. They sum to the target.
  short_lag           : adds lag 1 and lag 2, which are not known 7 days ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON, TARGET

VARIANTS = (None, "same_day_components", "short_lag")


def build_features(df: pd.DataFrame, variant: str | None = None) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    y = df[TARGET].astype(float)
    out = pd.DataFrame(index=df.index)

    for day in range(7):
        out[f"dow_{day}"] = (df["weekday"] == day).astype(float)
    angle = 2 * np.pi * (df["mnth"].astype(float) - 1) / 12
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["holiday"] = df["holiday"].astype(float)
    out["workingday"] = df["workingday"].astype(float)

    out[f"lag_{HORIZON}"] = y.shift(HORIZON)
    out[f"lag_{2 * HORIZON}"] = y.shift(2 * HORIZON)
    out[f"lag_{3 * HORIZON}"] = y.shift(3 * HORIZON)
    shifted = y.shift(HORIZON)
    out["roll7_at_horizon"] = shifted.rolling(7).mean()
    out["roll28_at_horizon"] = shifted.rolling(28).mean()

    if variant == "same_day_components":
        out["casual_same_day"] = df["casual"].astype(float)
        out["registered_same_day"] = df["registered"].astype(float)
    elif variant == "short_lag":
        out["lag_1"] = y.shift(1)
        out["lag_2"] = y.shift(2)
    return out


def first_valid_row(features: pd.DataFrame) -> int:
    """Index of the first row with no missing feature values."""
    ok = features.notna().all(axis=1).to_numpy()
    return int(np.argmax(ok)) if ok.any() else len(features)
