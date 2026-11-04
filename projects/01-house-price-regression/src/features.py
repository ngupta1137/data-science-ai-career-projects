"""Feature construction. The target and the group id are never features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES_RAW, TARGET


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix. Pure function: no fitting, so no train/test leakage."""
    features = df[FEATURES_RAW].copy()
    features["log_population"] = np.log1p(df["Population"])
    features["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"]
    assert TARGET not in features.columns
    return features
