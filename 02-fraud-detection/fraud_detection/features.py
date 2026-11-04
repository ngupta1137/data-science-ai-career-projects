"""Feature construction. Time and the label are never features.

Time is excluded on purpose: it is seconds since the first transaction in a two day
window, so any use of it would not transfer to a later period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PCA_FEATURES, TARGET


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function, no fitting, so no train/test leakage."""
    features = df[PCA_FEATURES].copy()
    features["log1p_amount"] = np.log1p(df["Amount"])
    assert TARGET not in features.columns and "Time" not in features.columns
    return features
