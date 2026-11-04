"""Synthetic frames with the credit card fraud schema. Used by tests and smoke runs only.

Nothing measured on this data says anything about real-data performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PCA_FEATURES


def make_synthetic_fraud(n: int = 20000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Imbalanced frame where fraud shifts a few features. Time is spread over two days."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    features = rng.normal(size=(n, len(PCA_FEATURES)))
    features[is_fraud, 0] += 1.8
    features[is_fraud, 1] -= 1.5
    features[is_fraud, 2] += 1.2
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=4.2, sigma=1.0, size=n),
        rng.lognormal(mean=3.5, sigma=1.2, size=n),
    )
    frame = pd.DataFrame(features, columns=PCA_FEATURES)
    frame.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    frame["Amount"] = amount
    frame["Class"] = is_fraud.astype(int)
    return frame
