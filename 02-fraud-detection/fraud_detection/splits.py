"""Split design.

Fraud patterns drift and fraudsters adapt. A model is used on transactions that happen
after the ones it learned from, so the split is by time: fit on the earliest rows,
validate on the next, test on the latest. A random split mixes the periods and flatters
the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def time_split(time_values, test_frac: float, val_frac: float):
    """Return (fit_idx, val_idx, test_idx) as positions into the input, ordered by time."""
    time_values = np.asarray(time_values, dtype=float)
    order = np.argsort(time_values, kind="stable")
    n = len(order)
    n_test = round(n * test_frac)
    n_trainval = n - n_test
    n_val = round(n_trainval * val_frac)
    n_fit = n_trainval - n_val
    return order[:n_fit], order[n_fit:n_trainval], order[n_trainval:]


def assert_time_order(time_values, fit_idx, val_idx, test_idx) -> None:
    t = np.asarray(time_values, dtype=float)
    if t[fit_idx].max() > t[val_idx].min():
        raise AssertionError("fit rows are later than validation rows")
    if t[val_idx].max() > t[test_idx].min():
        raise AssertionError("validation rows are later than test rows")
    all_idx = np.concatenate([fit_idx, val_idx, test_idx])
    if len(set(all_idx.tolist())) != len(all_idx):
        raise AssertionError("a row appears in more than one split")


def random_split(n_rows: int, test_frac: float, seed: int):
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed)
    return train_idx, test_idx
