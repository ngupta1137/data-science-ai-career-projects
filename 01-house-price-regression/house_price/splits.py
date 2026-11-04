"""Split design.

Houses near each other have similar prices. A random split puts neighbours on both
sides, so the test score rewards memorising a neighbourhood. The spatial split keeps
every 0.5 degree cell entirely in train or entirely in test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import GROUP_CELL_DEG


def add_geo_group(df: pd.DataFrame, cell_deg: float = GROUP_CELL_DEG) -> np.ndarray:
    lat_cell = np.floor(df["Latitude"].to_numpy() / cell_deg).astype(int)
    lon_cell = np.floor(df["Longitude"].to_numpy() / cell_deg).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(lat_cell, lon_cell, strict=True)])


def spatial_split(groups: np.ndarray, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def random_split(n_rows: int, test_size: float, seed: int):
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return train_idx, test_idx


def group_overlap(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> set:
    return set(groups[train_idx]) & set(groups[test_idx])


def assert_no_overlap(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    shared = group_overlap(groups, train_idx, test_idx)
    if shared:
        raise AssertionError(f"{len(shared)} spatial cells appear in both train and test")
