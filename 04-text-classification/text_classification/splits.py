"""Split design.

The official 20 Newsgroups test set is later in time than its training set, so it is
used as the test as-is. A stratified random share of the training rows is held out to
choose a model. Test documents are scored once.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def fit_val_split(labels, val_frac: float, seed: int):
    idx = np.arange(len(labels))
    fit_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=labels)
    return np.sort(fit_idx), np.sort(val_idx)


def assert_disjoint(*index_sets) -> None:
    seen: set[int] = set()
    for idx in index_sets:
        current = set(np.asarray(idx).tolist())
        if seen & current:
            raise AssertionError("a document appears in more than one split")
        seen |= current
