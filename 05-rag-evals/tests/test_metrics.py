import math

import numpy as np
import pytest

from rag_eval.metrics import bootstrap_ci, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_gold_found():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 0.0


def test_reciprocal_rank_positions():
    assert reciprocal_rank(["x", "y", "g"], ["g"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["g"], ["g"]) == 1.0
    assert reciprocal_rank(["x"], ["g"]) == 0.0


def test_ndcg_hand_computed():
    # one gold at rank 2: dcg = 1/log2(3); ideal = 1/log2(2) = 1
    assert ndcg_at_k(["x", "g"], ["g"], 5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["g", "h"], ["g", "h"], 5) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ["g"], 5) == 0.0


def test_bootstrap_ci_is_ordered_and_deterministic():
    values = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    low, high = bootstrap_ci(lambda i: values[i].mean(), len(values), n_boot=200, seed=1)
    assert 0.0 <= low <= values.mean() <= high <= 1.0
    assert (low, high) == bootstrap_ci(lambda i: values[i].mean(), len(values), 200, 1)
