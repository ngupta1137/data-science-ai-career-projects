import numpy as np
import pytest

from recsys_eval import metrics


def rng():
    return np.random.default_rng(0)


def test_topk_excludes_seen_and_unavailable_and_sorts():
    scores = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    seen = np.array([[True, False, False, False, False]])
    avail = np.array([True, True, False, True, True])
    top = metrics.topk(scores, seen, avail, 2, rng())
    assert top.tolist() == [[1, 3]]  # item 0 seen, item 2 not yet released


def test_ndcg_hand_example():
    top = np.array([[0, 1, 2]])
    relevant = np.zeros((1, 5), bool)
    relevant[0, [1, 3]] = True
    # gains 0, 1, 0 -> dcg = 1 / log2(3). ideal puts both relevant items first: 1 + 1 / log2(3).
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert metrics.ndcg_at_k(top, relevant)[0] == pytest.approx(expected)
    assert metrics.recall_at_k(top, relevant)[0] == 0.5
    assert metrics.hit_rate_at_k(top, relevant)[0] == 1.0


def test_ndcg_is_one_for_a_perfect_ranking_and_nan_without_relevant_items():
    relevant = np.zeros((2, 4), bool)
    relevant[0, [2, 0]] = True
    top = np.array([[2, 0, 1], [0, 1, 2]])
    out = metrics.ndcg_at_k(top, relevant)
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1])


def test_true_like_rate_is_the_mean_probability_of_the_shown_items():
    prob = np.array([[0.1, 0.9, 0.5, 0.3]])
    assert metrics.true_like_rate(np.array([[1, 2]]), prob)[0] == pytest.approx(0.7)


def test_coverage_and_gini():
    top = np.array([[0, 1], [0, 1], [0, 1]])
    assert metrics.coverage(top, 4) == 0.5
    even = np.array([[0], [1], [2], [3]])
    assert metrics.gini(even, 4) == pytest.approx(0.0)
    one = np.zeros((4, 1), int)
    assert metrics.gini(one, 4) == pytest.approx(0.75)  # (n - 1) / n for one item taking everything


def test_sampled_hit_rate_perfect_and_terrible_scores():
    n_u, n_i = 5, 300
    relevant = np.zeros((n_u, n_i), bool)
    relevant[np.arange(n_u), np.arange(n_u)] = True
    seen = np.zeros_like(relevant)
    avail = np.ones(n_i, bool)
    good = np.where(relevant, 10.0, 0.0)
    bad = np.where(relevant, -10.0, 0.0)
    assert np.nanmean(metrics.sampled_hit_rate(good, relevant, seen, avail, 10, 100, rng())) == 1.0
    assert np.nanmean(metrics.sampled_hit_rate(bad, relevant, seen, avail, 10, 100, rng())) == 0.0


def test_sampled_metric_is_easier_than_the_full_one(rng=None):
    g = np.random.default_rng(1)
    n_u, n_i = 200, 500
    scores = g.random((n_u, n_i))
    relevant = np.zeros((n_u, n_i), bool)
    relevant[np.arange(n_u), g.integers(0, n_i, n_u)] = True
    seen = np.zeros_like(relevant)
    avail = np.ones(n_i, bool)
    sampled = np.nanmean(metrics.sampled_hit_rate(scores, relevant, seen, avail, 10, 100, g))
    top = metrics.topk(scores, seen, avail, 10, g)
    full = np.nanmean(metrics.hit_rate_at_k(top, relevant))
    assert sampled > full + 0.03  # about 0.1 against 0.02 for random scores


def test_spearman():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_paired_bootstrap_of_identical_arrays_is_zero():
    x = np.random.default_rng(0).random(100)
    assert metrics.boot_diff_ci(x, x, np.random.default_rng(1), 50) == (0.0, 0.0)


def test_bootstrap_ignores_nan():
    x = np.array([1.0, np.nan, 3.0] * 20)
    lo, hi = metrics.boot_mean_ci(x, np.random.default_rng(1), 100)
    assert 1.0 <= lo <= 2.0 <= hi <= 3.0
