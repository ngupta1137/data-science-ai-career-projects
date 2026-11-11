import numpy as np
import pytest

from recsys_eval import config, data, models


def events(pairs):
    u, i = zip(*pairs, strict=True)
    return data.Events(np.array(u), np.array(i), np.arange(len(pairs), dtype=float))


def test_interaction_matrix_counts_repeat_clicks_once():
    m = models.interaction_matrix(events([(0, 1), (0, 1), (1, 2)]), 2, 3)
    assert m.toarray().tolist() == [[0, 1, 0], [0, 0, 1]]


def test_popularity_counts():
    ev = events([(0, 1), (1, 1), (2, 1), (0, 2)])
    s = models.score_popularity(ev, 3, 4)
    assert s[0].tolist() == [0, 3, 1, 0] and (s == s[0]).all()


def test_recent_popularity_only_counts_the_window():
    ev = data.Events(np.array([0, 1, 2]), np.array([0, 1, 1]), np.array([1.0, 90.0, 95.0]))
    s = models.score_recent_popularity(ev, 3, 2, now=96.0)
    assert s[0].tolist() == [0.0, 2.0]  # item 0's only click is outside the 14-day window


def test_item_knn_hand_example():
    # user 0 clicked items 0 and 1, user 1 clicked items 1 and 2.
    ev = events([(0, 0), (0, 1), (1, 1), (1, 2)])
    s = models.score_item_knn(ev, 2, 3)
    c = 1 / np.sqrt(2)  # cosine of items 0 and 1: one co-click, norms 1 and sqrt(2)
    assert s[0].tolist() == pytest.approx([c, c, c])
    assert s[1].tolist() == pytest.approx([c, c, c])


def test_svd_recovers_block_structure(monkeypatch):
    monkeypatch.setattr(config, "SVD_RANK", 2)  # full rank would just reproduce the matrix
    pairs = []
    for u in range(10):
        pairs += [(u, i) for i in range(5)]
    for u in range(10, 20):
        pairs += [(u, i) for i in range(5, 10)]
    pairs.remove((0, 4))  # user 0 has not clicked item 4, which everyone in its block has
    s = models.score_svd(events(pairs), 20, 10)
    assert s[0, 4] > s[0, 9] + 0.5


def test_ips_weight_lowers_the_pull_of_popular_items():
    pairs = [(u, 0) for u in range(30)] + [(u, 1) for u in range(5)] + [(u, 2) for u in range(5)]
    ev = events(pairs)
    plain = models.score_svd(ev, 30, 3)
    ips = models.score_svd(ev, 30, 3, ips_alpha=0.5)
    assert ips[:, 0].mean() / ips[:, 1].mean() < plain[:, 0].mean() / plain[:, 1].mean()


def test_oracle_is_the_true_like_probability():
    w = data.make_world(np.random.default_rng(0))
    assert np.array_equal(models.score_oracle(w, 50.0), w.like_prob_matrix(50.0))
