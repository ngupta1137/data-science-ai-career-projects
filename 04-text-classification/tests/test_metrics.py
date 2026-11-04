import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from text_classification import metrics

Y = ["a", "a", "b", "b"]
P = ["a", "b", "b", "b"]


def test_accuracy_and_macro_f1_hand_values():
    # Class a: precision 1, recall 1/2, F1 2/3. Class b: precision 2/3, recall 1, F1 4/5.
    assert metrics.accuracy(Y, P) == pytest.approx(0.75)
    assert metrics.macro_f1(Y, P) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_macro_f1_counts_an_unpredicted_class_as_zero():
    assert metrics.macro_f1(["a", "b"], ["a", "a"]) == pytest.approx((2 / 3 + 0.0) / 2)


def test_per_class_hand_values():
    table = metrics.per_class(Y, P, ["a", "b"])
    assert table["a"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert table["b"]["recall"] == 1.0 and table["b"]["support"] == 2


def test_top_confused_pairs_reads_the_off_diagonal():
    matrix = np.array([[3, 2, 0], [1, 4, 0], [0, 0, 5]])
    pairs = metrics.top_confused_pairs(matrix, ["a", "b", "c"], k=5)
    assert pairs[0] == {"true": "a", "predicted": "b", "count": 2}
    assert pairs[1] == {"true": "b", "predicted": "a", "count": 1}
    assert len(pairs) == 2  # zero cells are not reported


def test_length_buckets_follow_the_edges():
    counts = np.array([1, 5, 10, 50])
    assert metrics.length_buckets(counts, [5, 20]).tolist() == ["short", "medium", "medium", "long"]


def test_word_counts_handles_empty_text():
    assert metrics.word_counts(["", "one two", "  "]).tolist() == [0, 2, 0]


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.8, y, rng.integers(0, 3, 300))

    def stat(i):
        return metrics.accuracy(y[i], p[i])

    a = metrics.bootstrap_ci(stat, 300, n_boot=200, seed=5)
    assert a == metrics.bootstrap_ci(stat, 300, n_boot=200, seed=5)
    assert a[0] <= metrics.accuracy(y, p) <= a[1]


@pytest.mark.parametrize("clf", [MultinomialNB(), LogisticRegression(max_iter=500)])
def test_top_features_finds_the_topical_word(clf):
    texts = [
        "apple pear fruit",
        "apple fruit juice",
        "car engine road",
        "car road wheel",
        "cat dog pet",
        "dog pet leash",
    ]
    labels = ["food", "food", "auto", "auto", "pets", "pets"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)]).fit(texts, labels)
    top = metrics.top_features(pipe, ["auto", "food", "pets"], k=2)
    assert "car" in top["auto"] and "fruit" in top["food"] and "pet" in top["pets"]
