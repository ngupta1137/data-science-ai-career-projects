"""Text models and search grids. A majority-class baseline comes first, on purpose."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASELINE = "baseline_majority"
TUNED = ("tfidf_nb", "tfidf_logreg", "tfidf_linear_svc")
SIMPLEST = "tfidf_nb"


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=True, min_df=2, lowercase=True)


def make_model(name: str, seed: int) -> Pipeline:
    if name == BASELINE:
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "tfidf_nb":
        return Pipeline([("tfidf", _vectorizer()), ("clf", MultinomialNB())])
    if name == "tfidf_logreg":
        return Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    if name == "tfidf_linear_svc":
        return Pipeline([("tfidf", _vectorizer()), ("clf", LinearSVC(random_state=seed))])
    raise KeyError(name)


def param_grid(name: str, fast: bool) -> list[dict]:
    """Each dict maps pipeline parameter names to values."""
    if name == "tfidf_nb":
        return [{"clf__alpha": a} for a in ([0.1] if fast else [0.01, 0.03, 0.1, 0.3, 1.0])]
    if name == "tfidf_logreg":
        return [{"clf__C": c} for c in ([10.0] if fast else [1.0, 10.0, 100.0])]
    if name == "tfidf_linear_svc":
        return [{"clf__C": c} for c in ([1.0] if fast else [0.1, 0.3, 1.0, 3.0])]
    raise KeyError(name)
