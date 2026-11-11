"""Uplift learners. Each is fitted on (x, t, y) and returns a predicted uplift for new customers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.1, random_state=0
    )


class SLinear:
    """One logistic regression on x, t and x*t. Uplift is prediction with t=1 minus with t=0."""

    def fit(self, x, t, y):
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(self._design(x, t), y)
        return self

    @staticmethod
    def _design(x, t):
        t = np.asarray(t, float)[:, None]
        return np.hstack([x, t, x * t])

    def uplift(self, x):
        one = np.ones(len(x))
        return (
            self.model.predict_proba(self._design(x, one))[:, 1]
            - self.model.predict_proba(self._design(x, 0 * one))[:, 1]
        )


class SGbm:
    """One boosted model on x and t. Uplift is prediction with t=1 minus with t=0."""

    def fit(self, x, t, y):
        self.model = _gbm().fit(np.hstack([x, np.asarray(t, float)[:, None]]), y)
        return self

    def uplift(self, x):
        one = np.ones((len(x), 1))
        return (
            self.model.predict_proba(np.hstack([x, one]))[:, 1]
            - self.model.predict_proba(np.hstack([x, 0 * one]))[:, 1]
        )


class TGbm:
    """Two boosted models, one on treated customers and one on controls. Uplift is the difference."""

    def fit(self, x, t, y):
        t = np.asarray(t)
        self.m1 = _gbm().fit(x[t == 1], y[t == 1])
        self.m0 = _gbm().fit(x[t == 0], y[t == 0])
        return self

    def uplift(self, x):
        return self.m1.predict_proba(x)[:, 1] - self.m0.predict_proba(x)[:, 1]


class CtGbm:
    """Class transformation: z = 1 when (treated and bought) or (control and did not buy).

    With a coin-flip offer, uplift = 2 * P(z=1 | x) - 1. Valid only for randomised data at 50%.
    """

    def fit(self, x, t, y):
        t = np.asarray(t)
        z = (t * y + (1 - t) * (1 - y)).astype(int)
        self.model = _gbm().fit(x, z)
        return self

    def uplift(self, x):
        return 2.0 * self.model.predict_proba(x)[:, 1] - 1.0


class ResponseGbm:
    """The common mistake: predict who will buy, ignoring the offer. Used as a trap."""

    def fit(self, x, t, y):
        self.model = _gbm().fit(x, y)
        return self

    def uplift(self, x):
        return self.model.predict_proba(x)[:, 1]  # a buying score, not an uplift


LEARNERS = {"s_linear": SLinear, "s_gbm": SGbm, "t_gbm": TGbm, "ct_gbm": CtGbm}
