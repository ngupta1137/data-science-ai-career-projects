"""Four ways to build a default model when outcomes are seen only for approved loans."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import Applicants


def _gbm(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=seed
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Model:
    def predict(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class AcceptedOnly(Model):
    """Fit on approved loans, as if they were the whole population."""

    def fit(self, h: Applicants, seed: int = 0):
        m = h.approved
        self.m = _gbm(seed).fit(h.x[m], h.y[m])
        return self

    def predict(self, x):
        return self.m.predict_proba(x)[:, 1]


class Ipw(Model):
    """Weight approved loans by one over their estimated chance of approval given recorded features.

    This removes selection on recorded features only. It cannot remove selection on the
    unrecorded trait the underwriter used.
    """

    def fit(self, h: Applicants, seed: int = 0):
        prop = LogisticRegression(max_iter=1000).fit(h.x, h.approved.astype(int))
        p = np.clip(prop.predict_proba(h.x)[:, 1], 0.02, 1.0)
        m = h.approved
        self.m = _gbm(seed).fit(h.x[m], h.y[m], sample_weight=1.0 / p[m])
        return self

    def predict(self, x):
        return self.m.predict_proba(x)[:, 1]


class RandomSampleOnly(Model):
    """Fit only on the loans approved at random. Unbiased but small."""

    def fit(self, h: Applicants, seed: int = 0):
        m = h.random_approve == 1
        self.m = LogisticRegression(max_iter=1000).fit(h.x[m], h.y[m])
        return self

    def predict(self, x):
        return self.m.predict_proba(x)[:, 1]


class Recalibrated(Model):
    """Accepted-only scores, then a two-parameter recalibration fitted on the random sample."""

    def fit(self, h: Applicants, seed: int = 0):
        self.base = AcceptedOnly().fit(h, seed)
        m = h.random_approve == 1
        z = _logit(self.base.predict(h.x[m]))[:, None]
        self.cal = LogisticRegression(max_iter=1000).fit(z, h.y[m])
        return self

    def predict(self, x):
        return self.cal.predict_proba(_logit(self.base.predict(x))[:, None])[:, 1]


MODELS = {
    "accepted_only": AcceptedOnly,
    "ipw": Ipw,
    "random_sample_only": RandomSampleOnly,
    "recalibrated": Recalibrated,
}
