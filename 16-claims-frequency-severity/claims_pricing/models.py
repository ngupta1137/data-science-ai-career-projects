"""Pricing models for insurance claims frequency and severity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

from .data import Policies


class PricingModel(Protocol):
    def fit(self, p: Policies) -> PricingModel: ...

    def predict_pure_premium(self, p: Policies) -> np.ndarray: ...


@dataclass
class FlatBaseline:
    mean_rate: float = 0.0

    def fit(self, p: Policies) -> FlatBaseline:
        total_loss = float(np.sum(p.claim_amount))
        total_exposure = float(np.sum(p.exposure))
        self.mean_rate = total_loss / max(total_exposure, 1e-6)
        return self

    def predict_pure_premium(self, p: Policies) -> np.ndarray:
        return np.full(len(p), self.mean_rate, dtype=float)


@dataclass
class NoExposureModel:
    """Naive model: regresses claim count without exposure offset/weight."""

    freq_model: PoissonRegressor | None = None
    sev_model: GammaRegressor | None = None

    def fit(self, p: Policies) -> NoExposureModel:
        X = p.features()
        self.freq_model = PoissonRegressor(alpha=1e-4, max_iter=300)
        self.freq_model.fit(X, p.claim_count)

        has_claims = p.claim_count > 0
        self.sev_model = GammaRegressor(alpha=1e-4, max_iter=300)
        if np.sum(has_claims) > 10:
            sev_target = p.claim_amount[has_claims] / p.claim_count[has_claims]
            self.sev_model.fit(X[has_claims], sev_target, sample_weight=p.claim_count[has_claims])
        else:
            mean_sev = float(np.mean(p.claim_amount[has_claims]) if np.any(has_claims) else 1000.0)
            self.sev_model.fit(X[:2], np.array([mean_sev, mean_sev]))
        return self

    def predict_pure_premium(self, p: Policies) -> np.ndarray:
        assert self.freq_model is not None and self.sev_model is not None
        X = p.features()
        pred_freq = self.freq_model.predict(X)
        pred_sev = self.sev_model.predict(X)
        return pred_freq * pred_sev


@dataclass
class SingleTweedieModel:
    """Compound Poisson-Gamma model: single Tweedie GLM (p=1.5) on pure premium."""

    model: TweedieRegressor | None = None

    def fit(self, p: Policies) -> SingleTweedieModel:
        X = p.features()
        self.model = TweedieRegressor(power=1.5, alpha=1e-4, max_iter=300)
        pure_prem = p.claim_amount / np.maximum(p.exposure, 1e-6)
        self.model.fit(X, pure_prem, sample_weight=p.exposure)
        return self

    def predict_pure_premium(self, p: Policies) -> np.ndarray:
        assert self.model is not None
        X = p.features()
        return self.model.predict(X)


@dataclass
class FreqSevTwoPartModel:
    """Headline model: Poisson GLM with exposure offset times Gamma GLM for severity."""

    freq_model: PoissonRegressor | None = None
    sev_model: GammaRegressor | None = None

    def fit(self, p: Policies) -> FreqSevTwoPartModel:
        X = p.features()
        self.freq_model = PoissonRegressor(alpha=1e-4, max_iter=300)
        freq_target = p.claim_count / np.maximum(p.exposure, 1e-6)
        self.freq_model.fit(X, freq_target, sample_weight=p.exposure)

        has_claims = p.claim_count > 0
        self.sev_model = GammaRegressor(alpha=1e-4, max_iter=300)
        if np.sum(has_claims) > 10:
            sev_target = p.claim_amount[has_claims] / p.claim_count[has_claims]
            self.sev_model.fit(X[has_claims], sev_target, sample_weight=p.claim_count[has_claims])
        else:
            mean_sev = float(np.mean(p.claim_amount[has_claims]) if np.any(has_claims) else 1000.0)
            self.sev_model.fit(X[:2], np.array([mean_sev, mean_sev]))
        return self

    def predict_pure_premium(self, p: Policies) -> np.ndarray:
        assert self.freq_model is not None and self.sev_model is not None
        X = p.features()
        pred_freq = self.freq_model.predict(X)
        pred_sev = self.sev_model.predict(X)
        return pred_freq * pred_sev


@dataclass
class OracleModel:
    """Simulator ground truth."""

    def fit(self, p: Policies) -> OracleModel:
        return self

    def predict_pure_premium(self, p: Policies) -> np.ndarray:
        return p.true_pure_premium
