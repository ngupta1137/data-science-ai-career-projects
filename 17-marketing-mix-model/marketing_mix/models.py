"""Marketing mix candidate models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from . import config, data


def extract_base_features(weeks: np.ndarray) -> np.ndarray:
    """Build deterministic baseline features: intercept, trend, 52-week sin/cos, holiday."""
    t = weeks.astype(float)
    sin52 = np.sin(2.0 * np.pi * t / 52.0)
    cos52 = np.cos(2.0 * np.pi * t / 52.0)
    week_of_year = t % 52.0
    holiday = ((week_of_year >= 44) & (week_of_year <= 51)).astype(float)
    return np.column_stack([np.ones_like(t), t, sin52, cos52, holiday])


class NaiveLinearModel:
    """Baseline: fits linear regression on contemporaneous spend (no adstock, no saturation)."""

    def __init__(self) -> None:
        self.reg = LinearRegression(fit_intercept=False)
        self.channel_coefs_: dict[str, float] = {}

    def fit(self, train_data: data.MarketingData) -> NaiveLinearModel:
        X_base = extract_base_features(train_data.weeks)
        X_spend = np.column_stack([train_data.spend[ch] for ch in config.CHANNELS])
        X = np.column_stack([X_base, X_spend])
        self.reg.fit(X, train_data.revenue)
        # Channel slopes are the last 3 coefficients
        for i, ch in enumerate(config.CHANNELS):
            self.channel_coefs_[ch] = float(self.reg.coef_[X_base.shape[1] + i])
        return self

    def predict(self, test_data: data.MarketingData) -> np.ndarray:
        X_base = extract_base_features(test_data.weeks)
        X_spend = np.column_stack([test_data.spend[ch] for ch in config.CHANNELS])
        X = np.column_stack([X_base, X_spend])
        return self.reg.predict(X)

    def marginal_roi(self, at_spend: dict[str, float]) -> dict[str, float]:
        """Linear model assumes constant marginal ROI equal to regression slope."""
        return {ch: self.channel_coefs_[ch] for ch in config.CHANNELS}


class AdstockOnlyModel:
    """Fits geometric adstock decay per channel, but assumes linear spend response (no saturation)."""

    def __init__(self) -> None:
        self.alphas_: dict[str, float] = {}
        self.reg = LinearRegression(fit_intercept=False)
        self.channel_coefs_: dict[str, float] = {}

    def fit(self, train_data: data.MarketingData) -> AdstockOnlyModel:
        X_base = extract_base_features(train_data.weeks)

        def loss(alphas: np.ndarray) -> float:
            Z = [
                data.geometric_adstock(train_data.spend[ch], alphas[i])
                for i, ch in enumerate(config.CHANNELS)
            ]
            X = np.column_stack([X_base, *Z])
            reg = LinearRegression(fit_intercept=False).fit(X, train_data.revenue)
            pred = reg.predict(X)
            return float(np.mean((train_data.revenue - pred) ** 2))

        init_alphas = np.array([0.5, 0.2, 0.3])
        bounds = [(0.05, 0.85)] * 3
        res = minimize(loss, init_alphas, method="L-BFGS-B", bounds=bounds)
        self.alphas_ = {ch: float(res.x[i]) for i, ch in enumerate(config.CHANNELS)}

        Z_opt = [
            data.geometric_adstock(train_data.spend[ch], self.alphas_[ch]) for ch in config.CHANNELS
        ]
        X_opt = np.column_stack([X_base, *Z_opt])
        self.reg.fit(X_opt, train_data.revenue)
        for i, ch in enumerate(config.CHANNELS):
            self.channel_coefs_[ch] = float(self.reg.coef_[X_base.shape[1] + i])
        return self

    def predict(self, test_data: data.MarketingData) -> np.ndarray:
        X_base = extract_base_features(test_data.weeks)
        Z = [
            data.geometric_adstock(test_data.spend[ch], self.alphas_[ch]) for ch in config.CHANNELS
        ]
        X = np.column_stack([X_base, *Z])
        return self.reg.predict(X)

    def marginal_roi(self, at_spend: dict[str, float]) -> dict[str, float]:
        return {ch: self.channel_coefs_[ch] for ch in config.CHANNELS}


class SaturationOnlyModel:
    """Fits Hill saturation diminishing returns, but assumes zero carryover decay."""

    def __init__(self) -> None:
        self.Ks_: dict[str, float] = {}
        self.max_lifts_: dict[str, float] = {}
        self.params_: np.ndarray | None = None

    def fit(self, train_data: data.MarketingData) -> SaturationOnlyModel:
        t = train_data.weeks

        def predict_local(params: np.ndarray) -> np.ndarray:
            b0, b_tr, b_sin, b_cos, b_hol = params[:5]
            base = (
                b0
                + b_tr * t
                + b_sin * np.sin(2.0 * np.pi * t / 52.0)
                + b_cos * np.cos(2.0 * np.pi * t / 52.0)
            )
            week_of_year = t % 52.0
            holiday = ((week_of_year >= 44) & (week_of_year <= 51)).astype(float)
            base = base + b_hol * holiday

            lift = np.zeros_like(t)
            for i, ch in enumerate(config.CHANNELS):
                k = params[5 + 2 * i]
                m = params[5 + 2 * i + 1]
                spend = train_data.spend[ch]
                lift += m * (spend / (spend + k))
            return base + lift

        def loss(params: np.ndarray) -> float:
            pred = predict_local(params)
            err = np.mean((train_data.revenue - pred) ** 2)
            prior = 1e-3 * ((params[0] - 100_000.0) / 10_000.0) ** 2
            return float(err + prior)

        init_params = np.array(
            [
                100_000.0,
                50.0,
                15_000.0,
                0.0,
                20_000.0,
                35_000.0,
                45_000.0,
                18_000.0,
                25_000.0,
                25_000.0,
                30_000.0,
            ]
        )
        bounds = [
            (80_000.0, 120_000.0),
            (0.0, 100.0),
            (5_000.0, 25_000.0),
            (-5_000.0, 5_000.0),
            (5_000.0, 35_000.0),
            (10_000.0, 70_000.0),
            (10_000.0, 70_000.0),
            (5_000.0, 40_000.0),
            (5_000.0, 50_000.0),
            (10_000.0, 50_000.0),
            (10_000.0, 60_000.0),
        ]
        res = minimize(loss, init_params, method="L-BFGS-B", bounds=bounds)
        self.params_ = res.x
        for i, ch in enumerate(config.CHANNELS):
            self.Ks_[ch] = float(res.x[5 + 2 * i])
            self.max_lifts_[ch] = float(res.x[5 + 2 * i + 1])
        return self

    def predict(self, test_data: data.MarketingData) -> np.ndarray:
        assert self.params_ is not None
        t = test_data.weeks
        b0, b_tr, b_sin, b_cos, b_hol = self.params_[:5]
        base = (
            b0
            + b_tr * t
            + b_sin * np.sin(2.0 * np.pi * t / 52.0)
            + b_cos * np.cos(2.0 * np.pi * t / 52.0)
        )
        week_of_year = t % 52.0
        holiday = ((week_of_year >= 44) & (week_of_year <= 51)).astype(float)
        base = base + b_hol * holiday

        lift = np.zeros_like(t)
        for ch in config.CHANNELS:
            k = self.Ks_[ch]
            m = self.max_lifts_[ch]
            spend = test_data.spend[ch]
            lift += m * (spend / (spend + k))
        return base + lift

    def marginal_roi(self, at_spend: dict[str, float]) -> dict[str, float]:
        mrois = {}
        for ch in config.CHANNELS:
            k = self.Ks_[ch]
            m = self.max_lifts_[ch]
            x = at_spend[ch]
            mrois[ch] = float(m * k / ((x + k) ** 2))
        return mrois


class AdstockHillMMM:
    """Headline model: joint geometric adstock carryover and Hill saturation diminishing returns."""

    def __init__(self) -> None:
        self.params_: np.ndarray | None = None
        self.alphas_: dict[str, float] = {}
        self.Ks_: dict[str, float] = {}
        self.max_lifts_: dict[str, float] = {}

    def fit(self, train_data: data.MarketingData) -> AdstockHillMMM:
        t = train_data.weeks

        def predict_local(params: np.ndarray) -> np.ndarray:
            b0, b_tr, b_sin, b_cos, b_hol = params[:5]
            base = (
                b0
                + b_tr * t
                + b_sin * np.sin(2.0 * np.pi * t / 52.0)
                + b_cos * np.cos(2.0 * np.pi * t / 52.0)
            )
            week_of_year = t % 52.0
            holiday = ((week_of_year >= 44) & (week_of_year <= 51)).astype(float)
            base = base + b_hol * holiday

            lift = np.zeros_like(t)
            for i, ch in enumerate(config.CHANNELS):
                alpha = params[5 + 3 * i]
                k = params[5 + 3 * i + 1]
                m = params[5 + 3 * i + 2]
                adstocked = data.geometric_adstock(train_data.spend[ch], alpha)
                lift += m * (adstocked / (adstocked + k))
            return base + lift

        def loss(params: np.ndarray) -> float:
            pred = predict_local(params)
            err = np.mean((train_data.revenue - pred) ** 2)
            prior = 1e-3 * ((params[0] - 100_000.0) / 10_000.0) ** 2
            return float(err + prior)

        init_params = np.array(
            [
                100_000.0,
                50.0,
                15_000.0,
                0.0,
                20_000.0,
                0.5,
                35_000.0,
                45_000.0,
                0.2,
                18_000.0,
                25_000.0,
                0.3,
                25_000.0,
                30_000.0,
            ]
        )
        bounds = [
            (80_000.0, 120_000.0),
            (0.0, 100.0),
            (5_000.0, 25_000.0),
            (-5_000.0, 5_000.0),
            (5_000.0, 35_000.0),
            (0.05, 0.85),
            (10_000.0, 70_000.0),
            (10_000.0, 70_000.0),
            (0.05, 0.85),
            (5_000.0, 40_000.0),
            (5_000.0, 50_000.0),
            (0.05, 0.85),
            (10_000.0, 50_000.0),
            (10_000.0, 60_000.0),
        ]
        res = minimize(loss, init_params, method="L-BFGS-B", bounds=bounds)
        self.params_ = res.x
        for i, ch in enumerate(config.CHANNELS):
            self.alphas_[ch] = float(res.x[5 + 3 * i])
            self.Ks_[ch] = float(res.x[5 + 3 * i + 1])
            self.max_lifts_[ch] = float(res.x[5 + 3 * i + 2])
        return self

    def predict(self, test_data: data.MarketingData) -> np.ndarray:
        assert self.params_ is not None
        t = test_data.weeks
        b0, b_tr, b_sin, b_cos, b_hol = self.params_[:5]
        base = (
            b0
            + b_tr * t
            + b_sin * np.sin(2.0 * np.pi * t / 52.0)
            + b_cos * np.cos(2.0 * np.pi * t / 52.0)
        )
        week_of_year = t % 52.0
        holiday = ((week_of_year >= 44) & (week_of_year <= 51)).astype(float)
        base = base + b_hol * holiday

        lift = np.zeros_like(t)
        for ch in config.CHANNELS:
            alpha = self.alphas_[ch]
            k = self.Ks_[ch]
            m = self.max_lifts_[ch]
            adstocked = data.geometric_adstock(test_data.spend[ch], alpha)
            lift += m * (adstocked / (adstocked + k))
        return base + lift

    def marginal_roi(self, at_spend: dict[str, float]) -> dict[str, float]:
        mrois = {}
        for ch in config.CHANNELS:
            k = self.Ks_[ch]
            m = self.max_lifts_[ch]
            x = at_spend[ch]
            mrois[ch] = float(m * k / ((x + k) ** 2))
        return mrois


class OracleModel:
    """Oracle upper bound: evaluates true underlying simulator response curves."""

    def __init__(self) -> None:
        pass

    def fit(self, train_data: data.MarketingData) -> OracleModel:
        return self

    def predict(self, test_data: data.MarketingData) -> np.ndarray:
        return test_data.true_base + test_data.true_incremental

    def marginal_roi(self, at_spend: dict[str, float]) -> dict[str, float]:
        return {ch: data.true_channel_marginal_roi(ch, at_spend[ch]) for ch in config.CHANNELS}
