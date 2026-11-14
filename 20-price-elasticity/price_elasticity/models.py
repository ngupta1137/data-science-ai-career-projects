"""Econometric demand models: Naive OLS, 2SLS Instrumental Variables, and Oracle."""

from __future__ import annotations

import numpy as np

from . import config
from .data import ElasticityDataset


def optimal_markup_price(
    wholesale_cost: float,
    elasticity: float,
    max_markup_ratio: float = 4.5,
) -> float:
    """Compute gross-profit maximizing retail price given unit cost and demand elasticity.

    For elastic goods (elasticity < -1.0), optimal markup price is:
    P* = cost * (elasticity / (1 + elasticity)) = cost * (|beta| / (|beta| - 1)).

    For inelastic estimates (elasticity >= -1.05), standard unconstrained formula diverges
    or turns negative; pricing is bounded at a ceiling markup ratio.
    """
    if elasticity < -1.05:
        markup = float(elasticity / (1.0 + elasticity))
        return float(wholesale_cost * markup)
    # Bounded fallback when model erroneously estimates inelastic demand
    return float(wholesale_cost * max_markup_ratio)


def _build_covariates(
    dataset: ElasticityDataset,
    include_price_or_cost: np.ndarray | None = None,
    omit_seasonality: bool = False,
) -> np.ndarray:
    """Construct regression design matrix with category controls."""
    obs = dataset.observations
    n = len(obs)
    cols: list[np.ndarray] = [np.ones(n, dtype=float)]

    if include_price_or_cost is not None:
        cols.append(np.asarray(include_price_or_cost, dtype=float))

    if not omit_seasonality:
        season = np.array([o.seasonality for o in obs], dtype=float)
        cols.append(season)

    comp_p = np.array([o.competitor_price for o in obs], dtype=float)
    cols.append(comp_p)

    # Category fixed effect controls (reference: first category)
    categories = sorted(config.PRODUCT_CATEGORIES.keys())
    for cat in categories[1:]:
        cat_dummy = np.array([1.0 if o.category == cat else 0.0 for o in obs], dtype=float)
        cols.append(cat_dummy)

    return np.column_stack(cols)


class NaiveOLS:
    """Naive Ordinary Least Squares demand regression ignoring price endogeneity."""

    name: str = "naive_ols"

    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None
        self.elasticity: float | None = None
        self.intercept: float | None = None

    def fit(self, dataset: ElasticityDataset) -> NaiveOLS:
        obs = dataset.observations
        ln_p = np.array([np.log(o.retail_price) for o in obs], dtype=float)
        ln_q = np.array([np.log(o.quantity_sold) for o in obs], dtype=float)

        X = _build_covariates(dataset, include_price_or_cost=ln_p)
        coefs = np.linalg.lstsq(X, ln_q, rcond=None)[0]

        self.coefficients = coefs
        self.intercept = float(coefs[0])
        self.elasticity = float(coefs[1])
        return self

    def predict_quantity(
        self,
        price: float,
        seasonality: float,
        competitor_price: float,
        category: str = "electronics",
    ) -> float:
        if self.coefficients is None:
            raise RuntimeError("Model must be fitted before predicting.")
        ln_p = np.log(price)
        cols = [1.0, ln_p, seasonality, competitor_price]
        categories = sorted(config.PRODUCT_CATEGORIES.keys())
        for cat in categories[1:]:
            cols.append(1.0 if category == cat else 0.0)
        x = np.array(cols, dtype=float)
        ln_q = float(np.dot(self.coefficients, x))
        return float(np.exp(ln_q))

    def recommend_price(self, wholesale_cost: float, category: str | None = None) -> float:
        del category
        if self.elasticity is None:
            raise RuntimeError("Model must be fitted before recommending prices.")
        return optimal_markup_price(wholesale_cost, self.elasticity)


class TwoStageLeastSquares:
    """Two-Stage Least Squares (2SLS) Instrumental Variable estimator.

    Instruments endogenous retail price using exogenous wholesale cost shocks.
    """

    name: str = "two_stage_least_squares"

    def __init__(self) -> None:
        self.first_stage_coefs: np.ndarray | None = None
        self.second_stage_coefs: np.ndarray | None = None
        self.elasticity: float | None = None
        self.intercept: float | None = None
        self.first_stage_f_stat: float | None = None

    def fit(self, dataset: ElasticityDataset) -> TwoStageLeastSquares:
        obs = dataset.observations
        n = len(obs)

        ln_cost = np.array([np.log(o.wholesale_cost) for o in obs], dtype=float)
        ln_p = np.array([np.log(o.retail_price) for o in obs], dtype=float)
        ln_q = np.array([np.log(o.quantity_sold) for o in obs], dtype=float)

        # Stage 1: Regress endogenous ln_p on instrument ln_cost and covariates
        X1 = _build_covariates(dataset, include_price_or_cost=ln_cost)
        coefs1 = np.linalg.lstsq(X1, ln_p, rcond=None)[0]
        self.first_stage_coefs = coefs1

        ln_p_hat = X1 @ coefs1

        # Calculate first-stage F-statistic for instrument strength
        res_unrestricted = ln_p - ln_p_hat
        X1_restricted = _build_covariates(dataset, include_price_or_cost=None)
        coefs_restr = np.linalg.lstsq(X1_restricted, ln_p, rcond=None)[0]
        res_restricted = ln_p - (X1_restricted @ coefs_restr)

        rss_unrestr = float(np.sum(res_unrestricted**2))
        rss_restr = float(np.sum(res_restricted**2))
        df_num = 1
        df_denom = n - X1.shape[1]
        f_stat = ((rss_restr - rss_unrestr) / df_num) / (rss_unrestr / df_denom)
        self.first_stage_f_stat = float(f_stat)

        # Stage 2: Regress ln_q on fitted ln_p_hat and covariates
        X2 = _build_covariates(dataset, include_price_or_cost=ln_p_hat)
        coefs2 = np.linalg.lstsq(X2, ln_q, rcond=None)[0]
        self.second_stage_coefs = coefs2
        self.intercept = float(coefs2[0])
        self.elasticity = float(coefs2[1])

        return self

    def predict_quantity(
        self,
        price: float,
        seasonality: float,
        competitor_price: float,
        category: str = "electronics",
    ) -> float:
        if self.second_stage_coefs is None:
            raise RuntimeError("Model must be fitted before predicting.")
        ln_p = np.log(price)
        cols = [1.0, ln_p, seasonality, competitor_price]
        categories = sorted(config.PRODUCT_CATEGORIES.keys())
        for cat in categories[1:]:
            cols.append(1.0 if category == cat else 0.0)
        x = np.array(cols, dtype=float)
        ln_q = float(np.dot(self.second_stage_coefs, x))
        return float(np.exp(ln_q))

    def recommend_price(self, wholesale_cost: float, category: str | None = None) -> float:
        del category
        if self.elasticity is None:
            raise RuntimeError("Model must be fitted before recommending prices.")
        return optimal_markup_price(wholesale_cost, self.elasticity)


class OracleModel:
    """Theoretical benchmark knowing true causal elasticity parameters."""

    name: str = "oracle"

    def __init__(self, true_elasticity: float | None = None) -> None:
        self.elasticity = true_elasticity
        self.category_elasticities: dict[str, float] = {
            cat: float(cfg["true_elasticity"]) for cat, cfg in config.PRODUCT_CATEGORIES.items()
        }

    def fit(self, dataset: ElasticityDataset) -> OracleModel:
        # True population average elasticity
        true_betas = [o.true_elasticity for o in dataset.observations]
        self.elasticity = float(np.mean(true_betas))
        return self

    def recommend_price(self, wholesale_cost: float, category: str | None = None) -> float:
        if category is not None and category in self.category_elasticities:
            beta = self.category_elasticities[category]
        elif self.elasticity is not None:
            beta = self.elasticity
        else:
            beta = -2.13
        return optimal_markup_price(wholesale_cost, beta)
