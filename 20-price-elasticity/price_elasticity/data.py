"""Simulated e-commerce retail transactions with endogenous pricing and wholesale cost instruments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import config


class DataValidationError(ValueError):
    """Raised when transaction dataset fails schema or economic integrity checks."""


@dataclass
class ProductWeekObservation:
    product_id: int
    category: str
    week: int
    wholesale_cost: float  # Exogenous instrument Z (wholesale unit cost in USD)
    retail_price: float  # Endogenous price P (in USD)
    quantity_sold: float  # Observed sales volume Q
    seasonality: float  # Observable seasonal demand shifter X_1
    competitor_price: float  # Observable competitor average price X_2
    demand_shock: float  # Unobserved demand shock u (confounder)
    true_elasticity: float  # Known ground truth beta for this product
    base_cost: float


@dataclass
class ElasticityDataset:
    observations: list[ProductWeekObservation]
    total_observations: int
    n_products: int
    n_weeks: int
    category_counts: dict[str, int]
    mean_price: float
    mean_quantity: float
    sha256: str

    def __len__(self) -> int:
        return len(self.observations)


def generate_elasticity_data(
    n_products: int,
    n_weeks: int = config.WEEKS_PER_PRODUCT,
    seed: int = config.SEED,
) -> ElasticityDataset:
    """Generate simulated weekly sales transactions with known causal elasticity."""
    rng = np.random.default_rng(seed)
    categories = list(config.PRODUCT_CATEGORIES.keys())

    observations: list[ProductWeekObservation] = []
    category_counts: dict[str, int] = dict.fromkeys(categories, 0)
    prices: list[float] = []
    quantities: list[float] = []

    for pid in range(n_products):
        # Assign category cyclically
        cat_name = categories[pid % len(categories)]
        category_counts[cat_name] += 1
        cat_params = config.PRODUCT_CATEGORIES[cat_name]

        true_beta = float(cat_params["true_elasticity"])
        base_cost = float(cat_params["base_cost"])
        cost_std = float(cat_params["cost_std"])
        base_vol = float(cat_params["base_volume"])

        # Product-specific baseline intercept
        prod_alpha = base_vol + float(rng.normal(0, 0.2))

        for week in range(1, n_weeks + 1):
            # 1. Observable exogenous covariates:
            # Seasonality: cyclic annual pattern with noise
            season_angle = (2.0 * np.pi * (week % 52)) / 52.0
            seasonality = float(0.4 * np.sin(season_angle) + rng.normal(0, 0.15))

            # Competitor price index
            comp_price = float(np.log(base_cost * 1.6) + rng.normal(0, 0.2))

            # 2. Exogenous Instrument Z: Wholesale cost shock
            # (Raw material commodity fluctuation, import freight tariff)
            # Shifts retail price, but strictly independent of local retail demand shocks!
            cost_shock = float(rng.normal(0, cost_std))
            wholesale_cost = max(5.0, base_cost + cost_shock)
            ln_cost = float(np.log(wholesale_cost))

            # 3. Unobserved retail demand shock u:
            # (Viral social trend, sudden willingness to pay, store traffic surge)
            demand_shock = float(rng.normal(0, 0.45))

            # 4. Endogenous retail pricing equation:
            # Retail pricing managers and dynamic pricing algorithms mark up wholesale cost,
            # but ALSO raise prices when demand is strong (demand_shock > 0)!
            # In logs: ln P = 0.5 + 0.85 * ln_cost + 0.32 * demand_shock + 0.15 * seasonality + eps
            ln_price = (
                0.55
                + 0.85 * ln_cost
                + 0.32 * demand_shock
                + 0.15 * seasonality
                + 0.10 * comp_price
                + float(rng.normal(0, 0.08))
            )
            retail_price = float(np.exp(ln_price))

            # 5. True demand equation (Quantity sold Q):
            # ln Q = alpha + beta * ln P + 0.5 * seasonality + 0.4 * comp_price + demand_shock + eps
            ln_quantity = (
                prod_alpha
                + true_beta * ln_price
                + 0.50 * seasonality
                + 0.40 * comp_price
                + demand_shock
                + float(rng.normal(0, 0.12))
            )
            # Ensure positive sales volume
            quantity = float(max(0.1, np.exp(ln_quantity)))

            obs = ProductWeekObservation(
                product_id=pid,
                category=cat_name,
                week=week,
                wholesale_cost=wholesale_cost,
                retail_price=retail_price,
                quantity_sold=quantity,
                seasonality=seasonality,
                competitor_price=comp_price,
                demand_shock=demand_shock,
                true_elasticity=true_beta,
                base_cost=base_cost,
            )
            observations.append(obs)
            prices.append(retail_price)
            quantities.append(quantity)

    # Compute deterministic SHA256 fingerprint
    total_obs = len(observations)
    hasher = hashlib.sha256()
    hasher.update(f"products={n_products},weeks={n_weeks},seed={seed},obs={total_obs}".encode())
    for cat in sorted(categories):
        hasher.update(f"{cat}:{category_counts[cat]}".encode())
    sha = hasher.hexdigest()

    dataset = ElasticityDataset(
        observations=observations,
        total_observations=total_obs,
        n_products=n_products,
        n_weeks=n_weeks,
        category_counts=category_counts,
        mean_price=float(np.mean(prices)),
        mean_quantity=float(np.mean(quantities)),
        sha256=sha,
    )
    validate(dataset)
    return dataset


def validate(dataset: ElasticityDataset) -> None:
    """Validate elasticity transaction schema and logical bounds."""
    if dataset.total_observations <= 0:
        raise DataValidationError("Dataset must contain at least one observation.")
    if dataset.n_products <= 0 or dataset.n_weeks <= 0:
        raise DataValidationError("Products and weeks must be strictly positive.")
    if dataset.total_observations != dataset.n_products * dataset.n_weeks:
        raise DataValidationError("Total observations mismatch product x weeks count.")
    for cat, count in dataset.category_counts.items():
        if count <= 0:
            raise DataValidationError(f"Category {cat} has zero product count.")
    for obs in dataset.observations:
        if obs.wholesale_cost <= 0.0:
            raise DataValidationError(f"Wholesale cost must be positive: {obs.wholesale_cost}.")
        if obs.retail_price <= 0.0:
            raise DataValidationError(f"Retail price must be positive: {obs.retail_price}.")
        if obs.quantity_sold <= 0.0:
            raise DataValidationError(f"Quantity sold must be positive: {obs.quantity_sold}.")
        if obs.true_elasticity >= 0.0:
            raise DataValidationError(f"Demand elasticity must be negative: {obs.true_elasticity}.")
        if obs.category not in config.PRODUCT_CATEGORIES:
            raise DataValidationError(f"Unknown category: {obs.category}.")
