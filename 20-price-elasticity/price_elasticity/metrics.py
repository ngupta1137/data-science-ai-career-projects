"""Evaluation metrics, counterfactual profit simulation, and decision gate checks."""

from __future__ import annotations

import numpy as np

from . import config
from .data import ElasticityDataset


def calculate_elasticity_error(estimated_beta: float, true_beta: float) -> float:
    """Compute absolute estimation error against true causal elasticity."""
    return float(abs(estimated_beta - true_beta))


def simulate_pricing_policy(
    model,
    dataset: ElasticityDataset,
) -> dict:
    """Simulate profit outcomes on a holdout cohort under the model's pricing policy."""
    profits: list[float] = []
    prices: list[float] = []
    quantities: list[float] = []
    cat_profits: dict[str, list[float]] = {cat: [] for cat in config.PRODUCT_CATEGORIES}

    for obs in dataset.observations:
        # 1. Model recommends retail price given product's wholesale cost and category
        try:
            rec_price = model.recommend_price(obs.wholesale_cost, category=obs.category)
        except TypeError:
            rec_price = model.recommend_price(obs.wholesale_cost)

        # 2. Counterfactual quantity under true causal price elasticity
        # ln Q(P_rec) = ln Q(P_actual) + beta_true * (ln P_rec - ln P_actual)
        price_ratio = rec_price / obs.retail_price
        cf_quantity = obs.quantity_sold * (price_ratio**obs.true_elasticity)
        cf_quantity = float(max(0.001, cf_quantity))

        # 3. Gross profit: (Price - Unit Cost) * Volume
        unit_margin = rec_price - obs.wholesale_cost
        gross_profit = float(unit_margin * cf_quantity)

        profits.append(gross_profit)
        prices.append(rec_price)
        quantities.append(cf_quantity)
        cat_profits[obs.category].append(gross_profit)

    total_profit = float(np.sum(profits))
    mean_profit = float(np.mean(profits))
    mean_price = float(np.mean(prices))
    mean_qty = float(np.mean(quantities))

    category_summary = {}
    for cat, p_list in cat_profits.items():
        category_summary[cat] = {
            "total_profit": float(np.sum(p_list)),
            "mean_profit": float(np.mean(p_list)),
        }

    return {
        "model_name": getattr(model, "name", "unknown"),
        "estimated_elasticity": getattr(model, "elasticity", None),
        "total_profit": total_profit,
        "mean_profit_per_transaction": mean_profit,
        "mean_price": mean_price,
        "mean_quantity": mean_qty,
        "category_profits": category_summary,
    }


def bootstrap_elasticity_ci(
    dataset: ElasticityDataset,
    n_boot: int = 300,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval on 2SLS elasticity estimate."""
    rng = np.random.default_rng(seed)
    obs = dataset.observations
    n = len(obs)

    ln_cost = np.array([np.log(o.wholesale_cost) for o in obs], dtype=float)
    ln_p = np.array([np.log(o.retail_price) for o in obs], dtype=float)
    ln_q = np.array([np.log(o.quantity_sold) for o in obs], dtype=float)
    season = np.array([o.seasonality for o in obs], dtype=float)
    comp_p = np.array([o.competitor_price for o in obs], dtype=float)
    categories = sorted(config.PRODUCT_CATEGORIES.keys())
    cat_dummies = [
        np.array([1.0 if o.category == cat else 0.0 for o in obs], dtype=float)
        for cat in categories[1:]
    ]

    estimates: list[float] = []
    ones = np.ones(n, dtype=float)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_idx = season[idx]
        cp_idx = comp_p[idx]
        cd_idx = [cd[idx] for cd in cat_dummies]

        X1 = np.column_stack([ones, ln_cost[idx], s_idx, cp_idx, *cd_idx])
        coef1 = np.linalg.lstsq(X1, ln_p[idx], rcond=None)[0]
        p_hat = X1 @ coef1

        X2 = np.column_stack([ones, p_hat, s_idx, cp_idx, *cd_idx])
        coef2 = np.linalg.lstsq(X2, ln_q[idx], rcond=None)[0]
        estimates.append(float(coef2[1]))

    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def evaluate_decision_gate(
    two_sls_eval: dict,
    naive_eval: dict,
    oracle_eval: dict,
    ci_95: tuple[float, float],
    true_elasticity: float,
) -> dict:
    """Pre-registered decision gate check."""
    est_2sls = two_sls_eval["estimated_elasticity"]
    est_naive = naive_eval["estimated_elasticity"]

    err_2sls = calculate_elasticity_error(est_2sls, true_elasticity)
    err_naive = calculate_elasticity_error(est_naive, true_elasticity)

    oracle_prof = oracle_eval["total_profit"]
    recovery_2sls = float(two_sls_eval["total_profit"] / oracle_prof) if oracle_prof > 0 else 0.0
    recovery_naive = float(naive_eval["total_profit"] / oracle_prof) if oracle_prof > 0 else 0.0

    checks = {
        "elasticity_estimation_error_below_threshold": bool(err_2sls < config.MAX_ELASTICITY_ERROR),
        "beats_naive_ols_error": bool(err_2sls < err_naive),
        "profit_recovery_meets_threshold": bool(
            recovery_2sls >= config.MIN_PROFIT_RECOVERY
            and recovery_naive < config.MAX_NAIVE_RECOVERY
        ),
    }

    return {
        "rule": config.GATE_RULE,
        "headline_model": config.HEADLINE_MODEL,
        "true_elasticity": float(true_elasticity),
        "estimated_elasticity_2sls": float(est_2sls),
        "estimated_elasticity_naive": float(est_naive),
        "elasticity_error_2sls": float(err_2sls),
        "elasticity_error_naive": float(err_naive),
        "elasticity_ci_95": [float(ci_95[0]), float(ci_95[1])],
        "profit_recovery_ratio_2sls": float(recovery_2sls),
        "profit_recovery_ratio_naive": float(recovery_naive),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def compute_traps(
    dataset: ElasticityDataset,
    naive_eval: dict,
    two_sls_eval: dict,
    oracle_eval: dict,
    f_stat: float,
) -> dict:
    """Measure the three pre-registered econometric traps."""
    from .models import _build_covariates

    # Trap 1: Simultaneity / endogeneity attenuation bias
    true_betas = [o.true_elasticity for o in dataset.observations]
    true_avg_beta = float(np.mean(true_betas))

    naive_beta = naive_eval["estimated_elasticity"]
    two_sls_beta = two_sls_eval["estimated_elasticity"]

    # Correlation between price and unobserved demand shock
    prices = np.array([o.retail_price for o in dataset.observations], dtype=float)
    shocks = np.array([o.demand_shock for o in dataset.observations], dtype=float)
    price_shock_corr = float(np.corrcoef(prices, shocks)[0, 1])

    # Trap 2: Profit collapse from naive pricing
    oracle_prof = oracle_eval["total_profit"]
    naive_prof = naive_eval["total_profit"]
    two_sls_prof = two_sls_eval["total_profit"]

    profit_loss_naive = float((oracle_prof - naive_prof) / oracle_prof) if oracle_prof > 0 else 0.0
    profit_loss_2sls = float((oracle_prof - two_sls_prof) / oracle_prof) if oracle_prof > 0 else 0.0

    # Trap 3: Omitted variable bias without seasonality
    obs = dataset.observations
    ln_p = np.array([np.log(o.retail_price) for o in obs], dtype=float)
    ln_q = np.array([np.log(o.quantity_sold) for o in obs], dtype=float)

    X_with_season = _build_covariates(dataset, include_price_or_cost=ln_p, omit_seasonality=False)
    coef_with_season = np.linalg.lstsq(X_with_season, ln_q, rcond=None)[0]
    beta_with_season = float(coef_with_season[1])

    X_omitted = _build_covariates(dataset, include_price_or_cost=ln_p, omit_seasonality=True)
    coef_omitted = np.linalg.lstsq(X_omitted, ln_q, rcond=None)[0]
    omitted_beta = float(coef_omitted[1])

    return {
        "simultaneity_bias": {
            "price_demand_shock_correlation": float(price_shock_corr),
            "true_elasticity": float(true_avg_beta),
            "naive_ols_elasticity": float(naive_beta),
            "two_sls_elasticity": float(two_sls_beta),
            "first_stage_f_statistic": float(f_stat),
            "attenuation_bias_magnitude": float(naive_beta - true_avg_beta),
            "description": (
                "Retail prices rise during peak demand shocks, creating positive correlation "
                "between price and unobserved demand. Naive OLS misinterprets this as inelastic "
                "demand, while 2SLS purges the endogeneity via wholesale cost shocks."
            ),
        },
        "profit_destruction": {
            "oracle_maximum_profit": float(oracle_prof),
            "two_sls_recovered_profit": float(two_sls_prof),
            "naive_recovered_profit": float(naive_prof),
            "naive_profit_loss_pct": float(profit_loss_naive * 100.0),
            "two_sls_profit_loss_pct": float(profit_loss_2sls * 100.0),
            "description": (
                "Believing demand is inelastic, naive pricing sets exorbitant markups, "
                "crushing sales volume on truly elastic goods and destroying gross profit."
            ),
        },
        "omitted_variable_bias": {
            "naive_ols_with_seasonality": float(beta_with_season),
            "naive_ols_omitting_seasonality": float(omitted_beta),
            "additional_bias_from_omission": float(omitted_beta - beta_with_season),
            "description": (
                "Omitting macro demand shifters like holiday seasonality exacerbates price "
                "endogeneity, as pricing algorithms raise prices during holiday surges."
            ),
        },
    }
