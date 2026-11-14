import numpy as np
import pytest

from price_elasticity import config, data


def test_dataset_generation_shape():
    ds = data.generate_elasticity_data(n_products=6, n_weeks=10, seed=42)
    assert len(ds) == 60
    assert ds.total_observations == 60
    assert ds.n_products == 6
    assert ds.n_weeks == 10
    assert set(ds.category_counts.keys()) == set(config.PRODUCT_CATEGORIES.keys())


def test_data_validation_catches_invalid_values():
    ds = data.generate_elasticity_data(n_products=3, n_weeks=5, seed=42)

    # Valid dataset should pass
    data.validate(ds)

    # Corrupt wholesale cost
    corrupt_obs = list(ds.observations)
    corrupt_obs[0] = data.ProductWeekObservation(
        product_id=0,
        category="electronics",
        week=1,
        wholesale_cost=-10.0,
        retail_price=20.0,
        quantity_sold=5.0,
        seasonality=0.0,
        competitor_price=20.0,
        demand_shock=0.0,
        true_elasticity=-2.0,
        base_cost=15.0,
    )
    bad_ds = data.ElasticityDataset(
        observations=corrupt_obs,
        total_observations=len(corrupt_obs),
        n_products=ds.n_products,
        n_weeks=ds.n_weeks,
        category_counts=ds.category_counts,
        mean_price=ds.mean_price,
        mean_quantity=ds.mean_quantity,
        sha256="test",
    )
    with pytest.raises(data.DataValidationError, match="Wholesale cost must be positive"):
        data.validate(bad_ds)


def test_positive_elasticity_raises_error():
    ds = data.generate_elasticity_data(n_products=3, n_weeks=5, seed=42)
    corrupt_obs = list(ds.observations)
    corrupt_obs[0] = data.ProductWeekObservation(
        product_id=0,
        category="electronics",
        week=1,
        wholesale_cost=10.0,
        retail_price=20.0,
        quantity_sold=5.0,
        seasonality=0.0,
        competitor_price=20.0,
        demand_shock=0.0,
        true_elasticity=0.5,  # positive elasticity violates demand law
        base_cost=15.0,
    )
    bad_ds = data.ElasticityDataset(
        observations=corrupt_obs,
        total_observations=len(corrupt_obs),
        n_products=ds.n_products,
        n_weeks=ds.n_weeks,
        category_counts=ds.category_counts,
        mean_price=ds.mean_price,
        mean_quantity=ds.mean_quantity,
        sha256="test",
    )
    with pytest.raises(data.DataValidationError, match="Demand elasticity must be negative"):
        data.validate(bad_ds)


def test_deterministic_generation():
    ds1 = data.generate_elasticity_data(n_products=10, n_weeks=20, seed=123)
    ds2 = data.generate_elasticity_data(n_products=10, n_weeks=20, seed=123)
    assert ds1.sha256 == ds2.sha256
    assert ds1.mean_price == pytest.approx(ds2.mean_price)
    assert ds1.mean_quantity == pytest.approx(ds2.mean_quantity)


def test_endogeneity_correlation():
    ds = data.generate_elasticity_data(
        n_products=config.TRAIN_PRODUCTS,
        n_weeks=config.WEEKS_PER_PRODUCT,
        seed=config.SEED,
    )
    prices = np.array([o.retail_price for o in ds.observations])
    shocks = np.array([o.demand_shock for o in ds.observations])
    corr = float(np.corrcoef(prices, shocks)[0, 1])
    # Positive correlation between price and unobserved demand shock
    assert corr > 0.30


def test_instrument_relevance_correlation():
    ds = data.generate_elasticity_data(
        n_products=config.TRAIN_PRODUCTS,
        n_weeks=config.WEEKS_PER_PRODUCT,
        seed=config.SEED,
    )
    costs = np.array([np.log(o.wholesale_cost) for o in ds.observations])
    prices = np.array([np.log(o.retail_price) for o in ds.observations])
    corr = float(np.corrcoef(costs, prices)[0, 1])
    # Strong instrument relevance: wholesale cost strongly shifts retail price
    assert corr > 0.75
