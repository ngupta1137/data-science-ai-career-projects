import pytest

from price_elasticity import config, data, metrics, models


def test_elasticity_error():
    assert metrics.calculate_elasticity_error(-2.10, -2.15) == pytest.approx(0.05)
    assert metrics.calculate_elasticity_error(-0.50, -2.15) == pytest.approx(1.65)


def test_optimal_markup_price_formula():
    # Elastic good (beta = -2.0): optimal markup is -2.0 / (1 - 2.0) = 2.0x cost
    p_elastic = models.optimal_markup_price(wholesale_cost=10.0, elasticity=-2.0)
    assert p_elastic == pytest.approx(20.0)

    # Elastic good (beta = -3.0): optimal markup is -3.0 / (1 - 3.0) = 1.5x cost
    p_elastic_3 = models.optimal_markup_price(wholesale_cost=10.0, elasticity=-3.0)
    assert p_elastic_3 == pytest.approx(15.0)

    # Inelastic good (beta = -0.5): unconstrained formula diverges; capped at max_markup_ratio
    p_inelastic = models.optimal_markup_price(
        wholesale_cost=10.0, elasticity=-0.5, max_markup_ratio=4.5
    )
    assert p_inelastic == pytest.approx(45.0)


def test_models_fit_and_instruments():
    ds = data.generate_elasticity_data(
        n_products=config.FAST_TRAIN_PRODUCTS,
        n_weeks=config.FAST_WEEKS_PER_PRODUCT,
        seed=config.SEED,
    )

    ols = models.NaiveOLS().fit(ds)
    assert ols.elasticity is not None
    # Attenuated toward zero due to simultaneity
    assert ols.elasticity > -1.5

    tsls = models.TwoStageLeastSquares().fit(ds)
    assert tsls.elasticity is not None
    # 2SLS recovers elastic demand near -2.1
    assert tsls.elasticity < -1.8
    assert tsls.first_stage_f_stat is not None
    assert tsls.first_stage_f_stat > 100.0

    pred_q_ols = ols.predict_quantity(price=50.0, seasonality=0.0, competitor_price=3.5)
    pred_q_tsls = tsls.predict_quantity(price=50.0, seasonality=0.0, competitor_price=3.5)
    assert pred_q_ols > 0.0
    assert pred_q_tsls > 0.0


def test_pricing_simulation_and_gate_evaluation():
    train_ds = data.generate_elasticity_data(
        n_products=config.FAST_TRAIN_PRODUCTS,
        n_weeks=config.FAST_WEEKS_PER_PRODUCT,
        seed=config.SEED,
    )
    test_ds = data.generate_elasticity_data(
        n_products=config.FAST_TEST_PRODUCTS,
        n_weeks=config.FAST_WEEKS_PER_PRODUCT,
        seed=config.SEED + 100,
    )

    ols = models.NaiveOLS().fit(train_ds)
    tsls = models.TwoStageLeastSquares().fit(train_ds)
    oracle = models.OracleModel().fit(train_ds)

    ev_ols = metrics.simulate_pricing_policy(ols, test_ds)
    ev_tsls = metrics.simulate_pricing_policy(tsls, test_ds)
    ev_oracle = metrics.simulate_pricing_policy(oracle, test_ds)

    assert ev_oracle["total_profit"] > 0.0
    assert ev_tsls["total_profit"] > ev_ols["total_profit"]

    gate = metrics.evaluate_decision_gate(
        two_sls_eval=ev_tsls,
        naive_eval=ev_ols,
        oracle_eval=ev_oracle,
        ci_95=(-2.3, -1.9),
        true_elasticity=-2.15,
    )
    assert "checks" in gate
    assert gate["checks"]["beats_naive_ols_error"] is True
