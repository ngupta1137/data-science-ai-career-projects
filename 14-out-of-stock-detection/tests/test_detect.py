import numpy as np
import pytest

from oos_detect import config, data, detect, metrics


def test_zero_runs_hand_computed():
    s = np.array([[1, 0, 0, 2, 0]])
    assert detect.zero_runs(s).tolist() == [[0, 1, 2, 0, 1]]


def test_clean_rate_drops_long_zero_runs():
    h = np.array([[2, 2, 0, 0, 0, 0, 0, 2, 2, 0]])
    # the five-day run is dropped; the single zero at the end is kept
    assert detect.rate_clean(h, min_run=5)[0] == pytest.approx(8 / 5)
    assert detect.rate_naive(h)[0] == pytest.approx(8 / 10)


def test_posterior_hand_computed():
    p = detect.posterior_empty(np.array([1.0]), np.array([2]), prior=0.1)[0]
    like = np.exp(-2.0)
    assert p == pytest.approx(0.1 / (0.1 + 0.9 * like))
    assert detect.posterior_empty(np.array([1.0]), np.array([0]), prior=0.1)[0] == pytest.approx(
        0.1
    )


def test_rule_eligibility():
    run = np.array([0, 2, 3, 5])
    _, ok = detect.score("zero_run_rule", run, np.ones(4), np.ones(4))
    assert ok.tolist() == [False, False, True, True]


def test_no_visits_has_no_visits_and_oracle_is_perfectly_precise(rng):
    w = data.generate(rng, 2, 50, config.HISTORY_DAYS + 30)
    none = detect.simulate(w, "no_visits", 10)
    orc = detect.simulate(w, "oracle", 10)
    assert none["visits"].sum() == 0
    assert orc["useful_visits"].sum() == orc["visits"].sum()
    assert orc["lost_margin"].sum() <= none["lost_margin"].sum()


def test_capacity_is_respected(rng):
    w = data.generate(rng, 2, 100, config.HISTORY_DAYS + 20)
    o = detect.simulate(w, "zero_run_rule", 7)
    assert o["visits"].sum() <= 7 * 20


def test_net_value_and_bootstrap(rng):
    out = {"lost_margin": np.array([1.0, 0.0]), "visits": np.array([0, 1])}
    v = metrics.net_value_per_store_day(np.array([4.0, 6.0]), out, np.array([0, 1]), 2)
    assert v.tolist() == pytest.approx(
        [3 / config.EVAL_DAYS, (6 - config.VISIT_COST) / config.EVAL_DAYS]
    )
    ci = metrics.boot_mean_ci(np.array([1.0, 1.0, 1.0]), rng, 50)
    assert ci["ci95"] == [1.0, 1.0]


def test_validate_and_fingerprint(rng):
    w = data.generate(rng, 1, 5, config.HISTORY_DAYS + 2)
    data.validate(w)
    w.duration[0, 0] = 0
    with pytest.raises(data.DataValidationError):
        data.validate(w)
    assert data.fingerprint() == data.fingerprint()
