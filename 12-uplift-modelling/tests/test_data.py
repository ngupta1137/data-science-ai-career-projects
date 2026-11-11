import numpy as np
import pytest

from uplift_eval import config, data


def test_tau_is_the_difference_of_the_two_chances():
    c = data.generate(np.random.default_rng(0), 1000)
    assert np.allclose(c.tau, c.p1 - c.p0)


def test_trial_propensity_is_a_constant_coin_flip():
    c = data.generate(np.random.default_rng(1), 20000)
    assert (c.propensity == config.TREATMENT_SHARE).all()
    assert c.t.mean() == pytest.approx(0.5, abs=0.02)


def test_outcomes_follow_the_assigned_condition():
    c = data.generate(np.random.default_rng(2), 200000)
    assert c.y[c.t == 1].mean() == pytest.approx(c.p1[c.t == 1].mean(), abs=0.01)
    assert c.y[c.t == 0].mean() == pytest.approx(c.p0[c.t == 0].mean(), abs=0.01)


def test_some_customers_are_harmed_by_the_offer():
    c = data.generate(np.random.default_rng(3), 50000)
    assert 0.2 < (c.tau < 0).mean() < 0.5
    assert c.tau.mean() > 0


def test_observational_offers_follow_buying_signals_and_bias_the_naive_effect():
    rng = np.random.default_rng(4)
    trial = data.generate(rng, 100000)
    obs = data.generate(rng, 100000, randomised=False)
    naive_trial = trial.y[trial.t == 1].mean() - trial.y[trial.t == 0].mean()
    naive_obs = obs.y[obs.t == 1].mean() - obs.y[obs.t == 0].mean()
    assert naive_trial == pytest.approx(trial.tau.mean(), abs=0.01)
    assert naive_obs > obs.tau.mean() + 0.1  # confounded
    assert obs.p0[obs.t == 1].mean() > obs.p0[obs.t == 0].mean()


def test_validate():
    c = data.generate(np.random.default_rng(5), 100)
    data.validate(c)
    c.t[:] = 1
    with pytest.raises(data.DataValidationError):
        data.validate(c)


def test_deterministic_and_splits_independent():
    a, b = data.make_splits(1), data.make_splits(1)
    assert np.array_equal(a["test"].y, b["test"].y)
    assert not np.array_equal(a["train"].x[:50], a["test"].x[:50])


def test_fingerprint_roundtrip_and_mismatch(tmp_path):
    f = tmp_path / "sum.txt"
    digest = data.write_fingerprint(f)
    assert data.verify_fingerprint(f) == digest
    f.write_text("0" * 64 + "  x\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(f)


def test_committed_fingerprint_matches_generator():
    assert data.verify_fingerprint() == data.fingerprint()
