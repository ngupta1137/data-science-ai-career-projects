import numpy as np
import pytest

from uplift_eval import models


def trial(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    t = rng.integers(0, 2, n)
    tau = 0.25 * np.tanh(2 * x[:, 0])  # the offer helps when x0 is high and hurts when it is low
    base = 0.3
    y = (rng.random(n) < np.clip(base + t * tau, 0, 1)).astype(int)
    return x, t, y, tau


@pytest.mark.parametrize("name", list(models.LEARNERS))
def test_learner_recovers_the_direction_of_the_effect(name):
    x, t, y, _ = trial()
    xt, _, _, tau_t = trial(seed=1)
    pred = models.LEARNERS[name]().fit(x, t, y).uplift(xt)
    assert pred.shape == (len(xt),)
    assert np.corrcoef(pred, tau_t)[0, 1] > 0.5


def test_class_transformation_matches_the_average_effect_in_a_trial():
    x, t, y, tau = trial()
    pred = models.CtGbm().fit(x, t, y).uplift(x)
    assert pred.mean() == pytest.approx(tau.mean(), abs=0.03)


def test_s_linear_design_has_interactions():
    x = np.ones((2, 3))
    d = models.SLinear._design(x, np.array([0, 1]))
    assert (
        d.shape == (2, 7)
        and d[0].tolist() == [1, 1, 1, 0, 0, 0, 0]
        and d[1].tolist() == [1, 1, 1, 1, 1, 1, 1]
    )


def test_response_model_is_a_probability_not_an_uplift():
    x, t, y, tau = trial()
    score = models.ResponseGbm().fit(x, t, y).uplift(x)
    assert score.min() >= 0 and score.max() <= 1
    # it ignores the offer, so it tells nothing about who the offer changes
    assert abs(np.corrcoef(score, tau)[0, 1]) < abs(
        np.corrcoef(models.TGbm().fit(x, t, y).uplift(x), tau)[0, 1]
    )
