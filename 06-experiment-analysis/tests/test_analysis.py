import numpy as np
import pytest

from experiment_analysis import analysis, data


def test_diff_and_se_hand_values():
    y_t = np.array([[2.0, 4.0, 6.0]])
    y_c = np.array([[1.0, 2.0, 3.0]])
    diff, se = analysis.diff_and_se(y_t, y_c)
    assert diff[0] == pytest.approx(2.0)
    # var_t = 4, var_c = 1, se = sqrt(4/3 + 1/3)
    assert se[0] == pytest.approx(np.sqrt(5 / 3))


def test_cuped_removes_a_covariate_that_fully_explains_the_outcome(rng):
    x_t, x_c = rng.normal(size=(1, 400)), rng.normal(size=(1, 400))
    yt, yc = analysis.cuped_adjust(x_t, 2 * x_t, x_c, 2 * x_c)
    assert np.allclose(yt, yt[0, 0]) and np.allclose(yc, yt[0, 0])


def test_cuped_adjusted_outcome_is_uncorrelated_with_covariate(rng):
    d = data.draw(rng, reps=1, n=5000, rho=0.7)
    yt, yc = analysis.cuped_adjust(d["x_t"], d["y_t"], d["x_c"], d["y_c"])
    x = np.concatenate([d["x_t"], d["x_c"]], axis=-1)[0]
    y = np.concatenate([yt, yc], axis=-1)[0]
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-10


def test_z_crit_and_p_value_agree():
    assert analysis.z_crit(0.05) == pytest.approx(1.959964, abs=1e-5)
    assert analysis.p_value(np.array([1.959964]), np.array([1.0]))[0] == pytest.approx(
        0.05, abs=1e-6
    )


def test_one_look_equals_the_final_z(rng):
    d = data.draw(rng, reps=4, n=200)
    diff, se = analysis.diff_and_se(d["y_t"], d["y_c"])
    assert np.allclose(analysis.looks_z(d["y_t"], d["y_c"], 1)[:, 0], diff / se)


def test_looks_require_divisible_n(rng):
    d = data.draw(rng, reps=1, n=101)
    with pytest.raises(ValueError):
        analysis.looks_z(d["y_t"], d["y_c"], 5)


def test_holm_hand_example_matches_bonferroni_when_only_one_passes():
    p = np.array([[0.01, 0.04, 0.03]])
    expected = np.array([[True, False, False]])
    assert (analysis.holm_reject(p, 0.05) == expected).all()
    assert (analysis.bonferroni_reject(p, 0.05) == expected).all()


def test_holm_rejects_more_than_bonferroni():
    p = np.array([[0.001, 0.012, 0.02]])
    assert analysis.holm_reject(p, 0.05).tolist() == [[True, True, True]]
    assert analysis.bonferroni_reject(p, 0.05).tolist() == [[True, True, False]]


def test_holm_is_order_independent():
    p = np.array([[0.03, 0.01, 0.04]])
    assert analysis.holm_reject(p, 0.05).tolist() == [[False, True, False]]
