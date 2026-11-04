from itertools import pairwise

import numpy as np
import pytest

from demand_forecast import metrics


def test_mae_rmse_bias_hand_values():
    y = [10.0, 20.0, 30.0]
    p = [12.0, 18.0, 33.0]
    # Errors: 2, -2, 3 (forecast minus actual).
    assert metrics.mae(y, p) == pytest.approx(7 / 3)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert metrics.bias(y, p) == pytest.approx(1.0)


def test_smape_hand_value():
    # |100 - 110| / ((100 + 110) / 2) = 10 / 105
    assert metrics.smape([100.0], [110.0]) == pytest.approx(10 / 105 * 100)


def test_smape_skips_double_zero():
    assert metrics.smape([0.0, 100.0], [0.0, 110.0]) == pytest.approx(10 / 105 * 100)


def test_seasonal_naive_scale_and_mase_hand_values():
    history = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Period 7: 8-1 and 9-2 are both 7, so the scale is 7.
    scale = metrics.seasonal_naive_scale(history, 7)
    assert scale == pytest.approx(7.0)
    assert metrics.mase([10.0, 10.0], [12.0, 8.0], scale) == pytest.approx(2.0 / 7.0)


def test_scale_needs_enough_history():
    with pytest.raises(ValueError):
        metrics.seasonal_naive_scale([1, 2, 3], 7)


def test_rolling_origins_never_run_past_the_end():
    origins = metrics.rolling_origins(n_rows=20, initial=10, step=3, horizon=2)
    assert origins == [10, 13, 16]
    assert origins[-1] + 2 <= 20


def test_rolling_origins_blocks_do_not_overlap_when_step_equals_horizon():
    origins = metrics.rolling_origins(n_rows=60, initial=20, step=7, horizon=7)
    blocks = [set(range(o, o + 7)) for o in origins]
    for a, b in pairwise(blocks):
        assert a.isdisjoint(b)


def test_block_bootstrap_is_deterministic_and_ordered():
    errors = np.random.default_rng(1).uniform(0, 10, size=(20, 7))

    def stat(idx):
        return errors[idx].mean()

    a = metrics.bootstrap_blocks_ci(stat, 20, n_boot=200, seed=3)
    b = metrics.bootstrap_blocks_ci(stat, 20, n_boot=200, seed=3)
    assert a == b
    assert a[0] <= errors.mean() <= a[1]


def test_segment_table_counts():
    table = metrics.segment_table([1.0, 2.0, 3.0], [2.0, 2.0, 5.0], ["a", "a", "b"])
    assert table["a"]["n"] == 2 and table["a"]["mae"] == pytest.approx(0.5)
    assert table["b"]["mae"] == pytest.approx(2.0)
