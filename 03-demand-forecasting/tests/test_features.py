import numpy as np
import pytest

from demand_forecast import config
from demand_forecast.features import build_features, first_valid_row

TARGET_DERIVED = ["lag_7", "lag_14", "lag_21", "roll7_at_horizon", "roll28_at_horizon"]


def test_first_valid_row_is_set_by_the_longest_window(synthetic_frame):
    # roll28 of the series shifted by 7 needs 7 + 27 = 34 earlier days.
    assert first_valid_row(build_features(synthetic_frame)) == 34


def test_features_at_row_t_do_not_depend_on_the_last_horizon_minus_one_days(synthetic_frame):
    """The core no-leakage check. Row t may use only y[: t - HORIZON + 1]."""
    t = 400
    base = build_features(synthetic_frame).loc[t, TARGET_DERIVED]
    altered = synthetic_frame.copy()
    altered.loc[t - config.HORIZON + 1 :, config.TARGET] += 10_000
    after = build_features(altered).loc[t, TARGET_DERIVED]
    assert np.allclose(base.to_numpy(), after.to_numpy())


def test_the_perturbation_test_can_fail(synthetic_frame):
    """Test the test: the short-lag variant must be caught by the same check."""
    t = 400
    base = build_features(synthetic_frame, "short_lag").loc[t, ["lag_1", "lag_2"]]
    altered = synthetic_frame.copy()
    altered.loc[t - 2 :, config.TARGET] += 10_000
    after = build_features(altered, "short_lag").loc[t, ["lag_1", "lag_2"]]
    assert not np.allclose(base.to_numpy(), after.to_numpy())


def test_honest_features_exclude_same_day_components(synthetic_frame):
    columns = set(build_features(synthetic_frame).columns)
    assert not {"casual_same_day", "registered_same_day", "lag_1", "lag_2"} & columns
    assert config.TARGET not in columns


def test_leaky_variants_add_the_expected_columns(synthetic_frame):
    leak = set(build_features(synthetic_frame, "same_day_components").columns)
    assert {"casual_same_day", "registered_same_day"} <= leak
    short = set(build_features(synthetic_frame, "short_lag").columns)
    assert {"lag_1", "lag_2"} <= short


def test_unknown_variant_is_rejected(synthetic_frame):
    with pytest.raises(ValueError):
        build_features(synthetic_frame, "nope")


def test_calendar_features_are_one_hot_and_complete(synthetic_frame):
    features = build_features(synthetic_frame)
    dow = features[[f"dow_{d}" for d in range(7)]]
    assert (dow.sum(axis=1) == 1).all()
