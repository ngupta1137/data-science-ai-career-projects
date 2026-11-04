import numpy as np
import pytest

from fraud_detection import config, splits
from fraud_detection.features import build_features


def test_time_split_orders_and_partitions(synthetic_frame):
    t = synthetic_frame["Time"]
    fit, val, test = splits.time_split(t, config.TEST_FRAC, config.VAL_FRAC)
    splits.assert_time_order(t, fit, val, test)
    assert len(fit) + len(val) + len(test) == len(synthetic_frame)


def test_time_split_sizes_follow_config(synthetic_frame):
    n = len(synthetic_frame)
    _, val, test = splits.time_split(synthetic_frame["Time"], config.TEST_FRAC, config.VAL_FRAC)
    assert len(test) == round(n * config.TEST_FRAC)
    assert len(val) == round((n - len(test)) * config.VAL_FRAC)


def test_time_split_works_on_shuffled_input(synthetic_frame):
    shuffled = synthetic_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    fit, val, test = splits.time_split(shuffled["Time"], config.TEST_FRAC, config.VAL_FRAC)
    splits.assert_time_order(shuffled["Time"], fit, val, test)


def test_assert_time_order_detects_a_swapped_split(synthetic_frame):
    t = synthetic_frame["Time"]
    fit, val, test = splits.time_split(t, config.TEST_FRAC, config.VAL_FRAC)
    with pytest.raises(AssertionError):
        splits.assert_time_order(t, test, val, fit)


def test_random_split_does_mix_periods(synthetic_frame):
    """This is the trap. A random split puts late rows in train and early rows in test."""
    t = synthetic_frame["Time"].to_numpy()
    train, test = splits.random_split(len(synthetic_frame), config.TEST_FRAC, seed=1)
    assert t[train].max() > t[test].min()


def test_features_exclude_time_and_label(synthetic_frame):
    features = build_features(synthetic_frame)
    assert config.TARGET not in features.columns
    assert "Time" not in features.columns


def test_features_do_not_depend_on_the_label(synthetic_frame):
    altered = synthetic_frame.copy()
    altered[config.TARGET] = 1 - altered[config.TARGET]
    assert build_features(synthetic_frame).equals(build_features(altered))


def test_features_are_finite(synthetic_frame):
    assert np.isfinite(build_features(synthetic_frame).to_numpy(dtype=float)).all()
