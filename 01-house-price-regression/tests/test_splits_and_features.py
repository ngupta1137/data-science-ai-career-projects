import numpy as np
import pytest

from house_price import config, splits
from house_price.features import build_features


def test_spatial_split_has_no_cell_overlap(synthetic_frame):
    groups = splits.add_geo_group(synthetic_frame)
    train_idx, test_idx = splits.spatial_split(groups, config.TEST_SIZE, seed=1)
    assert splits.group_overlap(groups, train_idx, test_idx) == set()
    splits.assert_no_overlap(groups, train_idx, test_idx)


def test_spatial_split_partitions_all_rows(synthetic_frame):
    groups = splits.add_geo_group(synthetic_frame)
    train_idx, test_idx = splits.spatial_split(groups, config.TEST_SIZE, seed=1)
    assert len(train_idx) + len(test_idx) == len(synthetic_frame)
    assert set(train_idx).isdisjoint(set(test_idx))


def test_random_split_does_leak_cells(synthetic_frame):
    """This is the trap. A random split puts the same cell on both sides."""
    groups = splits.add_geo_group(synthetic_frame)
    train_idx, test_idx = splits.random_split(len(synthetic_frame), config.TEST_SIZE, seed=1)
    assert len(splits.group_overlap(groups, train_idx, test_idx)) > 0
    with pytest.raises(AssertionError):
        splits.assert_no_overlap(groups, train_idx, test_idx)


def test_split_is_deterministic(synthetic_frame):
    groups = splits.add_geo_group(synthetic_frame)
    a = splits.spatial_split(groups, config.TEST_SIZE, seed=3)
    b = splits.spatial_split(groups, config.TEST_SIZE, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_features_exclude_the_target(synthetic_frame):
    features = build_features(synthetic_frame)
    assert config.TARGET not in features.columns


def test_features_have_no_missing_or_infinite_values(synthetic_frame):
    features = build_features(synthetic_frame)
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_features_do_not_depend_on_the_target(synthetic_frame):
    """Changing the target must not change any feature."""
    altered = synthetic_frame.copy()
    altered[config.TARGET] = altered[config.TARGET] + 10.0
    assert build_features(synthetic_frame).equals(build_features(altered))
