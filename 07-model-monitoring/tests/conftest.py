import numpy as np
import pytest

from drift_monitor import data


@pytest.fixture(scope="session")
def ref():
    return data.make_reference(42)


@pytest.fixture
def rng():
    return np.random.default_rng(7)
