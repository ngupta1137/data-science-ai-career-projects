import pytest

from demand_forecast.synthetic import make_synthetic_demand


@pytest.fixture
def synthetic_frame():
    return make_synthetic_demand()
