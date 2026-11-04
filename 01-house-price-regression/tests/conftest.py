import pytest

from house_price.synthetic import make_synthetic_housing


@pytest.fixture
def synthetic_frame():
    return make_synthetic_housing()
