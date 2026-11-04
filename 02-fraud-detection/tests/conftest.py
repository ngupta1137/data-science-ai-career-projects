import pytest

from fraud_detection.synthetic import make_synthetic_fraud


@pytest.fixture
def synthetic_frame():
    return make_synthetic_fraud()
