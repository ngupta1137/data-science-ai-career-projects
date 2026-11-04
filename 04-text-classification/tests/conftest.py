import pytest

from text_classification.synthetic import make_synthetic_text


@pytest.fixture
def synthetic_frame():
    return make_synthetic_text()
