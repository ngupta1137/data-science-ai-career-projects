import numpy as np
import pytest

from credit_score import data, models


def test_shapes_and_validation(rng):
    a = data.generate(rng, 2000)
    data.validate(a)
    assert a.x.shape == (2000, 5)


def test_old_policy_approves_lower_risk(rng):
    a = data.generate(rng, 20000)
    assert a.y[a.old_approve == 1].mean() < a.y[a.old_approve == 0].mean()


def test_random_sample_is_representative(rng):
    a = data.generate(rng, 40000)
    r = a.random_approve == 1
    assert a.y[r].mean() == pytest.approx(a.y.mean(), abs=0.03)


def test_approved_are_lower_risk_than_their_features_suggest(rng):
    # selection on the unrecorded trait: same features, lower default among approved
    a = data.generate(rng, 60000)
    band = (np.abs(a.x[:, 0]) < 0.3) & (np.abs(a.x[:, 1]) < 0.3)
    assert a.y[band & (a.old_approve == 1)].mean() < a.y[band & (a.old_approve == 0)].mean()


def test_models_fit_and_predict_probabilities(rng):
    h = data.generate(rng, 6000)
    for cls in models.MODELS.values():
        p = cls().fit(h).predict(h.x[:50])
        assert ((p > 0) & (p < 1)).all()


def test_validate_rejects_bad_labels(rng):
    a = data.generate(rng, 500)
    a.y[0] = 2
    with pytest.raises(data.DataValidationError):
        data.validate(a)


def test_fingerprint_is_stable():
    assert data.fingerprint() == data.fingerprint()
