import numpy as np
import pytest

from claims_pricing import config, data


def test_generator_shapes_and_bounds(rng):
    p = data.generate(rng, 100)
    data.validate(p)
    assert len(p) == 100
    assert p.features().shape == (100, 5)
    assert (p.exposure > 0.0).all() and (p.exposure <= 1.0).all()
    assert (p.driver_age >= 18.0).all() and (p.driver_age <= 75.0).all()
    assert (p.annual_mileage >= 4000.0).all()
    assert (p.vehicle_age >= 0.0).all()
    assert (p.vehicle_value > 0.0).all()
    assert set(np.unique(p.urban)).issubset({0, 1})
    assert (p.claim_count >= 0).all()
    assert (p.claim_amount >= 0.0).all()
    assert ((p.claim_count == 0) == (p.claim_amount == 0.0)).all()


def test_validation_detects_bad_data(rng):
    p = data.generate(rng, 50)
    # Negative count
    bad_count = p.claim_count.copy()
    bad_count[0] = -1
    bad_p = data.Policies(
        driver_age=p.driver_age,
        annual_mileage=p.annual_mileage,
        vehicle_age=p.vehicle_age,
        vehicle_value=p.vehicle_value,
        urban=p.urban,
        exposure=p.exposure,
        claim_count=bad_count,
        claim_amount=p.claim_amount,
        true_freq=p.true_freq,
        true_sev=p.true_sev,
    )
    with pytest.raises(data.DataValidationError):
        data.validate(bad_p)


def test_fingerprint_matches_checksum():
    digest = data.verify_fingerprint(config.CHECKSUM_FILE)
    assert len(digest) == 64
