import pytest

from touch_attribution import config, data


def test_journey_generation_schema():
    ds = data.generate_journeys(n_journeys=100, seed=42)
    assert len(ds) == 100
    assert 0 < ds.conversions < 100
    assert set(ds.channel_touch_counts.keys()) == set(config.CHANNELS)
    for j in ds.journeys:
        assert len(j.path) >= 1
        assert j.converted in (0, 1)
        assert 0.0 < j.conversion_prob < 1.0
        assert 0.0 < j.organic_prob < 1.0
        assert set(j.true_channel_lifts.keys()) == set(config.CHANNELS)


def test_validation_catches_invalid_data():
    ds = data.generate_journeys(n_journeys=50, seed=1)

    # Corrupt total journeys
    ds.total_journeys = 0
    with pytest.raises(data.DataValidationError, match="at least one journey"):
        data.validate(ds)
    ds.total_journeys = 50

    # Corrupt conversions count
    ds.conversions = 100
    with pytest.raises(data.DataValidationError, match="cannot exceed total journeys"):
        data.validate(ds)
    ds.conversions = 15

    # Corrupt a journey path
    bad_journey = data.Journey(
        user_id=999,
        path=[],
        converted=1,
        conversion_prob=0.5,
        organic_prob=0.1,
        true_channel_lifts=dict.fromkeys(config.CHANNELS, 0.1),
    )
    ds.journeys.append(bad_journey)
    with pytest.raises(data.DataValidationError, match="empty touchpoint path"):
        data.validate(ds)


def test_true_causal_credit_sums_to_one():
    ds = data.generate_journeys(n_journeys=200, seed=42)
    credits = data.calculate_true_causal_credit(ds)
    assert set(credits.keys()) == set(config.CHANNELS)
    assert sum(credits.values()) == pytest.approx(1.0, rel=1e-5)
    for val in credits.values():
        assert val > 0.0


def test_organic_probability_bounds():
    ds = data.generate_journeys(n_journeys=100, seed=42)
    for j in ds.journeys:
        # With active marketing touches, total conversion prob must be >= organic baseline
        assert j.conversion_prob >= j.organic_prob - 1e-9
