import pytest

from search_ranking import config, data


def test_generate_ranking_data_shapes():
    n_q = 50
    ds = data.generate_ranking_data(n_queries=n_q, seed=42)
    assert len(ds) == n_q
    assert ds.total_queries == n_q
    assert ds.total_impressions == n_q * config.CANDIDATES_PER_QUERY
    assert ds.total_clicks > 0
    assert len(ds.clicks_by_position) == config.CANDIDATES_PER_QUERY

    # Check candidates per query
    for q in ds.queries:
        assert len(q.candidates) == config.CANDIDATES_PER_QUERY
        assert q.query_type in config.QUERY_PROBS
        positions = [c.pos for c in q.candidates]
        assert sorted(positions) == list(range(1, config.CANDIDATES_PER_QUERY + 1))


def test_examination_curve_decay():
    curve = data.get_examination_curve(k=10, eta=config.POSITION_DECAY_ETA)
    assert len(curve) == 10
    assert curve[0] == 1.0  # Normalized to 1.0 at pos 1
    # Strictly monotonically decreasing
    for i in range(len(curve) - 1):
        assert curve[i] > curve[i + 1]
    assert curve[-1] < 0.10  # Deep drop at position 10


def test_deterministic_generation():
    ds1 = data.generate_ranking_data(n_queries=30, seed=123)
    ds2 = data.generate_ranking_data(n_queries=30, seed=123)
    assert ds1.sha256 == ds2.sha256
    assert ds1.total_clicks == ds2.total_clicks
    assert ds1.clicks_by_position == ds2.clicks_by_position


def test_position_click_concentration():
    ds = data.generate_ranking_data(n_queries=200, seed=42)
    # Position 1 must have substantially more clicks than position 10
    c1 = ds.clicks_by_position[1]
    c10 = ds.clicks_by_position[10]
    assert c1 > 5 * c10


def test_validation_errors():
    ds = data.generate_ranking_data(n_queries=10, seed=42)

    # Corrupt total impressions
    ds.total_impressions = 999
    with pytest.raises(data.DataValidationError, match="Total impressions mismatch"):
        data.validate(ds)
    ds.total_impressions = ds.total_queries * config.CANDIDATES_PER_QUERY

    # Corrupt zero queries
    ds.total_queries = 0
    with pytest.raises(data.DataValidationError, match="at least one query"):
        data.validate(ds)
    ds.total_queries = len(ds.queries)

    # Corrupt candidates list
    q0 = ds.queries[0]
    orig_cand = q0.candidates
    q0.candidates = orig_cand[:-1]
    with pytest.raises(data.DataValidationError, match="expected 10"):
        data.validate(ds)
    q0.candidates = orig_cand
