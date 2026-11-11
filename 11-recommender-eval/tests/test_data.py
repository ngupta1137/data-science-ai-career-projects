import numpy as np
import pytest

from recsys_eval import config, data


@pytest.fixture(scope="module")
def world_events():
    w = data.make_world(np.random.default_rng(3))
    return w, data.simulate_events(np.random.default_rng(4), w, 60000)


def test_events_are_valid(world_events):
    data.validate(*world_events)


def test_no_click_before_release(world_events):
    w, ev = world_events
    assert (ev.time >= w.release[ev.item]).all()


def test_taste_drifts_over_time(world_events):
    w, _ = world_events
    a0, a1 = w.affinity_matrix(0.0), w.affinity_matrix(100.0)
    assert not np.allclose(a0, a1)
    assert np.corrcoef(a0.ravel(), a1.ravel())[0, 1] < 0.99


def test_affinity_is_unrelated_to_item_popularity(world_events):
    w, _ = world_events
    assert abs(np.corrcoef(w.pop, w.affinity_matrix(50.0).mean(axis=0))[0, 1]) < 0.15


def test_stronger_exposure_concentrates_clicks():
    def top_share(gamma):
        w = data.make_world(np.random.default_rng(5), exposure_strength=gamma)
        ev = data.simulate_events(np.random.default_rng(6), w, 60000)
        counts = np.sort(np.bincount(ev.item, minlength=w.n_items))[::-1]
        return counts[:30].sum() / counts.sum()

    assert top_share(2.5) > top_share(0.0) + 0.2


def test_click_rate_matches_the_model():
    w = data.make_world(np.random.default_rng(7), exposure_strength=0.0)
    ev = data.simulate_events(np.random.default_rng(8), w, 100000)
    rate = len(ev) / 100000
    mean_prob = w.like_prob_matrix(50.0).mean()
    assert rate == pytest.approx(mean_prob, abs=0.04)


def test_deterministic():
    a = data.simulate_events(
        np.random.default_rng(9), data.make_world(np.random.default_rng(1)), 5000
    )
    b = data.simulate_events(
        np.random.default_rng(9), data.make_world(np.random.default_rng(1)), 5000
    )
    assert np.array_equal(a.item, b.item) and np.array_equal(a.time, b.time)


def test_validate_rejects_unsorted_and_early_clicks(world_events):
    w, ev = world_events
    bad = data.Events(ev.user[::-1], ev.item[::-1], ev.time[::-1])
    with pytest.raises(data.DataValidationError):
        data.validate(w, bad)
    early = data.Events(ev.user.copy(), ev.item.copy(), ev.time.copy())
    j = int(np.argmax(w.release[early.item]))
    early.time[:] = 0.0
    assert w.release[early.item[j]] > 0
    with pytest.raises(data.DataValidationError):
        data.validate(w, early)


def test_fingerprint_roundtrip_and_mismatch(tmp_path):
    f = tmp_path / "sum.txt"
    digest = data.write_fingerprint(f)
    assert data.verify_fingerprint(f) == digest
    f.write_text("0" * 64 + "  x\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(f)


def test_committed_fingerprint_matches_generator():
    assert data.verify_fingerprint() == data.fingerprint()


def test_config_windows_are_ordered():
    assert 0 < config.VAL_START < config.TEST_START < config.HORIZON


def test_popular_items_are_liked_less_by_users_who_have_not_clicked_them():
    """The mechanism behind popularity scoring below random once clicked items are removed."""
    w = data.make_world(np.random.default_rng(3))
    ev = data.simulate_events(np.random.default_rng(4), w)
    past = ev.select(ev.time < 85)
    pop = np.bincount(past.item, minlength=w.n_items)
    top = np.argsort(-pop)[:10]
    like = w.like_prob_matrix(92.5)[:, top]
    clicked = np.zeros((w.n_users, w.n_items), bool)
    clicked[past.user, past.item] = True
    unseen = ~clicked[:, top]
    overall = w.like_prob_matrix(92.5)[:, w.release <= 100].mean()
    assert abs(like.mean() - overall) < 0.05  # popular items are not liked more or less on average
    assert like[unseen].mean() < like.mean() - 0.03  # but those still unclicked are liked less
