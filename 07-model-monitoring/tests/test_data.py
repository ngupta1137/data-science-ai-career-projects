import numpy as np
import pytest

from drift_monitor import config, data


def test_shapes_and_types(rng):
    s = data.draw_stream(rng, "null", n_weeks=10, size=50, onset=5)
    assert s["x"].shape == (10, 50, config.N_NUM)
    assert s["cat"].shape == s["y"].shape == (10, 50)
    assert set(np.unique(s["cat"])) <= {0, 1, 2}


def test_covariate_shift_starts_at_onset(rng):
    s = data.draw_stream(rng, "covariate_abrupt", n_weeks=40, size=4000, onset=20)
    means = s["x"][:, :, 0].mean(axis=1)
    assert abs(means[:20].mean()) < 0.05
    assert means[20:].mean() == pytest.approx(1.0, abs=0.05)
    assert abs(s["x"][:, :, 1].mean()) < 0.05  # other features untouched


def test_gradual_shift_ramps(rng):
    s = data.draw_stream(rng, "covariate_gradual", n_weeks=40, size=4000, onset=10)
    means = s["x"][:, :, 0].mean(axis=1)
    assert means[10] < means[16] < means[25]
    assert means[25] == pytest.approx(1.0, abs=0.06)


def test_variance_and_category_shift(rng):
    v = data.draw_stream(rng, "variance_shift", n_weeks=20, size=4000, onset=10)
    assert v["x"][10:, :, 1].std() == pytest.approx(1.8, rel=0.05)
    assert v["x"][:10, :, 1].std() == pytest.approx(1.0, rel=0.05)
    c = data.draw_stream(rng, "category_shift", n_weeks=20, size=4000, onset=10)
    assert (c["cat"][10:] == 2).mean() == pytest.approx(0.5, abs=0.03)
    assert (c["cat"][:10] == 2).mean() == pytest.approx(0.2, abs=0.03)


def test_concept_drift_changes_labels_but_not_inputs(rng):
    s = data.draw_stream(rng, "concept_drift", n_weeks=20, size=6000, onset=10)
    x0, y = s["x"][:, :, 0], s["y"]
    before = np.corrcoef(x0[:10].ravel(), y[:10].ravel())[0, 1]
    after = np.corrcoef(x0[10:].ravel(), y[10:].ravel())[0, 1]
    assert before > 0.15 and after < 0
    assert abs(x0[10:].mean()) < 0.05  # the inputs did not move


def test_unknown_scenario_kind_fails(rng, monkeypatch):
    monkeypatch.setitem(config.SCENARIOS, "bad", {"kind": "mystery"})
    with pytest.raises(ValueError):
        data.draw_stream(rng, "bad")


def test_reference_model_is_sound(ref):
    assert 0.75 < ref.ref_auc < 0.9
    assert ref.coef[0] > ref.coef[3] > 0 > ref.coef[2]
    assert ref.props.shape == (config.N_NUM, data.N_BINS)
    assert np.allclose(ref.props, 0.1, atol=0.01)


def test_fingerprint_matches_committed_file_and_detects_mismatch(tmp_path):
    assert data.verify_fingerprint() == data.fingerprint()
    bad = tmp_path / "CHECKSUMS.txt"
    bad.write_text("0" * 64 + "  x\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(bad)
