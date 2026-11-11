import numpy as np
import pytest

from experiment_analysis import config, data


@pytest.mark.parametrize("dist", ["normal", "lognormal"])
def test_draws_are_standardized(dist):
    rng = np.random.default_rng(0)
    d = data.draw(rng, reps=1, n=400_000, dist=dist, rho=0.0)
    assert abs(d["y_c"].mean()) < 0.02 and d["y_c"].var() == pytest.approx(1.0, rel=0.1)


def test_correlation_matches_rho():
    rng = np.random.default_rng(1)
    d = data.draw(rng, reps=1, n=100_000, rho=0.6)
    assert np.corrcoef(d["x_c"][0], d["y_c"][0])[0, 1] == pytest.approx(0.6, abs=0.01)


def test_effect_and_contamination_shift_only_the_treated_arm():
    rng = np.random.default_rng(2)
    d = data.draw(rng, reps=1, n=200_000, effect=0.3, contamination=0.5)
    assert d["y_t"].mean() - d["y_c"].mean() == pytest.approx(0.3, abs=0.02)
    assert d["x_t"].mean() - d["x_c"].mean() == pytest.approx(0.5, abs=0.02)


def test_lognormal_is_skewed():
    rng = np.random.default_rng(3)
    x = data.draw(rng, reps=1, n=200_000, dist="lognormal")["y_c"][0]
    assert ((x - x.mean()) ** 3).mean() > 1.0


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        data.draw(rng, 1, 10, dist="cauchy")
    with pytest.raises(ValueError):
        data.draw(rng, 1, 10, rho=1.0)


def test_draw_metrics_shapes_and_effects():
    rng = np.random.default_rng(4)
    t, c = data.draw_metrics(rng, reps=2, n=50_000, effects=[0.5, 0.0])
    assert t.shape == c.shape == (2, 2, 50_000)
    assert t[:, 0].mean() - c[:, 0].mean() == pytest.approx(0.5, abs=0.02)
    assert abs(t[:, 1].mean() - c[:, 1].mean()) < 0.02


def test_fingerprint_is_deterministic_and_matches_committed_file():
    assert data.fingerprint() == data.fingerprint()
    assert data.verify_fingerprint() == data.fingerprint()
    assert config.CHECKSUM_FILE.read_text(encoding="utf-8").split()[0] == data.fingerprint()


def test_verify_fingerprint_detects_mismatch(tmp_path):
    bad = tmp_path / "CHECKSUMS.txt"
    bad.write_text("0" * 64 + "  generator_probe(seed=0)\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(bad)
