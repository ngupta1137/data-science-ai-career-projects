import numpy as np
import pandas as pd
import pytest

from fairness_audit import config, data


def test_group_share_and_base_rates():
    df = data.generate(np.random.default_rng(0), 200_000)
    assert df["g"].mean() == pytest.approx(config.MINORITY_SHARE, abs=0.01)
    by = df.groupby("g")["y"].mean()
    assert by[1] < by[0] - 0.05  # group B has the lower base rate


def test_group_b_second_measurement_is_noisier():
    df = data.generate(np.random.default_rng(1), 200_000)
    resid = (df["x2"] - df["x1"]).groupby(df["g"]).std()
    assert resid[1] > resid[0] * 1.2


def test_proxy_carries_group_and_no_direct_outcome_signal():
    df = data.generate(np.random.default_rng(2), 200_000)
    assert df.groupby("g")["x3"].mean().diff().iloc[-1] == pytest.approx(
        config.PROXY_STRENGTH, abs=0.02
    )
    within = df[df["g"] == 0]
    assert abs(np.corrcoef(within["x3"], within["y"])[0, 1]) < 0.01


def test_splits_are_independent_and_sized():
    s = data.make_splits(5)
    assert (len(s["fit"]), len(s["dev"]), len(s["test"])) == (
        config.N_FIT,
        config.N_DEV,
        config.N_TEST,
    )
    assert not s["dev"].head(50).equals(s["test"].head(50))
    assert data.make_splits(5)["test"].equals(s["test"])


def test_validate_rejects_bad_frames():
    ok = data.make_splits(1)["dev"]
    data.validate(ok)
    with pytest.raises(data.DataValidationError):
        data.validate(ok.drop(columns=["x3"]))
    bad = ok.copy()
    bad["g"] = 0
    with pytest.raises(data.DataValidationError):
        data.validate(bad)
    bad = ok.copy()
    bad.loc[bad.index[0], "x1"] = np.nan
    with pytest.raises(data.DataValidationError):
        data.validate(bad)
    one_outcome = ok.copy()
    one_outcome.loc[one_outcome["g"] == 1, "y"] = 1
    with pytest.raises(data.DataValidationError):
        data.validate(one_outcome)
    assert isinstance(ok, pd.DataFrame)


def test_fingerprint_matches_committed_file_and_detects_mismatch(tmp_path):
    assert data.verify_fingerprint() == data.fingerprint()
    bad = tmp_path / "CHECKSUMS.txt"
    bad.write_text("0" * 64 + "  x\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(bad)
