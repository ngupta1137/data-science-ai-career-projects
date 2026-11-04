import numpy as np
import pytest

from fraud_detection import config, data


def test_valid_frame_passes(synthetic_frame):
    data.validate_frame(synthetic_frame)


def test_missing_column_fails(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="missing columns"):
        data.validate_frame(synthetic_frame.drop(columns=["Amount"]))


def test_missing_values_fail(synthetic_frame):
    synthetic_frame.loc[0, "V3"] = np.nan
    with pytest.raises(data.DataValidationError, match="V3"):
        data.validate_frame(synthetic_frame)


def test_bad_label_fails(synthetic_frame):
    synthetic_frame.loc[0, config.TARGET] = 2
    with pytest.raises(data.DataValidationError, match="Class"):
        data.validate_frame(synthetic_frame)


def test_negative_amount_fails(synthetic_frame):
    synthetic_frame.loc[0, "Amount"] = -1.0
    with pytest.raises(data.DataValidationError, match="Amount"):
        data.validate_frame(synthetic_frame)


def test_no_fraud_fails(synthetic_frame):
    synthetic_frame[config.TARGET] = 0
    with pytest.raises(data.DataValidationError, match="no fraud"):
        data.validate_frame(synthetic_frame)


def test_expected_counts_are_enforced(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="expected"):
        data.validate_frame(synthetic_frame, expected_rows=config.EXPECTED_ROWS)
    with pytest.raises(data.DataValidationError, match="fraud rows"):
        data.validate_frame(synthetic_frame, expected_fraud=config.EXPECTED_FRAUD)


def test_all_problems_are_reported_together(synthetic_frame):
    synthetic_frame.loc[0, "Amount"] = -1.0
    synthetic_frame.loc[1, "Time"] = -5.0
    with pytest.raises(data.DataValidationError) as info:
        data.validate_frame(synthetic_frame)
    assert "Amount" in str(info.value) and "Time" in str(info.value)


def test_checksum_mismatch_is_detected(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    checks = tmp_path / "CHECKSUMS.txt"
    digest = data.write_checksum(csv, checks)
    assert data.verify_checksum(csv, checks) == digest
    csv.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(data.DataValidationError, match="checksum mismatch"):
        data.verify_checksum(csv, checks)
