import numpy as np
import pytest

from house_price import config, data


def test_valid_frame_passes(synthetic_frame):
    data.validate_frame(synthetic_frame)


def test_missing_column_fails(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="missing columns"):
        data.validate_frame(synthetic_frame.drop(columns=["MedInc"]))


def test_missing_values_fail(synthetic_frame):
    synthetic_frame.loc[0, "HouseAge"] = np.nan
    with pytest.raises(data.DataValidationError, match="HouseAge"):
        data.validate_frame(synthetic_frame)


def test_out_of_range_latitude_fails(synthetic_frame):
    synthetic_frame.loc[0, "Latitude"] = 60.0
    with pytest.raises(data.DataValidationError, match="Latitude"):
        data.validate_frame(synthetic_frame)


def test_target_above_cap_fails(synthetic_frame):
    synthetic_frame.loc[0, config.TARGET] = config.TARGET_CAP + 1.0
    with pytest.raises(data.DataValidationError, match="cap"):
        data.validate_frame(synthetic_frame)


def test_nonpositive_income_fails(synthetic_frame):
    synthetic_frame.loc[0, "MedInc"] = 0.0
    with pytest.raises(data.DataValidationError, match="MedInc"):
        data.validate_frame(synthetic_frame)


def test_row_count_is_enforced_when_requested(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="expected"):
        data.validate_frame(synthetic_frame, expected_rows=config.EXPECTED_ROWS)


def test_all_problems_are_reported_together(synthetic_frame):
    synthetic_frame.loc[0, "Latitude"] = 60.0
    synthetic_frame.loc[1, "MedInc"] = -1.0
    with pytest.raises(data.DataValidationError) as info:
        data.validate_frame(synthetic_frame)
    message = str(info.value)
    assert "Latitude" in message and "MedInc" in message


def test_checksum_mismatch_is_detected(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    checks = tmp_path / "CHECKSUMS.txt"
    digest = data.write_checksum(csv, checks)
    assert data.verify_checksum(csv, checks) == digest
    csv.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(data.DataValidationError, match="checksum mismatch"):
        data.verify_checksum(csv, checks)
